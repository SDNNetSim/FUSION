"""
Standard slicing pipeline implementation.

This module provides StandardSlicingPipeline for dividing large requests
across multiple lightpaths when a single allocation cannot accommodate
the full bandwidth.

Phase: P3.1 - Pipeline Factory Scaffolding
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fusion.domain.config import SimulationConfig
    from fusion.domain.network_state import NetworkState
    from fusion.domain.request import Request
    from fusion.domain.results import SlicingResult
    from fusion.interfaces.pipelines import SNRPipeline, SpectrumPipeline

logger = logging.getLogger(__name__)


class StandardSlicingPipeline:
    """
    Standard request slicing pipeline.

    Slices large requests across multiple lightpaths when single
    allocation fails due to spectrum fragmentation or modulation
    limitations.

    Slicing Strategy:
        1. Start with minimum slices (2)
        2. Divide bandwidth evenly across slices
        3. Check if each slice can be allocated
        4. Increment slice count until success or max_slices reached

    Attributes:
        _config: Simulation configuration
        _max_slices: Maximum number of slices allowed

    Usage:
        >>> pipeline = StandardSlicingPipeline(config)
        >>> result = pipeline.try_slice(
        ...     request, path, "QPSK", 400, network_state
        ... )
        >>> if result.success:
        ...     print(f"Sliced into {result.num_slices} lightpaths")

    Phase: P3.1 - Pipeline Factory Scaffolding
    """

    def __init__(self, config: SimulationConfig) -> None:
        """
        Initialize slicing pipeline.

        Args:
            config: Simulation configuration
        """
        self._config = config
        self._max_slices = getattr(config, "max_slices", 4)

        logger.debug(f"StandardSlicingPipeline initialized (max_slices={self._max_slices})")

    def try_slice(
        self,
        request: Request,
        path: list[str],
        modulation: str,
        bandwidth_gbps: int,
        network_state: NetworkState,
        *,
        max_slices: int | None = None,
        spectrum_pipeline: SpectrumPipeline | None = None,
        snr_pipeline: SNRPipeline | None = None,
    ) -> SlicingResult:
        """
        Attempt to slice request into multiple smaller allocations.

        Tries progressively more slices until all can be allocated
        or max_slices is reached.

        Args:
            request: The request being processed
            path: Route to use for slicing
            modulation: Modulation format for slices
            bandwidth_gbps: Total bandwidth to slice
            network_state: Current network state
            max_slices: Override config max_slices (optional)
            spectrum_pipeline: Pipeline for finding spectrum (optional)
            snr_pipeline: Pipeline for SNR validation (optional)

        Returns:
            SlicingResult indicating success/failure and slice details

        Notes:
            - When spectrum_pipeline is None, only checks feasibility
            - When spectrum_pipeline is provided, attempts actual allocation
            - Each slice gets equal bandwidth (bandwidth_gbps // num_slices)
        """
        from fusion.domain.results import SlicingResult

        limit = max_slices or self._max_slices

        if limit < 2:
            logger.debug("Slicing disabled (max_slices < 2)")
            return SlicingResult.failed()

        # Try progressively more slices
        for num_slices in range(2, limit + 1):
            slice_bandwidth = bandwidth_gbps // num_slices

            if slice_bandwidth <= 0:
                logger.debug(f"Slice bandwidth too small at {num_slices} slices")
                continue

            # Check if all slices can be allocated
            if spectrum_pipeline is not None:
                # Attempt actual allocation
                result = self._try_allocate_slices(
                    request=request,
                    path=path,
                    modulation=modulation,
                    slice_bandwidth=slice_bandwidth,
                    num_slices=num_slices,
                    network_state=network_state,
                    spectrum_pipeline=spectrum_pipeline,
                    snr_pipeline=snr_pipeline,
                )
                if result is not None:
                    return result
            else:
                # Feasibility check only
                if self._check_slice_feasibility(
                    path=path,
                    modulation=modulation,
                    slice_bandwidth=slice_bandwidth,
                    num_slices=num_slices,
                    network_state=network_state,
                ):
                    logger.debug(
                        f"Slicing feasible: {num_slices} slices of {slice_bandwidth} Gbps"
                    )
                    return SlicingResult(
                        success=True,
                        num_slices=num_slices,
                        slice_bandwidth_gbps=slice_bandwidth,
                        lightpaths_created=(),  # Not allocated, just feasibility
                        total_bandwidth_gbps=num_slices * slice_bandwidth,
                    )

        logger.debug(f"Slicing failed: could not allocate up to {limit} slices")
        return SlicingResult.failed()

    def _check_slice_feasibility(
        self,
        path: list[str],
        modulation: str,
        slice_bandwidth: int,
        num_slices: int,
        network_state: NetworkState,
    ) -> bool:
        """
        Check if slicing is feasible (without actual allocation).

        This is a simplified check that estimates whether the path
        has enough free spectrum for all slices.

        Args:
            path: Route to use
            modulation: Modulation format
            slice_bandwidth: Bandwidth per slice
            num_slices: Number of slices
            network_state: Current network state

        Returns:
            True if slicing appears feasible
        """
        # Calculate slots needed per slice
        slots_per_slice = self._calculate_slots_needed(slice_bandwidth, modulation)
        total_slots_needed = slots_per_slice * num_slices

        # Get available slots on path
        available_slots = self._get_available_slots_on_path(path, network_state)

        # Simple feasibility: enough total slots
        # Note: This doesn't account for contiguity requirements per slice
        return available_slots >= total_slots_needed

    def _try_allocate_slices(
        self,
        request: Request,
        path: list[str],
        modulation: str,
        slice_bandwidth: int,
        num_slices: int,
        network_state: NetworkState,
        spectrum_pipeline: SpectrumPipeline,
        snr_pipeline: SNRPipeline | None,
    ) -> SlicingResult | None:
        """
        Attempt to actually allocate all slices.

        Args:
            request: The request being processed
            path: Route to use
            modulation: Modulation format (may be empty, will determine from config)
            slice_bandwidth: Bandwidth per slice
            num_slices: Number of slices
            network_state: Current network state
            spectrum_pipeline: Pipeline for spectrum assignment
            snr_pipeline: Pipeline for SNR validation (optional)

        Returns:
            SlicingResult if successful, None if any slice fails
        """
        from fusion.domain.results import SlicingResult

        allocated_lightpaths: list[int] = []

        # ALWAYS determine modulation based on SLICE bandwidth (matches legacy behavior)
        # Legacy slicing uses mod_per_bw[slice_bandwidth] to find modulation,
        # NOT the routing modulation for the full request bandwidth.
        # This is critical for spectral efficiency - a 50 Gbps slice may use
        # a higher-order modulation (16-QAM) than a 100 Gbps request (QPSK).
        slice_modulation = self._get_modulation_for_slice(slice_bandwidth, path, network_state)
        if not slice_modulation:
            logger.debug(f"No valid modulation for slice bandwidth {slice_bandwidth}")
            return None

        try:
            for i in range(num_slices):
                # Find spectrum for this slice
                spectrum_result = spectrum_pipeline.find_spectrum(
                    path=path,
                    modulation=slice_modulation,
                    bandwidth_gbps=slice_bandwidth,
                    network_state=network_state,
                )

                if not spectrum_result.is_free:
                    logger.debug(f"Slice {i + 1}/{num_slices} failed: no spectrum")
                    # Rollback already allocated slices
                    self.rollback_slices(allocated_lightpaths, network_state)
                    return None

                # Create lightpath for this slice
                # Calculate path weight for lightpath
                path_weight_km = self._calculate_path_weight(path, network_state)

                lightpath = network_state.create_lightpath(
                    path=path,
                    start_slot=spectrum_result.start_slot,
                    end_slot=spectrum_result.end_slot,
                    core=spectrum_result.core,
                    band=spectrum_result.band,
                    modulation=slice_modulation,
                    bandwidth_gbps=slice_bandwidth,
                    path_weight_km=path_weight_km,
                    guard_slots=getattr(self._config, "guard_slots", 0),
                )
                lightpath_id = lightpath.lightpath_id

                # Link request to lightpath and update remaining bandwidth
                lightpath.request_allocations[request.request_id] = slice_bandwidth
                lightpath.remaining_bandwidth_gbps -= slice_bandwidth
                request.lightpath_ids.append(lightpath_id)

                # Validate SNR if pipeline provided
                if snr_pipeline is not None:
                    lightpath = network_state.get_lightpath(lightpath_id)
                    if lightpath is not None:
                        snr_result = snr_pipeline.validate(lightpath, network_state)
                        if not snr_result.passed:
                            logger.debug(
                                f"Slice {i + 1}/{num_slices} failed SNR validation"
                            )
                            # Rollback this and all previous slices
                            allocated_lightpaths.append(lightpath_id)
                            self.rollback_slices(allocated_lightpaths, network_state)
                            return None

                allocated_lightpaths.append(lightpath_id)

            # All slices allocated successfully
            return SlicingResult.sliced(
                num_slices=num_slices,
                slice_bandwidth=slice_bandwidth,
                lightpath_ids=allocated_lightpaths,
            )

        except Exception as e:
            logger.error(f"Error during slice allocation: {e}")
            self.rollback_slices(allocated_lightpaths, network_state)
            return None

    def rollback_slices(
        self,
        lightpath_ids: list[int],
        network_state: NetworkState,
    ) -> None:
        """
        Rollback all slices on failure.

        Called when partial slice allocation fails and all created
        slices must be released.

        Args:
            lightpath_ids: IDs of lightpaths to release
            network_state: Network state to modify
        """
        if not lightpath_ids:
            return

        logger.debug(f"Rolling back {len(lightpath_ids)} sliced lightpaths")

        for lp_id in lightpath_ids:
            try:
                network_state.release_lightpath(lp_id)
            except Exception as e:
                logger.warning(f"Error rolling back lightpath {lp_id}: {e}")

    def _calculate_slots_needed(
        self,
        bandwidth_gbps: int,
        modulation: str,
    ) -> int:
        """Calculate spectrum slots needed for given bandwidth and modulation."""
        # Bits per symbol for each modulation
        bits_per_symbol = {
            "BPSK": 1,
            "QPSK": 2,
            "8-QAM": 3,
            "16-QAM": 4,
            "32-QAM": 5,
            "64-QAM": 6,
        }

        bps = bits_per_symbol.get(modulation, 2)  # Default to QPSK

        # Assume 12.5 GHz slot width, symbol rate ~= slot bandwidth
        # Simplified calculation: slots = bandwidth / (slot_width * bits_per_symbol)
        slot_bandwidth_gbps = 12.5 * bps  # Rough approximation

        slots = max(1, int(bandwidth_gbps / slot_bandwidth_gbps) + 1)

        # Add guard band
        guard_slots = getattr(self._config, "guard_slots", 1)
        return slots + guard_slots

    def _get_available_slots_on_path(
        self,
        path: list[str],
        network_state: NetworkState,
    ) -> int:
        """Get total available slots along path (minimum across all links)."""
        if len(path) < 2:
            return 0

        min_available = float("inf")

        for i in range(len(path) - 1):
            link = (path[i], path[i + 1])
            available = network_state.get_link_available_slots(link)
            min_available = min(min_available, available)

        return int(min_available) if min_available != float("inf") else 0

    def _calculate_path_weight(
        self,
        path: list[str],
        network_state: NetworkState,
    ) -> float:
        """Calculate total path weight (distance) in km."""
        total = 0.0
        topology = network_state.topology

        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if topology.has_edge(u, v):
                edge_data = topology.edges[u, v]
                total += float(edge_data.get("length", edge_data.get("weight", 0.0)))

        return total

    def _get_modulation_for_slice(
        self,
        slice_bandwidth: int,
        path: list[str],
        network_state: NetworkState,
    ) -> str:
        """
        Determine valid modulation for slice bandwidth and path.

        Looks up mod_per_bw for the slice bandwidth and finds a modulation
        that can reach the path length.

        Args:
            slice_bandwidth: Bandwidth of the slice in Gbps
            path: Route to use
            network_state: Network state for path length calculation

        Returns:
            Valid modulation format name, or empty string if none found
        """
        # Calculate path length
        path_length = 0.0
        topology = network_state.topology
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if topology.has_edge(u, v):
                edge_data = topology.edges[u, v]
                path_length += float(edge_data.get("length", edge_data.get("weight", 0.0)))

        # Look up modulation formats for slice bandwidth
        mod_per_bw = getattr(self._config, "mod_per_bw", {})
        bw_key = str(slice_bandwidth)

        if bw_key not in mod_per_bw:
            logger.debug(f"No mod_per_bw entry for bandwidth {slice_bandwidth}")
            return ""

        bw_mods = mod_per_bw[bw_key]

        # Find first modulation that can reach the path
        # Sort by efficiency (highest order first for spectral efficiency)
        mod_order = ["64-QAM", "32-QAM", "16-QAM", "8-QAM", "QPSK", "BPSK"]

        for mod_name in mod_order:
            if mod_name in bw_mods:
                mod_info = bw_mods[mod_name]
                if isinstance(mod_info, dict):
                    max_length = mod_info.get("max_length", 0)
                    if max_length >= path_length:
                        logger.debug(
                            f"Slice modulation for {slice_bandwidth} Gbps: {mod_name} "
                            f"(max_length={max_length}, path_length={path_length:.1f})"
                        )
                        return mod_name

        logger.debug(
            f"No valid modulation for {slice_bandwidth} Gbps on path length {path_length:.1f}"
        )
        return ""
