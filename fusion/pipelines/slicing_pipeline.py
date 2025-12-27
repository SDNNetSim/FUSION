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
        connection_index: int | None = None,
        path_index: int = 0,
    ) -> SlicingResult:
        """
        Attempt to slice request into multiple smaller allocations.

        Uses tier-based slicing (matching Legacy behavior):
        - Iterates through bandwidth tiers from largest to smallest
        - Allocates as many slices of each tier as possible
        - Allows mixing different tier sizes (e.g., 500 + 100 = 600 Gbps)

        Args:
            request: The request being processed
            path: Route to use for slicing
            modulation: Modulation format for slices
            bandwidth_gbps: Total bandwidth to slice
            network_state: Current network state
            max_slices: Override config max_slices (optional)
            spectrum_pipeline: Pipeline for finding spectrum (optional)
            snr_pipeline: Pipeline for SNR validation (optional)
            connection_index: External routing index for pre-calculated SNR lookup
            path_index: Index of which k-path is being tried (0, 1, 2...)

        Returns:
            SlicingResult indicating success/failure and slice details

        Notes:
            - When spectrum_pipeline is None, only checks feasibility
            - When spectrum_pipeline is provided, attempts actual allocation
            - Uses tier-based slicing with mixed sizes (matches Legacy)
        """
        from fusion.domain.results import SlicingResult

        limit = max_slices or self._max_slices

        if limit < 2:
            logger.debug("Slicing disabled (max_slices < 2)")
            return SlicingResult.failed()

        if spectrum_pipeline is not None:
            # Check if we should use dynamic slicing (GSNR-based modulation selection)
            # This matches legacy behavior when dynamic_lps=True (both fixed and flex grid)
            dynamic_lps = getattr(self._config, "dynamic_lps", False)

            if dynamic_lps:
                # Use dynamic slicing mode (1 slot at a time with GSNR modulation selection)
                result = self._try_allocate_dynamic(
                    request=request,
                    path=path,
                    bandwidth_gbps=bandwidth_gbps,
                    network_state=network_state,
                    spectrum_pipeline=spectrum_pipeline,
                    snr_pipeline=snr_pipeline,
                    connection_index=connection_index,
                    path_index=path_index,
                    max_slices=limit,
                )
            else:
                # Attempt actual tier-based allocation (matches Legacy flex-grid behavior)
                result = self._try_allocate_tier_based(
                    request=request,
                    path=path,
                    bandwidth_gbps=bandwidth_gbps,
                    network_state=network_state,
                    spectrum_pipeline=spectrum_pipeline,
                    snr_pipeline=snr_pipeline,
                    connection_index=connection_index,
                    path_index=path_index,
                    max_slices=limit,
                )
            if result is not None:
                return result
        else:
            # Feasibility check only - use simple tier check
            if self._check_tier_feasibility(
                path=path,
                bandwidth_gbps=bandwidth_gbps,
                network_state=network_state,
            ):
                logger.debug(f"Tier-based slicing appears feasible for {bandwidth_gbps} Gbps")
                return SlicingResult(
                    success=True,
                    num_slices=2,  # Estimate
                    slice_bandwidth_gbps=bandwidth_gbps // 2,
                    lightpaths_created=(),
                    total_bandwidth_gbps=bandwidth_gbps,
                )

        logger.debug(f"Slicing failed: could not allocate {bandwidth_gbps} Gbps with tier-based slicing")
        return SlicingResult.failed()

    def _try_allocate_tier_based(
        self,
        request: Request,
        path: list[str],
        bandwidth_gbps: int,
        network_state: NetworkState,
        spectrum_pipeline: SpectrumPipeline,
        snr_pipeline: SNRPipeline | None,
        connection_index: int | None,
        path_index: int,
        max_slices: int,
    ) -> SlicingResult | None:
        """
        Tier-based slicing allocation (matches Legacy behavior).

        Iterates through bandwidth tiers from largest to smallest,
        allocating as many slices of each tier as possible before
        moving to smaller tiers. Allows mixing different tier sizes.

        Args:
            request: The request being processed
            path: Route to use
            bandwidth_gbps: Total bandwidth to allocate
            network_state: Current network state
            spectrum_pipeline: Pipeline for spectrum assignment
            snr_pipeline: Pipeline for SNR validation (optional)
            connection_index: External routing index
            path_index: Index of which k-path is being tried
            max_slices: Maximum number of slices allowed

        Returns:
            SlicingResult if successful, None if slicing fails
        """
        from fusion.domain.results import SlicingResult

        # Get sorted bandwidth tiers from config (descending order)
        mod_per_bw = getattr(self._config, "mod_per_bw", {})
        sorted_tiers = sorted([int(k) for k in mod_per_bw.keys()], reverse=True)

        remaining_bw = bandwidth_gbps
        allocated_lightpaths: list[int] = []
        slice_bandwidths: list[int] = []  # Track bandwidth of each slice
        failed_snr_values: list[float] = []  # Track SNR from failed attempts (Legacy compatibility)

        for tier_bw in sorted_tiers:
            # Skip tiers >= original request bandwidth (matches Legacy)
            if tier_bw >= bandwidth_gbps:
                continue

            # Skip if remaining bandwidth is less than this tier
            if remaining_bw < tier_bw:
                continue

            # Get modulations valid for this tier on this path
            tier_modulations = self._get_modulation_for_slice(tier_bw, path, network_state)
            if not tier_modulations:
                continue

            # DEBUG: Show tier being attempted for req 43
            if request.request_id == 43:
                print(f'[DEBUG-R43-SLICING] Trying tier_bw={tier_bw} mods={tier_modulations}')

            # Try to allocate as many slices of this tier as possible
            while remaining_bw >= tier_bw:
                # Check max slices limit
                if len(allocated_lightpaths) >= max_slices:
                    break

                # Find spectrum for this slice
                # Pass slice_bandwidth to get_spectrum for proper slots calculation
                spectrum_result = spectrum_pipeline.find_spectrum(
                    path=path,
                    modulation=tier_modulations,
                    bandwidth_gbps=tier_bw,
                    network_state=network_state,
                    connection_index=connection_index,
                    path_index=path_index,
                    request_id=request.request_id,
                    slice_bandwidth=tier_bw,  # Key: pass tier bandwidth for slicing
                )

                if not spectrum_result.is_free:
                    # DEBUG: Show spectrum search failure for req 43
                    if request.request_id == 43:
                        print(f'[DEBUG-R43-SLICING] No spectrum for tier_bw={tier_bw}, moving to next tier')
                    break  # Move to smaller tier

                # Create lightpath for this slice
                path_weight_km = self._calculate_path_weight(path, network_state)
                actual_modulation = spectrum_result.modulation

                # DEBUG: Show successful spectrum allocation for req 43
                if request.request_id == 43:
                    print(f'[DEBUG-R43-SLICING] Got spectrum: start={spectrum_result.start_slot} end={spectrum_result.end_slot} mod={actual_modulation} snr={spectrum_result.snr_db}')

                lightpath = network_state.create_lightpath(
                    path=path,
                    start_slot=spectrum_result.start_slot,
                    end_slot=spectrum_result.end_slot,
                    core=spectrum_result.core,
                    band=spectrum_result.band,
                    modulation=actual_modulation,
                    bandwidth_gbps=tier_bw,
                    path_weight_km=path_weight_km,
                    guard_slots=getattr(self._config, "guard_slots", 0),
                    connection_index=connection_index,
                    snr_db=spectrum_result.snr_db,  # Pass SNR for metrics tracking
                )
                lightpath_id = lightpath.lightpath_id

                # Link request to lightpath
                lightpath.request_allocations[request.request_id] = tier_bw
                lightpath.remaining_bandwidth_gbps -= tier_bw
                request.lightpath_ids.append(lightpath_id)

                # SNR RECHECK: Check if EXISTING lightpaths would be degraded by this allocation
                # Legacy only does SNR recheck if snr_recheck config is True
                # This is different from validating the new LP's SNR - we check EXISTING LPs
                snr_recheck_enabled = getattr(self._config, "snr_recheck", False)
                if snr_pipeline is not None and snr_recheck_enabled:
                    # Call recheck_affected to check existing LPs (matches legacy behavior)
                    recheck_result = snr_pipeline.recheck_affected(
                        lightpath_id, network_state, slicing_flag=True
                    )
                    if not recheck_result.all_pass:
                        logger.debug(f"Tier slice failed SNR recheck - existing LPs degraded (tier_bw={tier_bw})")
                        # Track failed SNR for Legacy compatibility
                        if spectrum_result.snr_db is not None:
                            failed_snr_values.append(spectrum_result.snr_db)
                        network_state.release_lightpath(lightpath_id)
                        break  # Move to smaller tier

                allocated_lightpaths.append(lightpath_id)
                slice_bandwidths.append(tier_bw)
                remaining_bw -= tier_bw

            # Stop if fully allocated
            if remaining_bw <= 0:
                break

        # Check if we allocated anything
        if not allocated_lightpaths:
            return None

        # Check if fully allocated
        if remaining_bw <= 0:
            total_bw = sum(slice_bandwidths)
            # Create SlicingResult directly to handle mixed tier sizes correctly
            # The factory method SlicingResult.sliced() assumes equal slice sizes
            return SlicingResult(
                success=True,
                num_slices=len(allocated_lightpaths),
                slice_bandwidth_gbps=slice_bandwidths[0] if slice_bandwidths else 0,
                lightpaths_created=tuple(allocated_lightpaths),
                total_bandwidth_gbps=total_bw,  # Actual sum, not num_slices * slice_bw
                failed_attempt_snr_values=tuple(failed_snr_values),
            )

        # Partial allocation - check if we can accept partial service
        can_partial = getattr(self._config, "can_partially_serve", False)
        k_paths = getattr(self._config, "k_paths", 3)
        on_last_path = path_index >= k_paths - 1

        if can_partial and on_last_path:
            allocated_bw = sum(slice_bandwidths)
            # Create SlicingResult directly for mixed tier sizes
            return SlicingResult(
                success=True,
                num_slices=len(allocated_lightpaths),
                slice_bandwidth_gbps=slice_bandwidths[0] if slice_bandwidths else 0,
                lightpaths_created=tuple(allocated_lightpaths),
                total_bandwidth_gbps=allocated_bw,  # Actual sum
                failed_attempt_snr_values=tuple(failed_snr_values),
            )

        # Cannot accept partial - rollback all
        self.rollback_slices(allocated_lightpaths, network_state)
        return None

    def _try_allocate_dynamic(
        self,
        request: Request,
        path: list[str],
        bandwidth_gbps: int,
        network_state: NetworkState,
        spectrum_pipeline: SpectrumPipeline,
        snr_pipeline: SNRPipeline | None,
        connection_index: int | None,
        path_index: int,
        max_slices: int,
    ) -> SlicingResult | None:
        """
        Dynamic slicing allocation (matches Legacy dynamic_lps behavior).

        For fixed-grid: Allocates 1 slot at a time, using GSNR to determine modulation/bandwidth.
        For flex-grid: Iterates through bandwidth tiers, using GSNR-based modulation selection.

        Args:
            request: The request being processed
            path: Route to use
            bandwidth_gbps: Total bandwidth to allocate
            network_state: Current network state
            spectrum_pipeline: Pipeline for spectrum assignment
            snr_pipeline: Pipeline for SNR validation (optional)
            connection_index: External routing index
            path_index: Index of which k-path is being tried
            max_slices: Maximum number of slices allowed

        Returns:
            SlicingResult if successful, None if slicing fails
        """
        from fusion.domain.results import SlicingResult

        fixed_grid = getattr(self._config, "fixed_grid", False)

        if fixed_grid:
            # Fixed-grid dynamic slicing: 1 slot at a time
            return self._try_allocate_dynamic_fixed_grid(
                request, path, bandwidth_gbps, network_state, spectrum_pipeline,
                snr_pipeline, connection_index, path_index, max_slices
            )
        else:
            # Flex-grid dynamic slicing: iterate through bandwidth tiers
            return self._try_allocate_dynamic_flex_grid(
                request, path, bandwidth_gbps, network_state, spectrum_pipeline,
                snr_pipeline, connection_index, path_index, max_slices
            )

    def _try_allocate_dynamic_fixed_grid(
        self,
        request: Request,
        path: list[str],
        bandwidth_gbps: int,
        network_state: NetworkState,
        spectrum_pipeline: SpectrumPipeline,
        snr_pipeline: SNRPipeline | None,
        connection_index: int | None,
        path_index: int,
        max_slices: int,
    ) -> SlicingResult | None:
        """Fixed-grid dynamic slicing: 1 slot at a time."""
        from fusion.domain.results import SlicingResult

        remaining_bw = bandwidth_gbps
        allocated_lightpaths: list[int] = []
        slice_bandwidths: list[int] = []
        failed_snr_values: list[float] = []

        while remaining_bw > 0 and len(allocated_lightpaths) < max_slices:
            spectrum_result = spectrum_pipeline.find_spectrum(
                path=path,
                modulation="",
                bandwidth_gbps=bandwidth_gbps,
                network_state=network_state,
                connection_index=connection_index,
                path_index=path_index,
                use_dynamic_slicing=True,
                request_id=request.request_id,
            )

            if not spectrum_result.is_free:
                break

            achieved_bw = spectrum_result.achieved_bandwidth_gbps
            if achieved_bw is None or achieved_bw <= 0:
                break

            path_weight_km = self._calculate_path_weight(path, network_state)
            lightpath = network_state.create_lightpath(
                path=path,
                start_slot=spectrum_result.start_slot,
                end_slot=spectrum_result.end_slot,
                core=spectrum_result.core,
                band=spectrum_result.band,
                modulation=spectrum_result.modulation,
                bandwidth_gbps=achieved_bw,
                path_weight_km=path_weight_km,
                guard_slots=getattr(self._config, "guard_slots", 0),
                connection_index=connection_index,
                snr_db=spectrum_result.snr_db,
            )
            lightpath_id = lightpath.lightpath_id
            lightpath.request_allocations[request.request_id] = achieved_bw
            lightpath.remaining_bandwidth_gbps -= achieved_bw
            request.lightpath_ids.append(lightpath_id)

            snr_recheck_enabled = getattr(self._config, "snr_recheck", False)
            if snr_pipeline is not None and snr_recheck_enabled:
                recheck_result = snr_pipeline.recheck_affected(
                    lightpath_id, network_state, slicing_flag=True
                )
                if not recheck_result.all_pass:
                    if spectrum_result.snr_db is not None:
                        failed_snr_values.append(spectrum_result.snr_db)
                    network_state.release_lightpath(lightpath_id)
                    break

            allocated_lightpaths.append(lightpath_id)
            slice_bandwidths.append(achieved_bw)
            remaining_bw -= achieved_bw

        if not allocated_lightpaths:
            return None

        return self._finalize_dynamic_result(
            remaining_bw, bandwidth_gbps, allocated_lightpaths, slice_bandwidths,
            failed_snr_values, path_index, network_state
        )

    def _try_allocate_dynamic_flex_grid(
        self,
        request: Request,
        path: list[str],
        bandwidth_gbps: int,
        network_state: NetworkState,
        spectrum_pipeline: SpectrumPipeline,
        snr_pipeline: SNRPipeline | None,
        connection_index: int | None,
        path_index: int,
        max_slices: int,
    ) -> SlicingResult | None:
        """Flex-grid dynamic slicing: iterate through bandwidth tiers."""
        from fusion.domain.results import SlicingResult

        # Get sorted bandwidth tiers (descending)
        mod_per_bw = getattr(self._config, "mod_per_bw", {})
        sorted_tiers = sorted([int(k) for k in mod_per_bw.keys()], reverse=True)

        remaining_bw = bandwidth_gbps
        allocated_lightpaths: list[int] = []
        slice_bandwidths: list[int] = []
        failed_snr_values: list[float] = []

        # DEBUG: Show dynamic slicing for req 43
        if request.request_id == 43:
            print(f'[DEBUG-R43-DYNAMIC] Starting flex-grid dynamic slicing for {bandwidth_gbps} Gbps')

        for tier_bw in sorted_tiers:
            # Skip tiers >= original request bandwidth
            if tier_bw >= bandwidth_gbps:
                continue

            # DEBUG: Show tier for req 43
            if request.request_id == 43:
                print(f'[DEBUG-R43-DYNAMIC] Trying tier_bw={tier_bw}')

            while remaining_bw >= tier_bw and len(allocated_lightpaths) < max_slices:
                # Use dynamic slicing mode with slice_bandwidth for flex-grid
                spectrum_result = spectrum_pipeline.find_spectrum(
                    path=path,
                    modulation="",  # Will be determined by GSNR
                    bandwidth_gbps=bandwidth_gbps,
                    network_state=network_state,
                    connection_index=connection_index,
                    path_index=path_index,
                    use_dynamic_slicing=True,
                    request_id=request.request_id,
                    slice_bandwidth=tier_bw,  # Key: pass tier bandwidth
                )

                # DEBUG: Show result for req 43
                if request.request_id == 43:
                    print(f'[DEBUG-R43-DYNAMIC] spectrum: is_free={spectrum_result.is_free} start={spectrum_result.start_slot} end={spectrum_result.end_slot} mod={spectrum_result.modulation} bw={spectrum_result.achieved_bandwidth_gbps}')

                if not spectrum_result.is_free:
                    break  # Move to smaller tier

                # For flex-grid, bandwidth is the tier bandwidth
                achieved_bw = tier_bw

                path_weight_km = self._calculate_path_weight(path, network_state)
                lightpath = network_state.create_lightpath(
                    path=path,
                    start_slot=spectrum_result.start_slot,
                    end_slot=spectrum_result.end_slot,
                    core=spectrum_result.core,
                    band=spectrum_result.band,
                    modulation=spectrum_result.modulation,
                    bandwidth_gbps=achieved_bw,
                    path_weight_km=path_weight_km,
                    guard_slots=getattr(self._config, "guard_slots", 0),
                    connection_index=connection_index,
                    snr_db=spectrum_result.snr_db,
                )
                lightpath_id = lightpath.lightpath_id
                lightpath.request_allocations[request.request_id] = achieved_bw
                lightpath.remaining_bandwidth_gbps -= achieved_bw
                request.lightpath_ids.append(lightpath_id)

                snr_recheck_enabled = getattr(self._config, "snr_recheck", False)
                if snr_pipeline is not None and snr_recheck_enabled:
                    recheck_result = snr_pipeline.recheck_affected(
                        lightpath_id, network_state, slicing_flag=True
                    )
                    if not recheck_result.all_pass:
                        if spectrum_result.snr_db is not None:
                            failed_snr_values.append(spectrum_result.snr_db)
                        network_state.release_lightpath(lightpath_id)
                        break

                allocated_lightpaths.append(lightpath_id)
                slice_bandwidths.append(achieved_bw)
                remaining_bw -= achieved_bw

                # DEBUG: Show allocation for req 43
                if request.request_id == 43:
                    print(f'[DEBUG-R43-DYNAMIC] Allocated slice: lp_id={lightpath_id} mod={spectrum_result.modulation} bw={achieved_bw} remaining={remaining_bw}')

            if remaining_bw <= 0:
                break

        if not allocated_lightpaths:
            return None

        return self._finalize_dynamic_result(
            remaining_bw, bandwidth_gbps, allocated_lightpaths, slice_bandwidths,
            failed_snr_values, path_index, network_state
        )

    def _finalize_dynamic_result(
        self,
        remaining_bw: int,
        original_bw: int,
        allocated_lightpaths: list[int],
        slice_bandwidths: list[int],
        failed_snr_values: list[float],
        path_index: int,
        network_state: NetworkState,
    ) -> SlicingResult | None:
        """Finalize dynamic slicing result, handling partial allocation."""
        from fusion.domain.results import SlicingResult

        if remaining_bw <= 0:
            total_bw = sum(slice_bandwidths)
            return SlicingResult(
                success=True,
                num_slices=len(allocated_lightpaths),
                slice_bandwidth_gbps=slice_bandwidths[0] if slice_bandwidths else 0,
                lightpaths_created=tuple(allocated_lightpaths),
                total_bandwidth_gbps=total_bw,
                failed_attempt_snr_values=tuple(failed_snr_values),
            )

        can_partial = getattr(self._config, "can_partially_serve", False)
        k_paths = getattr(self._config, "k_paths", 3)
        on_last_path = path_index >= k_paths - 1

        if can_partial and on_last_path:
            allocated_bw = sum(slice_bandwidths)
            return SlicingResult(
                success=True,
                num_slices=len(allocated_lightpaths),
                slice_bandwidth_gbps=slice_bandwidths[0] if slice_bandwidths else 0,
                lightpaths_created=tuple(allocated_lightpaths),
                total_bandwidth_gbps=allocated_bw,
                failed_attempt_snr_values=tuple(failed_snr_values),
            )

        self.rollback_slices(allocated_lightpaths, network_state)
        return None

    def _check_tier_feasibility(
        self,
        path: list[str],
        bandwidth_gbps: int,
        network_state: NetworkState,
    ) -> bool:
        """Check if tier-based slicing is feasible (simplified check)."""
        # Get available slots on path
        available_slots = self._get_available_slots_on_path(path, network_state)

        # Rough estimate: need at least some slots
        return available_slots >= 4  # Minimum for any slicing

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
        connection_index: int | None = None,
        path_index: int = 0,
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
            connection_index: External routing index for pre-calculated SNR lookup
            path_index: Index of which k-path is being tried (0, 1, 2...)

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
        # Return ALL valid modulations so find_spectrum can try each one.
        slice_modulations = self._get_modulation_for_slice(slice_bandwidth, path, network_state)
        if not slice_modulations:
            logger.debug(f"No valid modulation for slice bandwidth {slice_bandwidth}")
            return None

        try:
            for i in range(num_slices):
                # Find spectrum for this slice - pass ALL valid modulations
                spectrum_result = spectrum_pipeline.find_spectrum(
                    path=path,
                    modulation=slice_modulations,  # Pass list of modulations
                    bandwidth_gbps=slice_bandwidth,
                    network_state=network_state,
                    connection_index=connection_index,
                    path_index=path_index,
                )
                if not spectrum_result.is_free:

                    # Check for partial serving (Legacy behavior)
                    # If can_partially_serve is enabled, some slices allocated, and on last path, accept partial
                    can_partial = getattr(self._config, "can_partially_serve", False)
                    k_paths = getattr(self._config, "k_paths", 3)
                    on_last_path = path_index >= k_paths - 1

                    if can_partial and len(allocated_lightpaths) > 0 and on_last_path:
                        # Accept partial service - don't rollback, return what we allocated
                        allocated_bw = len(allocated_lightpaths) * slice_bandwidth
                        return SlicingResult.sliced(
                            num_slices=len(allocated_lightpaths),
                            slice_bandwidth=slice_bandwidth,
                            lightpath_ids=allocated_lightpaths,
                        )

                    # Rollback already allocated slices
                    self.rollback_slices(allocated_lightpaths, network_state)
                    return None

                # Create lightpath for this slice
                # Calculate path weight for lightpath
                path_weight_km = self._calculate_path_weight(path, network_state)

                # Use the modulation that was actually selected by spectrum assignment
                actual_modulation = spectrum_result.modulation

                lightpath = network_state.create_lightpath(
                    path=path,
                    start_slot=spectrum_result.start_slot,
                    end_slot=spectrum_result.end_slot,
                    core=spectrum_result.core,
                    band=spectrum_result.band,
                    modulation=actual_modulation,  # Use selected modulation
                    bandwidth_gbps=slice_bandwidth,
                    path_weight_km=path_weight_km,
                    guard_slots=getattr(self._config, "guard_slots", 0),
                    connection_index=connection_index,
                    snr_db=spectrum_result.snr_db,  # Pass SNR for metrics tracking
                )
                lightpath_id = lightpath.lightpath_id

                # Link request to lightpath and update remaining bandwidth
                lightpath.request_allocations[request.request_id] = slice_bandwidth
                lightpath.remaining_bandwidth_gbps -= slice_bandwidth
                request.lightpath_ids.append(lightpath_id)

                # SNR RECHECK: Check if EXISTING lightpaths would be degraded by this allocation
                # This matches legacy behavior - check existing LPs, not the new LP
                snr_recheck_enabled = getattr(self._config, "snr_recheck", False)
                if snr_pipeline is not None and snr_recheck_enabled:
                    recheck_result = snr_pipeline.recheck_affected(
                        lightpath_id, network_state, slicing_flag=True
                    )
                    if not recheck_result.all_pass:
                        logger.debug(
                            f"Slice {i + 1}/{num_slices} failed SNR recheck - existing LPs degraded"
                        )
                        # Check for partial serving before rollback
                        can_partial = getattr(self._config, "can_partially_serve", False)
                        k_paths = getattr(self._config, "k_paths", 3)
                        on_last_path = path_index >= k_paths - 1

                        # Release the failed lightpath first
                        network_state.release_lightpath(lightpath_id)

                        if can_partial and len(allocated_lightpaths) > 0 and on_last_path:
                            # Accept partial service with what we have
                            allocated_bw = len(allocated_lightpaths) * slice_bandwidth
                            return SlicingResult.sliced(
                                num_slices=len(allocated_lightpaths),
                                slice_bandwidth=slice_bandwidth,
                                lightpath_ids=allocated_lightpaths,
                            )

                        # Rollback all previous slices
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
    ) -> list[str]:
        """
        Determine ALL valid modulations for slice bandwidth and path.

        Looks up mod_per_bw for the slice bandwidth and finds ALL modulations
        that can reach the path length, sorted by spectral efficiency.

        Args:
            slice_bandwidth: Bandwidth of the slice in Gbps
            path: Route to use
            network_state: Network state for path length calculation

        Returns:
            List of valid modulation format names, empty list if none found
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
            return []

        bw_mods = mod_per_bw[bw_key]

        # Find ALL modulations that can reach the path
        # Sort by efficiency (highest order first for spectral efficiency)
        mod_order = ["64-QAM", "32-QAM", "16-QAM", "8-QAM", "QPSK", "BPSK"]
        valid_mods = []

        for mod_name in mod_order:
            if mod_name in bw_mods:
                mod_info = bw_mods[mod_name]
                if isinstance(mod_info, dict):
                    max_length = mod_info.get("max_length", 0)
                    if max_length >= path_length:
                        valid_mods.append(mod_name)

        if valid_mods:
            logger.debug(
                f"Slice modulations for {slice_bandwidth} Gbps: {valid_mods} "
                f"(path_length={path_length:.1f})"
            )
        else:
            logger.debug(
                f"No valid modulation for {slice_bandwidth} Gbps on path length {path_length:.1f}"
            )

        return valid_mods
