"""
Pipeline protocol definitions for FUSION v5 architecture.

This module defines typing.Protocol classes for all pipeline components:
- RoutingPipeline: Find candidate routes between nodes
- SpectrumPipeline: Find available spectrum along paths
- GroomingPipeline: Pack requests onto existing lightpaths
- SNRPipeline: Validate signal quality
- SlicingPipeline: Divide requests into smaller allocations

Design Principles:
- Protocols are type-only definitions (no runtime behavior)
- All pipelines receive NetworkState as method parameter
- Pipelines do NOT store NetworkState as instance attribute
- Return structured result objects from fusion.domain.results

Usage:
    # Type hint a parameter as accepting any routing implementation
    def process_request(
        router: RoutingPipeline,
        network_state: NetworkState,
    ) -> RouteResult:
        return router.find_routes("A", "B", 100, network_state)

    # Check if object implements protocol (runtime_checkable)
    if isinstance(my_router, RoutingPipeline):
        result = my_router.find_routes(...)

Phase: P2.3 - Pipeline Protocols
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from fusion.domain.lightpath import Lightpath
    from fusion.domain.network_state import NetworkState
    from fusion.domain.request import Request
    from fusion.domain.results import (
        GroomingResult,
        RouteResult,
        SlicingResult,
        SNRRecheckResult,
        SNRResult,
        SpectrumResult,
    )


# =============================================================================
# RoutingPipeline
# =============================================================================


@runtime_checkable
class RoutingPipeline(Protocol):
    """
    Protocol for route finding algorithms.

    Implementations find candidate routes between source and destination
    nodes, returning paths with their weights and valid modulation formats.

    Design Notes:
        - Implementations should NOT store NetworkState as instance attribute
        - Receive NetworkState as method parameter for each call
        - Return RouteResult without modifying NetworkState
        - Configuration accessed via network_state.config

    Common Implementations:
        - K-shortest paths routing
        - Congestion-aware routing
        - Fragmentation-aware routing
        - NLI-aware routing (non-linear interference)

    Example::

        class KShortestPathRouter:
            def find_routes(
                self,
                source: str,
                destination: str,
                bandwidth_gbps: int,
                network_state: NetworkState,
                *,
                forced_path: list[str] | None = None,
            ) -> RouteResult:
                if forced_path is not None:
                    return RouteResult(paths=(tuple(forced_path),), ...)
                k = network_state.config.k_paths
                paths = find_k_shortest(network_state.topology, source, destination, k)
                return RouteResult(paths=paths, ...)
    """

    def find_routes(
        self,
        source: str,
        destination: str,
        bandwidth_gbps: int,
        network_state: NetworkState,
        *,
        forced_path: list[str] | None = None,
    ) -> RouteResult:
        """
        Find candidate routes between source and destination.

        Args:
            source: Source node identifier
            destination: Destination node identifier
            bandwidth_gbps: Required bandwidth (used for modulation selection)
            network_state: Current network state (topology, config)
            forced_path: If provided, use this path instead of searching
                        (typically from partial grooming)

        Returns:
            RouteResult containing:

            - paths: Candidate paths as tuples of node IDs
            - weights_km: Path distances/weights in kilometers
            - modulations: Valid modulation formats per path
            - strategy_name: Name of routing algorithm used

            Returns empty RouteResult if no routes found.

        Side Effects:
            None - this is a pure query method

        Notes:
            - Number of paths limited by network_state.config.k_paths
            - Modulation formats filtered by reach based on path weight
            - Empty RouteResult.paths indicates no valid routes
        """
        ...


# =============================================================================
# SpectrumPipeline
# =============================================================================


@runtime_checkable
class SpectrumPipeline(Protocol):
    """
    Protocol for spectrum assignment algorithms.

    Implementations find available spectrum slots along a path
    for a given bandwidth and modulation format.

    Design Notes:
        - Implementations should NOT store NetworkState as instance attribute
        - Return SpectrumResult without modifying NetworkState
        - Actual allocation done by NetworkState.create_lightpath()

    Common Implementations:
        - First-fit: Allocate lowest available slot range
        - Best-fit: Allocate smallest sufficient gap
        - Last-fit: Allocate highest available slot range

    Example::

        class FirstFitSpectrum:
            def find_spectrum(
                self,
                path: list[str],
                modulation: str,
                bandwidth_gbps: int,
                network_state: NetworkState,
            ) -> SpectrumResult:
                slots_needed = calculate_slots(bandwidth_gbps, modulation)
                for band in network_state.config.band_list:
                    for core in range(network_state.config.cores_per_link):
                        start = find_first_free(path, slots_needed, core, band)
                        if start is not None:
                            return SpectrumResult(is_free=True, start_slot=start, ...)
                return SpectrumResult.not_found(slots_needed)
    """

    def find_spectrum(
        self,
        path: list[str],
        modulation: str,
        bandwidth_gbps: int,
        network_state: NetworkState,
        *,
        connection_index: int | None = None,
        path_index: int = 0,
    ) -> SpectrumResult:
        """
        Find available spectrum along a path.

        Args:
            path: Ordered list of node IDs forming the route
            modulation: Modulation format name (e.g., "QPSK", "16-QAM")
            bandwidth_gbps: Required bandwidth in Gbps
            network_state: Current network state
            connection_index: External routing index for pre-calculated SNR lookup
            path_index: Index of which k-path is being tried (0, 1, 2...)

        Returns:
            SpectrumResult containing:
            - is_free: True if spectrum was found
            - start_slot, end_slot: Slot range (if found)
            - core: Core index (if found)
            - band: Band identifier (if found)
            - modulation: Confirmed modulation format
            - slots_needed: Number of slots required (including guard band)

        Side Effects:
            None - does NOT allocate spectrum.
            Caller must use NetworkState.create_lightpath() to actually allocate.

        Notes:
            - Searches across all bands and cores based on allocation policy
            - Returns SpectrumResult.is_free=False if no spectrum available
            - slots_needed includes guard band slots
        """
        ...

    def find_protected_spectrum(
        self,
        primary_path: list[str],
        backup_path: list[str],
        modulation: str,
        bandwidth_gbps: int,
        network_state: NetworkState,
    ) -> SpectrumResult:
        """
        Find spectrum for both primary and backup paths (1+1 protection).

        Args:
            primary_path: Primary route node sequence
            backup_path: Backup route node sequence (should be disjoint)
            modulation: Modulation format name
            bandwidth_gbps: Required bandwidth
            network_state: Current network state

        Returns:
            SpectrumResult containing:

            - Primary allocation in main fields (is_free, start_slot, etc.)
            - Backup allocation in backup_* fields

            Returns is_free=False if either path lacks spectrum.

        Side Effects:
            None - does NOT allocate spectrum

        Notes:
            - Both paths must have free spectrum for success
            - Primary and backup may use different cores/bands
        """
        ...


# =============================================================================
# GroomingPipeline
# =============================================================================


@runtime_checkable
class GroomingPipeline(Protocol):
    """
    Protocol for traffic grooming algorithms.

    Grooming attempts to pack multiple requests onto existing lightpaths
    that have available capacity, reducing new lightpath establishment.

    Design Notes:
        - MAY have side effects (modifies lightpath bandwidth allocations)
        - Must support rollback for failed allocations
        - Returns GroomingResult indicating success/partial/failure

    Grooming Strategies:
        - Full grooming: Entire request fits on existing lightpath(s)
        - Partial grooming: Some bandwidth groomed, rest needs new lightpath
        - No grooming: No suitable lightpaths found

    Example::

        class SimpleGrooming:
            def try_groom(
                self,
                request: Request,
                network_state: NetworkState,
            ) -> GroomingResult:
                candidates = network_state.get_lightpaths_between(
                    request.source, request.destination
                )
                for lp in candidates:
                    if lp.can_accommodate(request.bandwidth_gbps):
                        lp.allocate_bandwidth(request.request_id, request.bandwidth_gbps)
                        return GroomingResult.full(request.bandwidth_gbps, [lp.lightpath_id])
                return GroomingResult.no_grooming(request.bandwidth_gbps)
    """

    def try_groom(
        self,
        request: Request,
        network_state: NetworkState,
    ) -> GroomingResult:
        """
        Attempt to groom request onto existing lightpaths.

        Args:
            request: The request to groom (source, destination, bandwidth)
            network_state: Current network state with active lightpaths

        Returns:
            GroomingResult containing:
            - fully_groomed: True if entire request was groomed
            - partially_groomed: True if some bandwidth was groomed
            - bandwidth_groomed_gbps: Amount successfully groomed
            - remaining_bandwidth_gbps: Amount still needing allocation
            - lightpaths_used: IDs of lightpaths used for grooming
            - forced_path: If partially groomed, suggested path for remainder

        Side Effects:
            If grooming succeeds (full or partial), modifies lightpath
            bandwidth allocations via Lightpath.allocate_bandwidth()

        Notes:
            - Searches lightpaths between request endpoints
            - May use multiple lightpaths for partial grooming
            - Does NOT modify NetworkState spectrum (lightpaths already exist)
        """
        ...

    def rollback_groom(
        self,
        request: Request,
        lightpath_ids: list[int],
        network_state: NetworkState,
    ) -> None:
        """
        Rollback grooming allocations (e.g., after downstream failure).

        Args:
            request: The request that was groomed
            lightpath_ids: Lightpath IDs to rollback
            network_state: Current network state

        Side Effects:
            Releases bandwidth from specified lightpaths via
            Lightpath.release_bandwidth()

        Notes:
            - Called when subsequent pipeline stages fail
            - Restores lightpath capacity
            - Safe to call even if request wasn't groomed on all lightpaths
        """
        ...


# =============================================================================
# SNRPipeline
# =============================================================================


@runtime_checkable
class SNRPipeline(Protocol):
    """
    Protocol for signal-to-noise ratio validation.

    Validates that lightpaths meet SNR requirements for their
    modulation format, considering interference from other channels.

    Design Notes:
        - validate() is a pure query method (no side effects)
        - recheck_affected() checks existing lightpaths after new allocation
        - Returns SNRResult/SNRRecheckResult with pass/fail and measurements

    SNR Considerations:
        - ASE noise from amplifiers
        - Non-linear interference (NLI) from other channels
        - Crosstalk in multi-core fibers (MCF)
        - Modulation-dependent thresholds

    Example::

        class GNModelSNR:
            def validate(
                self,
                lightpath: Lightpath,
                network_state: NetworkState,
            ) -> SNRResult:
                required_snr = get_threshold(lightpath.modulation)
                snr_db = calculate_gn_model_snr(lightpath, network_state)
                if snr_db >= required_snr:
                    return SNRResult.success(snr_db, required_snr)
                return SNRResult.failure(snr_db, required_snr, "SNR below threshold")
    """

    def validate(
        self,
        lightpath: Lightpath,
        network_state: NetworkState,
    ) -> SNRResult:
        """
        Validate SNR for a lightpath.

        Args:
            lightpath: The lightpath to validate (may be newly created
                      or existing lightpath being rechecked)
            network_state: Current network state with spectrum allocations

        Returns:
            SNRResult containing:
            - passed: True if SNR meets threshold
            - snr_db: Measured/calculated SNR value
            - required_snr_db: Threshold for this modulation
            - margin_db: SNR margin above threshold
            - failure_reason: If failed, why
            - link_snr_values: Per-link SNR breakdown (optional)

        Side Effects:
            None - this is a pure query method

        Notes:
            - Uses lightpath.modulation to determine threshold
            - Considers interference from all other active lightpaths
            - May return different results before/after allocation
        """
        ...

    def recheck_affected(
        self,
        new_lightpath_id: int,
        network_state: NetworkState,
        *,
        affected_range_slots: int = 5,
    ) -> SNRRecheckResult:
        """
        Recheck SNR of existing lightpaths after new allocation.

        Some existing lightpaths may be degraded by the new
        allocation's interference. This method identifies them.

        Args:
            new_lightpath_id: ID of newly created lightpath
            network_state: Current network state
            affected_range_slots: Consider lightpaths within this many
                                 slots of the new allocation (default: 5)

        Returns:
            SNRRecheckResult containing:
            - all_pass: True if all affected lightpaths still pass
            - degraded_lightpath_ids: List of lightpath IDs now failing
            - violations: Dict mapping lightpath_id -> SNR shortfall (dB)
            - checked_count: Number of lightpaths that were checked

        Side Effects:
            None - pure query method

        Notes:
            - Only checks lightpaths on same links as new allocation
            - Only checks lightpaths in nearby spectrum (within range)
            - Used to trigger rollback if existing lightpaths degraded
        """
        ...


# =============================================================================
# SlicingPipeline
# =============================================================================


@runtime_checkable
class SlicingPipeline(Protocol):
    """
    Protocol for request slicing algorithms.

    Slicing divides a large request into multiple smaller lightpaths
    when a single lightpath cannot accommodate the full bandwidth.

    Design Notes:
        - try_slice() is a pure query (checks feasibility)
        - rollback_slices() removes created lightpaths on failure
        - Used when normal allocation fails due to spectrum fragmentation
        - Limited by config.max_slices

    Slicing Strategies:
        - Static slicing: Fixed slice size (e.g., 50 Gbps per slice)
        - Dynamic slicing: Adaptive based on availability

    Example::

        class DynamicSlicing:
            def try_slice(
                self,
                request: Request,
                path: list[str],
                modulation: str,
                bandwidth_gbps: int,
                network_state: NetworkState,
                *,
                max_slices: int | None = None,
            ) -> SlicingResult:
                limit = max_slices or network_state.config.max_slices
                for num_slices in range(2, limit + 1):
                    slice_bw = bandwidth_gbps // num_slices
                    if can_allocate_slices(path, slice_bw, num_slices, network_state):
                        return SlicingResult.sliced(num_slices, slice_bw, ...)
                return SlicingResult.failed()
    """

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

        Args:
            request: The request being processed
            path: Route to use for slicing
            modulation: Modulation format for slices
            bandwidth_gbps: Total bandwidth to slice
            network_state: Current network state
            max_slices: Override config.max_slices (optional)
            spectrum_pipeline: Pipeline for finding spectrum per slice (optional).
                When provided, allows full allocation rather than feasibility check.
            snr_pipeline: Pipeline for validating each slice (optional).
                When provided with spectrum_pipeline, enables SNR validation.
            connection_index: External routing index for pre-calculated SNR lookup.
            path_index: Index of which k-path is being tried (0, 1, 2...).

        Returns:
            SlicingResult containing:
            - success: True if slicing is possible
            - num_slices: Number of slices needed
            - slice_bandwidth_gbps: Bandwidth per slice
            - lightpaths_created: Empty for feasibility check, or IDs if allocated
            - total_bandwidth_gbps: Total bandwidth allocated

        Side Effects:
            None - this is a pure query method by default.
            When spectrum_pipeline/snr_pipeline provided, may allocate slices.
            Caller creates lightpaths based on result if not allocated.

        Notes:
            - Only called when single lightpath allocation fails
            - Does NOT create lightpaths by default (just checks feasibility)
            - Limited by max_slices or config.max_slices
            - Each slice needs its own spectrum range
            - When pipelines are provided, can perform full allocation
        """
        ...

    def rollback_slices(
        self,
        lightpath_ids: list[int],
        network_state: NetworkState,
    ) -> None:
        """
        Rollback all slices on failure.

        Called when partial slice allocation fails and all
        created slices must be released.

        Args:
            lightpath_ids: IDs of lightpaths to release
            network_state: Network state to modify

        Side Effects:
            Releases/deletes specified lightpaths from network_state

        Notes:
            - Used when SNR validation fails on one of the slices
            - All created slices should be rolled back together
            - Safe to call with empty list
        """
        ...


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "RoutingPipeline",
    "SpectrumPipeline",
    "GroomingPipeline",
    "SNRPipeline",
    "SlicingPipeline",
]
