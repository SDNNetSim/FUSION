"""
Pipeline protocol definitions for FUSION architecture.

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

Example usage::

    # Type hint a parameter as accepting any routing implementation
    def process_request(
        router: RoutingPipeline,
        network_state: NetworkState,
    ) -> RouteResult:
        return router.find_routes("A", "B", 100, network_state)

    # Check if object implements protocol (runtime_checkable)
    if isinstance(my_router, RoutingPipeline):
        result = my_router.find_routes(...)
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

        :param source: Source node identifier
        :type source: str
        :param destination: Destination node identifier
        :type destination: str
        :param bandwidth_gbps: Required bandwidth (used for modulation selection)
        :type bandwidth_gbps: int
        :param network_state: Current network state (topology, config)
        :type network_state: NetworkState
        :param forced_path: If provided, use this path instead of searching
            (typically from partial grooming)
        :type forced_path: list[str] | None
        :return: RouteResult containing paths, weights_km, modulations, and
            strategy_name. Returns empty RouteResult if no routes found.
        :rtype: RouteResult

        .. note::
            - Number of paths limited by network_state.config.k_paths
            - Modulation formats filtered by reach based on path weight
            - This is a pure query method with no side effects
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
        modulation: str | list[str],
        bandwidth_gbps: int,
        network_state: NetworkState,
        *,
        connection_index: int | None = None,
        path_index: int = 0,
        use_dynamic_slicing: bool = False,
        snr_bandwidth: int | None = None,
        request_id: int | None = None,
        slice_bandwidth: int | None = None,
        excluded_modulations: set[str] | None = None,
    ) -> SpectrumResult:
        """
        Find available spectrum along a path.

        :param path: Ordered list of node IDs forming the route
        :type path: list[str]
        :param modulation: Modulation format name (e.g., "QPSK") or list of
            valid modulations to try
        :type modulation: str | list[str]
        :param bandwidth_gbps: Required bandwidth in Gbps
        :type bandwidth_gbps: int
        :param network_state: Current network state
        :type network_state: NetworkState
        :param connection_index: External routing index for pre-calculated SNR lookup
        :type connection_index: int | None
        :param path_index: Index of which k-path is being tried (0, 1, 2...)
        :type path_index: int
        :param use_dynamic_slicing: If True, use dynamic slicing to find spectrum
        :type use_dynamic_slicing: bool
        :param snr_bandwidth: Override bandwidth for SNR calculation
        :type snr_bandwidth: int | None
        :param request_id: Request ID for tracking/logging
        :type request_id: int | None
        :param slice_bandwidth: Bandwidth per slice when slicing is active
        :type slice_bandwidth: int | None
        :param excluded_modulations: Modulations to exclude from dynamic slicing
        :type excluded_modulations: set[str] | None
        :return: SpectrumResult with is_free, start_slot, end_slot, core, band,
            modulation, and slots_needed fields
        :rtype: SpectrumResult

        .. note::
            This is a pure query method - does NOT allocate spectrum.
            Caller must use NetworkState.create_lightpath() to allocate.
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

        :param primary_path: Primary route node sequence
        :type primary_path: list[str]
        :param backup_path: Backup route node sequence (should be disjoint)
        :type backup_path: list[str]
        :param modulation: Modulation format name
        :type modulation: str
        :param bandwidth_gbps: Required bandwidth
        :type bandwidth_gbps: int
        :param network_state: Current network state
        :type network_state: NetworkState
        :return: SpectrumResult with primary allocation in main fields and
            backup allocation in backup_* fields. Returns is_free=False if
            either path lacks spectrum.
        :rtype: SpectrumResult

        .. note::
            This is a pure query method - does NOT allocate spectrum.
            Both paths must have free spectrum for success.
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

        :param request: The request to groom (source, destination, bandwidth)
        :type request: Request
        :param network_state: Current network state with active lightpaths
        :type network_state: NetworkState
        :return: GroomingResult with fully_groomed, partially_groomed,
            bandwidth_groomed_gbps, remaining_bandwidth_gbps, lightpaths_used,
            and forced_path fields
        :rtype: GroomingResult

        .. note::
            If grooming succeeds (full or partial), modifies lightpath
            bandwidth allocations via Lightpath.allocate_bandwidth().
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

        :param request: The request that was groomed
        :type request: Request
        :param lightpath_ids: Lightpath IDs to rollback
        :type lightpath_ids: list[int]
        :param network_state: Current network state
        :type network_state: NetworkState

        .. note::
            Releases bandwidth from specified lightpaths via
            Lightpath.release_bandwidth(). Safe to call even if request
            wasn't groomed on all lightpaths.
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

        :param lightpath: The lightpath to validate (may be newly created
            or existing lightpath being rechecked)
        :type lightpath: Lightpath
        :param network_state: Current network state with spectrum allocations
        :type network_state: NetworkState
        :return: SNRResult with passed, snr_db, required_snr_db, margin_db,
            failure_reason, and link_snr_values fields
        :rtype: SNRResult

        .. note::
            This is a pure query method. Uses lightpath.modulation to determine
            threshold and considers interference from all other active lightpaths.
        """
        ...

    def recheck_affected(
        self,
        new_lightpath_id: int,
        network_state: NetworkState,
        *,
        _affected_range_slots: int = 5,
        slicing_flag: bool = False,
    ) -> SNRRecheckResult:
        """
        Recheck SNR of existing lightpaths after new allocation.

        Some existing lightpaths may be degraded by the new
        allocation's interference. This method identifies them.

        :param new_lightpath_id: ID of newly created lightpath
        :type new_lightpath_id: int
        :param network_state: Current network state
        :type network_state: NetworkState
        :param affected_range_slots: Consider lightpaths within this many
            slots of the new allocation (default: 5)
        :type affected_range_slots: int
        :param slicing_flag: If True, indicates this is a slicing context
        :type slicing_flag: bool
        :return: SNRRecheckResult with all_pass, degraded_lightpath_ids,
            violations, and checked_count fields
        :rtype: SNRRecheckResult

        .. note::
            This is a pure query method. Only checks lightpaths on same links
            and in nearby spectrum. Used to trigger rollback if existing
            lightpaths are degraded.
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
        snr_accumulator: list[float] | None = None,
        path_weight: float | None = None,
    ) -> SlicingResult:
        """
        Attempt to slice request into multiple smaller allocations.

        :param request: The request being processed
        :type request: Request
        :param path: Route to use for slicing
        :type path: list[str]
        :param modulation: Modulation format for slices
        :type modulation: str
        :param bandwidth_gbps: Total bandwidth to slice
        :type bandwidth_gbps: int
        :param network_state: Current network state
        :type network_state: NetworkState
        :param max_slices: Override config.max_slices
        :type max_slices: int | None
        :param spectrum_pipeline: Pipeline for finding spectrum per slice
        :type spectrum_pipeline: SpectrumPipeline | None
        :param snr_pipeline: Pipeline for validating each slice
        :type snr_pipeline: SNRPipeline | None
        :param connection_index: External routing index for pre-calculated SNR lookup
        :type connection_index: int | None
        :param path_index: Index of which k-path is being tried (0, 1, 2...)
        :type path_index: int
        :param snr_accumulator: List to accumulate SNR values from failed attempts
        :type snr_accumulator: list[float] | None
        :param path_weight: Path weight in km (from routing, for metrics tracking)
        :type path_weight: float | None
        :return: SlicingResult with success, num_slices, slice_bandwidth_gbps,
            lightpaths_created, and total_bandwidth_gbps fields
        :rtype: SlicingResult

        .. note::
            Pure query method by default. When spectrum_pipeline/snr_pipeline
            provided, may allocate slices.
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

        :param lightpath_ids: IDs of lightpaths to release
        :type lightpath_ids: list[int]
        :param network_state: Network state to modify
        :type network_state: NetworkState

        .. note::
            Releases/deletes specified lightpaths from network_state.
            Safe to call with empty list.
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
