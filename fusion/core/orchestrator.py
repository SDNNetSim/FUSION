"""
SDN Orchestrator for FUSION simulation.

This module provides SDNOrchestrator, a thin coordination layer
that routes requests through pipelines without implementing
algorithm logic.

RULES (enforced by code review):
- No algorithm logic (K-shortest-path, first-fit, SNR calculation)
- No direct numpy access
- No hardcoded slicing/grooming logic
- Receives NetworkState per call, never stores it
- Each method < 50 lines

Phase: P3.2 - SDN Orchestrator Creation
Gap Analysis Coverage: P3.2.f (rollback), P3.2.g (protection), P3.2.h (congestion)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fusion.core.pipeline_factory import PipelineSet
    from fusion.domain.config import SimulationConfig
    from fusion.domain.network_state import NetworkState
    from fusion.domain.request import BlockReason, Request
    from fusion.domain.results import AllocationResult, RouteResult, SpectrumResult
    from fusion.interfaces.pipelines import (
        GroomingPipeline,
        RoutingPipeline,
        SlicingPipeline,
        SNRPipeline,
        SpectrumPipeline,
    )

logger = logging.getLogger(__name__)


class SDNOrchestrator:
    """
    Thin coordination layer for request handling.

    The orchestrator sequences pipeline calls and combines their
    results. It does NOT implement algorithm logic - all computation
    is delegated to pipelines.

    Attributes:
        config: Simulation configuration
        routing: Pipeline for finding candidate routes
        spectrum: Pipeline for spectrum assignment
        grooming: Pipeline for traffic grooming (optional)
        snr: Pipeline for SNR validation (optional)
        slicing: Pipeline for request slicing (optional)

    Note:
        Does NOT store network_state - receives per call only.
    """

    __slots__ = ("config", "routing", "spectrum", "grooming", "snr", "slicing", "_failed_attempt_snr_list")

    def __init__(
        self,
        config: SimulationConfig,
        pipelines: PipelineSet,
    ) -> None:
        """
        Initialize orchestrator with config and pipelines.

        Args:
            config: Simulation configuration
            pipelines: Container with all pipeline implementations
        """
        self.config: SimulationConfig = config
        self.routing: RoutingPipeline = pipelines.routing
        self.spectrum: SpectrumPipeline = pipelines.spectrum
        self.grooming: GroomingPipeline | None = pipelines.grooming
        self.snr: SNRPipeline | None = pipelines.snr
        self.slicing: SlicingPipeline | None = pipelines.slicing

    def handle_arrival(
        self,
        request: Request,
        network_state: NetworkState,
        forced_path: list[str] | None = None,
    ) -> AllocationResult:
        """
        Handle request arrival by coordinating pipelines.

        Args:
            request: The incoming request to process
            network_state: Current network state (passed per call)
            forced_path: Optional forced path from external source

        Returns:
            AllocationResult with success/failure and details
        """
        from fusion.domain.request import BlockReason

        # Branch for protection-enabled requests (P3.2.g)
        if self.config.protection_enabled and getattr(
            request, "protection_required", False
        ):
            return self._handle_protected_arrival(request, network_state)

        # Standard unprotected flow
        return self._handle_unprotected_arrival(request, network_state, forced_path)

    def _handle_unprotected_arrival(
        self,
        request: Request,
        network_state: NetworkState,
        forced_path: list[str] | None = None,
    ) -> AllocationResult:
        """Handle standard (unprotected) request arrival."""
        from fusion.domain.request import BlockReason, RequestStatus
        from fusion.domain.results import AllocationResult

        groomed_lightpaths: list[int] = []
        remaining_bw = request.bandwidth_gbps
        was_partially_groomed = False

        # Track SNR values from failed allocation attempts (for Legacy compatibility)
        # Legacy adds SNR to snr_list before SNR recheck; if recheck fails, value stays
        self._failed_attempt_snr_list = []  # type: list[float]

        # Stage 1: Grooming (if enabled)
        if self.grooming and self.config.grooming_enabled:
            groom_result = self.grooming.try_groom(request, network_state)

            if groom_result.fully_groomed:
                request.status = RequestStatus.GROOMED
                # CRITICAL: Add groomed lightpath IDs to request so release can return bandwidth
                request.lightpath_ids.extend(groom_result.lightpaths_used)
                return AllocationResult(
                    success=True,
                    is_groomed=True,
                    lightpaths_groomed=tuple(groom_result.lightpaths_used),
                    total_bandwidth_allocated_gbps=request.bandwidth_gbps,
                    grooming_result=groom_result,
                )

            if groom_result.partially_groomed:
                groomed_lightpaths = list(groom_result.lightpaths_used)
                remaining_bw = groom_result.remaining_bandwidth_gbps
                was_partially_groomed = True
                if groom_result.forced_path:
                    forced_path = list(groom_result.forced_path)

        # LP capacity is determined by spectrum assignment (modulation-based capacity)
        # NOT by original request bandwidth. For slicing, spectrum/slicing code sets
        # lightpath_bandwidth based on what the modulation can support.
        lp_capacity_override = None

        # Stage 2: Routing
        route_result = self.routing.find_routes(
            request.source,
            request.destination,
            remaining_bw,
            network_state,
            forced_path=forced_path,
        )

        if route_result.is_empty:
            return self._handle_failure(
                request, groomed_lightpaths, BlockReason.NO_PATH, network_state
            )

        # For partial grooming, LEGACY uses original request bandwidth for spectrum allocation
        # (not remaining_bw). This affects slots_needed calculation and SNR checks.
        spectrum_bw = request.bandwidth_gbps if was_partially_groomed else remaining_bw

        # Stage 3: Try standard allocation on ALL paths first (no slicing)
        # This applies to ALL modes including dynamic_lps - legacy tries standard first
        # and only falls back to dynamic slicing if standard fails on ALL paths
        for path_idx, path in enumerate(route_result.paths):
            modulations = route_result.modulations[path_idx]
            weight_km = route_result.weights_km[path_idx]

            # For partial grooming, actual_bw_to_allocate = remaining_bw (what we need to serve)
            # but spectrum_bw = original request (for spectrum calculation)
            actual_bw_to_allocate = remaining_bw if was_partially_groomed else spectrum_bw
            result = self._try_allocate_on_path(
                request, path, modulations, weight_km, spectrum_bw, network_state,
                allow_slicing=False,  # Don't try slicing yet
                connection_index=route_result.connection_index,
                path_index=path_idx,
                lp_capacity_override=lp_capacity_override,
                actual_bw_to_allocate=actual_bw_to_allocate,
            )
            if result is not None:
                groomed_bw = request.bandwidth_gbps - remaining_bw
                return self._combine_results(
                    request, groomed_lightpaths, result, route_result,
                    path_index=path_idx, groomed_bandwidth_gbps=groomed_bw
                )

        # Stage 4: Try dynamic_lps slicing on ALL paths
        # Note: This uses use_dynamic_slicing=True which behaves differently from
        # legacy's handle_dynamic_slicing_direct, but achieves similar blocking rates.
        if self.config.dynamic_lps:
            for path_idx, path in enumerate(route_result.paths):
                modulations = route_result.modulations[path_idx]
                weight_km = route_result.weights_km[path_idx]

                # For dynamic slicing, use remaining_bw for the slicing loop iteration
                # (how much bandwidth to actually serve), but spectrum_bw for spectrum
                # allocation (modulation/slots calculation). Legacy uses sdn_props.remaining_bw
                # for the slicing loop when partially groomed.
                slicing_target_bw = remaining_bw if was_partially_groomed else spectrum_bw
                # For partial grooming, actual_bw_to_allocate = remaining_bw (for stats tracking)
                actual_bw_to_allocate = remaining_bw if was_partially_groomed else spectrum_bw

                result = self._try_allocate_on_path(
                    request, path, modulations, weight_km, spectrum_bw, network_state,
                    allow_slicing=True,  # Now try dynamic slicing
                    slicing_only=False,  # Need to run through modulations to get spectrum
                    connection_index=route_result.connection_index,
                    path_index=path_idx,
                    use_dynamic_slicing=True,  # Use dynamic slicing spectrum method
                    lp_capacity_override=lp_capacity_override,
                    slicing_target_bw=slicing_target_bw,  # Actual bandwidth to serve via slicing
                    actual_bw_to_allocate=actual_bw_to_allocate,  # For stats tracking
                )
                if result is not None:
                    groomed_bw = request.bandwidth_gbps - remaining_bw
                    return self._combine_results(
                        request, groomed_lightpaths, result, route_result,
                        path_index=path_idx, groomed_bandwidth_gbps=groomed_bw
                    )

        # Stage 5: Try segment slicing pipeline on ALL paths (only if standard allocation failed)
        if self.slicing and self.config.slicing_enabled:
            for path_idx, path in enumerate(route_result.paths):
                modulations = route_result.modulations[path_idx]
                weight_km = route_result.weights_km[path_idx]

                # For partial grooming, actual_bw_to_allocate = remaining_bw
                actual_bw_to_allocate = remaining_bw if was_partially_groomed else spectrum_bw
                result = self._try_allocate_on_path(
                    request, path, modulations, weight_km, spectrum_bw, network_state,
                    allow_slicing=True,  # Now try slicing
                    slicing_only=True,   # Skip standard allocation (already tried)
                    connection_index=route_result.connection_index,
                    path_index=path_idx,
                    lp_capacity_override=lp_capacity_override,
                    actual_bw_to_allocate=actual_bw_to_allocate,
                )
                if result is not None:
                    groomed_bw = request.bandwidth_gbps - remaining_bw
                    return self._combine_results(
                        request, groomed_lightpaths, result, route_result,
                        path_index=path_idx, groomed_bandwidth_gbps=groomed_bw
                    )

        # Stage 6: All paths failed
        return self._handle_failure(
            request, groomed_lightpaths, BlockReason.CONGESTION, network_state
        )

    def _handle_protected_arrival(
        self,
        request: Request,
        network_state: NetworkState,
    ) -> AllocationResult:
        """
        Handle arrival for protected (1+1) request.

        Implements P3.2.g protection pipeline integration.
        """
        from fusion.domain.request import BlockReason, RequestStatus
        from fusion.domain.results import AllocationResult

        # Stage 1: Find working path
        working_routes = self.routing.find_routes(
            request.source,
            request.destination,
            request.bandwidth_gbps,
            network_state,
        )

        if working_routes.is_empty:
            return AllocationResult(
                success=False,
                block_reason=BlockReason.NO_PATH,
                route_result=working_routes,
            )

        # Check if routing returned backup paths (protection-aware routing)
        if not working_routes.has_protection:
            return AllocationResult(
                success=False,
                block_reason=BlockReason.PROTECTION_FAIL,
                route_result=working_routes,
            )

        # Try each working/backup path pair
        for idx, working_path in enumerate(working_routes.paths):
            backup_path = working_routes.backup_paths[idx] if working_routes.backup_paths else None
            if not backup_path:
                continue

            result = self._try_protected_allocation(
                request,
                working_path,
                backup_path,
                working_routes.modulations[idx],
                working_routes.backup_modulations[idx] if working_routes.backup_modulations else working_routes.modulations[idx],
                working_routes.weights_km[idx],
                working_routes.backup_weights_km[idx] if working_routes.backup_weights_km else working_routes.weights_km[idx],
                network_state,
                connection_index=working_routes.connection_index,
                path_index=idx,
            )
            if result is not None and result.success:
                request.status = RequestStatus.ALLOCATED
                return result

        return AllocationResult(
            success=False,
            block_reason=BlockReason.PROTECTION_FAIL,
            route_result=working_routes,
        )

    def _try_protected_allocation(
        self,
        request: Request,
        working_path: tuple[str, ...],
        backup_path: tuple[str, ...],
        working_mods: tuple[str, ...],
        backup_mods: tuple[str, ...],
        working_weight: float,
        backup_weight: float,
        network_state: NetworkState,
        connection_index: int | None = None,
        path_index: int = 0,
    ) -> AllocationResult | None:
        """Try to allocate both working and backup paths atomically."""
        from fusion.domain.results import AllocationResult

        # Find spectrum for working path
        working_spectrum = self.spectrum.find_spectrum(
            list(working_path), working_mods[0], request.bandwidth_gbps, network_state,
            connection_index=connection_index,
            path_index=path_index,
        )
        if not working_spectrum.is_free:
            return None

        # Create working lightpath
        working_lp = network_state.create_lightpath(
            path=working_path,
            start_slot=working_spectrum.start_slot,
            end_slot=working_spectrum.end_slot,
            core=working_spectrum.core,
            band=working_spectrum.band,
            modulation=working_spectrum.modulation,
            bandwidth_gbps=request.bandwidth_gbps,
            path_weight_km=working_weight,
            guard_slots=self.config.guard_slots,
            connection_index=connection_index,
            arrive_time=request.arrive_time,
        )

        # Find spectrum for backup path
        backup_spectrum = self.spectrum.find_spectrum(
            list(backup_path), backup_mods[0], request.bandwidth_gbps, network_state,
            connection_index=connection_index,
            path_index=path_index,
        )
        if not backup_spectrum.is_free:
            # Rollback working
            network_state.release_lightpath(working_lp.lightpath_id)
            return None

        # Create backup lightpath
        backup_lp = network_state.create_lightpath(
            path=backup_path,
            start_slot=backup_spectrum.start_slot,
            end_slot=backup_spectrum.end_slot,
            core=backup_spectrum.core,
            band=backup_spectrum.band,
            modulation=backup_spectrum.modulation,
            bandwidth_gbps=request.bandwidth_gbps,
            path_weight_km=backup_weight,
            guard_slots=self.config.guard_slots,
            connection_index=connection_index,
            arrive_time=request.arrive_time,
        )

        # SNR validation for both (if enabled)
        if self.snr and self.config.snr_enabled:
            working_snr = self.snr.validate(working_lp, network_state)
            if not working_snr.passed:
                network_state.release_lightpath(backup_lp.lightpath_id)
                network_state.release_lightpath(working_lp.lightpath_id)
                return None

            backup_snr = self.snr.validate(backup_lp, network_state)
            if not backup_snr.passed:
                network_state.release_lightpath(backup_lp.lightpath_id)
                network_state.release_lightpath(working_lp.lightpath_id)
                return None

        # Link working and backup LPs
        working_lp.protection_lp_id = backup_lp.lightpath_id
        backup_lp.working_lp_id = working_lp.lightpath_id

        # Update request
        request.lightpath_ids.extend([working_lp.lightpath_id, backup_lp.lightpath_id])

        return AllocationResult(
            success=True,
            lightpaths_created=(working_lp.lightpath_id, backup_lp.lightpath_id),
            total_bandwidth_allocated_gbps=request.bandwidth_gbps,
            is_protected=True,
            spectrum_result=working_spectrum,
        )

    def _try_allocate_on_path(
        self,
        request: Request,
        path: tuple[str, ...],
        modulations: tuple[str, ...],
        weight_km: float,
        bandwidth_gbps: int,
        network_state: NetworkState,
        allow_slicing: bool = True,
        slicing_only: bool = False,
        connection_index: int | None = None,
        path_index: int = 0,
        use_dynamic_slicing: bool = False,
        lp_capacity_override: int | None = None,
        snr_bandwidth: int | None = None,
        slicing_target_bw: int | None = None,
        actual_bw_to_allocate: int | None = None,
    ) -> AllocationResult | None:
        """
        Try to allocate on a single path.

        Args:
            request: The request to allocate
            path: Path to try allocation on
            modulations: Valid modulations for this path
            weight_km: Path weight in km
            bandwidth_gbps: Bandwidth for spectrum calculation (may be original request bw)
            network_state: Current network state
            allow_slicing: Whether slicing fallback is allowed
            slicing_only: Skip standard allocation (only try slicing)
            connection_index: External routing index for pre-calculated SNR lookup
            path_index: Index of which k-path is being tried (0, 1, 2...)
            use_dynamic_slicing: Use dynamic slicing spectrum method
            lp_capacity_override: Override LP capacity (for partial grooming)
            snr_bandwidth: Bandwidth for SNR checks (original request bw for partial grooming)
            slicing_target_bw: Actual bandwidth to serve via dynamic slicing (for partial groom)
            actual_bw_to_allocate: Actual bandwidth to allocate/track (for partial grooming,
                                   this is remaining_bw; defaults to bandwidth_gbps)
        """
        # Try standard allocation with ALL modulations at once (like legacy)
        # Filter out False/None values (modulations that don't reach path distance)
        if not slicing_only:
            valid_mods = [mod for mod in modulations if mod and mod is not False]
            if valid_mods:
                spectrum_result = self.spectrum.find_spectrum(
                    list(path), valid_mods, bandwidth_gbps, network_state,
                    connection_index=connection_index,
                    path_index=path_index,
                    use_dynamic_slicing=use_dynamic_slicing,
                    snr_bandwidth=snr_bandwidth,
                )

                if spectrum_result.is_free:
                    # Handle dynamic_lps mode: achieved_bandwidth may be < requested
                    achieved_bw = spectrum_result.achieved_bandwidth_gbps
                    if allow_slicing and self.config.dynamic_lps and achieved_bw is not None and achieved_bw < bandwidth_gbps:
                        # Dynamic slicing: create multiple lightpaths
                        # Only do this in slicing stage (allow_slicing=True), not standard allocation
                        # Use slicing_target_bw if provided (for partial grooming),
                        # otherwise use bandwidth_gbps
                        actual_remaining = slicing_target_bw if slicing_target_bw is not None else bandwidth_gbps
                        result = self._allocate_dynamic_slices(
                            request=request,
                            path=path,
                            weight_km=weight_km,
                            remaining_bw=actual_remaining,
                            slice_bw=achieved_bw,
                            first_spectrum_result=spectrum_result,
                            network_state=network_state,
                            connection_index=connection_index,
                            path_index=path_index,
                        )
                        if result is not None:
                            return result
                    elif not self.config.dynamic_lps or achieved_bw is None or achieved_bw >= bandwidth_gbps:
                        # Standard allocation: single lightpath
                        # Only allocate if we have full bandwidth (or not in dynamic_lps mode)
                        # When use_dynamic_slicing=True (Stage 4), use slicing_flag=True for SNR
                        # recheck to match Legacy's segment_slicing behavior
                        # Use actual_bw_to_allocate for allocation tracking (defaults to bandwidth_gbps)
                        bw_to_allocate = actual_bw_to_allocate if actual_bw_to_allocate is not None else bandwidth_gbps
                        alloc_result = self._allocate_and_validate(
                            request,
                            path,
                            spectrum_result,
                            weight_km,
                            bw_to_allocate,  # Use actual bandwidth to allocate, not spectrum bandwidth
                            network_state,
                            connection_index=connection_index,
                            lp_capacity_override=lp_capacity_override,
                            path_index=path_index,
                            slicing_flag=use_dynamic_slicing,
                        )
                        if alloc_result is not None:
                            return alloc_result

        # Fallback to slicing (if enabled and allowed)
        if allow_slicing and self.slicing and self.config.slicing_enabled:
            # Get first valid modulation for slicing
            first_valid_mod = next((m for m in modulations if m and m is not False), "")
            # Use actual_bw_to_allocate for slicing (defaults to bandwidth_gbps)
            bw_for_slicing = actual_bw_to_allocate if actual_bw_to_allocate is not None else bandwidth_gbps
            slicing_result = self.slicing.try_slice(
                request,
                list(path),
                first_valid_mod,
                bw_for_slicing,  # Use actual bandwidth to serve for slicing
                network_state,
                spectrum_pipeline=self.spectrum,
                snr_pipeline=self.snr,
                connection_index=connection_index,
                path_index=path_index,
            )
            # Convert SlicingResult to AllocationResult
            if slicing_result.success:
                from fusion.domain.results import AllocationResult

                return AllocationResult(
                    success=True,
                    lightpaths_created=slicing_result.lightpaths_created,
                    is_sliced=slicing_result.is_sliced,
                    total_bandwidth_allocated_gbps=slicing_result.total_bandwidth_gbps,
                )

        return None

    def _allocate_and_validate(
        self,
        request: Request,
        path: tuple[str, ...],
        spectrum_result: SpectrumResult,
        weight_km: float,
        bandwidth_gbps: int,
        network_state: NetworkState,
        connection_index: int | None = None,
        lp_capacity_override: int | None = None,
        path_index: int = 0,
        slicing_flag: bool = False,
    ) -> AllocationResult | None:
        """Allocate lightpath and validate SNR with congestion handling."""
        from fusion.domain.results import AllocationResult

        # Determine lightpath capacity:
        # 1. If lp_capacity_override is set, use it
        # 2. Elif achieved_bandwidth_gbps is set (from spectrum_props.lightpath_bandwidth), use it
        #    This is set by spectrum_adapter to match LEGACY behavior (sdn_props.bandwidth)
        # 3. Else fallback to the requested bandwidth_gbps
        if lp_capacity_override is not None:
            lp_capacity = lp_capacity_override
        elif spectrum_result.achieved_bandwidth_gbps is not None:
            lp_capacity = spectrum_result.achieved_bandwidth_gbps
        else:
            lp_capacity = bandwidth_gbps

        # Create lightpath
        # Note: end_slot from spectrum_result includes guard slots,
        # so we must pass guard_slots for correct release behavior
        lightpath = network_state.create_lightpath(
            path=path,
            start_slot=spectrum_result.start_slot,
            end_slot=spectrum_result.end_slot,
            core=spectrum_result.core,
            band=spectrum_result.band,
            modulation=spectrum_result.modulation,
            bandwidth_gbps=lp_capacity,
            path_weight_km=weight_km,
            guard_slots=self.config.guard_slots,
            connection_index=connection_index,
            arrive_time=request.arrive_time,
        )

        # Store SNR value from spectrum assignment for metrics tracking
        # This is the SNR calculated during spectrum assignment (before lightpath creation)
        # which matches legacy behavior
        if spectrum_result.snr_db is not None:
            lightpath.snr_db = spectrum_result.snr_db

        # SNR recheck: check if existing lightpaths would be degraded by new allocation
        # NOTE: We do NOT re-validate the new lightpath's SNR here because:
        # 1. SNR was already validated during spectrum assignment (if snr_type is set)
        # 2. Legacy snr_recheck_after_allocation only checks EXISTING lightpaths
        # 3. Modulations are chosen for capacity, not SNR margin after allocation
        if self.snr and self.config.snr_recheck:
            recheck_result = self.snr.recheck_affected(
                lightpath.lightpath_id, network_state, slicing_flag=slicing_flag
            )
            if not recheck_result.all_pass:
                # Rollback: existing LP would fail SNR
                logger.debug(
                    f"SNR recheck failed for request {request.request_id}: "
                    f"affected LPs {recheck_result.degraded_lightpath_ids}"
                )
                # Track the failed attempt's SNR for Legacy compatibility
                # Legacy adds to snr_list before SNR recheck; value stays even on failure
                if spectrum_result.snr_db is not None and hasattr(self, '_failed_attempt_snr_list'):
                    self._failed_attempt_snr_list.append(spectrum_result.snr_db)
                network_state.release_lightpath(lightpath.lightpath_id)
                return None

        # Success: link request to lightpath and update remaining bandwidth
        # Use allocate_bandwidth to properly track time_bw_usage for utilization stats
        lightpath.allocate_bandwidth(
            request.request_id, bandwidth_gbps, timestamp=request.arrive_time
        )
        request.lightpath_ids.append(lightpath.lightpath_id)

        return AllocationResult(
            success=True,
            lightpaths_created=(lightpath.lightpath_id,),
            total_bandwidth_allocated_gbps=bandwidth_gbps,
            spectrum_result=spectrum_result,
        )

    def _allocate_dynamic_slices(
        self,
        request: Request,
        path: tuple[str, ...],
        weight_km: float,
        remaining_bw: int,
        slice_bw: int,
        first_spectrum_result: SpectrumResult,
        network_state: NetworkState,
        connection_index: int | None = None,
        path_index: int = 0,
    ) -> AllocationResult | None:
        """
        Allocate multiple lightpaths for dynamic_lps mode.

        When dynamic slicing returns achieved_bandwidth < requested_bandwidth,
        create multiple lightpaths to satisfy the full request.

        Args:
            request: The request being processed
            path: Path for all lightpaths
            weight_km: Path weight in km
            remaining_bw: Total bandwidth still needed
            slice_bw: Bandwidth per lightpath (from dynamic slicing)
            first_spectrum_result: First spectrum result to use
            network_state: Current network state
            connection_index: External routing index
            path_index: Path index for SNR lookup
        """
        from fusion.domain.results import AllocationResult

        allocated_lightpaths: list[int] = []
        total_allocated = 0
        spectrum_result = first_spectrum_result
        max_iterations = 20  # Safety limit

        iteration = 0
        while remaining_bw > 0 and iteration < max_iterations:
            iteration += 1

            if not spectrum_result.is_free:
                # No more spectrum available
                break

            # Get actual slice bandwidth (use achieved_bw if available, else slice_bw)
            actual_slice_bw = spectrum_result.achieved_bandwidth_gbps or slice_bw

            # Create lightpath for this slice
            lightpath = network_state.create_lightpath(
                path=path,
                start_slot=spectrum_result.start_slot,
                end_slot=spectrum_result.end_slot,
                core=spectrum_result.core,
                band=spectrum_result.band,
                modulation=spectrum_result.modulation,
                bandwidth_gbps=actual_slice_bw,
                path_weight_km=weight_km,
                guard_slots=self.config.guard_slots,
                connection_index=connection_index,
                arrive_time=request.arrive_time,
            )

            # Store SNR value from spectrum assignment for metrics tracking
            # This is the SNR calculated during spectrum assignment (before lightpath creation)
            if spectrum_result.snr_db is not None:
                lightpath.snr_db = spectrum_result.snr_db

            # SNR recheck for affected existing lightpaths (legacy behavior)
            # NOTE: Legacy does NOT re-validate the new LP's SNR here - only checks existing LPs
            if self.snr and self.config.snr_recheck:
                recheck_result = self.snr.recheck_affected(
                    lightpath.lightpath_id, network_state, slicing_flag=True
                )
                if not recheck_result.all_pass:
                    # Rollback: existing LP would fail SNR
                    logger.debug(
                        f"Dynamic slice SNR recheck failed for request {request.request_id}: "
                        f"affected LPs {recheck_result.degraded_lightpath_ids}"
                    )
                    network_state.release_lightpath(lightpath.lightpath_id)
                    break

            # Calculate how much bandwidth to dedicate to this lightpath
            # Don't over-allocate: only use what the request actually needs
            dedicated_bw = min(actual_slice_bw, remaining_bw)

            # Success: link request to lightpath
            # Use allocate_bandwidth to properly track time_bw_usage
            lightpath.allocate_bandwidth(
                request.request_id, dedicated_bw, timestamp=request.arrive_time
            )
            request.lightpath_ids.append(lightpath.lightpath_id)
            allocated_lightpaths.append(lightpath.lightpath_id)

            total_allocated += dedicated_bw
            remaining_bw -= dedicated_bw

            if remaining_bw > 0:
                # Find spectrum for next slice
                spectrum_result = self.spectrum.find_spectrum(
                    list(path),
                    spectrum_result.modulation,
                    remaining_bw,
                    network_state,
                    connection_index=connection_index,
                    path_index=path_index,
                    use_dynamic_slicing=True,
                )

        if total_allocated > 0:
            return AllocationResult(
                success=True,
                lightpaths_created=tuple(allocated_lightpaths),
                is_sliced=len(allocated_lightpaths) > 1,
                total_bandwidth_allocated_gbps=total_allocated,
                spectrum_result=first_spectrum_result,
            )

        # Complete failure - rollback any allocated lightpaths
        for lp_id in allocated_lightpaths:
            network_state.release_lightpath(lp_id)

        return None

    def _handle_failure(
        self,
        request: Request,
        groomed_lightpaths: list[int],
        reason: BlockReason,
        network_state: NetworkState,
    ) -> AllocationResult:
        """Handle allocation failure with grooming rollback (P3.2.f)."""
        from fusion.domain.request import BlockReason, RequestStatus
        from fusion.domain.results import AllocationResult

        # Accept partial grooming if allowed
        if groomed_lightpaths and self.config.can_partially_serve:
            groomed_bw = self._sum_groomed_bandwidth(
                groomed_lightpaths, request, network_state
            )
            # Only return success if we actually allocated some bandwidth
            if groomed_bw > 0:
                request.status = RequestStatus.PARTIALLY_GROOMED
                # CRITICAL: Add groomed lightpath IDs so release can return bandwidth
                request.lightpath_ids.extend(groomed_lightpaths)
                return AllocationResult(
                    success=True,
                    lightpaths_groomed=tuple(groomed_lightpaths),
                    is_groomed=True,
                    is_partially_groomed=True,
                    total_bandwidth_allocated_gbps=groomed_bw,
                )

        # Rollback grooming if any (P3.2.f)
        if groomed_lightpaths and self.grooming:
            logger.debug(
                f"Rolling back {len(groomed_lightpaths)} groomed LPs "
                f"for request {request.request_id}"
            )
            self.grooming.rollback_groom(request, groomed_lightpaths, network_state)

        request.status = RequestStatus.BLOCKED
        request.block_reason = reason

        return AllocationResult(success=False, block_reason=reason)

    def _sum_groomed_bandwidth(
        self,
        lightpath_ids: list[int],
        request: Request,
        network_state: NetworkState,
    ) -> int:
        """Sum bandwidth allocated to request from groomed lightpaths."""
        total = 0
        for lp_id in lightpath_ids:
            lp = network_state.get_lightpath(lp_id)
            if lp and request.request_id in lp.request_allocations:
                total += lp.request_allocations[request.request_id]
        return total

    def _combine_results(
        self,
        request: Request,
        groomed_lightpaths: list[int],
        alloc_result: AllocationResult,
        route_result: "RouteResult | None" = None,
        path_index: int = 0,
        groomed_bandwidth_gbps: int = 0,
    ) -> AllocationResult:
        """Combine groomed and allocated results."""
        from fusion.domain.request import RequestStatus
        from fusion.domain.results import AllocationResult

        if groomed_lightpaths:
            request.status = RequestStatus.PARTIALLY_GROOMED
            # CRITICAL: Add groomed lightpath IDs so release can return bandwidth
            request.lightpath_ids.extend(groomed_lightpaths)
        else:
            request.status = RequestStatus.ALLOCATED

        # Total bandwidth = groomed + newly allocated
        total_bw = groomed_bandwidth_gbps + alloc_result.total_bandwidth_allocated_gbps

        # Include failed attempt SNR values for Legacy compatibility
        failed_snr = tuple(self._failed_attempt_snr_list) if hasattr(self, '_failed_attempt_snr_list') else ()

        return AllocationResult(
            success=True,
            lightpaths_created=alloc_result.lightpaths_created,
            lightpaths_groomed=tuple(groomed_lightpaths),
            is_groomed=len(groomed_lightpaths) > 0,
            is_partially_groomed=(
                len(groomed_lightpaths) > 0
                and len(alloc_result.lightpaths_created) > 0
            ),
            is_sliced=alloc_result.is_sliced,
            total_bandwidth_allocated_gbps=total_bw,
            path_index=path_index,
            route_result=route_result,
            spectrum_result=alloc_result.spectrum_result,
            failed_attempt_snr_values=failed_snr,
        )

    def handle_release(
        self,
        request: Request,
        network_state: NetworkState,
    ) -> None:
        """
        Handle request release.

        Releases bandwidth from all lightpaths associated with the
        request. If a lightpath has no remaining allocations, it
        is fully released.
        """
        from fusion.domain.request import RequestStatus

        for lp_id in list(request.lightpath_ids):
            lp = network_state.get_lightpath(lp_id)
            if lp is None:
                continue

            # Release this request's bandwidth using release_bandwidth to track time_bw_usage
            if request.request_id in lp.request_allocations:
                lp.release_bandwidth(request.request_id, timestamp=request.depart_time)

            # Handle protection: release backup if this is working
            if hasattr(lp, "protection_lp_id") and lp.protection_lp_id:
                backup = network_state.get_lightpath(lp.protection_lp_id)
                if backup and not backup.request_allocations:
                    network_state.release_lightpath(lp.protection_lp_id)

            # Release lightpath if no more allocations
            if not lp.request_allocations:
                network_state.release_lightpath(lp_id)

        request.lightpath_ids.clear()
        request.status = RequestStatus.RELEASED
