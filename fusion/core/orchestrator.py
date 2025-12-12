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

    __slots__ = ("config", "routing", "spectrum", "grooming", "snr", "slicing")

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

        # Stage 1: Grooming (if enabled)
        if self.grooming and self.config.grooming_enabled:
            groom_result = self.grooming.try_groom(request, network_state)

            if groom_result.fully_groomed:
                request.status = RequestStatus.GROOMED
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
                if groom_result.forced_path:
                    forced_path = list(groom_result.forced_path)

        # Stage 2: Routing
        route_result = self.routing.find_routes(
            request.source,
            request.destination,
            remaining_bw,
            network_state,
            forced_path=forced_path,
        )

        # DEBUG: Print routing result
        print(f"[DEBUG] Orchestrator routing result:")
        print(f"  request_id={request.request_id}, src={request.source}, dst={request.destination}")
        print(f"  paths={route_result.paths}")
        print(f"  modulations={route_result.modulations}")
        print(f"  connection_index={route_result.connection_index}")
        print(f"  is_empty={route_result.is_empty}")

        if route_result.is_empty:
            return self._handle_failure(
                request, groomed_lightpaths, BlockReason.NO_PATH, network_state
            )

        # Stage 3: Try standard allocation on ALL paths first (no slicing)
        # This matches legacy behavior: try all paths with standard allocation,
        # then only if all fail, try slicing on all paths
        for path_idx, path in enumerate(route_result.paths):
            modulations = route_result.modulations[path_idx]
            weight_km = route_result.weights_km[path_idx]

            result = self._try_allocate_on_path(
                request, path, modulations, weight_km, remaining_bw, network_state,
                allow_slicing=False,  # Don't try slicing yet
                connection_index=route_result.connection_index,
                path_index=path_idx,
            )
            if result is not None:
                return self._combine_results(
                    request, groomed_lightpaths, result, route_result
                )

        # Stage 4: Try slicing on ALL paths (only if standard allocation failed)
        if self.slicing and self.config.slicing_enabled:
            for path_idx, path in enumerate(route_result.paths):
                modulations = route_result.modulations[path_idx]
                weight_km = route_result.weights_km[path_idx]

                result = self._try_allocate_on_path(
                    request, path, modulations, weight_km, remaining_bw, network_state,
                    allow_slicing=True,  # Now try slicing
                    slicing_only=True,   # Skip standard allocation (already tried)
                    connection_index=route_result.connection_index,
                    path_index=path_idx,
                )
                if result is not None:
                    return self._combine_results(
                        request, groomed_lightpaths, result, route_result
                    )

        # Stage 5: All paths failed
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
    ) -> AllocationResult | None:
        """
        Try to allocate on a single path.

        Args:
            request: The request to allocate
            path: Path to try allocation on
            modulations: Valid modulations for this path
            weight_km: Path weight in km
            bandwidth_gbps: Bandwidth to allocate
            network_state: Current network state
            allow_slicing: Whether slicing fallback is allowed
            slicing_only: Skip standard allocation (only try slicing)
            connection_index: External routing index for pre-calculated SNR lookup
            path_index: Index of which k-path is being tried (0, 1, 2...)
        """
        # Try standard allocation with first valid modulation
        # Skip False/None values (modulations that don't reach path distance)
        print(f"[V5_ORCH] _try_allocate_on_path: bw_gbps={bandwidth_gbps}, dynamic_lps={self.config.dynamic_lps}")
        if not slicing_only:
            for mod in modulations:
                if not mod or mod is False:
                    continue
                spectrum_result = self.spectrum.find_spectrum(
                    list(path), mod, bandwidth_gbps, network_state,
                    connection_index=connection_index,
                    path_index=path_index,
                )
                print(f"[V5_ORCH] find_spectrum result: is_free={spectrum_result.is_free}, slots={spectrum_result.start_slot}-{spectrum_result.end_slot}")

                if spectrum_result.is_free:
                    alloc_result = self._allocate_and_validate(
                        request,
                        path,
                        spectrum_result,
                        weight_km,
                        bandwidth_gbps,
                        network_state,
                    )
                    print(f"[V5_ORCH] _allocate_and_validate returned: {alloc_result is not None}")
                    if alloc_result is not None:
                        return alloc_result

        # Fallback to slicing (if enabled and allowed)
        if allow_slicing and self.slicing and self.config.slicing_enabled:
            # Get first valid modulation for slicing
            first_valid_mod = next((m for m in modulations if m and m is not False), "")
            slicing_result = self.slicing.try_slice(
                request,
                list(path),
                first_valid_mod,
                bandwidth_gbps,
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
    ) -> AllocationResult | None:
        """Allocate lightpath and validate SNR with congestion handling."""
        from fusion.domain.results import AllocationResult

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
            bandwidth_gbps=bandwidth_gbps,
            path_weight_km=weight_km,
            guard_slots=self.config.guard_slots,
        )

        # Stage 1: Validate SNR for new lightpath
        if self.snr and self.config.snr_enabled:
            snr_result = self.snr.validate(lightpath, network_state)
            if not snr_result.passed:
                network_state.release_lightpath(lightpath.lightpath_id)
                return None

        # Stage 2: Congestion check - recheck affected existing LPs (P3.2.h)
        if self.snr and self.config.snr_recheck:
            recheck_result = self.snr.recheck_affected(
                lightpath.lightpath_id, network_state
            )
            if not recheck_result.all_pass:
                # Rollback: existing LP would fail SNR
                logger.debug(
                    f"Congestion rollback for request {request.request_id}: "
                    f"affected LPs {recheck_result.degraded_lightpath_ids}"
                )
                network_state.release_lightpath(lightpath.lightpath_id)
                return None

        # Success: link request to lightpath and update remaining bandwidth
        lightpath.request_allocations[request.request_id] = bandwidth_gbps
        lightpath.remaining_bandwidth_gbps -= bandwidth_gbps
        request.lightpath_ids.append(lightpath.lightpath_id)

        return AllocationResult(
            success=True,
            lightpaths_created=(lightpath.lightpath_id,),
            total_bandwidth_allocated_gbps=bandwidth_gbps,
            spectrum_result=spectrum_result,
        )

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
            request.status = RequestStatus.PARTIALLY_GROOMED
            return AllocationResult(
                success=True,
                lightpaths_groomed=tuple(groomed_lightpaths),
                is_groomed=True,
                is_partially_groomed=True,
                total_bandwidth_allocated_gbps=self._sum_groomed_bandwidth(
                    groomed_lightpaths, request, network_state
                ),
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
    ) -> AllocationResult:
        """Combine groomed and allocated results."""
        from fusion.domain.request import RequestStatus
        from fusion.domain.results import AllocationResult

        if groomed_lightpaths:
            request.status = RequestStatus.PARTIALLY_GROOMED
        else:
            request.status = RequestStatus.ALLOCATED

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
            total_bandwidth_allocated_gbps=alloc_result.total_bandwidth_allocated_gbps,
            route_result=route_result,
            spectrum_result=alloc_result.spectrum_result,
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

            # Release this request's bandwidth
            if request.request_id in lp.request_allocations:
                bw = lp.request_allocations.pop(request.request_id)
                lp.remaining_bandwidth_gbps += bw

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
