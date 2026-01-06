"""RLSimulationAdapter - coordination layer between RL and V4 simulation.

This module provides the adapter that connects RL environments to the V4
simulation stack (SDNOrchestrator, pipelines). The adapter ensures RL agents
use the SAME pipeline instances as the orchestrator, eliminating duplicated
simulation logic.

Key Invariants:
- Pipeline identity: adapter.routing IS orchestrator.routing (same object)
- Stateless: adapter never stores NetworkState
- Read-only queries: get_path_options() doesn't allocate
- Write-through: apply_action() routes through orchestrator

Phase: P4.1 - RLSimulationAdapter Scaffolding
Chunk: 2 - Adapter skeleton
Chunk: 3 - get_path_options method
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fusion.modules.rl.adapter.path_option import PathOption, PathOptionList

if TYPE_CHECKING:
    from fusion.core.orchestrator import SDNOrchestrator
    from fusion.domain.network_state import NetworkState
    from fusion.domain.request import Request
    from fusion.interfaces.pipelines import RoutingPipeline, SpectrumPipeline


class RLSimulationAdapter:
    """Adapter layer between RL environments and V4 simulation stack.

    This adapter provides RL agents with access to routing and spectrum
    pipelines WITHOUT duplicating any simulation logic. It uses the SAME
    pipeline instances as the SDNOrchestrator.

    Key Invariants:
    - Adapter shares pipeline references with orchestrator (identity, not copy)
    - Adapter never stores NetworkState (receives per-call)
    - Adapter never directly mutates spectrum (goes through orchestrator)

    Example:
        orchestrator = SDNOrchestrator(config, pipelines)
        adapter = RLSimulationAdapter(orchestrator)

        # Verify pipeline identity
        assert adapter.routing is orchestrator.routing
        assert adapter.spectrum is orchestrator.spectrum

    Attributes:
        routing: Reference to shared routing pipeline
        spectrum: Reference to shared spectrum pipeline
    """

    def __init__(
        self,
        orchestrator: SDNOrchestrator,
    ) -> None:
        """Initialize adapter with orchestrator reference.

        The adapter stores references to the orchestrator's pipelines,
        ensuring RL code uses the exact same instances as non-RL simulation.

        Args:
            orchestrator: SDNOrchestrator instance (shares pipelines)

        Raises:
            ValueError: If orchestrator is None
        """
        if orchestrator is None:
            raise ValueError("orchestrator cannot be None")

        self._orchestrator = orchestrator

        # Store pipeline references - these are the SAME instances used by orchestrator
        # This is critical: adapter.routing IS orchestrator.routing (identity)
        self._routing = orchestrator.routing
        self._spectrum = orchestrator.spectrum

    @property
    def routing(self) -> RoutingPipeline:
        """Access to shared routing pipeline (same instance as orchestrator)."""
        return self._routing

    @property
    def spectrum(self) -> SpectrumPipeline:
        """Access to shared spectrum pipeline (same instance as orchestrator)."""
        return self._spectrum

    @property
    def orchestrator(self) -> SDNOrchestrator:
        """Access to the underlying orchestrator."""
        return self._orchestrator

    def get_path_options(
        self,
        request: Request,
        network_state: NetworkState,
    ) -> PathOptionList:
        """Get candidate paths with feasibility information.

        This method is READ-ONLY. It queries routing and spectrum pipelines
        but does NOT allocate any spectrum or modify network state.

        Args:
            request: Current request to route
            network_state: Current network state (not stored by adapter)

        Returns:
            List of PathOption, one per candidate path, with:
            - Path geometry (nodes, length, hops)
            - Modulation and slots needed
            - Feasibility from real spectrum check
            - Congestion and availability metrics

        Note:
            The number of options returned may be less than k_paths
            if routing cannot find enough paths.
        """
        # 1. Get candidate paths from routing pipeline
        route_result = self._routing.find_routes(
            source=request.source,
            destination=request.destination,
            bandwidth_gbps=request.bandwidth_gbps,
            network_state=network_state,
        )

        # If no paths found, return empty list
        if route_result.is_empty:
            return []

        options: list[PathOption] = []

        # 2. For each candidate path, check spectrum feasibility
        for path_idx in range(len(route_result.paths)):
            path = route_result.paths[path_idx]
            weight_km = route_result.weights_km[path_idx]
            modulations = route_result.modulations[path_idx]

            # Pick the first (best) modulation if available
            modulation = modulations[0] if modulations else None

            # Default values for spectrum-related fields
            is_feasible = False
            slots_needed = 0
            spectrum_start = None
            spectrum_end = None
            core_index = None
            band = None

            # 3. Check spectrum feasibility if we have a modulation
            if modulation is not None:
                spectrum_result = self._spectrum.find_spectrum(
                    path=list(path),
                    modulation=modulation,
                    bandwidth_gbps=request.bandwidth_gbps,
                    network_state=network_state,
                )

                is_feasible = spectrum_result.is_free
                slots_needed = spectrum_result.slots_needed

                if is_feasible:
                    spectrum_start = spectrum_result.start_slot
                    spectrum_end = spectrum_result.end_slot
                    core_index = spectrum_result.core
                    band = spectrum_result.band

            # 4. Compute congestion metrics
            # TODO: These require NetworkState methods that may not exist yet
            # Using placeholder values for now
            congestion = self._compute_path_congestion(path, network_state)
            available_slots = self._compute_available_slots(path, network_state)

            # 5. Create PathOption
            option = PathOption(
                path_index=path_idx,
                path=path,  # Already a tuple from RouteResult
                weight_km=weight_km,
                num_hops=len(path) - 1,
                modulation=modulation,
                slots_needed=slots_needed,
                is_feasible=is_feasible,
                congestion=congestion,
                available_slots=available_slots,
                spectrum_start=spectrum_start,
                spectrum_end=spectrum_end,
                core_index=core_index,
                band=band,
            )
            options.append(option)

        return options

    def _compute_path_congestion(
        self,
        path: tuple[str, ...],
        network_state: NetworkState,
    ) -> float:
        """Compute congestion metric for a path.

        Congestion = max link utilization along the path.
        Returns value in [0, 1].

        Note: Currently returns 0.0 as placeholder until NetworkState
        provides get_link_utilization() method.
        """
        # TODO: Implement when NetworkState has get_link_utilization()
        # For now, return 0.0 (no congestion)
        return 0.0

    def _compute_available_slots(
        self,
        path: tuple[str, ...],
        network_state: NetworkState,
    ) -> float:
        """Compute available slots ratio for a path.

        Returns min(available/total) across all links.
        Value in [0, 1] where 1 = fully available.

        Note: Currently returns 1.0 as placeholder until NetworkState
        provides get_available_slots() method.
        """
        # TODO: Implement when NetworkState has get_available_slots()
        # For now, return 1.0 (fully available)
        return 1.0
