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
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fusion.core.orchestrator import SDNOrchestrator
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
