"""
Pipeline implementations for FUSION simulation.

This package contains pipeline implementations that can be selected
by PipelineFactory based on SimulationConfig.

Current implementations:
- ProtectedRoutingPipeline: 1+1 protection routing
- StandardSlicingPipeline: Request slicing across multiple lightpaths

Routing Strategies (P3.1.e):
- KShortestPathStrategy: Basic k-shortest paths routing
- LoadBalancedStrategy: Routing considering link utilization
- ProtectionAwareStrategy: Routing for disjoint path pairs

Protection Pipeline (P5.4):
- DisjointPathFinder: Link-disjoint and node-disjoint path algorithms
- ProtectionPipeline: 1+1 dedicated protection allocation

Phase: P3.1 - Pipeline Factory Scaffolding
Phase: P5.4 - Protection Pipeline
"""

from fusion.pipelines.disjoint_path_finder import DisjointPathFinder, DisjointnessType
from fusion.pipelines.protection_pipeline import (
    ProtectedAllocationResult,
    ProtectionPipeline,
)
from fusion.pipelines.routing_strategies import (
    KShortestPathStrategy,
    LoadBalancedStrategy,
    ProtectionAwareStrategy,
    RouteConstraints,
    RoutingStrategy,
)

__all__ = [
    # Routing strategies (P3.1.e)
    "RoutingStrategy",
    "RouteConstraints",
    "KShortestPathStrategy",
    "LoadBalancedStrategy",
    "ProtectionAwareStrategy",
    # Protection pipeline (P5.4)
    "DisjointnessType",
    "DisjointPathFinder",
    "ProtectionPipeline",
    "ProtectedAllocationResult",
]
