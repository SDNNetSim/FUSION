"""
Pipeline implementations for FUSION simulation.

This package contains pipeline implementations that can be selected
by PipelineFactory based on SimulationConfig.

Current implementations:

- ProtectedRoutingPipeline: 1+1 protection routing
- StandardSlicingPipeline: Request slicing across multiple lightpaths

Routing Strategies:

- KShortestPathStrategy: Basic k-shortest paths routing
- LoadBalancedStrategy: Routing considering link utilization
- ProtectionAwareStrategy: Routing for disjoint path pairs

Protection Pipeline:

- DisjointPathFinder: Link-disjoint and node-disjoint path algorithms
- ProtectionPipeline: 1+1 dedicated protection allocation
"""

from fusion.pipelines.disjoint_path_finder import DisjointnessType, DisjointPathFinder
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
    # Routing strategies
    "RoutingStrategy",
    "RouteConstraints",
    "KShortestPathStrategy",
    "LoadBalancedStrategy",
    "ProtectionAwareStrategy",
    # Protection pipeline
    "DisjointnessType",
    "DisjointPathFinder",
    "ProtectionPipeline",
    "ProtectedAllocationResult",
]
