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

Phase: P3.1 - Pipeline Factory Scaffolding
"""

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
]
