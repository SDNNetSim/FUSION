"""
FUSION Routing Module.

This package contains various routing algorithm implementations for optical network
path selection, including:

- K-Shortest Path routing
- Congestion-aware routing
- Fragmentation-aware routing
- Least congested link routing
- Non-linear impairment (NLI) aware routing
- Cross-talk (XT) aware routing

All algorithms implement the AbstractRoutingAlgorithm interface and can be accessed
through the RoutingRegistry for dynamic algorithm selection.
"""

from .congestion_aware import CongestionAwareRouting
from .fragmentation_aware import FragmentationAwareRouting
from .k_shortest_path import KShortestPath
from .least_congested import LeastCongestedRouting
from .nli_aware import NLIAwareRouting
from .one_plus_one_protection import OnePlusOneProtection
from .registry import (
    ROUTING_ALGORITHMS,
    RoutingRegistry,
    create_algorithm,
    get_algorithm,
    get_routing_algorithm_info,
    list_routing_algorithms,
)
from .xt_aware import XTAwareRouting

__all__ = [
    # Registry functions
    "RoutingRegistry",
    "create_algorithm",
    "get_algorithm",
    "list_routing_algorithms",
    "get_routing_algorithm_info",
    "ROUTING_ALGORITHMS",
    # Algorithm classes
    "KShortestPath",
    "CongestionAwareRouting",
    "LeastCongestedRouting",
    "FragmentationAwareRouting",
    "NLIAwareRouting",
    "XTAwareRouting",
    "OnePlusOneProtection",
]
