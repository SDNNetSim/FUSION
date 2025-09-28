# Routing Module

## Purpose
This module provides various routing algorithm implementations for optical network path selection, including congestion-aware, fragmentation-aware, and impairment-aware routing strategies.

## Key Components
- `__init__.py`: Public API exports and module initialization
- `registry.py`: Centralized registry for routing algorithm discovery and instantiation
- `k_shortest_path.py`: K-Shortest Path routing algorithm implementation
- `congestion_aware.py`: Congestion-aware routing considering network utilization
- `least_congested.py`: Routing based on least congested bottleneck links
- `fragmentation_aware.py`: Routing considering spectrum fragmentation levels
- `nli_aware.py`: Non-linear impairment aware routing algorithm
- `xt_aware.py`: Cross-talk aware routing for multi-core fiber systems
- `utils.py`: Utility classes and helper functions for routing calculations

## Usage
```python
from fusion.modules.routing import (
    RoutingRegistry,
    create_algorithm,
    KShortestPath,
    CongestionAwareRouting
)

# Create algorithm instance via registry
algorithm = create_algorithm("k_shortest_path", engine_props, sdn_props)

# Or instantiate directly
ksp = KShortestPath(engine_props, sdn_props)
path = ksp.route(source=1, destination=5, request=None)
```

## Dependencies
- Internal: `fusion.interfaces.router`, `fusion.core.properties`, `fusion.sim.utils`
- External: `networkx`, `numpy`