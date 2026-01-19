# FUSION Failures Module

> **Beta Status**: This module is currently in beta. It has not been fully tested or used in production simulations yet. Use with caution and expect potential API changes.

Network failure injection and tracking for survivability testing.

## Overview

The Failures module provides comprehensive failure injection capabilities for testing network survivability and recovery mechanisms in elastic optical networks. It supports multiple failure types, tracks active failures, and provides path feasibility checking for routing decisions.

## Supported Failure Types

### F1: Link Failure
Fail a single network link.

```python
from fusion.modules.failures import FailureManager

manager = FailureManager(engine_props, topology)
event = manager.inject_failure(
    'link',
    t_fail=100.0,
    t_repair=200.0,
    link_id=(0, 1)
)
```

### F2: Node Failure
Fail a node and all its adjacent links.

```python
event = manager.inject_failure(
    'node',
    t_fail=100.0,
    t_repair=200.0,
    node_id=5
)
```

### F3: SRLG (Shared Risk Link Group) Failure
Fail multiple links simultaneously that share a common risk (e.g., same conduit).

```python
srlg_links = [(0, 1), (2, 3), (4, 5)]
event = manager.inject_failure(
    'srlg',
    t_fail=100.0,
    t_repair=200.0,
    srlg_links=srlg_links
)
```

### F4: Geographic Failure
Fail all links within a hop-radius of a center node (disaster scenario).

```python
event = manager.inject_failure(
    'geo',
    t_fail=100.0,
    t_repair=200.0,
    center_node=5,
    hop_radius=2
)
```

## Core Classes

### FailureManager

Main class for managing network failures.

**Methods:**
- `inject_failure()`: Inject a failure into the network
- `is_path_feasible()`: Check if a path avoids failed links
- `repair_failures()`: Repair failures at scheduled time
- `get_affected_links()`: Get list of currently failed links
- `get_failure_count()`: Get number of active failures
- `clear_all_failures()`: Clear all failures (for testing)

## Usage Example

```python
import networkx as nx
from fusion.modules.failures import FailureManager

# Create topology
topology = nx.Graph()
topology.add_edges_from([(0, 1), (1, 2), (2, 3), (0, 3)])

# Initialize FailureManager
engine_props = {'seed': 42}
manager = FailureManager(engine_props, topology)

# Inject geographic failure
event = manager.inject_failure(
    'geo',
    t_fail=100.0,
    t_repair=200.0,
    center_node=1,
    hop_radius=1
)

print(f"Failed links: {event['failed_links']}")
print(f"Affected nodes: {event['meta']['affected_nodes']}")

# Check path feasibility
path1 = [0, 1, 2]
path2 = [0, 3, 2]

print(f"Path {path1} feasible: {manager.is_path_feasible(path1)}")
print(f"Path {path2} feasible: {manager.is_path_feasible(path2)}")

# Repair failures at scheduled time
repaired_links = manager.repair_failures(200.0)
print(f"Repaired links: {repaired_links}")
```

## Architecture

### Module Structure

```
fusion/modules/failures/
├── __init__.py              # Public API exports
├── README.md                # This file
├── TODO.md                  # Development roadmap and beta status
├── errors.py                # Custom exceptions
├── failure_manager.py       # Main FailureManager class
├── failure_types.py         # Failure type implementations
├── registry.py              # Failure handler registry
└── tests/                   # Unit and integration tests
```

### Design Patterns

**Registry Pattern**: Failure types are registered in a central registry for dynamic lookup.

```python
from fusion.modules.failures import get_failure_handler

handler = get_failure_handler('link')
event = handler(topology, link_id=(0, 1), t_fail=10.0, t_repair=20.0)
```

**Extensibility**: Custom failure types can be registered:

```python
from fusion.modules.failures import register_failure_type

def custom_failure(topology, t_fail, t_repair, **kwargs):
    # Custom failure logic
    return {'failure_type': 'custom', ...}

register_failure_type('custom', custom_failure)
```

## Integration

### With SimulationEngine

```python
from fusion.core.simulation import SimulationEngine
from fusion.modules.failures import FailureManager

class SimulationEngine:
    def __init__(self, engine_props):
        # Initialize topology
        self.topology = load_topology(...)

        # Initialize FailureManager
        self.failure_manager = FailureManager(engine_props, self.topology)

    def run(self):
        while events:
            # Check for repairs
            repaired = self.failure_manager.repair_failures(current_time)
```

### With SDN Controller

```python
def route_request(self, request, sdn_props):
    path = self.router.route(request['source'], request['destination'])

    # Check feasibility
    if self.failure_manager and not self.failure_manager.is_path_feasible(path):
        return None  # Path blocked by failure

    return path
```

## Testing

Run the test suite:

```bash
pytest fusion/modules/failures/tests/ -v --cov=fusion.modules.failures
```

Target coverage: 85% (currently in beta - see TODO.md for testing roadmap)

## Error Handling

The module defines custom exceptions:

- `FailureError`: Base exception
- `FailureConfigError`: Invalid configuration
- `FailureNotFoundError`: Failure not found
- `InvalidFailureTypeError`: Unknown failure type

```python
from fusion.modules.failures import FailureConfigError

try:
    manager.inject_failure('link', t_fail=20.0, t_repair=10.0, link_id=(0, 1))
except FailureConfigError as e:
    print(f"Configuration error: {e}")
```

## Performance

- **Failure injection**: O(1) for link failures, O(d) for node failures (d=node degree), O(k) for SRLG failures (k=SRLG size), O(V+E) for geographic failures
- **Path feasibility**: O(H) where H is path hops
- **Memory**: O(L) where L is total failed links (note: one node/geo failure can affect multiple links)

## Dependencies

- **NetworkX**: Graph operations and shortest path algorithms
- **Python 3.11+**: Type hints and modern syntax

## Version History

- **v1.0.0-beta** (2025-10-15): Initial beta implementation
  - F1, F2, F3, F4 failure types
  - FailureManager with full API
  - Requires additional testing and validation before production use

## Related Documentation

- [FUSION Coding Standards](../../../CODING_STANDARDS.md)
- [Testing Standards](../../../TESTING_STANDARDS.md)
- **Note**: Detailed specifications are being migrated to Sphinx documentation

## Contact

For questions or issues, please refer to the main FUSION repository.
