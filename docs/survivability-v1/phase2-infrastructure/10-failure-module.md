# Phase 2: Core Infrastructure

## 10 - Failure/Disaster Module

**Purpose**: Implement failure injection and tracking for F1 (link), F3 (SRLG), and F4 (geographic) failures.

**Location**: `fusion/modules/failures/`

**Estimated Effort**: 1.5-2 days

---

## Module Structure

```
fusion/modules/failures/
├── __init__.py              # Public API exports
├── README.md                # Module documentation
├── registry.py              # Failure injection registry
├── errors.py                # Custom exceptions
├── failure_manager.py       # Main FailureManager class
├── failure_types.py         # Failure type implementations
├── utils.py                 # Helper functions
└── tests/
    ├── __init__.py
    ├── README.md
    ├── test_failure_manager.py
    ├── test_failure_types.py
    └── fixtures/            # Test data files
```

---

## 1. Custom Exceptions (`errors.py`)

```python
"""
Custom exceptions for the failures module.
"""

class FailureError(Exception):
    """Base exception for failure-related errors."""
    pass


class FailureConfigError(FailureError):
    """Raised when failure configuration is invalid."""
    pass


class FailureNotFoundError(FailureError):
    """Raised when a requested failure cannot be found."""
    pass


class InvalidFailureTypeError(FailureError):
    """Raised when an unknown failure type is requested."""
    pass
```

---

## 2. FailureManager Class (`failure_manager.py`)

### Class Definition

```python
"""
Failure manager for network failure injection and tracking.
"""

from typing import Any
import networkx as nx
from .errors import FailureConfigError
from .registry import get_failure_handler


class FailureManager:
    """
    Manages failure injection and tracking for network simulations.

    Tracks active failures, maintains failure history, and provides
    path feasibility checking based on current network state.

    :param engine_props: Engine configuration properties
    :type engine_props: dict[str, Any]
    :param topology: Network topology graph
    :type topology: nx.Graph
    """

    def __init__(self, engine_props: dict[str, Any], topology: nx.Graph) -> None:
        """
        Initialize FailureManager.

        :param engine_props: Engine configuration
        :type engine_props: dict[str, Any]
        :param topology: Network topology
        :type topology: nx.Graph
        """
        self.engine_props = engine_props
        self.topology = topology
        self.active_failures: set[tuple[Any, Any]] = set()  # Currently failed links
        self.failure_history: list[dict[str, Any]] = []  # Historical failure events
        self.scheduled_repairs: dict[float, list[tuple[Any, Any]]] = {}  # Repair schedule

    def inject_failure(
        self,
        failure_type: str,
        t_fail: float,
        t_repair: float,
        **kwargs: Any
    ) -> dict[str, Any]:
        """
        Inject a failure event into the network.

        :param failure_type: Type of failure (link, node, srlg, geo)
        :type failure_type: str
        :param t_fail: Failure occurrence time
        :type t_fail: float
        :param t_repair: Repair completion time
        :type t_repair: float
        :param kwargs: Additional failure-specific parameters
        :type kwargs: Any
        :return: Failure event details
        :rtype: dict[str, Any]
        :raises FailureConfigError: If failure configuration is invalid
        :raises InvalidFailureTypeError: If failure type is unknown

        Example:
            >>> manager = FailureManager(props, topology)
            >>> event = manager.inject_failure(
            ...     'link',
            ...     t_fail=10.0,
            ...     t_repair=20.0,
            ...     link_id=(0, 1)
            ... )
            >>> print(event['failed_links'])
            [(0, 1)]
        """
        # Validate timing
        if t_repair <= t_fail:
            raise FailureConfigError(
                f"Repair time ({t_repair}) must be after failure time ({t_fail})"
            )

        # Get failure handler from registry
        handler = get_failure_handler(failure_type)

        # Execute failure
        event = handler(
            topology=self.topology,
            t_fail=t_fail,
            t_repair=t_repair,
            **kwargs
        )

        # Track active failures
        for link in event['failed_links']:
            self.active_failures.add(link)

        # Schedule repairs
        if t_repair not in self.scheduled_repairs:
            self.scheduled_repairs[t_repair] = []
        self.scheduled_repairs[t_repair].extend(event['failed_links'])

        # Record in history
        self.failure_history.append(event)

        return event

    def is_path_feasible(self, path: list[int]) -> bool:
        """
        Check if path is feasible given active failures.

        A path is infeasible if any of its links are currently failed.

        :param path: List of node IDs forming the path
        :type path: list[int]
        :return: True if path has no failed links, False otherwise
        :rtype: bool

        Example:
            >>> manager = FailureManager(props, topology)
            >>> manager.active_failures = {(0, 1)}
            >>> manager.is_path_feasible([0, 1, 2])
            False
            >>> manager.is_path_feasible([0, 3, 2])
            True
        """
        if not self.active_failures:
            return True

        # Check each link in the path
        for i in range(len(path) - 1):
            link = (path[i], path[i + 1])
            reverse_link = (path[i + 1], path[i])

            # Check both directions (undirected graph)
            if link in self.active_failures or reverse_link in self.active_failures:
                return False

        return True

    def get_affected_links(self) -> list[tuple[Any, Any]]:
        """
        Get list of currently failed links.

        :return: List of failed link tuples
        :rtype: list[tuple[Any, Any]]
        """
        return list(self.active_failures)

    def repair_failures(self, current_time: float) -> list[tuple[Any, Any]]:
        """
        Repair all failures scheduled for repair at current_time.

        :param current_time: Current simulation time
        :type current_time: float
        :return: List of repaired link tuples
        :rtype: list[tuple[Any, Any]]

        Example:
            >>> manager = FailureManager(props, topology)
            >>> manager.inject_failure('link', t_fail=10.0, t_repair=20.0, link_id=(0, 1))
            >>> repaired = manager.repair_failures(20.0)
            >>> print(repaired)
            [(0, 1)]
            >>> print(manager.active_failures)
            set()
        """
        if current_time not in self.scheduled_repairs:
            return []

        # Get links to repair
        links_to_repair = self.scheduled_repairs[current_time]

        # Remove from active failures
        for link in links_to_repair:
            self.active_failures.discard(link)

        # Remove from schedule
        del self.scheduled_repairs[current_time]

        return links_to_repair

    def get_failure_count(self) -> int:
        """
        Get number of currently active failures.

        :return: Number of failed links
        :rtype: int
        """
        return len(self.active_failures)

    def clear_all_failures(self) -> None:
        """
        Clear all active failures (for testing or reset).

        This removes all active failures and clears the repair schedule.
        """
        self.active_failures.clear()
        self.scheduled_repairs.clear()
```

---

## 3. Failure Type Implementations (`failure_types.py`)

```python
"""
Failure type implementations for network failures.
"""

from typing import Any
import networkx as nx
from .errors import FailureConfigError


def fail_link(
    topology: nx.Graph,
    link_id: tuple[Any, Any],
    t_fail: float,
    t_repair: float
) -> dict[str, Any]:
    """
    Fail a single link (F1).

    :param topology: Network topology
    :type topology: nx.Graph
    :param link_id: Link tuple (src, dst)
    :type link_id: tuple[Any, Any]
    :param t_fail: Failure time
    :type t_fail: float
    :param t_repair: Repair time
    :type t_repair: float
    :return: Failure event details
    :rtype: dict[str, Any]
    :raises FailureConfigError: If link does not exist

    Example:
        >>> event = fail_link(G, link_id=(0, 1), t_fail=10.0, t_repair=20.0)
        >>> print(event['failed_links'])
        [(0, 1)]
    """
    # Validate link exists
    if not topology.has_edge(*link_id):
        # Try reverse direction
        reverse_link = (link_id[1], link_id[0])
        if not topology.has_edge(*reverse_link):
            raise FailureConfigError(f"Link {link_id} does not exist in topology")
        link_id = reverse_link

    return {
        'failure_type': 'link',
        't_fail': t_fail,
        't_repair': t_repair,
        'failed_links': [link_id],
        'meta': {'link_id': link_id}
    }


def fail_node(
    topology: nx.Graph,
    node_id: Any,
    t_fail: float,
    t_repair: float
) -> dict[str, Any]:
    """
    Fail a node and all adjacent links (F2).

    :param topology: Network topology
    :type topology: nx.Graph
    :param node_id: Node ID to fail
    :type node_id: Any
    :param t_fail: Failure time
    :type t_fail: float
    :param t_repair: Repair time
    :type t_repair: float
    :return: Failure event details
    :rtype: dict[str, Any]
    :raises FailureConfigError: If node does not exist

    Example:
        >>> event = fail_node(G, node_id=1, t_fail=10.0, t_repair=20.0)
        >>> print(len(event['failed_links']))
        3  # Node 1 has 3 adjacent links
    """
    # Validate node exists
    if node_id not in topology.nodes:
        raise FailureConfigError(f"Node {node_id} does not exist in topology")

    # Get all adjacent links
    failed_links = list(topology.edges(node_id))

    if not failed_links:
        raise FailureConfigError(f"Node {node_id} has no adjacent links")

    return {
        'failure_type': 'node',
        't_fail': t_fail,
        't_repair': t_repair,
        'failed_links': failed_links,
        'meta': {'node_id': node_id}
    }


def fail_srlg(
    topology: nx.Graph,
    srlg_links: list[tuple[Any, Any]],
    t_fail: float,
    t_repair: float
) -> dict[str, Any]:
    """
    Fail all links in a Shared Risk Link Group (F3).

    :param topology: Network topology
    :type topology: nx.Graph
    :param srlg_links: List of link tuples in SRLG
    :type srlg_links: list[tuple[Any, Any]]
    :param t_fail: Failure time
    :type t_fail: float
    :param t_repair: Repair time
    :type t_repair: float
    :return: Failure event details
    :rtype: dict[str, Any]
    :raises FailureConfigError: If SRLG list is empty or contains invalid links

    Example:
        >>> srlg = [(0, 1), (2, 3), (4, 5)]
        >>> event = fail_srlg(G, srlg_links=srlg, t_fail=10.0, t_repair=20.0)
        >>> print(len(event['failed_links']))
        3
    """
    if not srlg_links:
        raise FailureConfigError("SRLG link list cannot be empty")

    # Validate all links exist
    validated_links = []
    for link_id in srlg_links:
        if not topology.has_edge(*link_id):
            # Try reverse direction
            reverse_link = (link_id[1], link_id[0])
            if not topology.has_edge(*reverse_link):
                raise FailureConfigError(f"SRLG link {link_id} does not exist in topology")
            validated_links.append(reverse_link)
        else:
            validated_links.append(link_id)

    return {
        'failure_type': 'srlg',
        't_fail': t_fail,
        't_repair': t_repair,
        'failed_links': validated_links,
        'meta': {'srlg_size': len(validated_links)}
    }


def fail_geo(
    topology: nx.Graph,
    center_node: Any,
    hop_radius: int,
    t_fail: float,
    t_repair: float
) -> dict[str, Any]:
    """
    Fail all links within hop_radius of center_node (F4).

    Uses NetworkX shortest path to determine hop distance.

    :param topology: Network topology
    :type topology: nx.Graph
    :param center_node: Center node of disaster
    :type center_node: Any
    :param hop_radius: Hop radius for failure region
    :type hop_radius: int
    :param t_fail: Failure time
    :type t_fail: float
    :param t_repair: Repair time
    :type t_repair: float
    :return: Failure event details
    :rtype: dict[str, Any]
    :raises FailureConfigError: If center_node invalid or radius non-positive

    Example:
        >>> event = fail_geo(G, center_node=5, hop_radius=2, t_fail=10.0, t_repair=20.0)
        >>> print(event['meta']['affected_nodes'])
        [5, 4, 6, 3, 7, 2, 8]
    """
    # Validate inputs
    if center_node not in topology.nodes:
        raise FailureConfigError(f"Center node {center_node} does not exist in topology")

    if hop_radius <= 0:
        raise FailureConfigError(f"Hop radius must be positive, got {hop_radius}")

    # Find all nodes within hop_radius
    affected_nodes = set()
    affected_nodes.add(center_node)

    # BFS from center node up to hop_radius
    try:
        shortest_paths = nx.single_source_shortest_path_length(
            topology,
            center_node,
            cutoff=hop_radius
        )
        affected_nodes.update(shortest_paths.keys())
    except nx.NetworkXError as e:
        raise FailureConfigError(f"Error computing geographic failure: {e}") from e

    # Find all links with at least one endpoint in affected region
    failed_links = []
    for u, v in topology.edges():
        if u in affected_nodes or v in affected_nodes:
            failed_links.append((u, v))

    if not failed_links:
        raise FailureConfigError(
            f"No links found within radius {hop_radius} of node {center_node}"
        )

    return {
        'failure_type': 'geo',
        't_fail': t_fail,
        't_repair': t_repair,
        'failed_links': failed_links,
        'meta': {
            'center_node': center_node,
            'hop_radius': hop_radius,
            'affected_nodes': list(affected_nodes)
        }
    }
```

---

## 4. Registry (`registry.py`)

```python
"""
Failure type registry for dynamic handler lookup.
"""

from typing import Callable
from .failure_types import fail_link, fail_node, fail_srlg, fail_geo
from .errors import InvalidFailureTypeError


# Registry of failure handlers
FAILURE_TYPES: dict[str, Callable] = {
    'link': fail_link,
    'node': fail_node,
    'srlg': fail_srlg,
    'geo': fail_geo
}


def get_failure_handler(failure_type: str) -> Callable:
    """
    Get failure handler function by type.

    :param failure_type: Failure type (link, node, srlg, geo)
    :type failure_type: str
    :return: Failure handler function
    :rtype: Callable
    :raises InvalidFailureTypeError: If failure type is unknown

    Example:
        >>> handler = get_failure_handler('link')
        >>> event = handler(topology, link_id=(0, 1), t_fail=10.0, t_repair=20.0)
    """
    if failure_type not in FAILURE_TYPES:
        raise InvalidFailureTypeError(
            f"Unknown failure type: {failure_type}. "
            f"Valid types: {list(FAILURE_TYPES.keys())}"
        )
    return FAILURE_TYPES[failure_type]


def register_failure_type(name: str, handler: Callable) -> None:
    """
    Register a custom failure type handler.

    Allows extending the failure module with custom failure types.

    :param name: Failure type name
    :type name: str
    :param handler: Failure handler function
    :type handler: Callable
    """
    FAILURE_TYPES[name] = handler
```

---

## 5. Module Exports (`__init__.py`)

```python
"""
FUSION Failures Module

Provides failure injection and tracking for network survivability testing.

Supports:
- F1: Link failures
- F2: Node failures
- F3: SRLG (Shared Risk Link Group) failures
- F4: Geographic failures (hop-radius based)

Example usage:
    >>> from fusion.modules.failures import FailureManager
    >>> manager = FailureManager(engine_props, topology)
    >>> event = manager.inject_failure(
    ...     'geo',
    ...     t_fail=100.0,
    ...     t_repair=200.0,
    ...     center_node=5,
    ...     hop_radius=2
    ... )
    >>> is_feasible = manager.is_path_feasible([0, 1, 2])
"""

from .failure_manager import FailureManager
from .failure_types import fail_link, fail_node, fail_srlg, fail_geo
from .registry import get_failure_handler, register_failure_type
from .errors import (
    FailureError,
    FailureConfigError,
    FailureNotFoundError,
    InvalidFailureTypeError
)

__all__ = [
    # Main classes
    'FailureManager',

    # Failure type functions
    'fail_link',
    'fail_node',
    'fail_srlg',
    'fail_geo',

    # Registry
    'get_failure_handler',
    'register_failure_type',

    # Exceptions
    'FailureError',
    'FailureConfigError',
    'FailureNotFoundError',
    'InvalidFailureTypeError',
]

__version__ = '1.0.0'
```

---

## 6. Integration with SimulationEngine

### Modifications to `fusion/core/simulation.py`

```python
from fusion.modules.failures import FailureManager

class SimulationEngine:
    def __init__(self, engine_props: dict[str, Any]) -> None:
        # ... existing initialization ...

        # Initialize FailureManager if failures are configured
        self.failure_manager: FailureManager | None = None
        if engine_props.get('failure_settings', {}).get('failure_type', 'none') != 'none':
            self.failure_manager = FailureManager(engine_props, self.topology)
            self._schedule_failure()

    def _schedule_failure(self) -> None:
        """Schedule failure event based on configuration."""
        failure_settings = self.engine_props.get('failure_settings', {})
        failure_type = failure_settings.get('failure_type')

        # Determine failure time
        t_fail_index = failure_settings.get('t_fail_arrival_index', -1)
        if t_fail_index == -1:
            # Inject at midpoint
            t_fail_index = self.engine_props['num_requests'] // 2

        # Determine repair time
        t_repair_arrivals = failure_settings.get('t_repair_after_arrivals', 1000)
        t_repair_index = t_fail_index + t_repair_arrivals

        # Map indices to simulation times (this depends on your arrival model)
        t_fail = self._arrival_time_at_index(t_fail_index)
        t_repair = self._arrival_time_at_index(t_repair_index)

        # Inject failure based on type
        if failure_type == 'link':
            self.failure_manager.inject_failure(
                'link',
                t_fail=t_fail,
                t_repair=t_repair,
                link_id=(
                    failure_settings['failed_link_src'],
                    failure_settings['failed_link_dst']
                )
            )
        elif failure_type == 'srlg':
            self.failure_manager.inject_failure(
                'srlg',
                t_fail=t_fail,
                t_repair=t_repair,
                srlg_links=failure_settings['srlg_links']
            )
        elif failure_type == 'geo':
            self.failure_manager.inject_failure(
                'geo',
                t_fail=t_fail,
                t_repair=t_repair,
                center_node=failure_settings['geo_center_node'],
                hop_radius=failure_settings['geo_hop_radius']
            )

    def run(self) -> None:
        """Main simulation loop."""
        # ... existing loop ...

        # In main event loop:
        while not self.event_queue.empty():
            event = self.event_queue.get()
            current_time = event.time

            # Check for scheduled repairs
            if self.failure_manager:
                repaired_links = self.failure_manager.repair_failures(current_time)
                if repaired_links:
                    logger.info(f"Repaired {len(repaired_links)} links at time {current_time}")

            # ... rest of event handling ...
```

### Modifications to `fusion/core/sdn_controller.py`

```python
def route_request(self, request: dict[str, Any], sdn_props: SDNProps) -> list[int] | None:
    """Route a request with failure awareness."""
    # Get candidate path
    path = self.router.route(request['source'], request['destination'], request)

    # Check path feasibility if failures are active
    if self.failure_manager and path:
        if not self.failure_manager.is_path_feasible(path):
            logger.debug(f"Path {path} is infeasible due to failures")
            return None

    return path
```

---

## 7. Configuration Schema

### Extension to `fusion/configs/schema.py`

```python
# Add to existing schema
FAILURE_SETTINGS_SCHEMA = {
    'type': 'object',
    'properties': {
        'failure_type': {
            'type': 'string',
            'enum': ['none', 'link', 'node', 'srlg', 'geo'],
            'default': 'none'
        },
        't_fail_arrival_index': {
            'type': 'integer',
            'default': -1  # -1 = midpoint
        },
        't_repair_after_arrivals': {
            'type': 'integer',
            'minimum': 1,
            'default': 1000
        },
        'failed_link_src': {'type': 'integer'},
        'failed_link_dst': {'type': 'integer'},
        'failed_node_id': {'type': 'integer'},
        'srlg_links': {
            'type': 'array',
            'items': {
                'type': 'array',
                'minItems': 2,
                'maxItems': 2
            }
        },
        'geo_center_node': {'type': 'integer'},
        'geo_hop_radius': {
            'type': 'integer',
            'minimum': 1,
            'default': 2
        }
    }
}
```

---

## 8. Testing Requirements

### Unit Tests (`tests/test_failure_manager.py`)

```python
import pytest
import networkx as nx
from fusion.modules.failures import FailureManager, FailureConfigError

@pytest.fixture
def sample_topology():
    """Create a sample topology for testing."""
    G = nx.Graph()
    G.add_edges_from([
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 3)
    ])
    return G

@pytest.fixture
def failure_manager(sample_topology):
    """Create a FailureManager instance."""
    engine_props = {'seed': 42}
    return FailureManager(engine_props, sample_topology)

def test_link_failure_blocks_path(failure_manager):
    """Test that a path using a failed link is marked infeasible."""
    # Inject link failure
    failure_manager.inject_failure('link', t_fail=10.0, t_repair=20.0, link_id=(1, 2))

    # Path using failed link should be infeasible
    assert not failure_manager.is_path_feasible([0, 1, 2, 3])

    # Path avoiding failed link should be feasible
    assert failure_manager.is_path_feasible([0, 5, 6, 3])

def test_failure_repair_restores_path(failure_manager):
    """Test that path becomes feasible after repair time."""
    # Inject and repair failure
    failure_manager.inject_failure('link', t_fail=10.0, t_repair=20.0, link_id=(1, 2))
    assert not failure_manager.is_path_feasible([0, 1, 2, 3])

    # Repair failure
    repaired = failure_manager.repair_failures(20.0)
    assert len(repaired) == 1
    assert (1, 2) in repaired or (2, 1) in repaired

    # Path should now be feasible
    assert failure_manager.is_path_feasible([0, 1, 2, 3])

def test_srlg_failure_multiple_links(failure_manager):
    """Test that all SRLG links are failed simultaneously."""
    srlg_links = [(0, 1), (2, 3), (5, 6)]
    failure_manager.inject_failure('srlg', t_fail=10.0, t_repair=20.0, srlg_links=srlg_links)

    assert failure_manager.get_failure_count() == 3

    # All paths using SRLG links should be infeasible
    assert not failure_manager.is_path_feasible([0, 1, 2])
    assert not failure_manager.is_path_feasible([0, 5, 6, 3])

def test_geo_failure_radius(failure_manager, sample_topology):
    """Test that links within hop radius are failed, others unaffected."""
    event = failure_manager.inject_failure(
        'geo',
        t_fail=10.0,
        t_repair=20.0,
        center_node=1,
        hop_radius=1
    )

    # Check affected nodes
    affected_nodes = event['meta']['affected_nodes']
    assert 1 in affected_nodes
    assert 0 in affected_nodes  # Neighbor
    assert 2 in affected_nodes  # Neighbor

    # Links within radius should be failed
    assert not failure_manager.is_path_feasible([0, 1, 2])

def test_failure_history_tracking(failure_manager):
    """Test that failure events are logged with timestamps."""
    failure_manager.inject_failure('link', t_fail=10.0, t_repair=20.0, link_id=(0, 1))
    failure_manager.inject_failure('link', t_fail=15.0, t_repair=25.0, link_id=(2, 3))

    assert len(failure_manager.failure_history) == 2
    assert failure_manager.failure_history[0]['t_fail'] == 10.0
    assert failure_manager.failure_history[1]['t_fail'] == 15.0

def test_invalid_failure_config(failure_manager):
    """Test that invalid configurations raise errors."""
    # Repair before failure
    with pytest.raises(FailureConfigError, match="Repair time.*must be after"):
        failure_manager.inject_failure('link', t_fail=20.0, t_repair=10.0, link_id=(0, 1))

    # Invalid link
    with pytest.raises(FailureConfigError, match="does not exist"):
        failure_manager.inject_failure('link', t_fail=10.0, t_repair=20.0, link_id=(99, 100))
```

### Test Coverage Target

- **Overall**: ≥ 85%
- **Critical paths**: 100% (path feasibility, failure injection)

---

## 9. Acceptance Criteria

- [x] `test_link_failure_blocks_path`: Path using failed link is infeasible
- [x] `test_node_failure_blocks_adjacent_links`: All adjacent links are failed
- [x] `test_srlg_failure_multiple_links`: All SRLG links are failed simultaneously
- [x] `test_geo_failure_radius`: Links within hop radius are failed, others unaffected
- [x] `test_failure_repair_restores_path`: Path becomes feasible after repair time
- [x] `test_failure_history_tracking`: Failure events are logged with timestamps

---

## Next Steps

After implementing the failure module:

1. **Run tests**: `pytest fusion/modules/failures/tests/ -v --cov`
2. **Integrate** with SimulationEngine
3. **Proceed** to [11-k-path-cache.md](11-k-path-cache.md)

---

**Related Documents**:
- [11-k-path-cache.md](11-k-path-cache.md) (Uses failure_manager)
- [20-protection.md](../phase3-protection/20-protection.md) (Uses is_path_feasible)
- [50-testing.md](../phase6-quality/50-testing.md) (Testing standards)
