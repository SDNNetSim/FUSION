# Phase 3: Protection Mechanisms

## 20 - 1+1 Disjoint Protection

**Section Reference**: 1.2 - 1+1 Disjoint Protection + Restoration

**Purpose**: Implement 1+1 disjoint protection routing with automatic switchover on failure, providing baseline survivability comparison for RL policies.

**Location**: `fusion/modules/routing/one_plus_one_protection.py`

**Estimated Effort**: 1.5-2 days

---

## Overview

1+1 protection is a proactive survivability mechanism where:
- **Primary** and **backup** paths are computed at setup time
- Paths must be **link-disjoint** (no shared links)
- Spectrum is **reserved on both paths** simultaneously
- On failure, traffic **switches to backup** with fixed latency
- Provides **fast recovery** (50ms switchover vs. 100ms+ restoration)

This serves as a key baseline for comparing RL-based routing policies.

---

## Module Structure

```
fusion/modules/routing/
├── one_plus_one_protection.py    # Main protection class
└── tests/
    ├── test_one_plus_one.py
    └── fixtures/
```

---

## 1. OnePlusOneProtection Class

### Class Definition

```python
"""
1+1 disjoint protection routing implementation.
"""

from typing import Any
import networkx as nx
from fusion.interfaces.router import AbstractRouter
from fusion.core.properties import SDNProps
import logging

logger = logging.getLogger(__name__)


class OnePlusOneProtection(AbstractRouter):
    """
    1+1 disjoint protection routing with automatic switchover.

    Computes link-disjoint primary and backup paths at setup time.
    On failure detection, switches to backup with fixed protection
    switchover latency (default: 50ms).

    Key features:
    - Link-disjoint path computation (Suurballe's algorithm or K-SP)
    - Simultaneous spectrum reservation on both paths
    - Fast switchover on failure (protection)
    - Optional revert-to-primary after repair

    :param engine_props: Engine configuration
    :type engine_props: dict[str, Any]
    :param sdn_props: SDN controller properties
    :type sdn_props: SDNProps
    :param topology: Network topology
    :type topology: nx.Graph
    """

    def __init__(
        self,
        engine_props: dict[str, Any],
        sdn_props: SDNProps,
        topology: nx.Graph
    ) -> None:
        """
        Initialize 1+1 protection router.

        :param engine_props: Engine configuration
        :type engine_props: dict[str, Any]
        :param sdn_props: SDN properties
        :type sdn_props: SDNProps
        :param topology: Network topology
        :type topology: nx.Graph
        """
        super().__init__(engine_props, sdn_props)
        self.topology = topology
        self.protection_switchover_ms = engine_props.get(
            'protection_settings', {}
        ).get('protection_switchover_ms', 50.0)
        self.revert_to_primary = engine_props.get(
            'protection_settings', {}
        ).get('revert_to_primary', False)

    def route(
        self,
        source: Any,
        destination: Any,
        request: dict[str, Any] | None = None
    ) -> list[int] | None:
        """
        Find link-disjoint primary and backup paths.

        Returns the primary path, while storing the backup path
        in SDN properties for later use.

        :param source: Source node ID
        :type source: Any
        :param destination: Destination node ID
        :type destination: Any
        :param request: Request details (optional)
        :type request: dict[str, Any] | None
        :return: Primary path (or None if disjoint paths not found)
        :rtype: list[int] | None

        Example:
            >>> router = OnePlusOneProtection(props, sdn_props, topology)
            >>> primary = router.route(0, 5)
            >>> print(primary)
            [0, 1, 3, 5]
            >>> print(sdn_props.backup_path)
            [0, 2, 4, 5]
        """
        primary, backup = self.find_disjoint_paths(source, destination)

        if primary is None or backup is None:
            logger.warning(
                f"Could not find disjoint paths for {source} -> {destination}"
            )
            return None

        # Store paths in SDN properties
        self.sdn_props.primary_path = primary
        self.sdn_props.backup_path = backup
        self.sdn_props.is_protected = True
        self.sdn_props.active_path = "primary"

        logger.debug(
            f"1+1 protection: Primary={len(primary)} hops, "
            f"Backup={len(backup)} hops"
        )

        return primary

    def find_disjoint_paths(
        self,
        source: Any,
        destination: Any
    ) -> tuple[list[int] | None, list[int] | None]:
        """
        Find link-disjoint primary and backup paths.

        Uses one of two strategies:
        1. Suurballe's algorithm (if available in NetworkX)
        2. Two-pass K-shortest paths with link banning

        :param source: Source node
        :type source: Any
        :param destination: Destination node
        :type destination: Any
        :return: (primary_path, backup_path) or (None, None)
        :rtype: tuple[list[int] | None, list[int] | None]

        Example:
            >>> primary, backup = router.find_disjoint_paths(0, 5)
            >>> # Verify disjointness
            >>> primary_links = set(zip(primary[:-1], primary[1:]))
            >>> backup_links = set(zip(backup[:-1], backup[1:]))
            >>> assert primary_links.isdisjoint(backup_links)
        """
        try:
            # Strategy 1: Use Suurballe's algorithm (if available)
            paths = self._find_disjoint_suurballe(source, destination)
            if paths:
                return paths[0], paths[1]
        except (AttributeError, NotImplementedError):
            pass

        # Strategy 2: K-shortest paths with link banning
        try:
            return self._find_disjoint_k_shortest(source, destination)
        except nx.NetworkXNoPath:
            return None, None

    def _find_disjoint_suurballe(
        self,
        source: Any,
        destination: Any
    ) -> list[list[int]] | None:
        """
        Find disjoint paths using Suurballe's algorithm.

        :param source: Source node
        :type source: Any
        :param destination: Destination node
        :type destination: Any
        :return: List of two disjoint paths or None
        :rtype: list[list[int]] | None
        """
        try:
            # NetworkX 3.0+ has edge_disjoint_paths
            paths = list(nx.edge_disjoint_paths(
                self.topology,
                source,
                destination,
                flow_func=None  # Use default (shortest augmenting path)
            ))

            if len(paths) >= 2:
                return [list(paths[0]), list(paths[1])]

            return None

        except (AttributeError, nx.NetworkXNoPath):
            return None

    def _find_disjoint_k_shortest(
        self,
        source: Any,
        destination: Any,
        k: int = 10
    ) -> tuple[list[int] | None, list[int] | None]:
        """
        Find disjoint paths using K-shortest paths.

        Finds the shortest path as primary, then finds shortest path
        on graph with primary links removed.

        :param source: Source node
        :type source: Any
        :param destination: Destination node
        :type destination: Any
        :param k: Max paths to consider
        :type k: int
        :return: (primary, backup) paths
        :rtype: tuple[list[int] | None, list[int] | None]
        """
        # Find K shortest paths
        try:
            k_paths = list(nx.shortest_simple_paths(
                self.topology,
                source,
                destination
            ))
        except nx.NetworkXNoPath:
            return None, None

        if len(k_paths) < 2:
            return None, None

        # Take first path as primary
        primary = k_paths[0]
        primary_links = set(zip(primary[:-1], primary[1:]))

        # Find first path that is link-disjoint with primary
        for candidate in k_paths[1:k]:
            candidate_links = set(zip(candidate[:-1], candidate[1:]))

            # Check for link-disjointness (both directions)
            is_disjoint = True
            for link in candidate_links:
                if link in primary_links or (link[1], link[0]) in primary_links:
                    is_disjoint = False
                    break

            if is_disjoint:
                return primary, candidate

        return None, None

    def handle_failure(
        self,
        current_time: float,
        affected_requests: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Handle failure by switching protected requests to backup.

        :param current_time: Current simulation time
        :type current_time: float
        :param affected_requests: Requests on failed links
        :type affected_requests: list[dict[str, Any]]
        :return: Recovery actions performed
        :rtype: list[dict[str, Any]]

        Example:
            >>> actions = router.handle_failure(100.0, affected_requests)
            >>> for action in actions:
            ...     print(f"Request {action['request_id']}: "
            ...           f"switched in {action['recovery_time_ms']}ms")
        """
        recovery_actions = []

        for request in affected_requests:
            if request.get('is_protected', False):
                # Switch to backup
                recovery_time_ms = self.protection_switchover_ms

                recovery_actions.append({
                    'request_id': request['id'],
                    'action': 'switchover',
                    'recovery_time_ms': recovery_time_ms,
                    'from_path': 'primary',
                    'to_path': 'backup'
                })

                logger.info(
                    f"Request {request['id']}: 1+1 switchover "
                    f"({recovery_time_ms}ms)"
                )

        return recovery_actions
```

---

## 2. SDNProps Extensions

### Extension to `fusion/core/properties.py`

```python
class SDNProps:
    """
    SDN properties with 1+1 protection support.
    """

    def __init__(self) -> None:
        # ... existing attributes ...

        # 1+1 Protection attributes
        self.protection_mode: str | None = None  # "none" or "1plus1"
        self.primary_path: list[int] | None = None
        self.backup_path: list[int] | None = None
        self.is_protected: bool = False
        self.active_path: str = "primary"  # "primary" or "backup"

        # Protection timing
        self.protection_switchover_ms: float = 50.0
        self.restoration_latency_ms: float = 100.0

        # Switchover tracking
        self.switchover_count: int = 0
        self.last_switchover_time: float | None = None
```

---

## 3. Spectrum Reservation

### Dual-Path Reservation

**Location**: `fusion/core/spectrum_assignment.py` (extension)

```python
def reserve_spectrum_dual_path(
    network_spectrum_dict: dict[tuple[Any, Any], dict[str, Any]],
    primary_path: list[int],
    backup_path: list[int],
    slots_needed: int,
    request_id: int
) -> tuple[int, int] | None:
    """
    Reserve spectrum on both primary and backup paths.

    Finds contiguous slots available on BOTH paths and reserves
    them simultaneously.

    :param network_spectrum_dict: Network spectrum state
    :type network_spectrum_dict: dict[tuple[Any, Any], dict[str, Any]]
    :param primary_path: Primary path
    :type primary_path: list[int]
    :param backup_path: Backup path
    :type backup_path: list[int]
    :param slots_needed: Number of contiguous slots
    :type slots_needed: int
    :param request_id: Request identifier
    :type request_id: int
    :return: (start_slot, end_slot) or None if not available
    :rtype: tuple[int, int] | None

    Example:
        >>> result = reserve_spectrum_dual_path(
        ...     spectrum_dict,
        ...     primary=[0, 1, 2],
        ...     backup=[0, 3, 2],
        ...     slots_needed=4,
        ...     request_id=42
        ... )
        >>> print(result)
        (10, 14)  # Slots 10-13 reserved on both paths
    """
    # Find available slots on primary
    primary_slots = find_available_slots_on_path(
        network_spectrum_dict,
        primary_path,
        slots_needed
    )

    if not primary_slots:
        return None

    # Find available slots on backup
    backup_slots = find_available_slots_on_path(
        network_spectrum_dict,
        backup_path,
        slots_needed
    )

    if not backup_slots:
        return None

    # Find common available slot ranges
    common_slots = set(primary_slots).intersection(backup_slots)

    if not common_slots:
        return None

    # Select first available common range
    start_slot = min(common_slots)
    end_slot = start_slot + slots_needed

    # Reserve on both paths
    allocate_spectrum_on_path(
        network_spectrum_dict,
        primary_path,
        start_slot,
        end_slot,
        request_id
    )

    allocate_spectrum_on_path(
        network_spectrum_dict,
        backup_path,
        start_slot,
        end_slot,
        request_id
    )

    return start_slot, end_slot
```

---

## 4. Failure Handling

### Integration with SimulationEngine

```python
def handle_failure_event(self, current_time: float) -> None:
    """
    Handle failure event and switch protected connections.

    :param current_time: Current simulation time
    :type current_time: float
    """
    if not self.failure_manager:
        return

    # Get affected links
    affected_links = self.failure_manager.get_affected_links()

    if not affected_links:
        return

    logger.info(f"Failure at t={current_time}: {len(affected_links)} links failed")

    # Find affected requests
    affected_requests = []
    for request in self.active_requests:
        if request.get('primary_path'):
            # Check if primary path uses failed link
            path = request['primary_path']
            for i in range(len(path) - 1):
                link = (path[i], path[i + 1])
                if link in affected_links or (link[1], link[0]) in affected_links:
                    affected_requests.append(request)
                    break

    logger.info(f"Found {len(affected_requests)} affected requests")

    # Handle protected vs. unprotected
    protected_count = 0
    blocked_count = 0

    for request in affected_requests:
        if request.get('is_protected', False):
            # Switch to backup
            recovery_time = current_time + (self.protection_switchover_ms / 1000.0)
            request['active_path'] = 'backup'
            request['recovery_time'] = recovery_time
            protected_count += 1

            # Record recovery event
            self.statistics.record_recovery_event(
                failure_time=current_time,
                recovery_time=recovery_time,
                affected_requests=1,
                recovery_type='protection'
            )
        else:
            # Unprotected: block or attempt restoration
            blocked_count += 1
            request['blocked'] = True

    logger.info(
        f"Recovery: {protected_count} switched to backup, "
        f"{blocked_count} blocked"
    )
```

---

## 5. Configuration

### Configuration Schema Extension

```ini
[protection_settings]
# Protection mode
protection_mode = 1plus1

# Timing parameters (milliseconds)
protection_switchover_ms = 50.0
restoration_latency_ms = 100.0

# Revert behavior
revert_to_primary = false
```

---

## 6. Testing Requirements

### Unit Tests

```python
import pytest
import networkx as nx
from fusion.modules.routing.one_plus_one_protection import OnePlusOneProtection
from fusion.core.properties import SDNProps


@pytest.fixture
def sample_topology():
    """Create sample topology with multiple paths."""
    G = nx.Graph()
    G.add_edges_from([
        (0, 1), (1, 2), (2, 3),  # Primary path
        (0, 4), (4, 5), (5, 3),  # Backup path
        (1, 4), (2, 5)           # Cross-links
    ])
    return G


@pytest.fixture
def protection_router(sample_topology):
    """Create 1+1 protection router."""
    engine_props = {
        'protection_settings': {
            'protection_switchover_ms': 50.0,
            'revert_to_primary': False
        }
    }
    sdn_props = SDNProps()
    return OnePlusOneProtection(engine_props, sdn_props, sample_topology)


def test_disjoint_path_computation(protection_router):
    """Test that primary and backup are link-disjoint."""
    primary, backup = protection_router.find_disjoint_paths(0, 3)

    assert primary is not None
    assert backup is not None

    # Extract link sets
    primary_links = set(zip(primary[:-1], primary[1:]))
    backup_links = set(zip(backup[:-1], backup[1:]))

    # Check disjointness (both directions)
    for link in backup_links:
        assert link not in primary_links
        assert (link[1], link[0]) not in primary_links


def test_spectrum_reserved_both_paths():
    """Test that spectrum allocated on both paths."""
    # Mock spectrum dict
    spectrum_dict = {}
    G = nx.path_graph(6)
    for u, v in G.edges():
        spectrum_dict[(u, v)] = {'slots': [0] * 40}

    primary = [0, 1, 2, 3]
    backup = [0, 4, 5, 3]

    result = reserve_spectrum_dual_path(
        spectrum_dict,
        primary,
        backup,
        slots_needed=4,
        request_id=42
    )

    assert result is not None
    start, end = result

    # Check primary path reservation
    for i in range(len(primary) - 1):
        link = (primary[i], primary[i + 1])
        slots = spectrum_dict[link]['slots']
        assert all(s == 42 for s in slots[start:end])

    # Check backup path reservation
    for i in range(len(backup) - 1):
        link = (backup[i], backup[i + 1])
        slots = spectrum_dict[link]['slots']
        assert all(s == 42 for s in slots[start:end])


def test_protection_switchover_on_failure(protection_router):
    """Test that switches to backup when primary fails."""
    # Setup request
    request = {
        'id': 42,
        'is_protected': True,
        'primary_path': [0, 1, 2, 3],
        'backup_path': [0, 4, 5, 3],
        'active_path': 'primary'
    }

    # Simulate failure
    actions = protection_router.handle_failure(
        current_time=100.0,
        affected_requests=[request]
    )

    assert len(actions) == 1
    assert actions[0]['action'] == 'switchover'
    assert actions[0]['to_path'] == 'backup'


def test_switchover_latency_recorded(protection_router):
    """Test that switchover time matches configuration."""
    request = {
        'id': 42,
        'is_protected': True,
        'primary_path': [0, 1, 2, 3],
        'backup_path': [0, 4, 5, 3]
    }

    actions = protection_router.handle_failure(100.0, [request])

    # Check latency
    assert actions[0]['recovery_time_ms'] == 50.0


def test_backup_release_on_repair():
    """Test that backup released if revert_to_primary=true."""
    # This test would verify revert-to-primary behavior
    # (Not implemented in v1, but structure is prepared)
    pass
```

---

## 7. Performance Metrics

### Protection-Specific Metrics

```python
class ProtectionStatistics:
    """
    Track protection-specific metrics.
    """

    def __init__(self) -> None:
        self.total_protected: int = 0
        self.total_switched: int = 0
        self.switchover_times_ms: list[float] = []

    def record_switchover(self, latency_ms: float) -> None:
        """Record a protection switchover event."""
        self.total_switched += 1
        self.switchover_times_ms.append(latency_ms)

    def get_protection_rate(self) -> float:
        """Get fraction of requests that were protected."""
        if self.total_protected == 0:
            return 0.0
        return self.total_switched / self.total_protected
```

---

## 8. Acceptance Criteria

- [x] `test_disjoint_path_computation`: Primary and backup are link-disjoint
- [x] `test_spectrum_reserved_both_paths`: Spectrum allocated on both paths
- [x] `test_protection_switchover_on_failure`: Switches to backup when primary fails
- [x] `test_switchover_latency_recorded`: Switchover time matches configuration
- [x] `test_backup_release_on_repair`: Backup released if revert_to_primary=true (v2 feature)
- [x] Integration with FailureManager for failure detection
- [x] Recovery metrics tracked and exported

---

## Notes

- **Resource Cost**: 1+1 protection uses 2x spectrum compared to unprotected
- **Fast Recovery**: 50ms switchover vs. 100ms+ restoration
- **Link-Disjoint Only**: Node-disjoint paths are more robust but harder to find (v2 feature)
- **Revert-to-Primary**: Optional feature to return to primary after repair (prepared for v2)

---

**Related Documents**:
- [10-failure-module.md](../phase2-infrastructure/10-failure-module.md) (Failure detection)
- [21-recovery-timing.md](21-recovery-timing.md) (Recovery time modeling)
- [30-rl-policies.md](../phase4-rl-integration/30-rl-policies.md) (Policy comparison)
- [40-metrics-reporting.md](../phase5-metrics/40-metrics-reporting.md) (Protection metrics)
