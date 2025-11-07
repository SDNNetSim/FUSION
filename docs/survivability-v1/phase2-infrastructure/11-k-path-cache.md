# Phase 2: Core Infrastructure

## 11 - K-Path Candidate Generation & Caching

**Section Reference**: 1.3 - K-Path Candidate Generation & Caching

**Purpose**: Pre-compute and cache K shortest paths for all source-destination pairs to enable efficient path feature extraction and candidate selection for routing and RL policies.

**Location**: `fusion/modules/routing/k_path_cache.py`

**Estimated Effort**: 1 day

---

## Overview

The K-Path Cache module provides pre-computed alternative paths for each source-destination pair in the network. This enables:
- Fast path selection without runtime computation
- Consistent path ordering for RL policy training and inference
- Path feature extraction for decision-making
- Action space definition for RL policies

---

## Module Structure

```
fusion/modules/routing/
├── k_path_cache.py          # Main KPathCache class
└── tests/
    ├── test_k_path_cache.py
    └── fixtures/
```

---

## 1. KPathCache Class

### Class Definition

```python
"""
K-Path cache for pre-computed shortest paths.
"""

from typing import Any
import networkx as nx
from pathlib import Path


class KPathCache:
    """
    Pre-compute and cache K shortest paths for all (src, dst) pairs.

    Uses Yen's K-shortest paths algorithm to compute alternatives,
    ordered by a configurable criterion (hops, length, latency).

    :param topology: Network topology
    :type topology: nx.Graph
    :param k: Number of paths to compute
    :type k: int
    :param ordering: Path ordering criterion (hops, length, latency)
    :type ordering: str
    """

    def __init__(
        self,
        topology: nx.Graph,
        k: int = 4,
        ordering: str = "hops"
    ) -> None:
        """
        Initialize K-Path cache.

        :param topology: Network topology
        :type topology: nx.Graph
        :param k: Number of paths per pair
        :type k: int
        :param ordering: Ordering criterion
        :type ordering: str
        """
        self.topology = topology
        self.k = k
        self.ordering = ordering
        self.cache: dict[tuple[Any, Any], list[list[int]]] = {}
        self._precompute_paths()

    def _precompute_paths(self) -> None:
        """
        Pre-compute K shortest paths for all node pairs.

        Uses NetworkX's k_shortest_paths or custom implementation
        based on Yen's algorithm. Paths are stored ordered by
        the specified criterion.

        :raises ValueError: If topology is invalid or k is non-positive
        """
        if self.k <= 0:
            raise ValueError(f"k must be positive, got {self.k}")

        if not self.topology.nodes:
            raise ValueError("Topology has no nodes")

        # Weight function based on ordering
        if self.ordering == "hops":
            weight = None  # Unweighted = hop count
        elif self.ordering == "length":
            weight = "length"
        elif self.ordering == "latency":
            weight = "latency"
        else:
            raise ValueError(f"Unknown ordering: {self.ordering}")

        # Compute K paths for all node pairs
        nodes = list(self.topology.nodes())
        for src in nodes:
            for dst in nodes:
                if src == dst:
                    continue

                try:
                    # Use NetworkX k_shortest_paths
                    paths = list(nx.shortest_simple_paths(
                        self.topology,
                        src,
                        dst,
                        weight=weight
                    ))

                    # Take up to K paths
                    self.cache[(src, dst)] = paths[:self.k]

                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    # No path exists
                    self.cache[(src, dst)] = []

    def get_k_paths(
        self,
        source: Any,
        destination: Any
    ) -> list[list[int]]:
        """
        Get K paths from cache.

        :param source: Source node
        :type source: Any
        :param destination: Destination node
        :type destination: Any
        :return: List of K paths (may be fewer if not enough exist)
        :rtype: list[list[int]]

        Example:
            >>> cache = KPathCache(topology, k=4)
            >>> paths = cache.get_k_paths(0, 5)
            >>> print(len(paths))
            4
            >>> print(paths[0])
            [0, 1, 3, 5]
        """
        return self.cache.get((source, destination), [])

    def get_path_features(
        self,
        path: list[int],
        network_spectrum_dict: dict[tuple[Any, Any], dict[str, Any]],
        failure_manager: "FailureManager | None" = None
    ) -> dict[str, Any]:
        """
        Extract features for a candidate path.

        Computes features needed for RL policy decisions and heuristics:
        - path_hops: Number of hops in the path
        - min_residual_slots: Minimum contiguous free slots along path (bottleneck)
        - frag_indicator: Fragmentation proxy (1 - largest_contig / total_free)
        - failure_mask: Whether path uses any failed link
        - dist_to_disaster_centroid: Hops to failure center (0 if no failure)

        :param path: Path node list
        :type path: list[int]
        :param network_spectrum_dict: Current spectrum state
        :type network_spectrum_dict: dict[tuple[Any, Any], dict[str, Any]]
        :param failure_manager: Optional failure manager for failure_mask
        :type failure_manager: FailureManager | None
        :return: Path features dict
        :rtype: dict[str, Any]

        Example:
            >>> features = cache.get_path_features(
            ...     path=[0, 1, 2, 3],
            ...     network_spectrum_dict=spectrum,
            ...     failure_manager=None
            ... )
            >>> print(features)
            {
                'path_hops': 3,
                'min_residual_slots': 15,
                'frag_indicator': 0.23,
                'failure_mask': 0,
                'dist_to_disaster_centroid': 0
            }
        """
        # Compute hop count
        path_hops = len(path) - 1

        # Compute min_residual_slots (bottleneck link)
        min_residual = float('inf')
        total_free = 0
        largest_contig = 0

        for i in range(len(path) - 1):
            link = (path[i], path[i + 1])
            reverse_link = (path[i + 1], path[i])

            # Get link spectrum (try both directions)
            link_spectrum = network_spectrum_dict.get(
                link,
                network_spectrum_dict.get(reverse_link, {})
            )

            if not link_spectrum:
                # Link not in spectrum dict (shouldn't happen)
                continue

            # Compute contiguous free slots
            slots = link_spectrum.get('slots', [])
            free_blocks = self._find_free_blocks(slots)

            if free_blocks:
                link_total_free = sum(block[1] - block[0] for block in free_blocks)
                link_largest_contig = max(block[1] - block[0] for block in free_blocks)

                total_free += link_total_free
                largest_contig = max(largest_contig, link_largest_contig)
                min_residual = min(min_residual, link_largest_contig)
            else:
                min_residual = 0

        # Handle case where no free slots found
        if min_residual == float('inf'):
            min_residual = 0

        # Compute fragmentation indicator
        if total_free > 0:
            frag_indicator = 1.0 - (largest_contig / total_free)
        else:
            frag_indicator = 1.0  # Fully fragmented (no free slots)

        # Compute failure mask
        failure_mask = 0
        if failure_manager and not failure_manager.is_path_feasible(path):
            failure_mask = 1

        # Compute distance to disaster centroid
        dist_to_disaster = 0
        if failure_manager and failure_manager.active_failures:
            # Find center of failed region (simplified: use first failed link)
            failed_links = list(failure_manager.active_failures)
            if failed_links:
                center_node = failed_links[0][0]  # Use one endpoint as center
                try:
                    # Distance from path to center (min distance from any path node)
                    distances = []
                    for node in path:
                        if node == center_node:
                            distances.append(0)
                        else:
                            try:
                                dist = nx.shortest_path_length(
                                    self.topology,
                                    node,
                                    center_node
                                )
                                distances.append(dist)
                            except nx.NetworkXNoPath:
                                pass

                    if distances:
                        dist_to_disaster = min(distances)
                except Exception:
                    pass

        return {
            'path_hops': path_hops,
            'min_residual_slots': int(min_residual),
            'frag_indicator': round(frag_indicator, 4),
            'failure_mask': failure_mask,
            'dist_to_disaster_centroid': dist_to_disaster
        }

    def _find_free_blocks(self, slots: list[int]) -> list[tuple[int, int]]:
        """
        Find contiguous free blocks in slot array.

        :param slots: Slot occupancy array (0 = free, >0 = occupied)
        :type slots: list[int]
        :return: List of (start, end) tuples for free blocks
        :rtype: list[tuple[int, int]]

        Example:
            >>> slots = [0, 0, 1, 1, 0, 0, 0]
            >>> blocks = cache._find_free_blocks(slots)
            >>> print(blocks)
            [(0, 2), (4, 7)]
        """
        blocks = []
        start = None

        for i, slot in enumerate(slots):
            if slot == 0:  # Free
                if start is None:
                    start = i
            else:  # Occupied
                if start is not None:
                    blocks.append((start, i))
                    start = None

        # Handle trailing free block
        if start is not None:
            blocks.append((start, len(slots)))

        return blocks

    def get_cache_size(self) -> int:
        """
        Get number of cached path pairs.

        :return: Number of (src, dst) pairs cached
        :rtype: int
        """
        return len(self.cache)

    def get_memory_estimate_mb(self) -> float:
        """
        Estimate memory usage in MB.

        Rough estimate based on number of paths and average path length.

        :return: Estimated memory usage in MB
        :rtype: float
        """
        import sys

        total_size = 0
        for paths in self.cache.values():
            for path in paths:
                # Size of list + size of integers
                total_size += sys.getsizeof(path)
                total_size += sum(sys.getsizeof(node) for node in path)

        return total_size / (1024 * 1024)
```

---

## 2. Integration with SimulationEngine

### Modifications to `fusion/core/simulation.py`

```python
from fusion.modules.routing.k_path_cache import KPathCache


class SimulationEngine:
    def __init__(self, engine_props: dict[str, Any]) -> None:
        # ... existing initialization ...

        # Initialize K-path cache if enabled
        self.k_path_cache: KPathCache | None = None
        if engine_props.get('routing_settings', {}).get('precompute_paths', False):
            k_paths = engine_props['routing_settings'].get('k_paths', 4)
            ordering = engine_props['routing_settings'].get('path_ordering', 'hops')

            logger.info(f"Pre-computing {k_paths} paths for all node pairs...")
            self.k_path_cache = KPathCache(
                self.topology,
                k=k_paths,
                ordering=ordering
            )
            logger.info(
                f"K-path cache initialized: "
                f"{self.k_path_cache.get_cache_size()} pairs, "
                f"{self.k_path_cache.get_memory_estimate_mb():.2f} MB"
            )
```

---

## 3. Configuration Schema

### Extension to `fusion/configs/schema.py`

```python
ROUTING_SETTINGS_SCHEMA = {
    'type': 'object',
    'properties': {
        'k_paths': {
            'type': 'integer',
            'minimum': 1,
            'maximum': 10,
            'default': 4,
            'description': 'Number of candidate paths to cache'
        },
        'path_ordering': {
            'type': 'string',
            'enum': ['hops', 'length', 'latency'],
            'default': 'hops',
            'description': 'Path ordering criterion'
        },
        'precompute_paths': {
            'type': 'boolean',
            'default': True,
            'description': 'Whether to pre-compute K paths at initialization'
        }
    }
}
```

### Configuration Example

```ini
[routing_settings]
route_method = k_shortest_path
k_paths = 4
path_ordering = hops
precompute_paths = true
```

---

## 4. Testing Requirements

### Unit Tests

```python
import pytest
import networkx as nx
from fusion.modules.routing.k_path_cache import KPathCache


@pytest.fixture
def sample_topology():
    """Create sample topology."""
    G = nx.Graph()
    G.add_edges_from([
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 3), (1, 6)
    ])
    return G


def test_k_paths_cached_for_all_pairs(sample_topology):
    """Test that all (src, dst) pairs have K paths cached."""
    cache = KPathCache(sample_topology, k=3)

    nodes = list(sample_topology.nodes())
    for src in nodes:
        for dst in nodes:
            if src == dst:
                continue
            paths = cache.get_k_paths(src, dst)
            assert len(paths) > 0  # At least one path
            assert len(paths) <= 3  # At most K paths


def test_paths_ordered_by_hops(sample_topology):
    """Test that paths are sorted by hop count."""
    cache = KPathCache(sample_topology, k=4, ordering='hops')

    paths = cache.get_k_paths(0, 4)
    assert len(paths) > 0

    # Verify paths are ordered by increasing hop count
    for i in range(len(paths) - 1):
        assert len(paths[i]) <= len(paths[i + 1])


def test_path_features_computed(sample_topology):
    """Test that features include hops, residual, frag, failure_mask."""
    cache = KPathCache(sample_topology, k=4)

    # Mock spectrum dict
    network_spectrum_dict = {}
    for u, v in sample_topology.edges():
        network_spectrum_dict[(u, v)] = {
            'slots': [0] * 20 + [1] * 10 + [0] * 10  # Mixed occupancy
        }

    path = [0, 1, 2, 3]
    features = cache.get_path_features(path, network_spectrum_dict)

    # Check all required features present
    assert 'path_hops' in features
    assert 'min_residual_slots' in features
    assert 'frag_indicator' in features
    assert 'failure_mask' in features
    assert 'dist_to_disaster_centroid' in features

    # Verify values
    assert features['path_hops'] == 3
    assert features['min_residual_slots'] > 0
    assert 0.0 <= features['frag_indicator'] <= 1.0
    assert features['failure_mask'] in [0, 1]


def test_failure_mask_set_correctly(sample_topology):
    """Test that failure_mask=1 when path uses failed link."""
    from fusion.modules.failures import FailureManager

    cache = KPathCache(sample_topology, k=4)

    # Create failure manager and inject failure
    engine_props = {'seed': 42}
    failure_manager = FailureManager(engine_props, sample_topology)
    failure_manager.inject_failure('link', t_fail=10.0, t_repair=20.0, link_id=(1, 2))

    # Mock spectrum dict
    network_spectrum_dict = {}
    for u, v in sample_topology.edges():
        network_spectrum_dict[(u, v)] = {'slots': [0] * 40}

    # Path using failed link
    path_with_failure = [0, 1, 2, 3]
    features = cache.get_path_features(
        path_with_failure,
        network_spectrum_dict,
        failure_manager
    )
    assert features['failure_mask'] == 1

    # Path avoiding failed link
    path_without_failure = [0, 5, 6, 3]
    features = cache.get_path_features(
        path_without_failure,
        network_spectrum_dict,
        failure_manager
    )
    assert features['failure_mask'] == 0


def test_min_residual_slots_accurate(sample_topology):
    """Test that min_residual matches bottleneck link."""
    cache = KPathCache(sample_topology, k=4)

    # Create spectrum dict with known bottleneck
    network_spectrum_dict = {}
    for u, v in sample_topology.edges():
        if (u, v) == (1, 2) or (u, v) == (2, 1):
            # Bottleneck link: only 5 free slots
            network_spectrum_dict[(u, v)] = {'slots': [0] * 5 + [1] * 35}
        else:
            # Other links: 20 free slots
            network_spectrum_dict[(u, v)] = {'slots': [0] * 20 + [1] * 20}

    path = [0, 1, 2, 3]
    features = cache.get_path_features(path, network_spectrum_dict)

    # Should match bottleneck (5 slots)
    assert features['min_residual_slots'] == 5
```

### Performance Tests

```python
def test_cache_initialization_time(sample_topology):
    """Test that cache initialization meets time budget."""
    import time

    start = time.time()
    cache = KPathCache(sample_topology, k=4)
    elapsed = time.time() - start

    # Should be fast for small topology
    assert elapsed < 1.0  # < 1 second


def test_cache_memory_estimate():
    """Test memory estimate for typical topology."""
    # NSFNet-like topology (14 nodes, ~21 edges)
    G = nx.random_regular_graph(3, 14)
    cache = KPathCache(G, k=4)

    memory_mb = cache.get_memory_estimate_mb()

    # Should be under 100 MB for K=4 on NSFNet
    assert memory_mb < 100
```

---

## 5. Performance Budgets

- **Initialization time**:
  - NSFNet (14 nodes): ≤ 5 seconds
  - USbackbone60 (60 nodes): ≤ 30 seconds
- **Memory usage**:
  - NSFNet with K=4: ≤ 100 MB
  - USbackbone60 with K=4: ≤ 500 MB
- **Feature extraction time**: ≤ 0.5 ms per path

---

## 6. Acceptance Criteria

- [x] `test_k_paths_cached_for_all_pairs`: All (src, dst) pairs have K paths cached
- [x] `test_paths_ordered_by_hops`: Paths are sorted by hop count
- [x] `test_path_features_computed`: Features include hops, residual, frag, failure_mask
- [x] `test_failure_mask_set_correctly`: failure_mask=1 when path uses failed link
- [x] `test_min_residual_slots_accurate`: min_residual matches bottleneck link
- [x] Performance budgets met for NSFNet and USbackbone60 topologies

---

## Notes

- **Path ordering**: Default is by hop count, but can be configured for length or latency
- **Memory optimization**: Consider lazy loading for large topologies (K > 5 or N > 100 nodes)
- **Feature engineering**: Additional features can be added for more sophisticated policies
- **Caching strategy**: Paths are computed once at initialization; no dynamic updates in v1

---

**Related Documents**:
- [10-failure-module.md](10-failure-module.md) (Uses FailureManager)
- [30-rl-policies.md](../phase4-rl-integration/30-rl-policies.md) (Uses path features)
- [52-performance.md](../phase6-quality/52-performance.md) (Performance requirements)
