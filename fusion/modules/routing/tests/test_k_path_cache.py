"""
Unit tests for K-Path Cache.
"""

from typing import Any

import networkx as nx
import pytest

from fusion.modules.failures import FailureManager
from fusion.modules.routing.k_path_cache import KPathCache


@pytest.fixture
def sample_topology() -> nx.Graph:
    """Create sample topology."""
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 3), (1, 6)])
    return G


@pytest.fixture
def k_path_cache(sample_topology: nx.Graph) -> KPathCache:
    """Create KPathCache instance."""
    return KPathCache(sample_topology, k=4, ordering="hops")


def test_k_paths_cached_for_all_pairs(sample_topology: nx.Graph) -> None:
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


def test_paths_ordered_by_hops(sample_topology: nx.Graph) -> None:
    """Test that paths are sorted by hop count."""
    cache = KPathCache(sample_topology, k=4, ordering="hops")

    paths = cache.get_k_paths(0, 4)
    assert len(paths) > 0

    # Verify paths are ordered by increasing hop count
    for i in range(len(paths) - 1):
        assert len(paths[i]) <= len(paths[i + 1])


def test_path_features_computed(sample_topology: nx.Graph) -> None:
    """Test that features include hops, residual, frag, failure_mask."""
    cache = KPathCache(sample_topology, k=4)

    # Mock spectrum dict
    network_spectrum_dict = {}
    for u, v in sample_topology.edges():
        network_spectrum_dict[(u, v)] = {
            "slots": [0] * 20 + [1] * 10 + [0] * 10  # Mixed occupancy
        }

    path = [0, 1, 2, 3]
    features = cache.get_path_features(path, network_spectrum_dict)

    # Check all required features present
    assert "path_hops" in features
    assert "min_residual_slots" in features
    assert "frag_indicator" in features
    assert "failure_mask" in features
    assert "dist_to_disaster_centroid" in features

    # Verify values
    assert features["path_hops"] == 3
    assert features["min_residual_slots"] > 0
    assert 0.0 <= features["frag_indicator"] <= 1.0
    assert features["failure_mask"] in [0, 1]


def test_failure_mask_set_correctly(sample_topology: nx.Graph) -> None:
    """Test that failure_mask=1 when path uses failed link."""
    cache = KPathCache(sample_topology, k=4)

    # Create failure manager and inject failure
    engine_props = {"seed": 42}
    failure_manager = FailureManager(engine_props, sample_topology)
    failure_manager.inject_failure("link", t_fail=10.0, t_repair=20.0, link_id=(1, 2))

    # Activate the failure
    failure_manager.activate_failures(10.0)

    # Mock spectrum dict
    network_spectrum_dict = {}
    for u, v in sample_topology.edges():
        network_spectrum_dict[(u, v)] = {"slots": [0] * 40}

    # Path using failed link
    path_with_failure = [0, 1, 2, 3]
    features = cache.get_path_features(
        path_with_failure, network_spectrum_dict, failure_manager
    )
    assert features["failure_mask"] == 1

    # Path avoiding failed link
    path_without_failure = [0, 5, 6, 3]
    features = cache.get_path_features(
        path_without_failure, network_spectrum_dict, failure_manager
    )
    assert features["failure_mask"] == 0


def test_min_residual_slots_accurate(sample_topology: nx.Graph) -> None:
    """Test that min_residual matches bottleneck link."""
    cache = KPathCache(sample_topology, k=4)

    # Create spectrum dict with known bottleneck
    network_spectrum_dict = {}
    for u, v in sample_topology.edges():
        if (u, v) == (1, 2) or (u, v) == (2, 1):
            # Bottleneck link: only 5 free slots
            network_spectrum_dict[(u, v)] = {"slots": [0] * 5 + [1] * 35}
        else:
            # Other links: 20 free slots
            network_spectrum_dict[(u, v)] = {"slots": [0] * 20 + [1] * 20}

    path = [0, 1, 2, 3]
    features = cache.get_path_features(path, network_spectrum_dict)

    # Should match bottleneck (5 slots)
    assert features["min_residual_slots"] == 5


def test_cache_initialization_time(sample_topology: nx.Graph) -> None:
    """Test that cache initialization meets time budget."""
    import time

    start = time.time()
    cache = KPathCache(sample_topology, k=4)
    elapsed = time.time() - start

    # Should be fast for small topology
    assert elapsed < 1.0  # < 1 second
    assert cache.get_cache_size() > 0


def test_cache_memory_estimate() -> None:
    """Test memory estimate for typical topology."""
    # NSFNet-like topology (14 nodes, ~21 edges)
    G = nx.random_regular_graph(3, 14, seed=42)
    cache = KPathCache(G, k=4)

    memory_mb = cache.get_memory_estimate_mb()

    # Should be under 100 MB for K=4 on NSFNet
    assert memory_mb < 100
    assert memory_mb > 0


def test_get_k_paths_returns_empty_for_disconnected() -> None:
    """Test that disconnected nodes return empty list."""
    G = nx.Graph()
    # Two disconnected components
    G.add_edges_from([(0, 1), (1, 2)])
    G.add_edges_from([(3, 4), (4, 5)])

    cache = KPathCache(G, k=4)

    # Should be empty for nodes in different components
    paths = cache.get_k_paths(0, 5)
    assert paths == []


def test_k_must_be_positive() -> None:
    """Test that k must be positive."""
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2)])

    with pytest.raises(ValueError, match="k must be positive"):
        KPathCache(G, k=0)

    with pytest.raises(ValueError, match="k must be positive"):
        KPathCache(G, k=-1)


def test_topology_must_have_nodes() -> None:
    """Test that topology must have nodes."""
    G = nx.Graph()  # Empty graph

    with pytest.raises(ValueError, match="has no nodes"):
        KPathCache(G, k=4)


def test_invalid_ordering() -> None:
    """Test that invalid ordering raises error."""
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2)])

    with pytest.raises(ValueError, match="Unknown ordering"):
        KPathCache(G, k=4, ordering="invalid")


def test_find_free_blocks(k_path_cache: KPathCache) -> None:
    """Test _find_free_blocks helper function."""
    # Test case 1: Mixed free and occupied
    slots = [0, 0, 1, 1, 0, 0, 0]
    blocks = k_path_cache._find_free_blocks(slots)
    assert blocks == [(0, 2), (4, 7)]

    # Test case 2: All free
    slots = [0, 0, 0, 0]
    blocks = k_path_cache._find_free_blocks(slots)
    assert blocks == [(0, 4)]

    # Test case 3: All occupied
    slots = [1, 1, 1, 1]
    blocks = k_path_cache._find_free_blocks(slots)
    assert blocks == []

    # Test case 4: Single free slot
    slots = [1, 0, 1]
    blocks = k_path_cache._find_free_blocks(slots)
    assert blocks == [(1, 2)]


def test_fragmentation_indicator_calculation(sample_topology: nx.Graph) -> None:
    """Test that fragmentation indicator is calculated correctly."""
    cache = KPathCache(sample_topology, k=4)

    # Highly fragmented: small blocks
    network_spectrum_dict = {}
    for u, v in sample_topology.edges():
        network_spectrum_dict[(u, v)] = {
            "slots": [0, 1, 0, 1, 0, 1]  # Highly fragmented
        }

    path = [0, 1, 2]
    features = cache.get_path_features(path, network_spectrum_dict)

    # Should have high fragmentation
    assert features["frag_indicator"] > 0.5

    # Not fragmented: one large block
    network_spectrum_dict = {}
    for u, v in sample_topology.edges():
        network_spectrum_dict[(u, v)] = {
            "slots": [0, 0, 0, 0, 0, 0, 1, 1]  # One large block
        }

    features = cache.get_path_features(path, network_spectrum_dict)

    # Should have low fragmentation
    # Note: With 2 links each having one block of 6 slots,
    # we get frag = 1 - (6/12) = 0.5
    assert features["frag_indicator"] <= 0.5


def test_no_free_slots_sets_min_residual_to_zero(sample_topology: nx.Graph) -> None:
    """Test that no free slots results in min_residual=0."""
    cache = KPathCache(sample_topology, k=4)

    # All slots occupied
    network_spectrum_dict = {}
    for u, v in sample_topology.edges():
        network_spectrum_dict[(u, v)] = {"slots": [1] * 40}

    path = [0, 1, 2]
    features = cache.get_path_features(path, network_spectrum_dict)

    assert features["min_residual_slots"] == 0


def test_path_features_without_failure_manager(sample_topology: nx.Graph) -> None:
    """Test that path features work without failure manager."""
    cache = KPathCache(sample_topology, k=4)

    network_spectrum_dict = {}
    for u, v in sample_topology.edges():
        network_spectrum_dict[(u, v)] = {"slots": [0] * 40}

    path = [0, 1, 2]
    features = cache.get_path_features(
        path, network_spectrum_dict, failure_manager=None
    )

    assert features["failure_mask"] == 0
    assert features["dist_to_disaster_centroid"] == 0


def test_dist_to_disaster_centroid(sample_topology: nx.Graph) -> None:
    """Test that distance to disaster centroid is calculated."""
    cache = KPathCache(sample_topology, k=4)

    # Create failure manager and inject failure
    engine_props = {"seed": 42}
    failure_manager = FailureManager(engine_props, sample_topology)
    failure_manager.inject_failure("link", t_fail=10.0, t_repair=20.0, link_id=(1, 2))

    # Activate the failure
    failure_manager.activate_failures(10.0)

    network_spectrum_dict = {}
    for u, v in sample_topology.edges():
        network_spectrum_dict[(u, v)] = {"slots": [0] * 40}

    # Path close to failure
    path_close = [0, 1, 6]
    features = cache.get_path_features(
        path_close, network_spectrum_dict, failure_manager
    )
    dist_close = features["dist_to_disaster_centroid"]

    # Path far from failure
    path_far = [4, 3, 6, 5]
    features = cache.get_path_features(path_far, network_spectrum_dict, failure_manager)
    dist_far = features["dist_to_disaster_centroid"]

    # Far path should have larger distance
    assert dist_far >= dist_close


def test_cache_handles_reverse_link_direction(sample_topology: nx.Graph) -> None:
    """Test that cache handles bidirectional links correctly."""
    cache = KPathCache(sample_topology, k=4)

    # Create spectrum dict with only one direction
    network_spectrum_dict = {}
    for u, v in sample_topology.edges():
        network_spectrum_dict[(u, v)] = {"slots": [0] * 40}

    path = [0, 1, 2]
    features1 = cache.get_path_features(path, network_spectrum_dict)

    # Should work with reversed direction too
    network_spectrum_dict_reversed = {}
    for u, v in sample_topology.edges():
        network_spectrum_dict_reversed[(v, u)] = {"slots": [0] * 40}

    features2 = cache.get_path_features(path, network_spectrum_dict_reversed)

    # Should get same results
    assert features1["path_hops"] == features2["path_hops"]
    assert features1["min_residual_slots"] == features2["min_residual_slots"]


def test_get_cache_size(k_path_cache: KPathCache) -> None:
    """Test that get_cache_size returns correct count."""
    size = k_path_cache.get_cache_size()
    assert size > 0

    # Should have N*(N-1) pairs for N nodes
    num_nodes = len(k_path_cache.topology.nodes())
    expected_max = num_nodes * (num_nodes - 1)
    assert size <= expected_max


def test_empty_spectrum_dict_handled_gracefully(sample_topology: nx.Graph) -> None:
    """Test that empty spectrum dict is handled gracefully."""
    cache = KPathCache(sample_topology, k=4)

    # Empty spectrum dict
    network_spectrum_dict: dict[tuple[Any, Any], dict[str, Any]] = {}

    path = [0, 1, 2]
    features = cache.get_path_features(path, network_spectrum_dict)

    # Should not crash, min_residual should be 0
    assert features["min_residual_slots"] == 0
    assert features["frag_indicator"] == 1.0  # Fully fragmented
