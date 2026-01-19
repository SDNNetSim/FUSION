"""
Unit tests for failure type implementations.
"""

import networkx as nx
import pytest

from fusion.modules.failures import (
    FailureConfigError,
    fail_geo,
    fail_link,
    fail_node,
    fail_srlg,
)


@pytest.fixture
def sample_topology() -> nx.Graph:
    """Create a sample topology for testing."""
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 3), (1, 6)])
    return G


def test_fail_link_valid(sample_topology: nx.Graph) -> None:
    """Test that valid link failure works correctly."""
    event = fail_link(sample_topology, link_id=(0, 1), t_fail=10.0, t_repair=20.0)

    assert event["failure_type"] == "link"
    assert event["t_fail"] == 10.0
    assert event["t_repair"] == 20.0
    assert len(event["failed_links"]) == 1
    assert (0, 1) in event["failed_links"] or (1, 0) in event["failed_links"]


def test_fail_link_reverse_direction(sample_topology: nx.Graph) -> None:
    """Test that link failure works with reverse direction."""
    # Try reverse direction
    event = fail_link(sample_topology, link_id=(1, 0), t_fail=10.0, t_repair=20.0)

    assert len(event["failed_links"]) == 1
    # Should normalize to one of the directions
    link = event["failed_links"][0]
    assert link == (1, 0) or link == (0, 1)


def test_fail_link_invalid_link(sample_topology: nx.Graph) -> None:
    """Test that invalid link raises error."""
    with pytest.raises(FailureConfigError, match="does not exist"):
        fail_link(sample_topology, link_id=(99, 100), t_fail=10.0, t_repair=20.0)


def test_fail_node_valid(sample_topology: nx.Graph) -> None:
    """Test that valid node failure works correctly."""
    event = fail_node(sample_topology, node_id=1, t_fail=10.0, t_repair=20.0)

    assert event["failure_type"] == "node"
    assert event["t_fail"] == 10.0
    assert event["t_repair"] == 20.0
    assert len(event["failed_links"]) > 0
    assert event["meta"]["node_id"] == 1


def test_fail_node_invalid_node(sample_topology: nx.Graph) -> None:
    """Test that invalid node raises error."""
    with pytest.raises(FailureConfigError, match="does not exist"):
        fail_node(sample_topology, node_id=99, t_fail=10.0, t_repair=20.0)


def test_fail_node_adjacent_links(sample_topology: nx.Graph) -> None:
    """Test that all adjacent links are failed."""
    # Node 1 has edges to 0, 2, and 6
    event = fail_node(sample_topology, node_id=1, t_fail=10.0, t_repair=20.0)

    failed_links = event["failed_links"]
    # Should have 3 adjacent links
    assert len(failed_links) == 3

    # Check that all neighbors are in failed links
    neighbors = [0, 2, 6]
    for neighbor in neighbors:
        assert (1, neighbor) in failed_links or (neighbor, 1) in failed_links


def test_fail_srlg_valid(sample_topology: nx.Graph) -> None:
    """Test that valid SRLG failure works correctly."""
    srlg_links = [(0, 1), (2, 3), (5, 6)]
    event = fail_srlg(sample_topology, srlg_links=srlg_links, t_fail=10.0, t_repair=20.0)

    assert event["failure_type"] == "srlg"
    assert len(event["failed_links"]) == 3
    assert event["meta"]["srlg_size"] == 3


def test_fail_srlg_empty_list(sample_topology: nx.Graph) -> None:
    """Test that empty SRLG list raises error."""
    with pytest.raises(FailureConfigError, match="cannot be empty"):
        fail_srlg(sample_topology, srlg_links=[], t_fail=10.0, t_repair=20.0)


def test_fail_srlg_invalid_link(sample_topology: nx.Graph) -> None:
    """Test that SRLG with invalid link raises error."""
    srlg_links = [(0, 1), (99, 100)]
    with pytest.raises(FailureConfigError, match="does not exist"):
        fail_srlg(sample_topology, srlg_links=srlg_links, t_fail=10.0, t_repair=20.0)


def test_fail_srlg_mixed_directions(sample_topology: nx.Graph) -> None:
    """Test that SRLG handles mixed link directions."""
    # Mix forward and reverse directions
    srlg_links = [(0, 1), (3, 2), (5, 6)]
    event = fail_srlg(sample_topology, srlg_links=srlg_links, t_fail=10.0, t_repair=20.0)

    assert len(event["failed_links"]) == 3


def test_fail_geo_valid(sample_topology: nx.Graph) -> None:
    """Test that valid geographic failure works correctly."""
    event = fail_geo(sample_topology, center_node=1, hop_radius=1, t_fail=10.0, t_repair=20.0)

    assert event["failure_type"] == "geo"
    assert event["meta"]["center_node"] == 1
    assert event["meta"]["hop_radius"] == 1
    assert len(event["failed_links"]) > 0

    # Center node should be in affected nodes
    affected_nodes = event["meta"]["affected_nodes"]
    assert 1 in affected_nodes


def test_fail_geo_invalid_center_node(sample_topology: nx.Graph) -> None:
    """Test that invalid center node raises error."""
    with pytest.raises(FailureConfigError, match="does not exist"):
        fail_geo(sample_topology, center_node=99, hop_radius=2, t_fail=10.0, t_repair=20.0)


def test_fail_geo_invalid_radius(sample_topology: nx.Graph) -> None:
    """Test that non-positive radius raises error."""
    with pytest.raises(FailureConfigError, match="must be positive"):
        fail_geo(sample_topology, center_node=1, hop_radius=0, t_fail=10.0, t_repair=20.0)

    with pytest.raises(FailureConfigError, match="must be positive"):
        fail_geo(sample_topology, center_node=1, hop_radius=-1, t_fail=10.0, t_repair=20.0)


def test_fail_geo_radius_calculation(sample_topology: nx.Graph) -> None:
    """Test that geographic failure correctly calculates affected nodes."""
    # Radius 1 from node 1
    event = fail_geo(sample_topology, center_node=1, hop_radius=1, t_fail=10.0, t_repair=20.0)
    affected = event["meta"]["affected_nodes"]

    # Should include node 1 and its immediate neighbors (0, 2, 6)
    assert 1 in affected
    assert 0 in affected
    assert 2 in affected
    assert 6 in affected

    # Should not include nodes 2 hops away
    # Node 3 is 2 hops away from node 1
    # (but might be included if there are links through node 6)


def test_fail_geo_increasing_radius(sample_topology: nx.Graph) -> None:
    """Test that larger radius affects more nodes."""
    event1 = fail_geo(sample_topology, center_node=1, hop_radius=1, t_fail=10.0, t_repair=20.0)
    event2 = fail_geo(sample_topology, center_node=1, hop_radius=2, t_fail=10.0, t_repair=20.0)

    affected1 = len(event1["meta"]["affected_nodes"])
    affected2 = len(event2["meta"]["affected_nodes"])

    # Larger radius should affect more or equal nodes
    assert affected2 >= affected1


def test_fail_geo_links_with_endpoint_in_region(sample_topology: nx.Graph) -> None:
    """Test that links with at least one endpoint in region are failed."""
    event = fail_geo(sample_topology, center_node=1, hop_radius=1, t_fail=10.0, t_repair=20.0)

    failed_links = event["failed_links"]
    affected_nodes = set(event["meta"]["affected_nodes"])

    # Every failed link should have at least one endpoint in affected region
    for u, v in failed_links:
        assert u in affected_nodes or v in affected_nodes


def test_failure_event_structure() -> None:
    """Test that all failure events have consistent structure."""
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 3)])

    # Test link failure
    event = fail_link(G, link_id=(0, 1), t_fail=10.0, t_repair=20.0)
    assert "failure_type" in event
    assert "t_fail" in event
    assert "t_repair" in event
    assert "failed_links" in event
    assert "meta" in event

    # Test node failure
    event = fail_node(G, node_id=1, t_fail=10.0, t_repair=20.0)
    assert "failure_type" in event
    assert "t_fail" in event
    assert "t_repair" in event
    assert "failed_links" in event
    assert "meta" in event

    # Test SRLG failure
    event = fail_srlg(G, srlg_links=[(0, 1), (1, 2)], t_fail=10.0, t_repair=20.0)
    assert "failure_type" in event
    assert "t_fail" in event
    assert "t_repair" in event
    assert "failed_links" in event
    assert "meta" in event

    # Test geo failure
    event = fail_geo(G, center_node=1, hop_radius=1, t_fail=10.0, t_repair=20.0)
    assert "failure_type" in event
    assert "t_fail" in event
    assert "t_repair" in event
    assert "failed_links" in event
    assert "meta" in event
