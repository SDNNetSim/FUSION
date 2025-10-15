"""
Unit tests for FailureManager class.
"""

import networkx as nx
import pytest

from fusion.modules.failures import FailureConfigError, FailureManager


@pytest.fixture
def sample_topology():
    """Create a sample topology for testing."""
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 3)])
    return G


@pytest.fixture
def failure_manager(sample_topology):
    """Create a FailureManager instance."""
    engine_props = {"seed": 42}
    return FailureManager(engine_props, sample_topology)


def test_link_failure_blocks_path(failure_manager):
    """Test that a path using a failed link is marked infeasible."""
    # Inject link failure
    failure_manager.inject_failure("link", t_fail=10.0, t_repair=20.0, link_id=(1, 2))

    # Path using failed link should be infeasible
    assert not failure_manager.is_path_feasible([0, 1, 2, 3])

    # Path avoiding failed link should be feasible
    assert failure_manager.is_path_feasible([0, 5, 6, 3])


def test_failure_repair_restores_path(failure_manager):
    """Test that path becomes feasible after repair time."""
    # Inject and repair failure
    failure_manager.inject_failure("link", t_fail=10.0, t_repair=20.0, link_id=(1, 2))
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
    failure_manager.inject_failure(
        "srlg", t_fail=10.0, t_repair=20.0, srlg_links=srlg_links
    )

    assert failure_manager.get_failure_count() == 3

    # All paths using SRLG links should be infeasible
    assert not failure_manager.is_path_feasible([0, 1, 2])
    assert not failure_manager.is_path_feasible([0, 5, 6, 3])


def test_geo_failure_radius(failure_manager, sample_topology):
    """Test that links within hop radius are failed, others unaffected."""
    event = failure_manager.inject_failure(
        "geo", t_fail=10.0, t_repair=20.0, center_node=1, hop_radius=1
    )

    # Check affected nodes
    affected_nodes = event["meta"]["affected_nodes"]
    assert 1 in affected_nodes
    assert 0 in affected_nodes  # Neighbor
    assert 2 in affected_nodes  # Neighbor

    # Links within radius should be failed
    assert not failure_manager.is_path_feasible([0, 1, 2])


def test_failure_history_tracking(failure_manager):
    """Test that failure events are logged with timestamps."""
    failure_manager.inject_failure("link", t_fail=10.0, t_repair=20.0, link_id=(0, 1))
    failure_manager.inject_failure("link", t_fail=15.0, t_repair=25.0, link_id=(2, 3))

    assert len(failure_manager.failure_history) == 2
    assert failure_manager.failure_history[0]["t_fail"] == 10.0
    assert failure_manager.failure_history[1]["t_fail"] == 15.0


def test_invalid_failure_config(failure_manager):
    """Test that invalid configurations raise errors."""
    # Repair before failure
    with pytest.raises(FailureConfigError, match="Repair time.*must be after"):
        failure_manager.inject_failure(
            "link", t_fail=20.0, t_repair=10.0, link_id=(0, 1)
        )

    # Invalid link
    with pytest.raises(FailureConfigError, match="does not exist"):
        failure_manager.inject_failure(
            "link", t_fail=10.0, t_repair=20.0, link_id=(99, 100)
        )


def test_node_failure_blocks_adjacent_links(failure_manager):
    """Test that all links adjacent to a failed node are blocked."""
    event = failure_manager.inject_failure(
        "node", t_fail=10.0, t_repair=20.0, node_id=1
    )

    # Check that multiple links are failed
    assert len(event["failed_links"]) > 0

    # All paths through node 1 should be infeasible
    assert not failure_manager.is_path_feasible([0, 1, 2])


def test_get_affected_links(failure_manager):
    """Test that get_affected_links returns currently failed links."""
    failure_manager.inject_failure("link", t_fail=10.0, t_repair=20.0, link_id=(1, 2))

    affected = failure_manager.get_affected_links()
    assert len(affected) == 1
    assert (1, 2) in affected or (2, 1) in affected


def test_clear_all_failures(failure_manager):
    """Test that clear_all_failures removes all active failures."""
    failure_manager.inject_failure("link", t_fail=10.0, t_repair=20.0, link_id=(1, 2))
    failure_manager.inject_failure("link", t_fail=10.0, t_repair=20.0, link_id=(3, 4))

    assert failure_manager.get_failure_count() == 2

    failure_manager.clear_all_failures()

    assert failure_manager.get_failure_count() == 0
    assert len(failure_manager.scheduled_repairs) == 0


def test_path_feasible_with_no_failures(failure_manager):
    """Test that all paths are feasible when no failures exist."""
    path = [0, 1, 2, 3, 4]
    assert failure_manager.is_path_feasible(path)


def test_repair_only_scheduled_failures(failure_manager):
    """Test that repair only removes failures at exact time."""
    failure_manager.inject_failure("link", t_fail=10.0, t_repair=20.0, link_id=(1, 2))

    # Try repairing at wrong time
    repaired = failure_manager.repair_failures(15.0)
    assert len(repaired) == 0
    assert failure_manager.get_failure_count() == 1

    # Repair at correct time
    repaired = failure_manager.repair_failures(20.0)
    assert len(repaired) == 1
    assert failure_manager.get_failure_count() == 0


def test_multiple_failures_at_same_repair_time(failure_manager):
    """Test that multiple failures can be scheduled for same repair time."""
    failure_manager.inject_failure("link", t_fail=10.0, t_repair=20.0, link_id=(1, 2))
    failure_manager.inject_failure("link", t_fail=10.0, t_repair=20.0, link_id=(3, 4))

    assert failure_manager.get_failure_count() == 2

    # Repair both at same time
    repaired = failure_manager.repair_failures(20.0)
    assert len(repaired) == 2
    assert failure_manager.get_failure_count() == 0


def test_bidirectional_link_checking(failure_manager):
    """Test that path feasibility checks both link directions."""
    failure_manager.inject_failure("link", t_fail=10.0, t_repair=20.0, link_id=(1, 2))

    # Both directions should be blocked
    assert not failure_manager.is_path_feasible([0, 1, 2, 3])
    assert not failure_manager.is_path_feasible([3, 2, 1, 0])
