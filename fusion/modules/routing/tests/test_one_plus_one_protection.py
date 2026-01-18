"""
Unit tests for OnePlusOneProtection routing algorithm.

Tests the 1+1 disjoint protection routing implementation including:
- Disjoint path computation
- Protection switchover
- Failure handling
- Spectrum reservation on dual paths
"""

import networkx as nx
import pytest

from fusion.core.properties import SDNProps
from fusion.modules.routing.one_plus_one_protection import OnePlusOneProtection


@pytest.fixture
def sample_topology() -> nx.Graph:
    """Create a sample topology with multiple disjoint paths."""
    G = nx.Graph()
    # Add edges with length attribute required by find_path_length
    edges = [
        (0, 1, 100.0),
        (1, 2, 100.0),
        (2, 3, 100.0),  # Primary path
        (0, 4, 100.0),
        (4, 5, 100.0),
        (5, 3, 100.0),  # Backup path
        (1, 4, 100.0),
        (2, 5, 100.0),  # Cross-links
    ]
    for u, v, length in edges:
        G.add_edge(u, v, length=length)
    return G


@pytest.fixture
def simple_topology() -> nx.Graph:
    """Create a simple topology for basic tests."""
    G = nx.Graph()
    for u, v in [(0, 1), (1, 2), (2, 3), (3, 4)]:
        G.add_edge(u, v, length=100.0)
    return G


@pytest.fixture
def protection_router(sample_topology: nx.Graph) -> OnePlusOneProtection:
    """Create a 1+1 protection router instance."""
    engine_props = {
        "topology": sample_topology,
        "protection_settings": {
            "protection_switchover_ms": 50.0,
            "revert_to_primary": False,
        },
    }
    sdn_props = SDNProps()
    sdn_props.topology = sample_topology
    return OnePlusOneProtection(engine_props, sdn_props)


class TestOnePlusOneProtection:
    """Test suite for OnePlusOneProtection routing algorithm."""

    def test_algorithm_name(self, protection_router: OnePlusOneProtection) -> None:
        """Test that algorithm name is correctly set."""
        assert protection_router.algorithm_name == "1plus1_protection"

    def test_supported_topologies(
        self, protection_router: OnePlusOneProtection
    ) -> None:
        """Test that supported topologies list is defined."""
        topologies = protection_router.supported_topologies
        assert isinstance(topologies, list)
        assert len(topologies) > 0
        assert "NSFNet" in topologies

    def test_validate_environment_connected(
        self, sample_topology: nx.Graph, protection_router: OnePlusOneProtection
    ) -> None:
        """Test environment validation for connected graph."""
        assert protection_router.validate_environment(sample_topology) is True

    def test_validate_environment_disconnected(
        self, protection_router: OnePlusOneProtection
    ) -> None:
        """Test environment validation fails for disconnected graph."""
        G = nx.Graph()
        G.add_edges_from([(0, 1), (2, 3)])  # Two disconnected components
        assert protection_router.validate_environment(G) is False

    def test_validate_environment_low_connectivity(
        self, simple_topology: nx.Graph, protection_router: OnePlusOneProtection
    ) -> None:
        """Test environment validation for low edge connectivity."""
        # Linear path has edge connectivity of 1
        result = protection_router.validate_environment(simple_topology)
        # Should fail as it requires edge connectivity >= 2
        assert result is False

    def test_disjoint_path_computation(
        self, protection_router: OnePlusOneProtection
    ) -> None:
        """Test that primary and backup paths are link-disjoint."""
        primary, backup = protection_router.find_disjoint_paths(0, 3)

        assert primary is not None
        assert backup is not None

        # Extract link sets
        primary_links = set(zip(primary[:-1], primary[1:], strict=False))
        backup_links = set(zip(backup[:-1], backup[1:], strict=False))

        # Check disjointness (both directions)
        for link in backup_links:
            assert link not in primary_links
            assert (link[1], link[0]) not in primary_links

    def test_route_stores_paths_in_sdn_props(
        self, protection_router: OnePlusOneProtection
    ) -> None:
        """Test that route() stores paths in SDN properties and route_props."""
        sdn_props = protection_router.sdn_props
        protection_router.route(0, 3, None)

        # Check SDN properties
        assert sdn_props.primary_path is not None
        assert sdn_props.backup_path is not None
        assert sdn_props.is_protected is True
        assert sdn_props.active_path == "primary"

        # Check route_props for SDN controller compatibility
        assert len(protection_router.route_props.paths_matrix) == 1
        assert protection_router.route_props.paths_matrix[0] == sdn_props.primary_path
        assert len(protection_router.route_props.weights_list) == 1

    def test_route_returns_none_no_disjoint_paths(
        self, protection_router: OnePlusOneProtection, simple_topology: nx.Graph
    ) -> None:
        """Test that route doesn't set paths when disjoint paths don't exist."""
        # Update router topology to simple linear topology
        protection_router.topology = simple_topology
        protection_router.sdn_props.primary_path = None
        protection_router.sdn_props.backup_path = None
        protection_router.route(0, 4, None)

        assert protection_router.sdn_props.primary_path is None
        assert protection_router.sdn_props.backup_path is None
        # Route props should also be empty
        assert len(protection_router.route_props.paths_matrix) == 0

    def test_handle_failure_switchover(
        self, protection_router: OnePlusOneProtection
    ) -> None:
        """Test failure handling switches to backup path."""
        affected_requests = [
            {
                "id": 42,
                "is_protected": True,
                "primary_path": [0, 1, 2, 3],
                "backup_path": [0, 4, 5, 3],
                "active_path": "primary",
            }
        ]

        actions = protection_router.handle_failure(
            current_time=100.0, affected_requests=affected_requests
        )

        assert len(actions) == 1
        assert actions[0]["action"] == "switchover"
        assert actions[0]["to_path"] == "backup"
        assert actions[0]["recovery_time_ms"] == 50.0

    def test_handle_failure_no_action_unprotected(
        self, protection_router: OnePlusOneProtection
    ) -> None:
        """Test that unprotected requests don't trigger switchover."""
        affected_requests = [{"id": 42, "is_protected": False, "path": [0, 1, 2, 3]}]

        actions = protection_router.handle_failure(
            current_time=100.0, affected_requests=affected_requests
        )

        assert len(actions) == 0

    def test_get_paths_returns_both_paths(
        self, protection_router: OnePlusOneProtection
    ) -> None:
        """Test that get_paths returns primary and backup."""
        paths = protection_router.get_paths(0, 3, k=2)

        assert len(paths) == 2
        # Verify paths are disjoint
        primary_links = set(zip(paths[0][:-1], paths[0][1:], strict=False))
        backup_links = set(zip(paths[1][:-1], paths[1][1:], strict=False))
        for link in backup_links:
            assert link not in primary_links

    def test_update_weights(
        self, protection_router: OnePlusOneProtection, sample_topology: nx.Graph
    ) -> None:
        """Test that update_weights sets uniform weights."""
        protection_router.update_weights(sample_topology)

        # Check that all edges have weight 1.0
        for u, v in sample_topology.edges():
            assert sample_topology[u][v]["weight"] == 1.0

    def test_get_metrics(self, protection_router: OnePlusOneProtection) -> None:
        """Test that metrics are returned correctly."""
        # Perform some operations
        protection_router.route(0, 3, None)
        protection_router.route(0, 3, None)

        metrics = protection_router.get_metrics()

        assert "algorithm" in metrics
        assert metrics["algorithm"] == "1plus1_protection"
        assert "disjoint_paths_found" in metrics
        assert metrics["disjoint_paths_found"] == 2
        assert "success_rate" in metrics

    def test_reset(self, protection_router: OnePlusOneProtection) -> None:
        """Test that reset clears metrics."""
        # Perform operations
        protection_router.route(0, 3, None)
        assert protection_router._disjoint_paths_found == 1

        # Reset
        protection_router.reset()

        assert protection_router._disjoint_paths_found == 0
        assert protection_router._disjoint_paths_failed == 0

    def test_disjoint_k_shortest_fallback(
        self, protection_router: OnePlusOneProtection, sample_topology: nx.Graph
    ) -> None:
        """Test that K-shortest fallback works when Suurballe fails."""
        # This tests the find_disjoint_paths_k_shortest method
        primary, backup = protection_router.find_disjoint_paths_k_shortest(0, 3, k=10)

        assert primary is not None
        assert backup is not None

        # Verify disjointness
        primary_links = set(zip(primary[:-1], primary[1:], strict=False))
        backup_links = set(zip(backup[:-1], backup[1:], strict=False))
        assert primary_links.isdisjoint(backup_links)

    def test_protection_switchover_time_configurable(self) -> None:
        """Test that protection switchover time is configurable."""
        engine_props = {
            "protection_settings": {
                "protection_switchover_ms": 100.0,
                "revert_to_primary": True,
            }
        }
        sdn_props = SDNProps()

        router = OnePlusOneProtection(engine_props, sdn_props)

        assert router.protection_switchover_ms == 100.0
        assert router.revert_to_primary is True


class TestDualPathDisjointness:
    """Test suite specifically for path disjointness verification."""

    def test_no_shared_links_simple_topology(self) -> None:
        """Test disjoint paths on a simple topology."""
        G = nx.Graph()
        for u, v in [(0, 1), (1, 2), (0, 3), (3, 2)]:
            G.add_edge(u, v, length=100.0)

        engine_props = {"topology": G, "protection_settings": {}}
        sdn_props = SDNProps()
        sdn_props.topology = G

        router = OnePlusOneProtection(engine_props, sdn_props)
        primary, backup = router.find_disjoint_paths(0, 2)

        assert primary is not None
        assert backup is not None

        # Verify complete disjointness
        primary_links = set(zip(primary[:-1], primary[1:], strict=False))
        backup_links = set(zip(backup[:-1], backup[1:], strict=False))

        for p_link in primary_links:
            assert p_link not in backup_links
            assert (p_link[1], p_link[0]) not in backup_links

    def test_multiple_disjoint_paths_available(self) -> None:
        """Test when multiple disjoint path pairs exist."""
        G = nx.Graph()
        # Create a grid-like topology with multiple disjoint paths
        edges = [
            (0, 1),
            (1, 2),
            (2, 3),  # Top path
            (0, 4),
            (4, 5),
            (5, 3),  # Middle path
            (0, 6),
            (6, 7),
            (7, 3),  # Bottom path
        ]
        for u, v in edges:
            G.add_edge(u, v, length=100.0)

        engine_props = {"topology": G, "protection_settings": {}}
        sdn_props = SDNProps()
        sdn_props.topology = G

        router = OnePlusOneProtection(engine_props, sdn_props)
        primary, backup = router.find_disjoint_paths(0, 3)

        assert primary is not None
        assert backup is not None

        # Verify they are different paths
        assert primary != backup

    def test_no_disjoint_paths_linear(self) -> None:
        """Test that linear topology returns None (no disjoint paths)."""
        G = nx.path_graph(5)  # Linear: 0-1-2-3-4
        # Add length attribute to all edges
        for u, v in G.edges():
            G[u][v]["length"] = 100.0

        engine_props = {"topology": G, "protection_settings": {}}
        sdn_props = SDNProps()
        sdn_props.topology = G

        router = OnePlusOneProtection(engine_props, sdn_props)
        primary, backup = router.find_disjoint_paths(0, 4)

        # Linear topology has no disjoint paths
        assert primary is None or backup is None
