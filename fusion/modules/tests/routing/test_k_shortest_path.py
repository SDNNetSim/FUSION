"""Unit tests for fusion.modules.routing.k_shortest_path module."""

from typing import Any
from unittest.mock import Mock

import networkx as nx
import pytest


@pytest.fixture
def topology() -> nx.Graph:
    """Create a test network topology.

    :return: NetworkX graph with test topology.
    :rtype: nx.Graph
    """
    graph = nx.Graph()
    graph.add_edge("A", "B", length=100.0, weight=1)
    graph.add_edge("B", "C", length=150.0, weight=1)
    graph.add_edge("A", "C", length=300.0, weight=3)
    graph.add_edge("B", "D", length=200.0, weight=2)
    graph.add_edge("C", "D", length=120.0, weight=1)
    return graph


@pytest.fixture
def engine_props(topology: nx.Graph) -> dict[str, Any]:
    """Create engine properties for testing.

    :param topology: Network topology graph.
    :type topology: nx.Graph
    :return: Engine configuration dictionary.
    :rtype: dict[str, Any]
    """
    return {
        "topology": topology,
        "k_paths": 3,
        "routing_weight": "length",
        "mod_per_bw": {
            "50GHz": {
                "QPSK": {"max_length": 1000, "slots_needed": 10},
                "16-QAM": {"max_length": 500, "slots_needed": 8},
                "64-QAM": {"max_length": 200, "slots_needed": 6},
            }
        },
    }


@pytest.fixture
def sdn_props() -> Mock:
    """Create mock SDN properties for testing.

    :return: Mock SDNProps object.
    :rtype: Mock
    """
    props = Mock()
    props.source = "A"
    props.destination = "D"
    props.bandwidth = "50GHz"
    props.slots_needed = 10
    props.topology = None
    props.modulation_formats_dict = {
        "QPSK": {"max_length": 1000, "slots_needed": 10},
        "16-QAM": {"max_length": 500, "slots_needed": 8},
        "64-QAM": {"max_length": 200, "slots_needed": 6},
    }
    return props


@pytest.fixture
def k_shortest_path(engine_props: dict[str, Any], sdn_props: Mock) -> Any:
    """Create KShortestPath instance for testing.

    :param engine_props: Engine configuration dictionary.
    :type engine_props: dict[str, Any]
    :param sdn_props: Mock SDN properties.
    :type sdn_props: Mock
    :return: Configured KShortestPath instance.
    :rtype: Any
    """
    from fusion.modules.routing.k_shortest_path import KShortestPath

    return KShortestPath(engine_props, sdn_props)


class TestKShortestPath:
    """Tests for KShortestPath routing algorithm."""

    def test_init_stores_properties(
        self, k_shortest_path: Any, engine_props: dict[str, Any]
    ) -> None:
        """Test that initialization stores all configuration properties."""
        # Assert
        assert k_shortest_path.engine_props == engine_props
        assert k_shortest_path.k_paths_count == 3
        assert k_shortest_path.routing_weight == "length"

    def test_algorithm_name_property(self, k_shortest_path: Any) -> None:
        """Test algorithm name property returns correct identifier."""
        # Assert
        assert k_shortest_path.algorithm_name == "k_shortest_path"

    def test_supported_topologies_property(self, k_shortest_path: Any) -> None:
        """Test supported topologies property returns expected list."""
        # Act
        topologies = k_shortest_path.supported_topologies

        # Assert
        assert isinstance(topologies, list)
        assert "NSFNet" in topologies
        assert "Generic" in topologies

    def test_validate_environment_with_connected_topology(
        self, k_shortest_path: Any, topology: nx.Graph
    ) -> None:
        """Test environment validation succeeds with connected topology."""
        # Act
        is_valid = k_shortest_path.validate_environment(topology)

        # Assert
        assert is_valid is True

    def test_validate_environment_with_disconnected_topology(
        self, k_shortest_path: Any
    ) -> None:
        """Test environment validation fails with disconnected topology."""
        # Arrange
        disconnected_topology = nx.Graph()
        disconnected_topology.add_edge("A", "B", length=100)
        disconnected_topology.add_edge("C", "D", length=100)

        # Act
        is_valid = k_shortest_path.validate_environment(disconnected_topology)

        # Assert
        assert is_valid is False

    def test_route_finds_shortest_path(self, k_shortest_path: Any) -> None:
        """Test that route method finds the shortest path."""
        # Act
        k_shortest_path.route("A", "D", request=None)

        # Assert
        assert len(k_shortest_path.route_props.paths_matrix) > 0
        path = k_shortest_path.route_props.paths_matrix[0]
        assert path[0] == "A"
        assert path[-1] == "D"
        assert len(path) >= 2

    def test_route_populates_route_props(self, k_shortest_path: Any) -> None:
        """Test that route method populates route_props matrices."""
        # Act
        k_shortest_path.route("A", "C", request=None)

        # Assert
        assert len(k_shortest_path.route_props.paths_matrix) > 0
        assert len(k_shortest_path.route_props.modulation_formats_matrix) > 0
        assert len(k_shortest_path.route_props.weights_list) > 0

    def test_route_with_no_path_returns_none(self, k_shortest_path: Any) -> None:
        """Test that route sets empty paths_matrix when no path exists."""
        # Act
        k_shortest_path.route("A", "Z", request=None)

        # Assert
        assert len(k_shortest_path.route_props.paths_matrix) == 0

    def test_get_paths_returns_k_paths(self, k_shortest_path: Any) -> None:
        """Test that get_paths returns up to k shortest paths."""
        # Act
        paths = k_shortest_path.get_paths("A", "D", k=3)

        # Assert
        assert isinstance(paths, list)
        assert len(paths) <= 3
        assert all(isinstance(path, list) for path in paths)

    def test_get_paths_respects_weight_parameter(self, k_shortest_path: Any) -> None:
        """Test that get_paths respects the routing_weight configuration."""
        # Act
        paths = k_shortest_path.get_paths("A", "C", k=2)

        # Assert
        assert len(paths) >= 1
        # First path should be shortest by configured weight
        assert paths[0][0] == "A"
        assert paths[0][-1] == "C"

    def test_get_paths_with_no_paths_returns_empty_list(
        self, k_shortest_path: Any
    ) -> None:
        """Test that get_paths returns empty list when no paths exist."""
        # Act
        paths = k_shortest_path.get_paths("A", "Z", k=3)

        # Assert
        assert paths == []

    @pytest.mark.parametrize(
        "k_value,expected_max_paths",
        [
            (1, 1),
            (2, 2),
            (5, 5),
        ],
    )
    def test_get_paths_respects_k_parameter(
        self, k_shortest_path: Any, k_value: int, expected_max_paths: int
    ) -> None:
        """Test that get_paths respects the k parameter.

        :param k_shortest_path: KShortestPath fixture instance.
        :type k_shortest_path: Any
        :param k_value: Number of paths to request.
        :type k_value: int
        :param expected_max_paths: Maximum expected paths returned.
        :type expected_max_paths: int
        """
        # Act
        paths = k_shortest_path.get_paths("A", "D", k=k_value)

        # Assert
        assert len(paths) <= expected_max_paths

    def test_update_weights_does_not_raise_error(
        self, k_shortest_path: Any, topology: nx.Graph
    ) -> None:
        """Test that update_weights method runs without error."""
        # Act & Assert (should not raise)
        k_shortest_path.update_weights(topology)

    def test_get_metrics_returns_algorithm_statistics(
        self, k_shortest_path: Any
    ) -> None:
        """Test that get_metrics returns performance statistics."""
        # Arrange
        k_shortest_path.route("A", "D", request=None)

        # Act
        metrics = k_shortest_path.get_metrics()

        # Assert
        assert "algorithm" in metrics
        assert metrics["algorithm"] == "k_shortest_path"
        assert "paths_computed" in metrics
        assert metrics["paths_computed"] == 1
        assert "average_hop_count" in metrics
        assert "k_value" in metrics
        assert "weight_metric" in metrics

    def test_reset_clears_algorithm_state(self, k_shortest_path: Any) -> None:
        """Test that reset clears internal state counters."""
        # Arrange
        k_shortest_path.route("A", "D", request=None)
        assert k_shortest_path._path_count > 0

        # Act
        k_shortest_path.reset()

        # Assert
        assert k_shortest_path._path_count == 0
        assert k_shortest_path._total_hops == 0

    def test_route_updates_metrics(self, k_shortest_path: Any) -> None:
        """Test that route method updates internal metrics."""
        # Arrange
        initial_count = k_shortest_path._path_count

        # Act
        k_shortest_path.route("A", "C", request=None)

        # Assert
        assert k_shortest_path._path_count == initial_count + 1
        assert k_shortest_path._total_hops > 0

    def test_route_with_unweighted_shortest_paths(self) -> None:
        """Test routing with unweighted (hop-count based) shortest paths."""
        from fusion.modules.routing.k_shortest_path import KShortestPath

        # Arrange
        topology = nx.Graph()
        topology.add_edge("A", "B", length=100.0)
        topology.add_edge("B", "C", length=100.0)
        topology.add_edge("A", "C", length=250.0)

        engine_props = {
            "topology": topology,
            "k_paths": 2,
            "routing_weight": None,
            "mod_per_bw": {
                "50GHz": {
                    "QPSK": {"max_length": 1000},
                    "16-QAM": {"max_length": 500},
                    "64-QAM": {"max_length": 200},
                }
            },
        }
        sdn_props = Mock()
        sdn_props.bandwidth = "50GHz"
        sdn_props.modulation_formats_dict = {
            "QPSK": {"max_length": 1000, "slots_needed": 10},
            "16-QAM": {"max_length": 500, "slots_needed": 8},
            "64-QAM": {"max_length": 200, "slots_needed": 6},
        }

        algorithm = KShortestPath(engine_props, sdn_props)

        # Act
        algorithm.route("A", "C", request=None)

        # Assert
        assert len(algorithm.route_props.paths_matrix) > 0
        path = algorithm.route_props.paths_matrix[0]
        assert len(path) == 2  # Direct path should be shortest by hops

    def test_route_with_modulation_format_selection(self, k_shortest_path: Any) -> None:
        """Test that route method properly selects modulation formats."""
        # Act
        k_shortest_path.route("A", "D", request=None)

        # Assert
        assert len(k_shortest_path.route_props.modulation_formats_matrix) > 0
        modulation_formats = k_shortest_path.route_props.modulation_formats_matrix[0]
        assert isinstance(modulation_formats, list)
        assert len(modulation_formats) > 0
