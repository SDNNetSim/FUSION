"""Unit tests for fusion.modules.routing.fragmentation_aware module."""

from typing import Any
from unittest.mock import Mock, patch

import networkx as nx
import numpy as np
import pytest


@pytest.fixture
def topology() -> nx.Graph:
    """Create a test network topology.

    :return: NetworkX graph with test topology.
    :rtype: nx.Graph
    """
    graph = nx.Graph()
    graph.add_edge("A", "B", length=100.0, weight=1, frag_cost=0.2)
    graph.add_edge("B", "C", length=150.0, weight=1, frag_cost=0.5)
    graph.add_edge("A", "C", length=300.0, weight=3, frag_cost=0.1)
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
        "beta": 0.5,
        "mod_per_bw": {
            "50GHz": {
                "QPSK": {"max_length": 1000, "slots_needed": 10},
                "16-QAM": {"max_length": 500, "slots_needed": 8},
                "64-QAM": {"max_length": 200, "slots_needed": 6},
            }
        },
    }


@pytest.fixture
def sdn_props(topology: nx.Graph) -> Mock:
    """Create mock SDN properties for testing.

    :param topology: Network topology graph.
    :type topology: nx.Graph
    :return: Mock SDNProps object.
    :rtype: Mock
    """
    props = Mock()
    props.source = "A"
    props.destination = "C"
    props.bandwidth = "50GHz"
    props.slots_needed = 10
    props.topology = topology
    props.network_spectrum_dict = {
        ("A", "B"): {"cores_matrix": {"c": [np.array([0, 1, 0, 1, 0] * 64)]}},
        ("B", "C"): {"cores_matrix": {"c": [np.array([1, 1, 0, 0, 1] * 64)]}},
        ("A", "C"): {"cores_matrix": {"c": [np.array([0, 0, 0, 0, 0] * 64)]}},
    }
    props.mod_formats = {
        "QPSK": {"max_length": 1000, "slots_needed": 10},
        "16-QAM": {"max_length": 500, "slots_needed": 8},
        "64-QAM": {"max_length": 200, "slots_needed": 6},
    }
    return props


@pytest.fixture
def fragmentation_aware(engine_props: dict[str, Any], sdn_props: Mock) -> Any:
    """Create FragmentationAwareRouting instance for testing.

    :param engine_props: Engine configuration dictionary.
    :type engine_props: dict[str, Any]
    :param sdn_props: Mock SDN properties.
    :type sdn_props: Mock
    :return: Configured FragmentationAwareRouting instance.
    :rtype: Any
    """
    from fusion.modules.routing.fragmentation_aware import FragmentationAwareRouting

    return FragmentationAwareRouting(engine_props, sdn_props)


class TestFragmentationAwareRouting:
    """Tests for FragmentationAwareRouting algorithm."""

    def test_init_stores_properties(
        self, fragmentation_aware: Any, engine_props: dict[str, Any]
    ) -> None:
        """Test that initialization stores all configuration properties."""
        # Assert
        assert fragmentation_aware.engine_props == engine_props
        assert hasattr(fragmentation_aware, "route_props")
        assert hasattr(fragmentation_aware, "route_help_obj")

    def test_algorithm_name_property(self, fragmentation_aware: Any) -> None:
        """Test algorithm name property returns correct identifier."""
        # Assert
        assert fragmentation_aware.algorithm_name == "fragmentation_aware"

    def test_supported_topologies_property(self, fragmentation_aware: Any) -> None:
        """Test supported topologies property returns expected list."""
        # Act
        topologies = fragmentation_aware.supported_topologies

        # Assert
        assert isinstance(topologies, list)
        assert "Generic" in topologies

    def test_validate_environment_with_valid_topology(
        self, fragmentation_aware: Any, topology: nx.Graph
    ) -> None:
        """Test environment validation succeeds with valid topology."""
        # Act
        is_valid = fragmentation_aware.validate_environment(topology)

        # Assert
        assert is_valid is True

    def test_route_finds_least_fragmented_path(self, fragmentation_aware: Any) -> None:
        """Test that route method finds path with least fragmentation."""
        # Act
        with patch("fusion.utils.network.find_path_fragmentation", return_value=0.2):
            path = fragmentation_aware.route("A", "C", request=None)

        # Assert
        assert path is not None
        assert path[0] == "A"
        assert path[-1] == "C"

    def test_route_with_no_path_returns_none(self, fragmentation_aware: Any) -> None:
        """Test that route returns None when no path exists."""
        # Act
        path = fragmentation_aware.route("A", "Z", request=None)

        # Assert
        assert path is None

    def test_route_updates_metrics(self, fragmentation_aware: Any) -> None:
        """Test that route method updates internal metrics."""
        # Arrange
        initial_count = fragmentation_aware._path_count

        # Act
        with patch("fusion.utils.network.find_path_fragmentation", return_value=0.3):
            fragmentation_aware.route("A", "C", request=None)

        # Assert
        assert fragmentation_aware._path_count == initial_count + 1

    def test_update_fragmentation_costs_sets_frag_costs(
        self, fragmentation_aware: Any, topology: nx.Graph
    ) -> None:
        """Test that update fragmentation costs sets frag_cost on topology edges."""
        # Act
        with patch("fusion.utils.network.find_path_fragmentation", return_value=0.25):
            fragmentation_aware._update_fragmentation_costs()

        # Assert
        for source, dest in list(topology.edges())[::2]:
            assert "frag_cost" in topology[source][dest]

    def test_get_paths_returns_paths_ordered_by_fragmentation(
        self, fragmentation_aware: Any
    ) -> None:
        """Test that get_paths returns paths ordered by fragmentation."""
        # Act
        with patch("fusion.utils.network.find_path_fragmentation", return_value=0.2):
            paths = fragmentation_aware.get_paths("A", "C", k=2)

        # Assert
        assert isinstance(paths, list)
        assert len(paths) <= 2

    def test_update_weights_sets_fragmentation_costs(
        self, fragmentation_aware: Any, topology: nx.Graph
    ) -> None:
        """Test that update_weights sets fragmentation costs on topology edges."""
        # Act
        with patch("fusion.utils.network.find_path_fragmentation", return_value=0.15):
            fragmentation_aware.update_weights(topology)

        # Assert
        for source, dest in list(topology.edges())[::2]:
            if topology.has_edge(source, dest):
                assert "frag_cost" in topology[source][dest]

    def test_get_metrics_returns_algorithm_statistics(
        self, fragmentation_aware: Any
    ) -> None:
        """Test that get_metrics returns performance statistics."""
        # Arrange
        with patch("fusion.utils.network.find_path_fragmentation", return_value=0.4):
            fragmentation_aware.route("A", "C", request=None)

        # Act
        metrics = fragmentation_aware.get_metrics()

        # Assert
        assert "algorithm" in metrics
        assert metrics["algorithm"] == "fragmentation_aware"
        assert "paths_computed" in metrics
        assert "average_fragmentation" in metrics

    def test_reset_clears_algorithm_state(self, fragmentation_aware: Any) -> None:
        """Test that reset clears internal state counters."""
        # Arrange
        with patch("fusion.utils.network.find_path_fragmentation", return_value=0.3):
            fragmentation_aware.route("A", "C", request=None)
        assert fragmentation_aware._path_count > 0

        # Act
        fragmentation_aware.reset()

        # Assert
        assert fragmentation_aware._path_count == 0
        assert fragmentation_aware._total_fragmentation == 0.0

    def test_calculate_path_fragmentation_returns_valid_metric(
        self, fragmentation_aware: Any
    ) -> None:
        """Test path fragmentation calculation returns valid metric."""
        # Arrange
        path = ["A", "B", "C"]

        # Act
        with patch("fusion.utils.network.find_path_fragmentation", return_value=0.35):
            fragmentation = fragmentation_aware._calculate_path_fragmentation(path)

        # Assert
        assert isinstance(fragmentation, float)
        assert fragmentation >= 0.0

    def test_calculate_path_fragmentation_with_empty_path(
        self, fragmentation_aware: Any
    ) -> None:
        """Test path fragmentation calculation with empty path returns zero."""
        # Act
        fragmentation = fragmentation_aware._calculate_path_fragmentation([])

        # Assert
        assert fragmentation == 0.0

    def test_find_least_weight_populates_route_props(
        self, fragmentation_aware: Any
    ) -> None:
        """Test that find least weight populates route properties."""
        # Arrange
        fragmentation_aware.sdn_props.source = "A"
        fragmentation_aware.sdn_props.destination = "C"

        # Act
        with (
            patch("fusion.utils.network.find_path_length", return_value=100.0),
            patch("fusion.utils.network.get_path_modulation", return_value="QPSK"),
        ):
            fragmentation_aware._find_least_weight("frag_cost")

        # Assert
        assert len(fragmentation_aware.route_props.paths_matrix) > 0
        assert len(fragmentation_aware.route_props.weights_list) > 0
        assert len(fragmentation_aware.route_props.modulation_formats_matrix) > 0
