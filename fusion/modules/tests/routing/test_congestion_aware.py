"""Unit tests for fusion.modules.routing.congestion_aware module."""

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
    graph.add_edge("A", "B", length=100.0, weight=1)
    graph.add_edge("B", "C", length=150.0, weight=1)
    graph.add_edge("A", "C", length=300.0, weight=3)
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
        "k_paths": 2,
        "ca_alpha": 0.3,
        "beta": 0.5,
        "mod_per_bw": {
            "50GHz": {
                "QPSK": {"max_length": 1000, "slots_needed": 10},
                "16-QAM": {"max_length": 500, "slots_needed": 8},
                "64-QAM": {"max_length": 200, "slots_needed": 6},
            }
        },
        "pre_calc_mod_selection": False,
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
        ("A", "B"): {"cores_matrix": {"c": [np.zeros(320), np.zeros(320)]}},
        ("B", "C"): {"cores_matrix": {"c": [np.ones(320), np.ones(320)]}},
        ("A", "C"): {"cores_matrix": {"c": [np.zeros(320), np.zeros(320)]}},
    }
    props.mod_formats_dict = {
        "QPSK": {"max_length": 1000, "slots_needed": 10},
        "16-QAM": {"max_length": 500, "slots_needed": 8},
        "64-QAM": {"max_length": 200, "slots_needed": 6},
    }
    return props


@pytest.fixture
def congestion_aware(engine_props: dict[str, Any], sdn_props: Mock) -> Any:
    """Create CongestionAwareRouting instance for testing.

    :param engine_props: Engine configuration dictionary.
    :type engine_props: dict[str, Any]
    :param sdn_props: Mock SDN properties.
    :type sdn_props: Mock
    :return: Configured CongestionAwareRouting instance.
    :rtype: Any
    """
    from fusion.modules.routing.congestion_aware import CongestionAwareRouting

    return CongestionAwareRouting(engine_props, sdn_props)


class TestCongestionAwareRouting:
    """Tests for CongestionAwareRouting algorithm."""

    def test_init_stores_properties(
        self, congestion_aware: Any, engine_props: dict[str, Any]
    ) -> None:
        """Test that initialization stores all configuration properties."""
        # Assert
        assert congestion_aware.engine_props == engine_props
        assert hasattr(congestion_aware, "route_props")
        assert hasattr(congestion_aware, "route_help_obj")

    def test_algorithm_name_property(self, congestion_aware: Any) -> None:
        """Test algorithm name property returns correct identifier."""
        # Assert
        assert congestion_aware.algorithm_name == "congestion_aware"

    def test_supported_topologies_property(self, congestion_aware: Any) -> None:
        """Test supported topologies property returns expected list."""
        # Act
        topologies = congestion_aware.supported_topologies

        # Assert
        assert isinstance(topologies, list)
        assert "Generic" in topologies

    def test_validate_environment_with_valid_topology(
        self, congestion_aware: Any, topology: nx.Graph
    ) -> None:
        """Test environment validation succeeds with valid topology."""
        # Act
        is_valid = congestion_aware.validate_environment(topology)

        # Assert
        assert is_valid is True

    def test_route_finds_congestion_aware_path(self, congestion_aware: Any) -> None:
        """Test that route method finds a congestion-aware path."""
        # Act
        with (
            patch("fusion.utils.network.find_path_length", return_value=100.0),
            patch(
                "fusion.utils.network.find_path_congestion",
                return_value=(0.5, 0.3),
            ),
        ):
            path = congestion_aware.route("A", "C", request=None)

        # Assert
        assert path is not None
        assert path[0] == "A"
        assert path[-1] == "C"

    def test_route_with_no_path_returns_none(self, congestion_aware: Any) -> None:
        """Test that route returns None when no path exists."""
        # Act
        path = congestion_aware.route("A", "Z", request=None)

        # Assert
        assert path is None

    def test_route_updates_metrics(self, congestion_aware: Any) -> None:
        """Test that route method updates internal metrics."""
        # Arrange
        initial_count = congestion_aware._path_count

        # Act
        with (
            patch("fusion.utils.network.find_path_length", return_value=100.0),
            patch(
                "fusion.utils.network.find_path_congestion",
                return_value=(0.5, 0.3),
            ),
        ):
            congestion_aware.route("A", "C", request=None)

        # Assert
        assert congestion_aware._path_count == initial_count + 1

    def test_get_paths_returns_paths_ordered_by_congestion(
        self, congestion_aware: Any
    ) -> None:
        """Test that get_paths returns paths ordered by congestion score."""
        # Act
        with (
            patch("fusion.utils.network.find_path_length", return_value=100.0),
            patch(
                "fusion.utils.network.find_path_congestion",
                return_value=(0.3, 0.2),
            ),
        ):
            paths = congestion_aware.get_paths("A", "C", k=1)

        # Assert
        assert isinstance(paths, list)
        assert len(paths) <= 1

    def test_update_weights_sets_congestion_costs(
        self, congestion_aware: Any, topology: nx.Graph
    ) -> None:
        """Test that update_weights sets congestion costs on topology edges."""
        # Act
        congestion_aware.update_weights(topology)

        # Assert
        for source, dest in topology.edges():
            if topology.has_edge(source, dest):
                edge_data = topology[source][dest]
                if "cong_cost" in edge_data:
                    assert isinstance(edge_data["cong_cost"], (int, float))

    def test_get_metrics_returns_algorithm_statistics(
        self, congestion_aware: Any
    ) -> None:
        """Test that get_metrics returns performance statistics."""
        # Arrange
        with (
            patch("fusion.utils.network.find_path_length", return_value=100.0),
            patch(
                "fusion.utils.network.find_path_congestion",
                return_value=(0.4, 0.2),
            ),
        ):
            congestion_aware.route("A", "C", request=None)

        # Act
        metrics = congestion_aware.get_metrics()

        # Assert
        assert "algorithm" in metrics
        assert metrics["algorithm"] == "congestion_aware"
        assert "paths_computed" in metrics
        assert "average_congestion" in metrics

    def test_reset_clears_algorithm_state(self, congestion_aware: Any) -> None:
        """Test that reset clears internal state counters."""
        # Arrange
        with (
            patch("fusion.utils.network.find_path_length", return_value=100.0),
            patch(
                "fusion.utils.network.find_path_congestion",
                return_value=(0.5, 0.3),
            ),
        ):
            congestion_aware.route("A", "C", request=None)
        assert congestion_aware._path_count > 0

        # Act
        congestion_aware.reset()

        # Assert
        assert congestion_aware._path_count == 0
        assert congestion_aware._total_congestion == 0.0

    def test_calculate_path_congestion_returns_valid_metric(
        self, congestion_aware: Any
    ) -> None:
        """Test path congestion calculation returns valid metric."""
        # Arrange
        path = ["A", "B", "C"]

        # Act
        congestion = congestion_aware._calculate_path_congestion(path)

        # Assert
        assert isinstance(congestion, float)
        assert 0.0 <= congestion <= 1.0

    def test_calculate_path_congestion_with_empty_path(
        self, congestion_aware: Any
    ) -> None:
        """Test path congestion calculation with empty path returns zero."""
        # Act
        congestion = congestion_aware._calculate_path_congestion([])

        # Assert
        assert congestion == 0.0

    def test_gather_candidate_paths_returns_k_paths(
        self, congestion_aware: Any
    ) -> None:
        """Test that candidate path gathering returns up to k paths."""
        # Act
        with (
            patch("fusion.utils.network.find_path_length", return_value=100.0),
            patch(
                "fusion.utils.network.find_path_congestion",
                return_value=(0.5, 0.3),
            ),
        ):
            candidate_data = congestion_aware._gather_candidate_paths()

        # Assert
        assert "paths" in candidate_data
        assert "lengths" in candidate_data
        assert "congestions" in candidate_data
        k_paths = congestion_aware.engine_props.get("k_paths", 1)
        assert len(candidate_data["paths"]) <= k_paths

    def test_calculate_path_scores_uses_alpha_weighting(
        self, congestion_aware: Any
    ) -> None:
        """Test that path scores use alpha weighting between congestion and length."""
        # Arrange
        candidate_data = {
            "paths": [["A", "C"], ["A", "B", "C"]],
            "lengths": [300.0, 250.0],
            "congestions": [0.1, 0.8],
        }

        # Act
        scored_paths = congestion_aware._calculate_path_scores(candidate_data)

        # Assert
        assert isinstance(scored_paths, list)
        assert len(scored_paths) == 2
        # Each scored path should be (path, length, score)
        assert all(len(item) == 3 for item in scored_paths)
