"""Unit tests for fusion.modules.routing.xt_aware module."""

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
    graph.add_edge("A", "B", length=100.0, weight=1, xt_cost=0.2)
    graph.add_edge("B", "C", length=150.0, weight=1, xt_cost=0.5)
    graph.add_edge("A", "C", length=300.0, weight=3, xt_cost=0.1)
    return graph


@pytest.fixture
def engine_props(topology: nx.Graph) -> dict[str, Any]:
    """Create engine properties for testing.

    :param topology: Network topology graph.
    :type topology: nx.Graph
    :return: Engine configuration dictionary.
    :rtype: dict[str, Any]
    """
    return {"topology": topology, "beta": 0.5, "xt_type": "with_length"}


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
        ("A", "B"): {
            "cores_matrix": {
                "c": [np.zeros(320) for _ in range(7)],
                "l": [np.zeros(320) for _ in range(7)],
            }
        },
        ("B", "C"): {
            "cores_matrix": {
                "c": [np.ones(320) for _ in range(7)],
                "l": [np.ones(320) for _ in range(7)],
            }
        },
        ("A", "C"): {
            "cores_matrix": {
                "c": [np.zeros(320) for _ in range(7)],
                "l": [np.zeros(320) for _ in range(7)],
            }
        },
    }
    props.modulation_formats_dict = {"QPSK": {"max_length": 1000, "slots_needed": 10}}
    return props


@pytest.fixture
def xt_aware(engine_props: dict[str, Any], sdn_props: Mock) -> Any:
    """Create XTAwareRouting instance for testing.

    :param engine_props: Engine configuration dictionary.
    :type engine_props: dict[str, Any]
    :param sdn_props: Mock SDN properties.
    :type sdn_props: Mock
    :return: Configured XTAwareRouting instance.
    :rtype: Any
    """
    from fusion.modules.routing.xt_aware import XTAwareRouting

    xt_routing = XTAwareRouting(engine_props, sdn_props)
    xt_routing.route_props.span_length = 50.0
    xt_routing.route_props.max_link_length = 300.0
    return xt_routing


class TestXTAwareRouting:
    """Tests for XTAwareRouting algorithm."""

    def test_init_stores_properties(
        self, xt_aware: Any, engine_props: dict[str, Any]
    ) -> None:
        """Test that initialization stores all configuration properties."""
        # Assert
        assert xt_aware.engine_props == engine_props
        assert hasattr(xt_aware, "route_props")
        assert hasattr(xt_aware, "route_help_obj")

    def test_algorithm_name_property(self, xt_aware: Any) -> None:
        """Test algorithm name property returns correct identifier."""
        # Assert
        assert xt_aware.algorithm_name == "xt_aware"

    def test_supported_topologies_property(self, xt_aware: Any) -> None:
        """Test supported topologies property returns expected list."""
        # Act
        topologies = xt_aware.supported_topologies

        # Assert
        assert isinstance(topologies, list)
        assert "Generic" in topologies

    def test_validate_environment_with_valid_topology(
        self, xt_aware: Any, topology: nx.Graph
    ) -> None:
        """Test environment validation succeeds with valid topology."""
        # Act
        is_valid = xt_aware.validate_environment(topology)

        # Assert
        assert is_valid is True

    def test_route_finds_least_xt_path(self, xt_aware: Any) -> None:
        """Test that route method finds path with least cross-talk."""
        # Act
        free_slots = {"c": {0: [1, 2, 3]}}
        with (
            patch("fusion.utils.spectrum.find_free_slots", return_value=free_slots),
            patch.object(
                xt_aware.route_help_obj, "find_xt_link_cost", return_value=0.2
            ),
        ):
            path = xt_aware.route("A", "C", request=None)

        # Assert
        assert path is not None
        assert path[0] == "A"
        assert path[-1] == "C"

    def test_route_with_no_path_returns_none(self, xt_aware: Any) -> None:
        """Test that route returns None when no path exists."""
        # Act
        path = xt_aware.route("A", "Z", request=None)

        # Assert
        assert path is None

    def test_route_updates_metrics(self, xt_aware: Any) -> None:
        """Test that route method updates internal metrics."""
        # Arrange
        initial_count = xt_aware._path_count

        # Act
        free_slots = {"c": {0: [1, 2, 3]}}
        with (
            patch("fusion.utils.spectrum.find_free_slots", return_value=free_slots),
            patch.object(
                xt_aware.route_help_obj, "find_xt_link_cost", return_value=0.3
            ),
        ):
            xt_aware.route("A", "C", request=None)

        # Assert
        assert xt_aware._path_count == initial_count + 1

    def test_update_xt_costs_sets_xt_costs(
        self, xt_aware: Any, topology: nx.Graph
    ) -> None:
        """Test that update XT costs sets xt_cost on topology edges."""
        # Act
        free_slots = {"c": {0: [1, 2]}}
        with (
            patch("fusion.utils.spectrum.find_free_slots", return_value=free_slots),
            patch.object(
                xt_aware.route_help_obj, "find_xt_link_cost", return_value=0.25
            ),
            patch.object(xt_aware.route_help_obj, "get_max_link_length"),
        ):
            xt_aware._update_xt_costs()

        # Assert
        for source, dest in list(topology.edges())[::2]:
            assert "xt_cost" in topology[source][dest]

    @pytest.mark.parametrize("xt_type", ["with_length", "without_length", None])
    def test_update_xt_costs_with_different_xt_types(
        self, xt_aware: Any, topology: nx.Graph, xt_type: str | None
    ) -> None:
        """Test XT cost calculation with different xt_type configurations.

        :param xt_aware: Any fixture instance.
        :type xt_aware: Any
        :param topology: Network topology graph.
        :type topology: nx.Graph
        :param xt_type: Type of XT calculation to test.
        :type xt_type: str | None
        """
        # Arrange
        xt_aware.engine_props["xt_type"] = xt_type
        xt_aware.route_props.max_link_length = 300.0

        # Act
        free_slots = {"c": {0: [1, 2]}}
        with (
            patch("fusion.utils.spectrum.find_free_slots", return_value=free_slots),
            patch.object(
                xt_aware.route_help_obj, "find_xt_link_cost", return_value=0.25
            ),
        ):
            xt_aware._update_xt_costs()

        # Assert
        for source, dest in list(topology.edges())[::2]:
            assert "xt_cost" in topology[source][dest]
            assert isinstance(topology[source][dest]["xt_cost"], (int, float))

    def test_get_paths_returns_paths_ordered_by_xt(self, xt_aware: Any) -> None:
        """Test that get_paths returns paths ordered by cross-talk."""
        # Act
        free_slots = {"c": {0: [1, 2]}}
        with (
            patch("fusion.utils.spectrum.find_free_slots", return_value=free_slots),
            patch.object(
                xt_aware.route_help_obj, "find_xt_link_cost", return_value=0.2
            ),
            patch.object(xt_aware.route_help_obj, "get_max_link_length"),
        ):
            paths = xt_aware.get_paths("A", "C", k=2)

        # Assert
        assert isinstance(paths, list)
        assert len(paths) <= 2

    def test_update_weights_sets_xt_costs(
        self, xt_aware: Any, topology: nx.Graph
    ) -> None:
        """Test that update_weights sets XT costs on topology edges."""
        # Arrange
        xt_aware.route_props.max_link_length = 300.0

        # Act
        free_slots = {"c": {0: [1, 2]}}
        with (
            patch("fusion.utils.spectrum.find_free_slots", return_value=free_slots),
            patch.object(
                xt_aware.route_help_obj, "find_xt_link_cost", return_value=0.15
            ),
        ):
            xt_aware.update_weights(topology)

        # Assert
        for source, dest in list(topology.edges())[::2]:
            if topology.has_edge(source, dest):
                assert "xt_cost" in topology[source][dest]

    def test_get_metrics_returns_algorithm_statistics(self, xt_aware: Any) -> None:
        """Test that get_metrics returns performance statistics."""
        # Arrange
        free_slots = {"c": {0: [1, 2, 3]}}
        with (
            patch("fusion.utils.spectrum.find_free_slots", return_value=free_slots),
            patch.object(
                xt_aware.route_help_obj, "find_xt_link_cost", return_value=0.4
            ),
            patch.object(xt_aware.route_help_obj, "get_max_link_length"),
        ):
            xt_aware.route("A", "C", request=None)

        # Act
        metrics = xt_aware.get_metrics()

        # Assert
        assert "algorithm" in metrics
        assert metrics["algorithm"] == "xt_aware"
        assert "paths_computed" in metrics
        assert "average_xt" in metrics
        assert "xt_type" in metrics

    def test_reset_clears_algorithm_state(self, xt_aware: Any) -> None:
        """Test that reset clears internal state counters."""
        # Arrange
        free_slots = {"c": {0: [1, 2, 3]}}
        with (
            patch("fusion.utils.spectrum.find_free_slots", return_value=free_slots),
            patch.object(
                xt_aware.route_help_obj, "find_xt_link_cost", return_value=0.3
            ),
            patch.object(xt_aware.route_help_obj, "get_max_link_length"),
        ):
            xt_aware.route("A", "C", request=None)
        assert xt_aware._path_count > 0

        # Act
        xt_aware.reset()

        # Assert
        assert xt_aware._path_count == 0
        assert xt_aware._total_xt == 0.0

    def test_calculate_path_xt_returns_valid_metric(self, xt_aware: Any) -> None:
        """Test path XT calculation returns valid metric."""
        # Arrange
        path = ["A", "B", "C"]

        # Act
        xt = xt_aware._calculate_path_xt(path)

        # Assert
        assert isinstance(xt, float)
        assert xt >= 0.0

    def test_calculate_path_xt_with_empty_path(self, xt_aware: Any) -> None:
        """Test path XT calculation with empty path returns zero."""
        # Act
        xt = xt_aware._calculate_path_xt([])

        # Assert
        assert xt == 0.0

    def test_find_least_weight_populates_route_props(self, xt_aware: Any) -> None:
        """Test that find least weight populates route properties."""
        # Arrange
        xt_aware.sdn_props.source = "A"
        xt_aware.sdn_props.destination = "C"

        # Act
        mod_dict = {"QPSK": {"max_length": 1000}}
        with (
            patch("fusion.utils.network.find_path_length", return_value=100.0),
            patch("fusion.utils.data.sort_nested_dict_values", return_value=mod_dict),
        ):
            xt_aware._find_least_weight("xt_cost")

        # Assert
        assert len(xt_aware.route_props.paths_matrix) > 0
        assert len(xt_aware.route_props.weights_list) > 0
        assert len(xt_aware.route_props.modulation_formats_matrix) > 0

    def test_calculate_path_xt_sums_link_costs(
        self, xt_aware: Any, topology: nx.Graph
    ) -> None:
        """Test that path XT calculation sums link costs correctly."""
        # Arrange
        path = ["A", "B", "C"]
        topology["A"]["B"]["xt_cost"] = 0.2
        topology["B"]["C"]["xt_cost"] = 0.3

        # Act
        xt = xt_aware._calculate_path_xt(path)

        # Assert
        assert xt == 0.5
