"""Unit tests for fusion.modules.routing.least_congested module."""

from typing import Any
from unittest.mock import Mock

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
    graph.add_edge('A', 'B', length=100.0, weight=1)
    graph.add_edge('B', 'C', length=150.0, weight=1)
    graph.add_edge('A', 'C', length=300.0, weight=3)
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
        'topology': topology,
        'k_paths': 2,
        'beta': 0.5
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
    props.source = 'A'
    props.destination = 'C'
    props.bandwidth = '50GHz'
    props.slots_needed = 10
    props.topology = topology
    props.network_spectrum_dict = {
        ('A', 'B'): {
            'cores_matrix': {
                'c': [np.zeros(320), np.zeros(320)]
            }
        },
        ('B', 'C'): {
            'cores_matrix': {
                'c': [np.ones(320), np.ones(320)]
            }
        },
        ('A', 'C'): {
            'cores_matrix': {
                'c': [np.zeros(320), np.zeros(320)]
            }
        }
    }
    return props


@pytest.fixture
def least_congested(
    engine_props: dict[str, Any], sdn_props: Mock
) -> Any:
    """Create LeastCongestedRouting instance for testing.

    :param engine_props: Engine configuration dictionary.
    :type engine_props: dict[str, Any]
    :param sdn_props: Mock SDN properties.
    :type sdn_props: Mock
    :return: Configured LeastCongestedRouting instance.
    :rtype: Any
    """
    from fusion.modules.routing.least_congested import LeastCongestedRouting
    return LeastCongestedRouting(engine_props, sdn_props)


class TestLeastCongestedRouting:
    """Tests for LeastCongestedRouting algorithm."""

    def test_init_stores_properties(
        self, least_congested: Any, engine_props: dict[str, Any]
    ) -> None:
        """Test that initialization stores all configuration properties."""
        # Assert
        assert least_congested.engine_props == engine_props
        assert hasattr(least_congested, 'route_props')
        assert hasattr(least_congested, 'route_help_obj')

    def test_algorithm_name_property(
        self, least_congested: Any
    ) -> None:
        """Test algorithm name property returns correct identifier."""
        # Assert
        assert least_congested.algorithm_name == 'least_congested'

    def test_supported_topologies_property(
        self, least_congested: Any
    ) -> None:
        """Test supported topologies property returns expected list."""
        # Act
        topologies = least_congested.supported_topologies

        # Assert
        assert isinstance(topologies, list)
        assert 'Generic' in topologies

    def test_validate_environment_with_valid_topology(
        self, least_congested: Any, topology: nx.Graph
    ) -> None:
        """Test environment validation succeeds with valid topology."""
        # Act
        is_valid = least_congested.validate_environment(topology)

        # Assert
        assert is_valid is True

    def test_route_finds_least_congested_path(
        self, least_congested: Any
    ) -> None:
        """Test that route method finds path with least congested bottleneck link."""
        # Act
        path = least_congested.route('A', 'C', request=None)

        # Assert
        assert path is not None
        assert path[0] == 'A'
        assert path[-1] == 'C'

    def test_route_with_no_path_returns_none(
        self, least_congested: Any
    ) -> None:
        """Test that route returns None when no path exists."""
        # Act
        path = least_congested.route('A', 'Z', request=None)

        # Assert
        assert path is None

    def test_route_updates_metrics(
        self, least_congested: Any
    ) -> None:
        """Test that route method updates internal metrics."""
        # Arrange
        initial_count = least_congested._path_count

        # Act
        least_congested.route('A', 'C', request=None)

        # Assert
        assert least_congested._path_count == initial_count + 1

    def test_find_most_cong_link_identifies_bottleneck(
        self, least_congested: Any
    ) -> None:
        """Test that find most congested link identifies the bottleneck."""
        # Arrange
        path_list = ['A', 'B', 'C']

        # Act
        least_congested._find_most_cong_link(path_list)

        # Assert
        assert len(least_congested.route_props.paths_matrix) == 1
        path_data = least_congested.route_props.paths_matrix[0]
        assert 'path_list' in path_data
        assert 'link_dict' in path_data

    def test_select_least_congested_sorts_by_free_slots(
        self, least_congested: Any
    ) -> None:
        """Test that select least congested sorts paths by free slots."""
        # Arrange
        least_congested.route_props.paths_matrix = [
            {'path_list': ['A', 'C'], 'link_dict': {'link': {}, 'free_slots': 100}},
            {'path_list': ['A', 'B', 'C'], 'link_dict': {'link': {}, 'free_slots': 200}}
        ]

        # Act
        least_congested._select_least_congested()

        # Assert
        # Best path should have most free slots (200)
        best_path = least_congested.route_props.paths_matrix[0]
        assert best_path['link_dict']['free_slots'] == 200

    def test_get_paths_returns_best_path(
        self, least_congested: Any
    ) -> None:
        """Test that get_paths returns the best path."""
        # Act
        paths = least_congested.get_paths('A', 'C', k=1)

        # Assert
        assert isinstance(paths, list)
        assert len(paths) <= 1

    def test_update_weights_does_not_raise_error(
        self, least_congested: Any, topology: nx.Graph
    ) -> None:
        """Test that update_weights method runs without error."""
        # Act & Assert (should not raise)
        least_congested.update_weights(topology)

    def test_get_metrics_returns_algorithm_statistics(
        self, least_congested: Any
    ) -> None:
        """Test that get_metrics returns performance statistics."""
        # Arrange
        least_congested.route('A', 'C', request=None)

        # Act
        metrics = least_congested.get_metrics()

        # Assert
        assert 'algorithm' in metrics
        assert metrics['algorithm'] == 'least_congested'
        assert 'paths_computed' in metrics
        assert 'average_congestion' in metrics

    def test_reset_clears_algorithm_state(
        self, least_congested: Any
    ) -> None:
        """Test that reset clears internal state counters."""
        # Arrange
        least_congested.route('A', 'C', request=None)
        assert least_congested._path_count > 0

        # Act
        least_congested.reset()

        # Assert
        assert least_congested._path_count == 0
        assert least_congested._total_congestion == 0.0

    def test_calculate_path_congestion_returns_valid_metric(
        self, least_congested: Any
    ) -> None:
        """Test path congestion calculation returns valid metric."""
        # Arrange
        path = ['A', 'B', 'C']

        # Act
        congestion = least_congested._calculate_path_congestion(path)

        # Assert
        assert isinstance(congestion, float)
        assert 0.0 <= congestion <= 1.0

    def test_calculate_path_congestion_with_empty_path(
        self, least_congested: Any
    ) -> None:
        """Test path congestion calculation with empty path returns zero."""
        # Act
        congestion = least_congested._calculate_path_congestion([])

        # Assert
        assert congestion == 0.0
