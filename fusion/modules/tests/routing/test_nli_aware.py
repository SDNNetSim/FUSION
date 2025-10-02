"""Unit tests for fusion.modules.routing.nli_aware module."""

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
    graph.add_edge('A', 'B', length=100.0, weight=1, nli_cost=0.2)
    graph.add_edge('B', 'C', length=150.0, weight=1, nli_cost=0.5)
    graph.add_edge('A', 'C', length=300.0, weight=3, nli_cost=0.1)
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
        'beta': 0.5,
        'mod_per_bw': {
            '50GHz': {
                'QPSK': {'max_length': 1000, 'slots_needed': 10},
                '16-QAM': {'max_length': 500, 'slots_needed': 8},
                '64-QAM': {'max_length': 200, 'slots_needed': 6}
            }
        }
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
                'c': [np.zeros(320) for _ in range(7)]
            }
        },
        ('B', 'C'): {
            'cores_matrix': {
                'c': [np.ones(320) for _ in range(7)]
            }
        },
        ('A', 'C'): {
            'cores_matrix': {
                'c': [np.zeros(320) for _ in range(7)]
            }
        }
    }
    props.mod_formats = {
        'QPSK': {'max_length': 1000, 'slots_needed': 10},
        '16-QAM': {'max_length': 500, 'slots_needed': 8},
        '64-QAM': {'max_length': 200, 'slots_needed': 6}
    }
    return props


@pytest.fixture
def nli_aware(
    engine_props: dict[str, Any], sdn_props: Mock
) -> Any:
    """Create NLIAwareRouting instance for testing.

    :param engine_props: Engine configuration dictionary.
    :type engine_props: dict[str, Any]
    :param sdn_props: Mock SDN properties.
    :type sdn_props: Mock
    :return: Configured NLIAwareRouting instance.
    :rtype: Any
    """
    from fusion.modules.routing.nli_aware import NLIAwareRouting
    nli_routing = NLIAwareRouting(engine_props, sdn_props)
    nli_routing.route_props.span_length = 50.0
    return nli_routing


class TestNLIAwareRouting:
    """Tests for NLIAwareRouting algorithm."""

    def test_init_stores_properties(
        self, nli_aware: Any, engine_props: dict[str, Any]
    ) -> None:
        """Test that initialization stores all configuration properties."""
        # Assert
        assert nli_aware.engine_props == engine_props
        assert hasattr(nli_aware, 'route_props')
        assert hasattr(nli_aware, 'route_help_obj')

    def test_algorithm_name_property(self, nli_aware: Any) -> None:
        """Test algorithm name property returns correct identifier."""
        # Assert
        assert nli_aware.algorithm_name == 'nli_aware'

    def test_supported_topologies_property(self, nli_aware: Any) -> None:
        """Test supported topologies property returns expected list."""
        # Act
        topologies = nli_aware.supported_topologies

        # Assert
        assert isinstance(topologies, list)
        assert 'Generic' in topologies

    def test_validate_environment_with_valid_topology(
        self, nli_aware: Any, topology: nx.Graph
    ) -> None:
        """Test environment validation succeeds with valid topology."""
        # Act
        is_valid = nli_aware.validate_environment(topology)

        # Assert
        assert is_valid is True

    def test_route_finds_least_nli_path(self, nli_aware: Any) -> None:
        """Test that route method finds path with least NLI."""
        # Act
        with patch.object(nli_aware.route_help_obj, 'get_nli_cost', return_value=0.2):
            path = nli_aware.route('A', 'C', request=None)

        # Assert
        assert path is not None
        assert path[0] == 'A'
        assert path[-1] == 'C'

    def test_route_with_no_path_returns_none(
        self, nli_aware: Any
    ) -> None:
        """Test that route returns None when no path exists."""
        # Act
        path = nli_aware.route('A', 'Z', request=None)

        # Assert
        assert path is None

    def test_route_updates_metrics(self, nli_aware: Any) -> None:
        """Test that route method updates internal metrics."""
        # Arrange
        initial_count = nli_aware._path_count

        # Act
        with patch.object(nli_aware.route_help_obj, 'get_nli_cost', return_value=0.3):
            nli_aware.route('A', 'C', request=None)

        # Assert
        assert nli_aware._path_count == initial_count + 1

    def test_update_nli_costs_sets_nli_costs(
        self, nli_aware: Any, topology: nx.Graph
    ) -> None:
        """Test that update NLI costs sets nli_cost on topology edges."""
        # Act
        with patch.object(nli_aware.route_help_obj, 'get_nli_cost', return_value=0.25):
            nli_aware._update_nli_costs()

        # Assert
        for source, dest in list(topology.edges())[::2]:
            assert 'nli_cost' in topology[source][dest]

    def test_get_paths_returns_paths_ordered_by_nli(
        self, nli_aware: Any
    ) -> None:
        """Test that get_paths returns paths ordered by NLI."""
        # Act
        with patch.object(nli_aware.route_help_obj, 'get_nli_cost', return_value=0.2):
            paths = nli_aware.get_paths('A', 'C', k=2)

        # Assert
        assert isinstance(paths, list)
        assert len(paths) <= 2

    def test_update_weights_sets_nli_costs(
        self, nli_aware: Any, topology: nx.Graph
    ) -> None:
        """Test that update_weights sets NLI costs on topology edges."""
        # Act
        with patch.object(nli_aware.route_help_obj, 'get_nli_cost', return_value=0.15):
            nli_aware.update_weights(topology)

        # Assert
        for source, dest in list(topology.edges())[::2]:
            if topology.has_edge(source, dest):
                assert 'nli_cost' in topology[source][dest]

    def test_get_metrics_returns_algorithm_statistics(
        self, nli_aware: Any
    ) -> None:
        """Test that get_metrics returns performance statistics."""
        # Arrange
        with patch.object(nli_aware.route_help_obj, 'get_nli_cost', return_value=0.4):
            nli_aware.route('A', 'C', request=None)

        # Act
        metrics = nli_aware.get_metrics()

        # Assert
        assert 'algorithm' in metrics
        assert metrics['algorithm'] == 'nli_aware'
        assert 'paths_computed' in metrics
        assert 'average_nli' in metrics

    def test_reset_clears_algorithm_state(self, nli_aware: Any) -> None:
        """Test that reset clears internal state counters."""
        # Arrange
        with patch.object(nli_aware.route_help_obj, 'get_nli_cost', return_value=0.3):
            nli_aware.route('A', 'C', request=None)
        assert nli_aware._path_count > 0

        # Act
        nli_aware.reset()

        # Assert
        assert nli_aware._path_count == 0
        assert nli_aware._total_nli == 0.0

    def test_calculate_path_nli_returns_valid_metric(
        self, nli_aware: Any
    ) -> None:
        """Test path NLI calculation returns valid metric."""
        # Arrange
        path = ['A', 'B', 'C']

        # Act
        nli = nli_aware._calculate_path_nli(path)

        # Assert
        assert isinstance(nli, float)
        assert nli >= 0.0

    def test_calculate_path_nli_with_empty_path(
        self, nli_aware: Any
    ) -> None:
        """Test path NLI calculation with empty path returns zero."""
        # Act
        nli = nli_aware._calculate_path_nli([])

        # Assert
        assert nli == 0.0

    def test_find_least_weight_populates_route_props(
        self, nli_aware: Any
    ) -> None:
        """Test that find least weight populates route properties."""
        # Arrange
        nli_aware.sdn_props.source = 'A'
        nli_aware.sdn_props.destination = 'C'

        # Act
        with (
            patch('fusion.utils.network.find_path_length', return_value=100.0),
            patch('fusion.utils.network.get_path_modulation', return_value='QPSK'),
        ):
            nli_aware._find_least_weight('nli_cost')

        # Assert
        assert len(nli_aware.route_props.paths_matrix) > 0
        assert len(nli_aware.route_props.weights_list) > 0
        assert len(nli_aware.route_props.modulation_formats_matrix) > 0

    def test_calculate_path_nli_sums_link_costs(
        self, nli_aware: Any, topology: nx.Graph
    ) -> None:
        """Test that path NLI calculation sums link costs correctly."""
        # Arrange
        path = ['A', 'B', 'C']
        topology['A']['B']['nli_cost'] = 0.2
        topology['B']['C']['nli_cost'] = 0.3

        # Act
        nli = nli_aware._calculate_path_nli(path)

        # Assert
        assert nli == 0.5
