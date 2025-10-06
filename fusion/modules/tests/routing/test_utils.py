"""Unit tests for fusion.modules.routing.utils module."""

from typing import Any
from unittest.mock import Mock, patch

import networkx as nx
import numpy as np
import pytest


@pytest.fixture
def route_props() -> Mock:
    """Create mock routing properties for testing.

    :return: Mock RoutingProps object with test data.
    :rtype: Mock
    """
    props = Mock()
    props.frequency_spacing = 50e9
    props.input_power = 0.001
    props.mci_worst = 1.0
    props.span_length = 50.0
    props.max_link_length = None
    return props


@pytest.fixture
def engine_props() -> dict[str, Any]:
    """Create engine properties for testing.

    :return: Engine configuration dictionary.
    :rtype: dict[str, Any]
    """
    topology = nx.Graph()
    topology.add_edge('A', 'B', length=100.0)
    topology.add_edge('B', 'C', length=150.0)

    return {
        'spectral_slots': 320,
        'guard_slots': 1,
        'topology': topology,
        'beta': 0.5
    }


@pytest.fixture
def sdn_props() -> Mock:
    """Create mock SDN properties for testing.

    :return: Mock SDNProps object with test data.
    :rtype: Mock
    """
    props = Mock()
    props.slots_needed = 10
    props.network_spectrum_dict = {
        ('A', 'B'): {
            'cores_matrix': {
                'c': [
                    np.zeros(320),
                    np.zeros(320),
                    np.zeros(320)
                ],
                'l': [
                    np.zeros(320),
                    np.zeros(320),
                    np.zeros(320)
                ]
            }
        }
    }
    return props


@pytest.fixture
def routing_helpers(
    route_props: Mock, engine_props: dict[str, Any], sdn_props: Mock
) -> Any:
    """Create RoutingHelpers instance for testing.

    :param route_props: Mock routing properties.
    :type route_props: Mock
    :param engine_props: Engine configuration dictionary.
    :type engine_props: dict[str, Any]
    :param sdn_props: Mock SDN properties.
    :type sdn_props: Mock
    :return: Configured RoutingHelpers instance.
    :rtype: Any
    """
    from fusion.modules.routing.utils import RoutingHelpers
    return RoutingHelpers(route_props, engine_props, sdn_props)


class TestRoutingHelpers:
    """Tests for RoutingHelpers class."""

    def test_init_stores_properties(
        self, routing_helpers: Any, route_props: Mock,
        engine_props: dict[str, Any], sdn_props: Mock
    ) -> None:
        """Test that initialization stores all properties correctly."""
        # Assert
        assert routing_helpers.route_props == route_props
        assert routing_helpers.engine_props == engine_props
        assert routing_helpers.sdn_props == sdn_props

    @pytest.mark.parametrize("slots_needed,center_index,expected_start,expected_end", [
        (10, 160, 155, 165),
        (11, 160, 155, 166),
        (8, 100, 96, 104),
        (1, 50, 50, 51),
    ])
    def test_get_indexes_with_various_slots_needed(
        self, routing_helpers: Any, sdn_props: Mock,
        slots_needed: int, center_index: int, expected_start: int, expected_end: int
    ) -> None:
        """Test get indexes calculation with various slots_needed values.

        :param routing_helpers: RoutingHelpers fixture instance.
        :type routing_helpers: Any
        :param sdn_props: Mock SDN properties.
        :type sdn_props: Mock
        :param slots_needed: Number of slots needed for connection.
        :type slots_needed: int
        :param center_index: Center frequency index.
        :type center_index: int
        :param expected_start: Expected start index.
        :type expected_start: int
        :param expected_end: Expected end index.
        :type expected_end: int
        """
        # Arrange
        sdn_props.slots_needed = slots_needed

        # Act
        start_index, end_index = routing_helpers._get_indexes(center_index)

        # Assert
        assert start_index == expected_start
        assert end_index == expected_end

    def test_get_simulated_link_creates_valid_spectrum(
        self, routing_helpers: Any
    ) -> None:
        """Test that simulated link has correct structure with free center channel."""
        # Act
        simulated_link = routing_helpers._get_simulated_link()

        # Assert
        assert len(simulated_link) == 320
        center_index = len(simulated_link) // 2
        start_idx, end_idx = routing_helpers._get_indexes(center_index)

        # Center channel should be free
        assert np.all(simulated_link[start_idx:end_idx] == 0)

        # Should have occupied channels before and after
        if start_idx > 0:
            assert np.any(simulated_link[:start_idx] != 0)
        if end_idx < len(simulated_link):
            assert np.any(simulated_link[end_idx:] != 0)

    def test_find_channel_mci_calculates_correctly(
        self, routing_helpers: Any
    ) -> None:
        """Test multi-core interference calculation for a channel."""
        # Arrange
        channels_list = [(1, 10), (12, 8)]
        center_freq = 10.0 * routing_helpers.route_props.frequency_spacing
        num_span = 2.0

        # Act
        total_mci = routing_helpers._find_channel_mci(
            channels_list, center_freq, num_span
        )

        # Assert
        assert isinstance(total_mci, float)
        assert total_mci > 0

    def test_find_link_cost_with_free_channels(
        self, routing_helpers: Any
    ) -> None:
        """Test link cost calculation with available free channels."""
        from fusion.modules.routing.utils import FULLY_CONGESTED_LINK_COST

        # Arrange
        free_channels_dict = {'c': {0: [(1, 10)]}}
        taken_channels_dict = {'c': {0: [(11, 10)]}}
        num_span = 2.0

        # Act
        link_cost = routing_helpers._find_link_cost(
            free_channels_dict, taken_channels_dict, num_span
        )

        # Assert
        assert isinstance(link_cost, float)
        assert link_cost > 0
        assert link_cost < FULLY_CONGESTED_LINK_COST

    def test_find_link_cost_with_fully_congested_link(
        self, routing_helpers: Any
    ) -> None:
        """Test that fully congested link returns maximum cost."""
        from fusion.modules.routing.utils import FULLY_CONGESTED_LINK_COST

        # Arrange
        free_channels_dict: dict[str, dict[int, list[Any]]] = {'c': {0: []}}
        taken_channels_dict = {'c': {0: [(1, 10), (11, 10)]}}
        num_span = 2.0

        # Act
        link_cost = routing_helpers._find_link_cost(
            free_channels_dict, taken_channels_dict, num_span
        )

        # Assert
        assert link_cost == FULLY_CONGESTED_LINK_COST

    def test_find_worst_nli_with_valid_network(
        self, routing_helpers: Any, sdn_props: Mock
    ) -> None:
        """Test worst NLI calculation for network with spectrum data."""
        # Arrange
        sdn_props.network_spectrum_dict = {
            ('A', 'B'): {
                'cores_matrix': {
                    'c': [
                        np.array([1, 1, -1, 0, 1, 1, 0, 0, 1, 0]),
                        np.array([0, 0, 1, 1, 0, 0, -1, 1, 0, 1]),
                        np.array([1, 0, -1, 1, 1, 0, 1, -1, 1, 1])
                    ]
                }
            }
        }
        sdn_props.slots_needed = 3
        span_count = 2.0

        # Act
        with patch('fusion.utils.spectrum.find_free_channels') as mock_free, \
             patch('fusion.utils.spectrum.find_taken_channels') as mock_taken:
            mock_free.return_value = {'c': {0: [(1, 3)]}}
            mock_taken.return_value = {'c': {0: [(5, 2)]}}

            nli_worst = routing_helpers.find_worst_nli(span_count)

        # Assert
        assert isinstance(nli_worst, float)
        assert nli_worst >= 0

    def test_find_worst_nli_with_no_network_spectrum(
        self, routing_helpers: Any, sdn_props: Mock
    ) -> None:
        """Test worst NLI returns zero when network spectrum is None."""
        # Arrange
        sdn_props.network_spectrum_dict = None

        # Act
        nli_worst = routing_helpers.find_worst_nli(span_count=2.0)

        # Assert
        assert nli_worst == 0.0

    @pytest.mark.parametrize("core_num,expected_neighbors", [
        (0, {5, 6, 1}),
        (1, {0, 6, 2}),
        (2, {1, 6, 3}),
        (3, {2, 6, 4}),
        (4, {3, 6, 5}),
        (5, {4, 6, 0}),
    ])
    def test_find_adjacent_cores_for_outer_cores(
        self, core_num: int, expected_neighbors: set[int]
    ) -> None:
        """Test adjacent core identification for outer cores.

        :param core_num: Core number to test.
        :type core_num: int
        :param expected_neighbors: Expected set of adjacent core numbers.
        :type expected_neighbors: set[int]
        """
        from fusion.modules.routing.utils import RoutingHelpers

        # Act
        adj_cores = RoutingHelpers._find_adjacent_cores(core_num)

        # Assert
        assert set(adj_cores) == expected_neighbors

    def test_find_num_overlapped_for_non_center_core(
        self, routing_helpers: Any
    ) -> None:
        """Test overlapped channels calculation for non-center core."""
        # Arrange
        core_info_dict = {
            'c': {i: np.zeros(10) for i in range(7)}
        }
        core_info_dict['c'][0][1] = 1
        core_info_dict['c'][5][1] = 1
        core_info_dict['c'][6][1] = 1

        # Act
        num_overlapped = routing_helpers._find_num_overlapped(
            channel=1, core_num=0, core_info_dict=core_info_dict, band='c'
        )

        # Assert
        assert num_overlapped == 2 / 3

    def test_find_num_overlapped_for_center_core(
        self, routing_helpers: Any
    ) -> None:
        """Test overlapped channels calculation for center core."""
        from fusion.modules.routing.utils import CENTER_CORE_INDEX

        # Arrange
        core_info_dict = {
            'c': {i: np.zeros(10) for i in range(7)}
        }
        core_info_dict['c'][0][1] = 1
        core_info_dict['c'][5][1] = 1
        core_info_dict['c'][6][1] = 1

        # Act
        num_overlapped = routing_helpers._find_num_overlapped(
            channel=1,
            core_num=CENTER_CORE_INDEX,
            core_info_dict=core_info_dict,
            band='c',
        )

        # Assert
        assert num_overlapped == 2 / 6

    def test_find_xt_link_cost_calculates_correctly(
        self, routing_helpers: Any, sdn_props: Mock
    ) -> None:
        """Test crosstalk link cost calculation."""
        # Arrange
        free_slots_dict = {
            'c': {0: [1, 2, 3]},
            'l': {1: [4, 5, 6]}
        }
        link_tuple = ('A', 'B')
        sdn_props.network_spectrum_dict = {
            link_tuple: {
                'cores_matrix': {
                    'c': np.ones((7, 10)),
                    'l': np.ones((7, 10))
                }
            }
        }

        # Act
        with patch.object(routing_helpers, '_find_num_overlapped', return_value=0.5):
            xt_cost = routing_helpers.find_xt_link_cost(free_slots_dict, link_tuple)

        # Assert
        assert isinstance(xt_cost, float)
        assert xt_cost > 0

    def test_find_xt_link_cost_with_no_free_slots(
        self, routing_helpers: Any, sdn_props: Mock
    ) -> None:
        """Test that XT cost is maximum when no slots are free."""
        from fusion.modules.routing.utils import FULLY_CONGESTED_LINK_COST

        # Arrange
        free_slots_dict: dict[str, dict[int, list[int]]] = {'c': {0: []}}
        link_tuple = ('A', 'B')

        # Act
        xt_cost = routing_helpers.find_xt_link_cost(free_slots_dict, link_tuple)

        # Assert
        assert xt_cost == FULLY_CONGESTED_LINK_COST

    def test_find_xt_link_cost_with_no_network_spectrum(
        self, routing_helpers: Any, sdn_props: Mock
    ) -> None:
        """Test XT cost returns zero when network spectrum is None."""
        # Arrange
        sdn_props.network_spectrum_dict = None
        free_slots_dict = {'c': {0: [1, 2, 3]}}
        link_tuple = ('A', 'B')

        # Act
        xt_cost = routing_helpers.find_xt_link_cost(free_slots_dict, link_tuple)

        # Assert
        assert xt_cost == 0.0

    def test_get_nli_path_sums_link_costs(
        self, routing_helpers: Any, route_props: Mock
    ) -> None:
        """Test NLI path calculation sums costs across all links."""
        # Arrange
        path_list = ['A', 'B', 'C']
        route_props.span_length = 50.0

        # Act
        with patch.object(
            routing_helpers, 'get_nli_cost', return_value=10.0
        ) as mock_get_nli:
            nli_cost = routing_helpers.get_nli_path(path_list)

        # Assert
        assert nli_cost == 20.0
        assert mock_get_nli.call_count == 2

    def test_get_max_link_length_finds_maximum(
        self, routing_helpers: Any, route_props: Mock, engine_props: dict[str, Any]
    ) -> None:
        """Test that maximum link length is correctly identified."""
        # Arrange
        engine_props['topology'].add_edge('C', 'D', length=200.0)

        # Act
        routing_helpers.get_max_link_length()

        # Assert
        assert route_props.max_link_length == 200.0

    def test_get_nli_cost_calculates_with_beta_weighting(
        self,
        routing_helpers: Any,
        route_props: Mock,
        engine_props: dict[str, Any],
        sdn_props: Mock,
    ) -> None:
        """Test NLI cost calculation with beta weighting."""
        # Arrange
        link_tuple = ('A', 'B')
        num_span = 2.0
        route_props.max_link_length = 100.0
        engine_props['beta'] = 0.5

        sdn_props.network_spectrum_dict = {
            link_tuple: {'cores_matrix': {'c': np.zeros((7, 10))}}
        }
        sdn_props.slots_needed = 3

        # Act
        with patch('fusion.utils.spectrum.find_free_channels') as mock_free, \
             patch('fusion.utils.spectrum.find_taken_channels') as mock_taken, \
             patch.object(routing_helpers, '_find_link_cost', return_value=10.0):
            mock_free.return_value = {0: [1, 2, 3]}
            mock_taken.return_value = {0: [4, 5, 6]}

            nli_cost = routing_helpers.get_nli_cost(link_tuple, num_span)

        # Assert
        expected_cost = (100.0 / 100.0) * 0.5 + (1 - 0.5) * 10.0
        assert nli_cost == expected_cost

    def test_get_nli_cost_auto_calculates_max_link_length(
        self, routing_helpers: Any, route_props: Mock, sdn_props: Mock
    ) -> None:
        """Test get_nli_cost auto-calculates max link length if not set."""
        # Arrange
        link_tuple = ('A', 'B')
        num_span = 2.0
        route_props.max_link_length = None

        sdn_props.network_spectrum_dict = {
            link_tuple: {'cores_matrix': {'c': np.zeros((7, 10))}}
        }

        # Act
        with patch('fusion.utils.spectrum.find_free_channels'), \
             patch('fusion.utils.spectrum.find_taken_channels'), \
             patch.object(routing_helpers, '_find_link_cost', return_value=5.0):
            routing_helpers.get_nli_cost(link_tuple, num_span)

        # Assert
        assert route_props.max_link_length is not None


class TestRoutingHelpersConstants:
    """Tests for module-level constants."""

    def test_fully_congested_link_cost_is_high_value(self) -> None:
        """Test that fully congested link cost constant is appropriately high."""
        from fusion.modules.routing.utils import FULLY_CONGESTED_LINK_COST

        # Assert
        assert FULLY_CONGESTED_LINK_COST == 1000.0

    def test_default_band_is_c_band(self) -> None:
        """Test that default band is C-band."""
        from fusion.modules.routing.utils import DEFAULT_BAND

        # Assert
        assert DEFAULT_BAND == 'c'

    def test_center_core_index_is_six(self) -> None:
        """Test that center core index is 6 for seven-core fiber."""
        from fusion.modules.routing.utils import CENTER_CORE_INDEX

        # Assert
        assert CENTER_CORE_INDEX == 6

    def test_max_outer_cores_is_six(self) -> None:
        """Test that maximum outer cores is 6 for seven-core fiber."""
        from fusion.modules.routing.utils import MAX_OUTER_CORES

        # Assert
        assert MAX_OUTER_CORES == 6
