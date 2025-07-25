# pylint: disable=protected-access
# pylint: disable=too-few-public-methods

import unittest
from unittest.mock import patch, call
import networkx as nx
import numpy as np
from fusion.helper_scripts.routing_helpers import RoutingHelpers


class RouteProps:
    """
    Mocks the routing props class.
    """
    def __init__(self, freq_spacing, input_power, mci_worst, span_len, max_link_length=None):
        self.freq_spacing = freq_spacing
        self.input_power = input_power
        self.mci_worst = mci_worst
        self.span_len = span_len
        self.max_link_length = max_link_length


class SDNProps:
    """
    Mocks the SDN props class.
    """
    def __init__(self, slots_needed, net_spec_dict):
        self.slots_needed = slots_needed
        self.net_spec_dict = net_spec_dict


class TestRoutingHelpers(unittest.TestCase):
    """
    Tests routing_helpers.py
    """

    def setUp(self):
        """Set up the environment for testing."""
        self.route_props = RouteProps(
            freq_spacing=50,
            input_power=0.001,
            mci_worst=1.0,
            span_len=50,
            max_link_length=None
        )
        self.engine_props = {
            'spectral_slots': 320,
            'guard_slots': 1,
            'topology': nx.DiGraph(),  # Topology is a DiGraph object
            'beta': 0.5
        }
        self.engine_props['topology'].add_edge('A', 'B', length=100)
        self.engine_props['topology'].add_edge('B', 'C', length=150)

        self.sdn_props = SDNProps(
            slots_needed=10,
            net_spec_dict={
                ('A', 'B'): {
                    'cores_matrix': {
                        'c': [  # 'c' band
                            np.array(np.zeros(320)),  # Core 0
                            np.array(np.zeros(320)),  # Core 1
                            np.array(np.zeros(320))  # Core 2
                        ],
                        'l': [  # 'l' band as an example
                            np.array(np.zeros(320)),  # Core 0
                            np.array(np.zeros(320)),  # Core 1
                            np.array(np.zeros(320))  # Core 2
                        ]
                    }
                }
            }
        )
        self.helpers = RoutingHelpers(self.route_props, self.engine_props, self.sdn_props)

    def test_get_indexes_even_slots_needed(self):
        """Test get indexes with an even number of slots needed."""
        self.sdn_props.slots_needed = 10
        start_index, end_index = self.helpers._get_indexes(center_index=160)
        self.assertEqual(start_index, 155)
        self.assertEqual(end_index, 165)

    def test_get_indexes_odd_slots_needed(self):
        """Test get indexes with an odd number of slots needed."""
        self.sdn_props.slots_needed = 11
        start_index, end_index = self.helpers._get_indexes(center_index=160)
        self.assertEqual(start_index, 155)
        self.assertEqual(end_index, 166)

    def test_get_simulated_link(self):
        """Test the simulated link generation."""
        simulated_link = self.helpers._get_simulated_link()
        center_start = 155
        center_end = 164

        self.assertTrue(np.all(simulated_link[center_start:center_end + 1] == 0), "Center channel is not free")
        if center_start > 0:
            self.assertTrue(np.any(simulated_link[:center_start] != 0),
                            "Indexes before the center channel are not all occupied")
        if center_end < len(simulated_link) - 1:
            self.assertTrue(np.any(simulated_link[center_end + 1:] != 0),
                            "Indexes after the center channel are not all occupied")

    def test_find_channel_mci(self):
        """Test find channel's multi-core interference."""
        channels_list = [(1, 10), (12, 8)]
        center_freq = 10.0
        num_span = 2.0

        total_mci = self.helpers._find_channel_mci(channels_list, center_freq, num_span)
        compare_mci = 2.8186640583738165e-10
        self.assertAlmostEqual(total_mci, compare_mci, places=15, msg="Total MCI was not calculated correctly.")

    def test_find_link_cost_valid(self):
        """Tests the find link cost method for valid input."""
        free_channels_dict = {'core1': {0: [(1, 10)]}}
        taken_channels_dict = {'core1': {0: [(11, 10)]}}
        num_span = 2
        link_cost = self.helpers._find_link_cost(free_channels_dict, taken_channels_dict, num_span)

        self.assertGreater(link_cost, 0, "Link cost should be positive for non-congested links")

    def test_find_link_cost_congested(self):
        """Tests the find link cost method for congested link."""
        free_channels_dict = {'core1': {0: []}}
        taken_channels_dict = {'core1': {0: [(1, 10), (11, 10)]}}
        num_span = 2
        link_cost = self.helpers._find_link_cost(free_channels_dict, taken_channels_dict, num_span)

        self.assertEqual(link_cost, 1000.0, "Link cost should be 1000.0 for fully congested links")

    def test_find_worst_nli(self):
        """Test the find worst NLI method."""
        # Adjust net_spec_dict to use the correct structure for cores_matrix with bands
        self.sdn_props.net_spec_dict = {('A', 'B'): {
            'cores_matrix': {
                'c': [  # 'c' band
                    np.array([1, 1, -1, 0, 1, 1, 0, 0, 1, 0]),  # Core 0
                    np.array([0, 0, 1, 1, 0, 0, -1, 1, 0, 1]),  # Core 1
                    np.array([1, 0, -1, 1, 1, 0, 1, -1, 1, 1])  # Core 2
                ],
                'l': [  # 'l' band as an example
                    np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]),  # Core 0
                    np.array([1, 1, 0, 0, 1, 1, 0, 0, 1, 1]),  # Core 1
                    np.array([1, 1, 1, 1, 0, 0, 1, 1, 1, 1])  # Core 2
                ]
            }
        }}

        self.sdn_props.slots_needed = 3
        num_span = 2

        # Mocking the functions to return appropriate test data
        with patch('fusion.helper_scripts.sim_helpers.find_free_channels', return_value={
            'c': {0: [1, 2, 3]},  # For the 'c' band, Core 0
            'l': {1: [4, 5, 6]}  # For the 'l' band, Core 1
        }), patch('fusion.helper_scripts.sim_helpers.find_taken_channels') as mock_find_taken_channels:
            # Mock find_taken_channels to return a structure that matches the expected output
            mock_find_taken_channels.return_value = {
                'c': {0: [[1, 1]], 1: [[1, 1]], 2: [[1]]},  # 'c' band
                'l': {0: [[1, 1]], 1: [[1, 1]], 2: [[1]]}  # 'l' band
            }

            # Call the function under test
            nli_worst = self.helpers.find_worst_nli(num_span=num_span)

            # Expected NLI value (mocked based on previous calculation)
            expected_nli_worst = 2.19e-10

            # Assertion to check if the calculation matches the expected value
            self.assertAlmostEqual(nli_worst, expected_nli_worst, places=2,
                                   msg="The worst NLI was not calculated properly.")

    def test_find_adjacent_cores(self):
        """Tests the find adjacent cores method."""
        for core_num in range(6):
            adj_cores = self.helpers._find_adjacent_cores(core_num)
            self.assertIn(6, adj_cores, f"Core 6 should be adjacent to core {core_num}")
            if core_num == 0:
                self.assertIn(5, adj_cores, "Core 5 should be adjacent to core 0")
            else:
                self.assertIn(core_num - 1, adj_cores, f"Core {core_num - 1} should be adjacent to core {core_num}")
            if core_num == 5:
                self.assertIn(0, adj_cores, "Core 0 should be adjacent to core 5")
            else:
                self.assertIn(core_num + 1, adj_cores, f"Core {core_num + 1} should be adjacent to core {core_num}")

    def test_find_num_overlapped(self):
        """Tests the find number of overlapped channels method."""
        core_info_dict = {band: {core: np.zeros(10) for core in range(7)} for band in ['core1']}
        core_info_dict['core1'][0][1] = 1
        core_info_dict['core1'][5][1] = 1
        core_info_dict['core1'][6][1] = 1

        num_overlapped = self.helpers._find_num_overlapped(channel=1, core_num=0, core_info_dict=core_info_dict,
                                                           band='core1')
        self.assertEqual(num_overlapped, 2 / 3, "Overlap calculation for non-central core is incorrect")

        num_overlapped = self.helpers._find_num_overlapped(channel=1, core_num=6, core_info_dict=core_info_dict,
                                                           band='core1')
        self.assertEqual(num_overlapped, 2 / 6, "Overlap calculation for central core is incorrect")

    def test_find_xt_link_cost(self):
        """Tests the find crosstalk link cost method."""
        # Adjust free_slots_dict to use string bands
        free_slots_dict = {
            'c': {0: [1, 2, 3]},  # Example for 'c' band, core 0
            'l': {1: [4, 5, 6]}  # Example for 'l' band, core 1
        }

        link_list = ('A', 'B')
        self.sdn_props.net_spec_dict = {
            link_list: {
                'cores_matrix': {
                    'c': np.ones((7, 10)),  # 'c' band with 7 cores, 10 channels
                    'l': np.ones((7, 10))  # 'l' band with 7 cores, 10 channels
                }
            }
        }

        with patch.object(self.helpers, '_find_num_overlapped', return_value=1.0):
            xt_cost = self.helpers.find_xt_link_cost(free_slots_dict, link_list)
            self.assertEqual(xt_cost, 1.0, msg="XT cost calculation is incorrect")

    def test_get_nli_path(self):
        """Tests the get NLI path method."""
        path_list = ['A', 'B', 'C']
        self.route_props.span_len = 50

        with patch.object(self.helpers, 'get_nli_cost', return_value=10.0) as mock_get_nli_cost:
            nli_cost = self.helpers.get_nli_path(path_list)
            expected_calls = [call(link_tuple=('A', 'B'), num_span=2), call(link_tuple=('B', 'C'), num_span=3)]
            mock_get_nli_cost.assert_has_calls(expected_calls, any_order=True)
            expected_nli_cost = 20.0
            self.assertEqual(nli_cost, expected_nli_cost, "NLI cost calculation for the path is incorrect")

    def test_get_max_link_length(self):
        """Tests the get max link length method."""
        self.engine_props['topology'] = nx.Graph()
        self.engine_props['topology'].add_edge('A', 'B', length=100)
        self.engine_props['topology'].add_edge('B', 'C', length=200)
        self.engine_props['topology'].add_edge('C', 'D', length=150)

        self.helpers.get_max_link_length()
        self.assertEqual(self.route_props.max_link_length, 200, "Incorrect maximum link length identified")

    def test_get_nli_cost(self):
        """Tests the get nli cost method."""
        self.engine_props['topology'] = nx.Graph()
        self.engine_props['topology'].add_edge('A', 'B', length=100)
        self.engine_props['beta'] = 0.5
        self.route_props.max_link_length = 100
        self.sdn_props.net_spec_dict = {('A', 'B'): {'cores_matrix': {'c': np.array(np.zeros((7, 10)))}}}
        self.sdn_props.slots_needed = 3

        with patch('fusion.helper_scripts.sim_helpers.find_free_channels', return_value={0: [1, 2, 3]}), \
                patch('fusion.helper_scripts.sim_helpers.find_taken_channels', return_value={0: [4, 5, 6]}), \
                patch.object(self.helpers, '_find_link_cost', return_value=10.0):
            nli_cost = self.helpers.get_nli_cost(link_tuple=('A', 'B'), num_span=2)

            expected_nli_cost = (100 / 100) * 0.5 + ((1 - 0.5) * 10.0)
            self.assertAlmostEqual(nli_cost, expected_nli_cost, msg="NLI cost calculation is incorrect")


if __name__ == '__main__':
    unittest.main()
