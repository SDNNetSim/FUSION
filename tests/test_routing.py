import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import networkx as nx

from src.routing import Routing
from arg_scripts.routing_args import RoutingProps


class TestRouting(unittest.TestCase):
    """
    Test methods in routing.py
    """

    def setUp(self):
        self.engine_props = {
            'topology': nx.Graph(),
            'mod_per_bw': {
                '50GHz': {'QPSK': {'max_length': 10}},
                '100GHz': {'QPSK': {'max_length': 20}}
            },
            'k_paths': 2,
            'xt_type': 'with_length',
            'beta': 0.5,
            'route_method': 'k_shortest_path',
            'pre_calc_mod_selection': False,
            'network': 'USbackbone60',
        }
        self.engine_props['topology'].add_edge('A', 'B', weight=1, xt_cost=10, length=1)
        self.engine_props['topology'].add_edge('B', 'C', weight=1, xt_cost=5, length=1)
        self.engine_props['topology'].add_edge('A', 'C', weight=3, xt_cost=100, length=2)

        self.sdn_props = MagicMock()
        self.sdn_props.net_spec_dict = {
            ('A', 'B'): {'cores_matrix': {'c': np.zeros((1, 10))}},
            ('B', 'C'): {'cores_matrix': {'c': np.ones((1, 10))}},
            ('A', 'C'): {'cores_matrix': {'c': np.zeros((1, 10))}}
        }
        self.sdn_props.source = 'A'
        self.sdn_props.destination = 'C'
        self.sdn_props.topology = self.engine_props['topology']
        self.sdn_props.mod_formats = {
            'QPSK': {'max_length': 10},
            '16-QAM': {'max_length': 20},
            '64-QAM': {'max_length': 30}
        }
        self.sdn_props.bandwidth = '50GHz'

        self.route_props = RoutingProps()
        self.route_props.loaded_data_dict_mock = [
            [
                None,  # Placeholder for irrelevant indices
                None,
                None,
                [[np.float64(10)]],  # Path lengths for index 3
                [None],  # Placeholder for irrelevant indices (index 4)
                [
                    [
                        [  # src_dest_index = 5
                            [1, 2, 3, 4],  # First valid path
                        ]
                    ]
                ],
            ]
        ]

        self.instance = Routing(engine_props=self.engine_props, sdn_props=self.sdn_props)
        self.instance.route_props = self.route_props

    def test_find_most_cong_link(self):
        """
        Test the find most congested link method.
        """
        path_list = ['A', 'B', 'C']
        self.instance._find_most_cong_link(path_list)  # pylint: disable=protected-access
        self.assertEqual(len(self.instance.route_props.paths_matrix), 1)

        cores_arr = self.instance.route_props.paths_matrix[0]['link_dict']['link']['cores_matrix']['c'][0]
        condition = np.all(cores_arr == 1)
        self.assertTrue(condition)

    def test_find_least_cong_path(self):
        """
        Test find the least congested path.
        """
        self.instance.find_least_cong()
        self.assertEqual(len(self.instance.route_props.paths_matrix), 2)
        self.assertEqual(self.instance.route_props.paths_matrix[0]['path_list'], ['A', 'C'])

    def test_least_xt_cost_path_selection(self):
        """
        Test find least cross-talk method.
        """
        self.instance.find_least_weight('xt_cost')

        expected_path = ['A', 'B', 'C']
        selected_path = self.instance.route_props.paths_matrix[0]
        self.assertEqual(selected_path, expected_path, f"Expected path {expected_path} but got {selected_path}")

    def test_least_weight_path_selection(self):
        """
        Test find the least weight path method.
        """
        self.instance.find_least_weight('weight')

        expected_path = ['A', 'B', 'C']
        selected_path = self.instance.route_props.paths_matrix[0]
        self.assertEqual(selected_path, expected_path, f"Expected path {expected_path} but got {selected_path}")

    def test_find_k_shortest_paths(self):
        """
        Test find the k-shortest paths method.
        """
        # Update mod_per_bw to include '16-QAM'
        self.engine_props['mod_per_bw'] = {
            '50GHz': {'QPSK': {'max_length': 10}, '16-QAM': {'max_length': 20}, '64-QAM': {'max_length': 40}},
            '100GHz': {'QPSK': {'max_length': 20}}
        }

        self.instance.find_k_shortest()
        self.assertEqual(len(self.instance.route_props.paths_matrix), self.engine_props['k_paths'],
                         "Did not find the expected number of shortest paths")
        for path in self.instance.route_props.paths_matrix:
            self.assertIsInstance(path, list, "Each path should be a list")
        for mod_format_list in self.instance.route_props.mod_formats_matrix:
            self.assertEqual(len(mod_format_list), 1, "Each path should have exactly one modulation format")
        for weight in self.instance.route_props.weights_list:
            self.assertIsInstance(weight, (int, float), "Each weight should be a number")

    def test_find_least_nli(self):
        """
        Test find the least non-linear impairment cost method.
        """
        self.sdn_props.bandwidth = '50GHz'
        self.engine_props['mod_per_bw'] = {
            '50GHz': {'QPSK': {'slots_needed': 10}}
        }
        with patch.object(self.instance.route_help_obj, 'get_nli_cost', return_value=1.0):
            self.instance.find_least_nli()

            for link_tuple in list(self.sdn_props.net_spec_dict.keys())[::2]:
                source, destination = link_tuple
                self.assertIn('nli_cost', self.sdn_props.topology[source][destination], "NLI cost not set for link")

    def test_find_least_xt(self):
        """
        Test find the least XT method.
        """
        self.route_props.max_link_length = 100  # Ensure max_link_length is set

        with patch('helper_scripts.sim_helpers.find_free_slots', return_value={'free_slots': []}), \
                patch.object(self.instance.route_help_obj, 'find_xt_link_cost', return_value=0.1), \
                patch.object(self.instance.route_help_obj, 'get_max_link_length', return_value=100):
            self.instance.find_least_xt()

            for link_list in list(self.sdn_props.net_spec_dict.keys())[::2]:
                source, destination = link_list
                self.assertIn('xt_cost', self.sdn_props.topology[source][destination], "XT cost not set for link")

    def test_load_k_shortest_success(self):
        """
        Test the successful loading of k-shortest paths from a file.
        """
        self.engine_props['k_paths'] = 2  # Set k-paths to test with
        self.sdn_props.source = '1'
        self.sdn_props.destination = '4'

        # Mock mod_formats_dict
        self.sdn_props.mod_formats_dict = {
            'QPSK': {'max_length': 10},
            '16-QAM': {'max_length': 20},
            '64-QAM': {'max_length': 30}
        }

        # Mock sort_nested_dict_vals
        with patch('helper_scripts.sim_helpers.sort_nested_dict_vals', return_value={
            '64-QAM': {'max_length': 30},
            '16-QAM': {'max_length': 20},
            'QPSK': {'max_length': 10}
        }):
            # Mock np.load to return the mock loaded_data_dict
            with patch('numpy.load', return_value=self.route_props.loaded_data_dict_mock):
                self.instance.load_k_shortest()

                # Verify that paths_matrix has been populated correctly
                expected_paths = [['1', '2', '3', '4']]
                expected_weights = [10]

                self.assertEqual(len(self.instance.route_props.paths_matrix), 1)
                self.assertEqual(self.instance.route_props.paths_matrix, expected_paths)
                self.assertEqual(self.instance.route_props.weights_list[:1], expected_weights)

                # Verify modulation formats were added in reverse order
                expected_mod_formats = ['64-QAM', '16-QAM', 'QPSK']
                self.assertGreater(len(self.instance.route_props.mod_formats_matrix[0]), 0)
                self.assertEqual(
                    self.instance.route_props.mod_formats_matrix[0],
                    expected_mod_formats
                )

    def test_load_k_shortest_no_matching_paths(self):
        """
        Test load_k_shortest when there are no matching paths for the source and destination.
        """
        self.engine_props['k_paths'] = 2  # Set k-paths to test with
        self.sdn_props.source = '5'
        self.sdn_props.destination = '6'

        # Reuse mock loaded_data_dict from context
        with patch('numpy.load', return_value=self.route_props.loaded_data_dict_mock):
            self.instance.load_k_shortest()

            # Verify that no paths were added
            self.assertEqual(len(self.instance.route_props.paths_matrix), 0)
            self.assertEqual(len(self.instance.route_props.weights_list), 0)

    def test_load_k_shortest_partial_match(self):
        """
        Test load_k_shortest when only partial matching paths exist.
        """
        self.engine_props['k_paths'] = 2  # Set k-paths to test with
        self.sdn_props.source = '1'
        self.sdn_props.destination = '4'

        # Reuse mock loaded_data_dict from context
        with patch('numpy.load', return_value=self.route_props.loaded_data_dict_mock):
            self.instance.load_k_shortest()

            # Verify that only the matching path was added
            expected_paths = [['1', '2', '3', '4']]
            expected_weights = [10]

            self.assertEqual(len(self.instance.route_props.paths_matrix), 1)
            self.assertEqual(self.instance.route_props.paths_matrix, expected_paths)
            self.assertEqual(self.instance.route_props.weights_list[:1], expected_weights)


if __name__ == '__main__':
    unittest.main()
