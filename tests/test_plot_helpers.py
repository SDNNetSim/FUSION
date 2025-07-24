# pylint: disable=protected-access

import unittest
from unittest.mock import patch, mock_open
import numpy as np
from fusion.helper_scripts.plot_helpers import PlotHelpers, find_times
from fusion.cli.args.plot_args import PlotArgs, PlotProps


class TestPlotHelpers(unittest.TestCase):
    """Unit tests for the PlotHelpers class and related functions."""

    def setUp(self):
        """Set up the environment for testing."""
        self.net_names_list = ['Network1', 'Network2']
        self.plot_props = PlotProps()
        self.plot_helpers = PlotHelpers(plot_props=self.plot_props, net_names_list=self.net_names_list)

        # Initialize the plot_dict with a valid structure
        self.plot_helpers.time = "2023-08-12"
        self.plot_helpers.sim_num = "1"
        self.plot_helpers._update_plot_dict()  # Ensure that the plot_dict is set up

        # Set up mock data
        self.plot_helpers.erlang = "10"
        self.plot_helpers.data_dict = {
            'network': 'TestNetwork',
            'date': '2023-08-12',
            'sim_dict': {
                '1': ['10', '20']
            }
        }
        self.plot_helpers.file_info = {
            "2023-08-12": {
                'network': 'TestNetwork',
                'date': '2023-08-12',
                'sim_dict': {
                    '1': ['10', '20']
                }
            }
        }

        # Example simulation dictionary with correct structure
        self.example_sim_dict = {
            "iter_stats": {
                "0": {
                    "mods_used_dict": {
                        "100": {"QPSK": 2, "16QAM": 3}
                    },
                    "snapshots_dict": {
                        "1": {"active_requests": 10, "blocking_prob": 0.05, "occ_slots": 50}
                    },
                    "lengths_mean": [100],
                    "hops_mean": [3],
                    "route_times_mean": [0.005],
                    "block_reasons_dict": {
                        "congestion": [0.1],
                        "distance": [1000]
                    }
                }
            },
            "blocking_mean": 0.02,
            "path_algorithm": "q_learning"
        }
        # Assign the initialized dict to plot_helpers object
        self.plot_helpers.erlang_dict = self.example_sim_dict

    @patch('fusion.helper_scripts.plot_helpers.PlotHelpers._read_json_file')
    def test_read_input_output(self, mock_read_json_file):
        """Test the _read_input_output method."""
        mock_read_json_file.side_effect = [self.example_sim_dict, self.example_sim_dict]

        input_dict, erlang_dict = self.plot_helpers._read_input_output()

        # Verify the method reads the correct input and output files
        self.assertEqual(input_dict, self.example_sim_dict)
        self.assertEqual(erlang_dict, self.example_sim_dict)

    def test_update_plot_dict(self):
        """Test the _update_plot_dict method."""
        self.plot_helpers._update_plot_dict()

        # Verify that the plot_dict is updated correctly
        self.assertIsInstance(self.plot_helpers.plot_props.plot_dict[self.plot_helpers.time][self.plot_helpers.sim_num],
                              PlotArgs)

    @patch('fusion.helper_scripts.plot_helpers.PlotHelpers._dict_to_np_array')
    def test_find_snapshot_usage(self, mock_dict_to_np_array):
        """Test the _find_snapshot_usage method."""
        mock_dict_to_np_array.side_effect = [
            np.array([10]), np.array([0.05]), np.array([50])
        ]

        self.plot_helpers.erlang_dict = self.example_sim_dict
        self.plot_helpers._find_snapshot_usage()

        # Verify the matrices are updated correctly
        self.assertTrue(np.array_equal(
            self.plot_helpers.plot_props.plot_dict[self.plot_helpers.time][self.plot_helpers.sim_num].active_req_matrix,
            np.array([10])))
        self.assertTrue(np.array_equal(
            self.plot_helpers.plot_props.plot_dict[self.plot_helpers.time][self.plot_helpers.sim_num].block_req_matrix,
            np.array([0.05])))
        self.assertTrue(np.array_equal(
            self.plot_helpers.plot_props.plot_dict[self.plot_helpers.time][self.plot_helpers.sim_num].occ_slot_matrix,
            np.array([50])))

    def test_find_mod_info(self):
        """Test the _find_mod_info method."""
        self.plot_helpers._update_plot_dict()

        # Run the method to find modulation info
        self.plot_helpers._find_mod_info()

        modulations_dict = self.plot_helpers.plot_props.plot_dict[self.plot_helpers.time][
            self.plot_helpers.sim_num].modulations_dict
        self.assertEqual(modulations_dict['100']['QPSK'], [2])
        self.assertEqual(modulations_dict['100']['16QAM'], [3])

    def test_find_misc_stats(self):
        """Test the _find_misc_stats method."""
        self.plot_helpers._update_plot_dict()

        # Test the _find_misc_stats method
        self.plot_helpers.erlang_dict = self.example_sim_dict
        self.plot_helpers._find_misc_stats()

        plot_args = self.plot_helpers.plot_props.plot_dict[self.plot_helpers.time][self.plot_helpers.sim_num]
        self.assertEqual(plot_args.lengths_list, [100])
        self.assertEqual(plot_args.hops_list, [3])
        self.assertEqual(plot_args.times_list, [5.0])
        self.assertEqual(plot_args.cong_block_list, [0.1])
        self.assertEqual(plot_args.dist_block_list, [1000])

    @patch('os.listdir')
    @patch('os.path.isdir')
    @patch('fusion.helper_scripts.plot_helpers._check_filters')
    @patch('builtins.open', new_callable=mock_open, read_data='{}')
    def test_find_times(self, mock_open_func, mock_check_filters, mock_isdir, mock_listdir):  # pylint: disable=unused-argument
        """Test the find_times function."""
        mock_listdir.side_effect = [
            ['time1', 'time2'],  # First call for times list
            ['network_sim_1.json', 'network_sim_2.json'],  # Second call for simulation files in time1
            ['network_sim_1.json', 'network_sim_2.json']  # Third call for simulation files in time2
        ]
        mock_isdir.return_value = True
        mock_check_filters.return_value = True

        dates_dict = {'2023-08-12': 'TestNetwork'}
        filter_dict = {'not_filter_list': [], 'or_filter_list': [], 'and_filter_list': []}

        resp = find_times(dates_dict, filter_dict)

        # Verify the response structure
        self.assertIn('times_matrix', resp)
        self.assertIn('sims_matrix', resp)
        self.assertIn('networks_matrix', resp)
        self.assertIn('dates_matrix', resp)

        # Assert that the times_matrix contains two entries, reflecting the mock data
        self.assertEqual(len(resp['times_matrix']), 2)
        self.assertEqual(len(resp['sims_matrix']), 2)
        self.assertEqual(len(resp['networks_matrix']), 2)
        self.assertEqual(len(resp['dates_matrix']), 2)


if __name__ == '__main__':
    unittest.main()
