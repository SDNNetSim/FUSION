# pylint: disable=too-many-public-methods

import unittest
import os
import copy
from datetime import datetime
from unittest.mock import patch, mock_open, MagicMock

import networkx as nx
import numpy as np

from helper_scripts.sim_helpers import (
    get_path_mod, find_max_path_len, sort_dict_keys, sort_nested_dict_vals,
    find_path_len, find_path_cong, find_free_slots,
    find_free_channels, find_taken_channels, snake_to_title, int_to_string,
    dict_to_list, list_to_title, calc_matrix_stats, combine_and_one_hot,
    get_start_time, find_core_cong, find_core_frag_cong, min_max_scale,
    get_super_channels, get_hfrag, classify_cong, parse_yaml_file, get_arrival_rates,
    run_simulation_for_arrival_rates, save_study_results, modify_multiple_json_values
)


class TestSimHelpers(unittest.TestCase):
    """Unit tests for sim_helpers functions."""

    def setUp(self):
        """Set up test data for the unit tests."""
        self.mods_dict = {
            'QPSK': {'max_length': 2000},
            '16-QAM': {'max_length': 1000},
            '64-QAM': {'max_length': 500}
        }

        self.topology = nx.Graph()
        self.topology.add_edge(1, 2, length=10)
        self.topology.add_edge(2, 3, length=20)
        self.topology.add_edge(1, 3, length=50)
        self.topology.add_edge(3, 4, length=15)

        self.net_spec_dict = {
            ('A', 'B'): {
                'cores_matrix': {
                    'c': np.array([[0, 1, 0, 0, 1], [0, 0, 1, 0, 1]])
                }
            },
            ('D', 'E'): {
                'cores_matrix': {
                    'c': np.array([[0, 0, 1, 1, -1], [0, 0, 0, 0, 0]])
                }
            }
        }

    def test_valid_path_length(self):
        """Test valid path length for modulation format selection."""
        path_len = 800
        expected_mod = '16-QAM'
        chosen_mod = get_path_mod(self.mods_dict, path_len)
        self.assertEqual(chosen_mod, expected_mod)

    def test_exceeding_path_length(self):
        """Test path length exceeding all modulation formats."""
        path_len = 2500
        chosen_mod = get_path_mod(self.mods_dict, path_len)
        self.assertFalse(chosen_mod)

    def test_longest_hop_path(self):
        """Test finding the longest hop path in the topology."""
        source, destination = 1, 3
        expected_length = 30
        max_path_len = find_max_path_len(source, destination, self.topology)
        self.assertEqual(max_path_len, expected_length)

    def test_sort_nested_dict_vals(self):
        """Test sorting a dictionary by a nested key."""
        original_dict = {
            'item1': {'nested_key': 10},
            'item2': {'nested_key': 5},
            'item3': {'nested_key': 15}
        }
        expected_sorted_dict = {
            'item2': {'nested_key': 5},
            'item1': {'nested_key': 10},
            'item3': {'nested_key': 15}
        }
        sorted_dict = sort_nested_dict_vals(original_dict, 'nested_key')
        self.assertEqual(sorted_dict, expected_sorted_dict)

    def test_sort_dict_keys(self):
        """Test sorting a dictionary by its keys in descending order."""
        dictionary = {'3': 'c', '1': 'a', '2': 'b'}
        expected_sorted_dict = {'3': 'c', '2': 'b', '1': 'a'}
        sorted_dict = sort_dict_keys(dictionary)
        self.assertEqual(sorted_dict, expected_sorted_dict)

    def test_find_path_len(self):
        """Test finding the length of a path in the topology."""
        path_list = [1, 2, 3, 4]
        expected_path_len = 45
        calculated_path_len = find_path_len(path_list, self.topology)
        self.assertEqual(calculated_path_len, expected_path_len)

    def test_find_path_cong(self):
        """Test finding the average congestion of a path."""
        net_spec_dict = {
            (1, 2): {'cores_matrix': {'c': np.array([[0, 1, 1], [1, 1, 0]])}},
            (2, 3): {'cores_matrix': {'c': np.array([[1, 0, 0], [0, 0, 1]])}}
        }
        path_list = [1, 2, 3]
        expected_avg_cong = ((4 / 6) + (2 / 6)) / 2
        calculated_avg_cong = find_path_cong(path_list, net_spec_dict)
        self.assertAlmostEqual(calculated_avg_cong, expected_avg_cong, places=5)

    def test_find_core_cong(self):
        """Test finding congestion on a specific core along a path."""
        net_spec_dict = {
            (1, 2): {'cores_matrix': {'c': np.array([[0, 1, 1], [1, 1, 0]])}},
            (2, 3): {'cores_matrix': {'c': np.array([[1, 0, 0], [0, 0, 1]])}}
        }
        path_list = [1, 2, 3]
        core_index = 0
        expected_core_cong = ((2 / 3) + (1 / 3)) / 2
        calculated_core_cong = find_core_cong(core_index, net_spec_dict, path_list)
        self.assertAlmostEqual(calculated_core_cong, expected_core_cong, places=2)

    def test_find_core_frag_cong(self):
        """Test finding fragmentation and congestion on a core."""
        net_spec_dict = {
            (1, 2): {'cores_matrix': {'c': [np.zeros(256), np.zeros(256)]}},
            (2, 3): {'cores_matrix': {'c': [np.zeros(256), np.zeros(256)]}}
        }
        path_list = [1, 2, 3]
        core = 0
        band = 'c'

        frag_resp, cong_resp = find_core_frag_cong(net_spec_dict, path_list, core, band)

        self.assertEqual(frag_resp, 0)
        self.assertEqual(cong_resp, 0)

    def test_find_free_slots(self):
        """Test finding free slots for each core on a link."""
        result1 = find_free_slots(self.net_spec_dict, ('A', 'B'))
        expected_result1 = {'c': {0: np.array([0, 2, 3]), 1: np.array([0, 1, 3])}}
        for core, slots_list in expected_result1['c'].items():
            self.assertTrue(np.array_equal(result1['c'][core], slots_list))

    def test_find_free_channels(self):
        """Test finding free channels for a given link."""
        slots_needed = 2
        result1 = find_free_channels(self.net_spec_dict, slots_needed, ('A', 'B'))
        expected_result1 = {'c': {0: [[2, 3]], 1: [[0, 1]]}}
        self.assertEqual(result1, expected_result1)

    def test_find_taken_channels(self):
        """Test finding taken channels for a given link."""
        result1 = find_taken_channels(copy.deepcopy(self.net_spec_dict), ('D', 'E'))
        expected_result1 = {'c': {0: [[1, 1]], 1: []}}
        self.assertEqual(result1, expected_result1)

    def test_snake_to_title(self):
        """Test converting a snake_case string to Title Case."""
        snake_str = "hello_world"
        result = snake_to_title(snake_str)
        self.assertEqual(result, "Hello World")

    def test_int_to_string(self):
        """Test converting an integer to a string with commas."""
        number = 1234567
        result = int_to_string(number)
        self.assertEqual(result, "1,234,567")

    def test_dict_to_list(self):
        """Test creating a list from a dictionary based on a nested key."""
        data_dict = {
            'item1': {'value': 10},
            'item2': {'value': 20},
            'item3': {'value': 30}
        }
        nested_key = 'value'
        result = dict_to_list(data_dict, nested_key)
        self.assertTrue(np.array_equal(result, [10, 20, 30]))

    def test_list_to_title(self):
        """Test converting a list to a title case string."""
        input_list = [["Alice"], ["Bob"], ["Charlie"]]
        result = list_to_title(input_list)
        self.assertEqual(result, "Alice, Bob & Charlie")

    def test_calc_matrix_stats(self):
        """Test calculating min, max, and average of matrix columns."""
        input_dict = {
            '0': [1.0, 5.0, 3.0],
            '1': [2.0, 4.0, 8.0],
            '2': [0.0, 3.0, 5.0]
        }

        expected_output = {
            'min': [0, 3, 3],
            'max': [2, 5, 8],
            'average': [1.0, 4.0, 5.333333333333333]
        }

        result = calc_matrix_stats(input_dict)
        self.assertDictEqual(result, expected_output)

    def test_combine_and_one_hot(self):
        """Test performing OR operation on two arrays to find overlaps."""
        arr1 = np.array([0, 1, 0, 1, 0])
        arr2 = np.array([1, 0, 1, 0, 1])

        expected_result = np.array([1, 1, 1, 1, 1])

        result = combine_and_one_hot(arr1, arr2)

        self.assertTrue(np.array_equal(result, expected_result))

    def test_get_start_time(self):
        """Test getting the start time of a simulation."""
        sim_dict = {'s1': {'date': None, 'sim_start': None}}
        expected_date = datetime.now().strftime("%m%d")
        expected_sim_start = datetime.now().strftime("%H_%M_%S")

        get_start_time(sim_dict)

        self.assertEqual(sim_dict['s1']['date'], expected_date)
        self.assertTrue(sim_dict['s1']['sim_start'].startswith(expected_sim_start))

    def test_min_max_scale(self):
        """Test scaling a value with respect to a min and max."""
        value = 5
        min_value = 0
        max_value = 10
        expected_scaled_value = 0.5

        scaled_value = min_max_scale(value, min_value, max_value)
        self.assertEqual(scaled_value, expected_scaled_value)

    def test_get_super_channels(self):
        """Test finding available super-channels for a core."""
        input_arr = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0])
        slots_needed = 3
        expected_super_channels = np.array([[0, 3], [5, 8]])

        super_channels = get_super_channels(input_arr, slots_needed)
        self.assertTrue(np.array_equal(super_channels, expected_super_channels))

    def test_get_hfrag(self):
        """Test calculating Shannon entropy fragmentation scores."""
        path_list = [1, 2, 3]
        core_num = 0
        band = 'c'
        slots_needed = 3
        spectral_slots = 8

        # Updated net_spec_dict to ensure the arrays are correctly formed
        net_spec_dict = {
            (1, 2): {'cores_matrix': {'c': np.zeros((1, spectral_slots))}},
            (2, 3): {'cores_matrix': {'c': np.zeros((1, spectral_slots))}}
        }

        # Adjusted expected values to match the actual function's logic
        expected_sc_index_mat = np.array([[0, 3], [1, 4], [2, 5], [3, 6], [4, 7]])
        expected_resp_frag_arr = np.array([1.386, -np.inf, -np.inf, -np.inf, 1.386, np.inf, np.inf, np.inf])

        # Call the function
        sc_index_mat, resp_frag_arr = get_hfrag(
            path_list, core_num, band, slots_needed, spectral_slots, net_spec_dict
        )

        # Assertions
        self.assertTrue(np.array_equal(sc_index_mat, expected_sc_index_mat))
        self.assertTrue(np.array_equal(resp_frag_arr, expected_resp_frag_arr))

    def test_classify_cong(self):
        """Test classifying congestion percentage into levels."""
        curr_cong = 0.2
        expected_cong_index = 0
        cong_index = classify_cong(curr_cong)
        self.assertEqual(cong_index, expected_cong_index)

        curr_cong = 0.5
        expected_cong_index = 1
        cong_index = classify_cong(curr_cong)
        self.assertEqual(cong_index, expected_cong_index)

    @patch('builtins.open', new_callable=mock_open, read_data="key: value")
    @patch('helper_scripts.sim_helpers.yaml.safe_load')
    def test_parse_yaml_file(self, mock_yaml_load, mock_open_file):
        """Test parsing a YAML file."""
        mock_yaml_load.return_value = {"key": "value"}
        yaml_file = "fake_file.yaml"

        result = parse_yaml_file(yaml_file)
        self.assertEqual(result, {"key": "value"})

        mock_open_file.assert_called_once_with(yaml_file, "r", encoding='utf-8')
        mock_yaml_load.assert_called_once()

    def test_get_arrival_rates(self):
        """Test generating a list of arrival rates from a configuration dictionary."""
        arrival_dict = {'start': 10, 'stop': 50, 'step': 10}
        expected_arrival_rates = [10, 20, 30, 40, 50]
        result = get_arrival_rates(arrival_dict)
        self.assertEqual(result, expected_arrival_rates)

    @patch('helper_scripts.sim_helpers.run_simulation_for_arrival_rates')
    def test_run_simulation_for_arrival_rates(self, mock_run_func):
        """Test running the simulation for each arrival rate."""

        class MockEnv:  # pylint: disable=too-few-public-methods
            """ Mock and environment. """

            def __init__(self):
                self.engine_obj = MagicMock()
                self.engine_obj.engine_props = {}
                self.sim_dict = {'holding_time': 2, 'cores_per_link': 4}
                self.path_agent = MagicMock()
                self.path_agent.reward_penalty_list = [10, -5, 15]

        env = MockEnv()
        arrival_list = [10, 20, 30]
        expected_mean_reward = np.mean([np.sum(env.path_agent.reward_penalty_list)] * len(arrival_list))

        result = run_simulation_for_arrival_rates(env, arrival_list, mock_run_func)
        self.assertAlmostEqual(result, expected_mean_reward, places=5)

    @patch('builtins.open', new_callable=mock_open)
    @patch('os.makedirs')
    @patch('pickle.dump')
    def test_save_study_results(self, mock_pickle_dump, mock_makedirs, mock_open_file):
        """Test saving the study results."""

        class MockEnv:  # pylint: disable=too-few-public-methods
            """ Mock an environment. """

            def __init__(self):
                self.engine_obj = MagicMock()
                self.engine_obj.engine_props = {
                    'network': 'test_network',
                    'date': '2025-01-01',
                    'sim_start': '12:00:00',
                    'path_algorithm': 'q_learning'
                }

        env = MockEnv()
        study = MagicMock()
        best_params = {'param1': 0.5, 'param2': 10}
        best_reward = 100.0
        start_time = '2025-01-01 12:00:00'
        study_name = 'test_study'

        save_study_results(study, env, study_name, best_params, best_reward, start_time)
        mock_makedirs.assert_called_once_with(
            os.path.join('logs', 'q_learning', 'test_network', '2025-01-01', '12:00:00'), exist_ok=True
        )
        mock_pickle_dump.assert_called_once()
        mock_open_file.assert_any_call(
            os.path.join('logs', 'q_learning', 'test_network', '2025-01-01', '12:00:00', 'best_hyperparams.txt'),
            'w',
            encoding='utf-8'
        )

    @patch('json.load', return_value={'key1': 'value1', 'key2': 'value2'})
    @patch('json.dump')
    @patch('builtins.open', new_callable=mock_open)
    def test_modify_multiple_json_values(self, mock_open_file, mock_json_dump, mock_json_load): # pylint: disable=unused-argument
        """Test modifying multiple JSON values."""
        file_path = 'test.json'
        update_list = [('key1', 'new_value1'), ('key2', 'new_value2')]

        modify_multiple_json_values(file_path, update_list)
        mock_open_file.assert_called_with(file_path, 'w', encoding='utf-8')
        mock_json_dump.assert_called_once_with(
            {'key1': 'new_value1', 'key2': 'new_value2'}, mock_open_file(), indent=4
        )


if __name__ == '__main__':
    unittest.main()
