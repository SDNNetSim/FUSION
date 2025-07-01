# pylint: disable=protected-access

import unittest
import shutil
from unittest.mock import MagicMock, patch, mock_open

import numpy as np
import networkx as nx

from helper_scripts.stats_helpers import SimStats
from arg_scripts.stats_args import StatsProps
from arg_scripts.stats_args import SNAP_KEYS_LIST


class TestSimStats(unittest.TestCase):
    """
    Tests stats_helpers.py
    """

    def setUp(self):
        """Set up the test environment for SimStats."""
        self.engine_props = {
            'num_requests': 100,
            'snapshot_step': 20,
            'cores_per_link': 4,
            'save_snapshots': True,
            'mod_per_bw': {'50GHz': {'QPSK': {}, '16QAM': {}}},
            'output_train_data': True,
            'band_list': ['C'],
            'save_start_end_slots': False,
        }
        self.sim_info = "Simulation Info"

        # Properly initialize stats_props as an instance of StatsProps
        self.stats_props = StatsProps()

        # Ensure block_reasons_dict is properly initialized
        self.stats_props.block_reasons_dict = {'congestion': 0, 'distance': 0, 'xt_threshold': 0}

        # Initialize a simple topology graph
        self.topology = nx.Graph()
        self.topology.add_edge('A', 'B', length=10)
        self.topology.add_edge('B', 'C', length=15)
        self.topology.add_edge('A', 'C', length=20)

        # Initialize SimStats with the correct stats_props object and topology
        self.sim_stats = SimStats(engine_props=self.engine_props, sim_info=self.sim_info, stats_props=self.stats_props)
        self.sim_stats.topology = self.topology  # Set the topology in sim_stats

    def test_end_iter_update(self):
        """Test end iteration update."""
        self.sim_stats.blocked_reqs = 20
        self.sim_stats.total_trans = 300
        # Set up stats_props as expected for this test
        self.stats_props.block_reasons_dict = {'congestion': 15, 'distance': 5}
        self.stats_props.snapshots_dict = {}
        self.stats_props.trans_list = []

        # Mock the method that calculates iteration means
        with patch.object(self.sim_stats, '_get_iter_means'):
            self.sim_stats.end_iter_update()

        # Calculate the expected transmission mean
        expected_trans_mean = 300 / (self.engine_props['num_requests'] - self.sim_stats.blocked_reqs)

        # Check that the expected transmission mean is in the trans_list
        self.assertIn(expected_trans_mean, self.stats_props.trans_list)
        self.assertEqual(self.stats_props.block_reasons_dict['congestion'], 15 / 20)
        self.assertEqual(self.stats_props.block_reasons_dict['distance'], 5 / 20)

    @classmethod
    def tearDownClass(cls):
        """
        Deletes previously created directories for tests.
        """
        try:
            remove_dir = 'mocked'
            shutil.rmtree(remove_dir)
        except FileNotFoundError:
            pass

    def test_get_snapshot_info(self):
        """
        Test get snapshot info.
        """
        net_spec_dict = {
            (0, 1): {'cores_matrix': [np.array([0, 1, 0, -1]), np.array([1, 0, -1, 0])]},
            (1, 0): {'cores_matrix': [np.array([0, -1, 1, 0]), np.array([-1, 1, 0, 0])]}
        }
        path_list = [(0, 1)]
        occupied_slots, guard_slots, active_reqs = SimStats._get_snapshot_info(net_spec_dict=net_spec_dict,
                                                                               path_list=path_list)

        self.assertEqual(occupied_slots, 4)
        self.assertEqual(guard_slots, 2)
        self.assertEqual(active_reqs, 1)

    def test_update_snapshot(self):
        """Test update snapshot."""
        # Manually initialize snapshots_dict for the specific request number
        self.sim_stats.stats_props.snapshots_dict = {
            2: {
                'occupied_slots': [],
                'guard_slots': [],
                'active_requests': [],
                'blocking_prob': [],
                'num_segments': [],
                'bit_rate_blocking_prob': [],
            }
        }

        self.sim_stats._get_snapshot_info = MagicMock(return_value=(3, 3, 1))
        self.sim_stats.blocked_reqs = 1
        self.sim_stats.bit_rate_request = 1
        req_num = 2
        path_list = [(0, 1)]

        self.sim_stats.update_snapshot(net_spec_dict={}, req_num=req_num, path_list=path_list)

        # Assertions to verify correct update
        self.assertIn(req_num, self.sim_stats.stats_props.snapshots_dict)
        snapshot = self.sim_stats.stats_props.snapshots_dict[req_num]
        self.assertEqual(snapshot['occupied_slots'][0], 3)
        self.assertEqual(snapshot['guard_slots'][0], 3)
        self.assertEqual(snapshot['active_requests'][0], 1)
        self.assertAlmostEqual(snapshot['blocking_prob'][0], 0.5)
        self.assertEqual(snapshot['num_segments'][0], self.sim_stats.curr_trans)

    def test_init_snapshots(self):
        """
        Test init snapshots.
        """
        self.sim_stats._init_snapshots()

        expected_req_nums = list(range(0, 101, 20))
        for req_num in expected_req_nums:
            self.assertIn(req_num, self.sim_stats.stats_props.snapshots_dict)
            for key in SNAP_KEYS_LIST:
                self.assertIn(key, self.sim_stats.stats_props.snapshots_dict[req_num])
                self.assertEqual(self.sim_stats.stats_props.snapshots_dict[req_num][key], [])

    def test_init_mods_weights_bws(self):
        """
        Test initialize modulations, weights, and bandwidths.
        """
        self.sim_stats._init_mods_weights_bws()
        for bandwidth, mod_obj in self.sim_stats.engine_props['mod_per_bw'].items():
            self.assertIn(bandwidth, self.sim_stats.stats_props.mods_used_dict)
            self.assertIn(bandwidth, self.sim_stats.stats_props.weights_dict)
            self.assertIn(bandwidth, self.sim_stats.stats_props.block_bw_dict)

            for modulation in mod_obj.keys():
                self.assertIn(modulation, self.sim_stats.stats_props.mods_used_dict[bandwidth])
                self.assertIn(modulation, self.sim_stats.stats_props.weights_dict[bandwidth])
                self.assertEqual(self.sim_stats.stats_props.mods_used_dict[bandwidth][modulation], 0)
                self.assertEqual(self.sim_stats.stats_props.weights_dict[bandwidth][modulation], [])

    def test_init_stat_dicts(self):
        """
        Test init statistic dictionaries.
        """
        self.sim_stats._init_mods_weights_bws = MagicMock()
        self.sim_stats._init_snapshots = MagicMock()
        self.sim_stats._init_stat_dicts()

        expected_cores_dict = {0: 0, 1: 0, 2: 0, 3: 0}
        self.assertEqual(self.sim_stats.stats_props.cores_dict, expected_cores_dict)

        expected_block_reasons_dict = {'distance': 0, 'congestion': 0, 'xt_threshold': 0}
        self.assertEqual(self.sim_stats.stats_props.block_reasons_dict, expected_block_reasons_dict)

    def test_init_stat_lists(self):
        """
        Test init stat lists.
        """
        self.sim_stats.stats_props.sim_block_list = [1, 2, 3]
        self.sim_stats.iteration = 0
        self.sim_stats._init_stat_lists()

        self.assertEqual(self.sim_stats.stats_props.sim_block_list, [])
        self.sim_stats.iteration = 1
        self.sim_stats.stats_props.sim_block_list = [1, 2, 3]
        self.sim_stats._init_stat_lists()
        self.assertEqual(self.sim_stats.stats_props.sim_block_list, [1, 2, 3])

    def test_init_iter_stats(self):
        """
        Test init iter stats.
        """
        self.sim_stats._init_stat_dicts = MagicMock()
        self.sim_stats._init_stat_lists = MagicMock()

        self.sim_stats.init_iter_stats()
        self.sim_stats._init_stat_dicts.assert_called_once()
        self.sim_stats._init_stat_lists.assert_called_once()

        self.assertEqual(self.sim_stats.blocked_reqs, 0)
        self.assertEqual(self.sim_stats.total_trans, 0)

    def test_get_blocking(self):
        """
        Test get blocking.
        """
        self.sim_stats.blocked_reqs = 20
        self.sim_stats.bit_rate_blocked = 100
        self.sim_stats.bit_rate_request = 500

        self.sim_stats.get_blocking()
        expected_blocking_prob = 20 / 100
        expected_bit_rate_blocking_prob = 100 / 500
        self.assertIn(expected_blocking_prob, self.sim_stats.stats_props.sim_block_list)
        self.assertIn(expected_bit_rate_blocking_prob, self.sim_stats.stats_props.sim_br_block_list)

    # def test_handle_iter_lists(self):
    #     """
    #     Test handle iter lists.
    #     """
    #     sdn_data = MagicMock()
    #     sdn_data.stat_key_list = ['core_list', 'modulation_list', 'xt_list']
    #     sdn_data.core_list = [1, 2]
    #     sdn_data.modulation_list = ['QPSK', '16QAM']
    #     sdn_data.bandwidth_list = ['50GHz', '75GHz']
    #     sdn_data.xt_list = [0.1, 0.2]
    #
    #     self.sim_stats.stats_props.cores_dict = {1: 0, 2: 0}
    #     self.sim_stats.stats_props.mods_used_dict = {'50GHz': {'QPSK': 0}, '75GHz': {'16QAM': 0}}
    #     self.sim_stats.stats_props.xt_list = []
    #
    #     self.sim_stats._handle_iter_lists(sdn_data=sdn_data)
    #     self.assertEqual(self.sim_stats.stats_props.cores_dict[1], 0)
    #     self.assertEqual(self.sim_stats.stats_props.cores_dict[2], 0)
    #     self.assertEqual(self.sim_stats.stats_props.mods_used_dict['50GHz']['QPSK'], 0)
    #     self.assertEqual(self.sim_stats.stats_props.mods_used_dict['75GHz']['16QAM'], 0)
    #     self.assertEqual(self.sim_stats.stats_props.xt_list, [])

    def test_handle_iter_lists(self):
        """
        Test handle iter lists.
        """
        # Mock sdn_data
        sdn_data = MagicMock()
        sdn_data.stat_key_list = ['core_list', 'modulation_list', 'xt_list']
        sdn_data.bandwidth_list = ['50GHz', '75GHz']
        sdn_data.band_list = ['C', 'L']
        sdn_data.path_weight = 100.0

        # Mock get_data for specific keys
        sdn_data.get_data.side_effect = lambda key: {
            'core_list': [1, 2],
            'modulation_list': ['QPSK', '16QAM'],
            'xt_list': [0.1, 0.2],  # XT specific mock
        }.get(key, [])

        # Initialize stats properties
        self.sim_stats.stats_props.cores_dict = {1: 0, 2: 0}
        self.sim_stats.stats_props.mods_used_dict = {
            '50GHz': {'QPSK': 0, 'length': {'C': [], 'overall': []}},
            '75GHz': {'16QAM': 0, 'length': {'L': [], 'overall': []}},
            'QPSK': {'C': 1, 'L': 1, 'length': {'C': [1], 'L': [1], 'overall': [2]}},
            '16QAM': {'C': 1, 'L': 1, 'length': {'C': [1], 'L': [1], 'overall': [2]}}
        }
        self.sim_stats.stats_props.xt_list = []

        # Call the method under test
        self.sim_stats._handle_iter_lists(sdn_data=sdn_data)

        # Assertions
        self.assertEqual(self.sim_stats.stats_props.cores_dict[1], 1)
        self.assertEqual(self.sim_stats.stats_props.cores_dict[2], 1)
        self.assertEqual(self.sim_stats.stats_props.mods_used_dict['50GHz']['QPSK'], 1)
        self.assertEqual(self.sim_stats.stats_props.mods_used_dict['75GHz']['16QAM'], 1)
        self.assertEqual(self.sim_stats.stats_props.xt_list, [])

    def test_iter_update(self):
        """Test iter update."""
        req_data_blocked = {'bandwidth': '50'}
        sdn_data_blocked = MagicMock()
        sdn_data_blocked.was_routed = False
        sdn_data_blocked.block_reason = 'congestion'
        sdn_data_blocked.bandwidth = '50'
        self.sim_stats.stats_props.block_bw_dict = {'50': 0}
        self.sim_stats.bit_rate_request = 1
        self.sim_stats.stats_props.weights_dict = {'50': {'QPSK': list()}}
        self.sim_stats.iter_update(req_data=req_data_blocked, sdn_data=sdn_data_blocked)

        self.assertEqual(self.sim_stats.blocked_reqs, 1)
        self.assertEqual(self.sim_stats.stats_props.block_reasons_dict['congestion'], 1)
        self.assertEqual(self.sim_stats.stats_props.block_bw_dict['50'], 1)

        req_data_routed = {'bandwidth': '50'}
        sdn_data_routed = MagicMock()
        sdn_data_routed.was_routed = True
        sdn_data_routed.path_list = ['A', 'B', 'C']
        sdn_data_routed.route_time = 10
        sdn_data_routed.num_trans = 2
        sdn_data_routed.path_weight = 5
        sdn_data_routed.modulation_list = ['QPSK']
        sdn_data_routed.bandwidth = '50'
        sdn_data_routed.stat_key_list = []

        self.sim_stats.iter_update(req_data=req_data_routed, sdn_data=sdn_data_routed)

        self.assertEqual(self.sim_stats.stats_props.hops_list[0], 2)
        self.assertEqual(self.sim_stats.stats_props.lengths_list[0], 25)  # Length from A->B->C
        self.assertEqual(self.sim_stats.stats_props.route_times_list[0], 10)
        self.assertEqual(self.sim_stats.total_trans, 2)
        self.assertEqual(self.sim_stats.stats_props.weights_dict['50']['QPSK'][0], 5)

    def test_get_iter_means(self):
        """
        Test get iter means.
        """
        self.sim_stats.stats_props.snapshots_dict = {1: {'occupied_slots': [10, 20, 30]}}
        self.sim_stats.stats_props.weights_dict = {'50GHz': {'QPSK': [1, 2, 3, 4]}}
        self.sim_stats.stats_props.mods_used_dict = {'QPSK': {'length': {'50': 50}}}

        self.sim_stats._get_iter_means()
        expected_mod_obj = {'mean': 2.5, 'std': 1.2909944487358056, 'min': 1, 'max': 4}
        self.assertEqual(self.sim_stats.stats_props.weights_dict['50GHz']['QPSK'], expected_mod_obj)
        self.assertEqual(self.sim_stats.stats_props.snapshots_dict[1]['occupied_slots'], 20)

    def test_get_conf_inter(self):
        """
        Test get confidence interval.
        """
        self.sim_stats.stats_props.sim_block_list = [0.1, 0.2, 0.15, 0.25]
        self.sim_stats.stats_props.sim_br_block_list = [0.1, 0.2, 0.15, 0.25]

        should_end = self.sim_stats.get_conf_inter()
        self.assertIsNotNone(self.sim_stats.block_mean)
        self.assertIsNotNone(self.sim_stats.block_ci)
        self.assertIsNotNone(self.sim_stats.block_ci_percent)
        self.assertFalse(should_end)

    @patch("builtins.open", new_callable=mock_open)
    @patch("os.path.join", return_value='mocked/path/to/simulation_results')
    @patch("helper_scripts.os_helpers.create_dir")
    def test_save_stats_json(self, mock_create_dir, mock_path_join, mock_file):  # pylint: disable=unused-argument
        """Test save stats with a json file."""
        self.sim_stats.engine_props = {
            'file_type': 'json',
            'erlang': 10,
            'thread_num': 'thread_1',
            'output_train_data': True,
            'max_iters': 10,
            'save_start_end_slots': False,
        }
        self.sim_stats.sim_info = 'sim_test'
        self.sim_stats.block_mean = 0.2
        self.sim_stats.block_variance = 0.02
        self.sim_stats.block_ci = 0.05
        self.sim_stats.block_ci_percent = 5
        self.sim_stats.iteration = 1

        # Properly initialize stats_props as an instance of StatsProps
        self.sim_stats.stats_props = StatsProps()

        # Set an example value to ensure there's something to save
        self.sim_stats.stats_props.trans_list = [3.5]

        self.sim_stats.save_stats(base_fp=None)
        mock_file.assert_called_once_with('mocked/path/to/simulation_results/10_erlang.json', 'w', encoding='utf-8')


if __name__ == '__main__':
    unittest.main()
