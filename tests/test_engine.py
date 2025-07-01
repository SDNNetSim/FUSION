import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import networkx as nx

from src.engine import Engine


class TestEngine(unittest.TestCase):
    """
    Methods related to testing engine.py
    """

    def setUp(self):
        engine_props = {
            'ai_algorithm': None,
            'network': 'USNet',
            'date': '2024-02-02',
            'sim_start': '00:00',
            'save_snapshots': True,
            'snapshot_step': 1,
            'output_train_data': False,
            'core_nodes': False,
            'topology_info': {
                'nodes': {'A': {}, 'B': {}},  # Nodes as a dictionary
                'links': {
                    1: {"source": "A", "destination": "B", "fiber": {"num_cores": 2}, "length": 100,
                        "spectral_slots": 320}
                }
            },
            'band_list': [],
            'c_band': 320,
            'l_band': 320,
            'max_iters': 10,
            'print_step': 1,
            'save_step': 1,
            'is_training': False,
            'deploy_model': False,
            'seeds': [42],
            'erlang': 10,
            'thread_num': 1,
            'mod_per_bw': {  # Adding mod_per_bw to engine_props
                '50GHz': {'QPSK': {}, '16QAM': {}}
            },
        }
        self.engine = Engine(engine_props=engine_props)
        self.engine.reqs_dict = {1.0: {'req_id': 10, 'request_type': 'arrival'}}

        # Mocking sdn_obj and stats_obj
        self.engine.sdn_obj = MagicMock()
        self.engine.sdn_obj.sdn_props = MagicMock()
        self.engine.sdn_obj.sdn_props.was_routed = True
        self.engine.sdn_obj.sdn_props.num_trans = 3
        self.engine.sdn_obj.sdn_props.path_list = ['A', 'B', 'C']
        self.engine.sdn_obj.sdn_props.spectrum_object.modulation = 'QAM'
        self.engine.sdn_obj.sdn_props.is_sliced = False
        self.engine.sdn_obj.sdn_props.core_list = [0, 1]
        self.engine.sdn_obj.sdn_props.curr_band = 'c'
        self.engine.sdn_obj.sdn_props.net_spec_dict = {}
        self.engine.sdn_obj.sdn_props.update_params = MagicMock()

        self.engine.stats_obj = MagicMock()

        self.engine.topology = nx.Graph()
        self.engine.net_spec_dict = {}
        self.engine.reqs_status_dict = {}
        self.engine.ml_model = None

    @patch('src.engine.nx.Graph')
    def test_create_topology(self, mock_graph):
        """
        Tests the create topology method.
        """
        self.engine.topology = mock_graph.return_value
        self.engine.engine_props['topology_info']['nodes'] = ['A', 'B']
        self.engine.engine_props['topology_info']['links'] = {
            1: {"source": "A", "destination": "B", "fiber": {"num_cores": 2}, "length": 100, "spectral_slots": 320}
        }
        self.engine.create_topology()

        self.engine.topology.add_nodes_from.assert_called_once_with(['A', 'B'])
        self.engine.topology.add_edge.assert_called_once_with('A', 'B', length=100, nli_cost=None)
        expected_cores_matrix = np.zeros((2, 320))
        expected_net_spec = {
            ('A', 'B'): {'cores_matrix': {'c': expected_cores_matrix, 'l': expected_cores_matrix}, 'link_num': 1},
            ('B', 'A'): {'cores_matrix': {'c': expected_cores_matrix, 'l': expected_cores_matrix}, 'link_num': 1}
        }
        for link, link_data in self.engine.net_spec_dict.items():
            for band in ['c', 'l']:
                self.assertTrue(
                    np.array_equal(link_data['cores_matrix'][band], expected_net_spec[link]['cores_matrix'][band]))
            self.assertEqual(link_data['link_num'], expected_net_spec[link]['link_num'])

        self.assertEqual(self.engine.engine_props['topology'], self.engine.topology)
        self.assertEqual(self.engine.stats_obj.topology, self.engine.topology)
        self.assertEqual(self.engine.sdn_obj.sdn_props.net_spec_dict, self.engine.net_spec_dict)
        self.assertEqual(self.engine.sdn_obj.sdn_props.topology, self.engine.topology)

    def test_end_iter(self):
        """
        Tests the end_iter method.
        """
        iteration = 0
        self.engine.engine_props['print_step'] = 1  # Ensure print_step triggers the print
        self.engine.engine_props['is_training'] = False  # Ensure it's not training to allow printing
        self.engine.engine_props['save_step'] = 1  # Ensure save_step triggers the saving data

        with patch.object(self.engine.stats_obj, 'get_conf_inter', return_value=True), \
                patch.object(self.engine.stats_obj, 'print_iter_stats') as _:
            self.engine.end_iter(iteration=iteration)
            self.engine.stats_obj.get_blocking.assert_called_once()
            self.engine.stats_obj.end_iter_update.assert_called_once()
            self.engine.stats_obj.get_conf_inter.assert_called_once()

    def test_handle_release_with_req(self):
        """
        Test handle release with an existing request in the reqs_status_dict.
        """
        curr_time = 1.0
        req_id = self.engine.reqs_dict[curr_time]['req_id']
        self.engine.reqs_status_dict[req_id] = {
            'path': ['A', 'B', 'C']
        }
        self.engine.handle_release(curr_time=curr_time)

        self.engine.sdn_obj.handle_event.assert_called_once_with(
            req_dict=self.engine.reqs_dict[curr_time], request_type='release'
        )
        self.assertEqual(self.engine.net_spec_dict, self.engine.sdn_obj.sdn_props.net_spec_dict)

    def test_init_iter(self):
        """
        Tests the init_iter method.
        """
        iteration = 0
        self.engine.engine_props['request_distribution'] = {'50GHz': 1.0}
        self.engine.engine_props['arrival_rate'] = 0.2
        self.engine.engine_props['holding_time'] = 100
        self.engine.engine_props['sim_type'] = 'yue'
        self.engine.engine_props['num_requests'] = 5000
        self.engine.engine_props['seeds'] = [42]
        self.engine.engine_props['topology_info']['nodes'] = {'A': {}, 'B': {}}  # Ensure nodes are a dictionary
        self.engine.engine_props['is_only_core_node'] = True  # Define nodes permitted to send requests

        with patch('src.engine.load_model', autospec=True) as mock_load_model:
            self.engine.init_iter(iteration=iteration)
            self.assertEqual(self.engine.iteration, iteration)
            self.engine.stats_obj.init_iter_stats.assert_called_once()

            if self.engine.engine_props['deploy_model']:
                mock_load_model.assert_called_once_with(engine_props=self.engine.engine_props)
            else:
                mock_load_model.assert_not_called()

    def test_end_iter_non_training_conf_inter(self):
        """
        Test end_iter during non-training with confidence interval check.
        """
        iteration = 0
        self.engine.engine_props['is_training'] = False  # Non-training scenario

        with patch.object(self.engine.stats_obj, 'get_conf_inter', return_value=True) as mock_get_conf_inter:
            resp = self.engine.end_iter(iteration=iteration, print_flag=True)
            mock_get_conf_inter.assert_called_once()
            self.assertTrue(resp)

    def test_end_iter_training_conf_inter(self):
        """
        Test end_iter during training (ignores confidence interval).
        """
        iteration = 0
        self.engine.engine_props['is_training'] = True  # Training scenario

        with patch.object(self.engine.stats_obj, 'get_conf_inter') as mock_get_conf_inter:
            resp = self.engine.end_iter(iteration=iteration, print_flag=True)
            mock_get_conf_inter.assert_not_called()  # Should not check confidence intervals during training
            self.assertFalse(resp)

    def test_end_iter_print_iter_stats(self):
        """
        Test end_iter ensures print_iter_stats is called.
        """
        iteration = 0
        self.engine.engine_props['print_step'] = 1

        with patch.object(self.engine.stats_obj, 'print_iter_stats') as mock_print_iter_stats:
            self.engine.end_iter(iteration=iteration, print_flag=True)
            mock_print_iter_stats.assert_called_once_with(max_iters=self.engine.engine_props['max_iters'],
                                                          print_flag=True)

    def test_end_iter_save_stats(self):
        """
        Test end_iter ensures save_stats is called.
        """
        iteration = 0
        base_fp = "/path/to/output"
        self.engine.engine_props['save_step'] = 1

        with patch.object(self.engine.stats_obj, 'save_stats') as mock_save_stats:
            self.engine.end_iter(iteration=iteration, base_fp=base_fp)
            mock_save_stats.assert_called_once_with(base_fp=base_fp)


if __name__ == '__main__':
    unittest.main()
