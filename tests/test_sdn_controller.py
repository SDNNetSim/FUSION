# pylint: disable=protected-access

import unittest
from unittest.mock import patch
import numpy as np
import networkx as nx

from src.sdn_controller import SDNController
from arg_scripts.sdn_args import SDNProps  # Class import for sdn_props
from src.snr_measurements import SnrMeasurements



class TestSDNController(unittest.TestCase):
    """
    Tests the SDNController class.
    """

    def setUp(self):
        self.engine_props = {
            'cores_per_link': 7,
            'guard_slots': 1,
            'route_method': 'shortest_path',
            'max_segments': 1,
            'band_list': ['c'],
            'is_grooming_enabled': False,
        }
        self.controller = SDNController(engine_props=self.engine_props)

        # Mock SDNProps to ensure it's treated as a class object
        self.controller.sdn_props = SDNProps()
        self.controller.sdn_props.path_list = ['A', 'B', 'C']
        self.controller.sdn_props.req_id = 1
        self.controller.sdn_props.net_spec_dict = {
            ('A', 'B'): {
                'cores_matrix': {'c': np.zeros((7, 10))},
                'usage_count': 0,
                'throughput': 0
            },
            ('B', 'A'): {
                'cores_matrix': {'c': np.zeros((7, 10))},
                'usage_count': 0,
                'throughput': 0
            },
            ('B', 'C'): {
                'cores_matrix': {'c': np.zeros((7, 10))},
                'usage_count': 0,
                'throughput': 0
            },
            ('C', 'B'): {
                'cores_matrix': {'c': np.zeros((7, 10))},
                'usage_count': 0,
                'throughput': 0
            }
        }
        self.controller.sdn_props.transponder_usage_dict = { 'A': {
                                                                    "available_transponder": 0,
                                                                    "total_transponder": 0,},
                                                            'B': {
                                                                    "available_transponder": 0,
                                                                    "total_transponder": 0,},
                                                            'C': {
                                                                    "available_transponder": 0,
                                                                    "total_transponder": 0,},
        }
        self.controller.sdn_props.lightpath_status_dict = {('A','B'):{},
                                                           ('A','C'):{},
                                                           ('B','C'):{},}
        self.controller.sdn_props.lp_bw_utilization_dict = {}

    def test_release(self):
        """
        Test the release method.
        """
        self.controller.sdn_props.net_spec_dict[('A', 'B')]['cores_matrix']['c'][0][:3] = 10
        self.controller.sdn_props.net_spec_dict[('A', 'B')]['cores_matrix']['c'][0][3] = -10
        self.controller.sdn_props.transponder_usage_dict['A']['total_transponder'] = 1
        self.controller.sdn_props.transponder_usage_dict['B']['total_transponder'] = 1
        self.controller.sdn_props.source = 'A'
        self.controller.sdn_props.destination = 'B'
        self.controller.sdn_props.path_list = ['A', 'B']
        self.controller.sdn_props.lightpath_status_dict[('A','B')] = {10:{"time_bw_usage": {1: 100}}}
        self.controller.sdn_props.lightpath_status_dict[('A','B')][10]['band'] = 'c'
        self.controller.sdn_props.lightpath_status_dict[('A','B')][10]['core'] = 0
        self.controller.sdn_props.lightpath_status_dict[('A','B')][10]['lightpath_bandwidth'] = 100
        self.controller.sdn_props.lightpath_status_dict[('A','B')][10]['requests_dict'] = [2]
        self.controller.sdn_props.depart = 10

        self.controller.release( lightpath_id = 10)

        for link in zip(self.controller.sdn_props.path_list, self.controller.sdn_props.path_list[1:]):
            for core_num in range(self.engine_props['cores_per_link']):
                core_arr = self.controller.sdn_props.net_spec_dict[link]['cores_matrix']['c'][core_num]
                self.assertTrue(np.all(core_arr[:4] == 0), "Request and guard band not properly cleared")

    def test_allocate(self):
        """
        Test the allocate method.
        """
        self.controller.spectrum_obj.spectrum_props.start_slot = 0
        self.controller.spectrum_obj.spectrum_props.end_slot = 3
        self.controller.spectrum_obj.spectrum_props.core_num = 0
        self.controller.spectrum_obj.spectrum_props.curr_band = 'c'
        self.controller.spectrum_obj.spectrum_props.lightpath_id = 10
        self.controller.engine_props['guard_slots'] = 1
        self.controller.allocate()

        for link in zip(self.controller.sdn_props.path_list, self.controller.sdn_props.path_list[1:]):
            core_matrix = self.controller.sdn_props.net_spec_dict[link]['cores_matrix']['c'][0]
            self.assertTrue(np.all(core_matrix[:2] == self.controller.spectrum_obj.spectrum_props.lightpath_id),
                            "Request not properly allocated")

            self.assertEqual(core_matrix[2], self.controller.spectrum_obj.spectrum_props.lightpath_id * -1,
                             msg="Guard band not properly allocated.")

    def test_update_req_stats(self):
        """
        Test the update request statistics method.
        """
        # Properly initialize sdn_props with necessary attributes
        self.controller.sdn_props.xt_cost = []
        self.controller.sdn_props.core_num = []
        self.controller.spectrum_obj.spectrum_props.xt_cost = 10
        self.controller.spectrum_obj.spectrum_props.core_num = 2

        # Set sdn_props.stat_key_list to ensure the relevant keys are present
        self.controller.sdn_props.stat_key_list = ['xt_cost', 'core_num']

        # Call the method to update request statistics
        self.controller._update_req_stats(bandwidth='100G', remaining = "0")

        # Verify that the bandwidth and other stats are updated correctly
        self.assertIn('100G', self.controller.sdn_props.bandwidth_list)
        self.assertIn(10, self.controller.sdn_props.xt_cost)  # Check if xt_cost was updated correctly
        self.assertIn(2, self.controller.sdn_props.core_num)  # Check if core_num was updated correctly

    @patch('src.sdn_controller.SDNController.allocate')
    @patch('src.sdn_controller.SDNController._update_req_stats')
    @patch('src.spectrum_assignment.SpectrumAssignment.get_spectrum')
    def test_allocate_slicing(self, mock_get_spectrum, mock_update_req_stats, mock_allocate):
        """
        Tests the allocate slicing method.
        """
        self.controller.spectrum_obj.spectrum_props.is_free = True  # Ensure spectrum is free
        mock_get_spectrum.return_value = None  # Ensure that the method is mocked correctly

        # Call the allocate slicing method
        self.controller._allocate_slicing(num_segments=2, mod_format='QPSK', path_list=['A', 'B'], bandwidth='50G')

        # Verify that the relevant methods were called
        mock_get_spectrum.assert_called()
        mock_allocate.assert_called()
        mock_update_req_stats.assert_called_with(bandwidth='50G')

    @patch('src.sdn_controller.SDNController.allocate')
    @patch('src.sdn_controller.SDNController._update_req_stats')
    @patch('src.routing.Routing.get_route')
    @patch('src.spectrum_assignment.SpectrumAssignment.get_spectrum')
    def test_handle_event_arrival(self, mock_allocate, mock_stats, mock_route, mock_spectrum):  # pylint: disable=unused-argument
        """
        Tests the handle event with an arrival request.
        """
        mock_route.return_value = None
        self.controller.route_obj.route_props.paths_matrix = [['A', 'B', 'C']]
        self.controller.route_obj.route_props.mod_formats_matrix = [['QPSK']]
        self.controller.route_obj.route_props.weights_list = [10]

        self.controller.spectrum_obj.spectrum_props.is_free = True
        mock_spectrum.return_value = None

        self.controller.handle_event(req_dict={}, request_type="arrival")

        mock_allocate.assert_called_once()
        self.assertTrue(self.controller.sdn_props.was_routed)

    def test_allocate_without_guard_band(self):
        """
        Test the allocate method when no guard band is present.
        """
        self.controller.spectrum_obj.spectrum_props.start_slot = 0
        self.controller.spectrum_obj.spectrum_props.end_slot = 2
        self.controller.spectrum_obj.spectrum_props.core_num = 0
        self.controller.spectrum_obj.spectrum_props.curr_band = 'c'
        self.controller.spectrum_obj.spectrum_props.lightpath_id = 10
        self.controller.engine_props['guard_slots'] = 0

        # Perform allocation
        self.controller.allocate()

        for link in zip(self.controller.sdn_props.path_list, self.controller.sdn_props.path_list[1:]):
            core_matrix = self.controller.sdn_props.net_spec_dict[link]['cores_matrix']['c'][0]

            # Check allocation
            self.assertTrue(np.all(core_matrix[:3] == self.controller.spectrum_obj.spectrum_props.lightpath_id),
                            "Request not properly allocated")

            # Ensure no guard band is allocated
            self.assertNotIn(self.controller.spectrum_obj.spectrum_props.lightpath_id * -1, core_matrix,
                             "Guard band should not be allocated.")

    def test_handle_dynamic_slicing_success(self):
        """
        Test _handle_dynamic_slicing for a successful allocation scenario.
        """
        self.controller.sdn_props.bandwidth = '100'
        self.controller.engine_props['mod_per_bw'] = {
            '50': {'QPSK': {'max_length': 100}},
            '100': {'16QAM': {'max_length': 200}}
        }
        self.controller.engine_props['fixed_grid'] = True
        self.controller.engine_props['topology'] = {
            ('A', 'B'): {'length': 50},
            ('B', 'C'): {'length': 50}
        }
        self.engine_props['can_partially_serve'] = False
        self.controller.sdn_props.path_list = ['A', 'B', 'C']
        self.controller.sdn_props.num_trans = 0
        self.controller.sdn_props.path_index = 0
        self.controller.sdn_props.depart = 10

        # Set spectrum to free
        self.controller.spectrum_obj.spectrum_props.is_free = True

        with patch('src.spectrum_assignment.SpectrumAssignment.get_spectrum_dynamic_slicing',
                   return_value=('16QAM', 50)) as mock_get_spectrum, \
                patch.object(self.controller, 'allocate') as mock_allocate, \
                patch.object(self.controller, '_update_req_stats') as mock_update_stats, \
                patch('src.sdn_controller.find_path_len', return_value=100) as mock_find_path_len:
            self.controller._handle_dynamic_slicing(path_list=['A', 'B', 'C'], forced_segments=1)

            # Verify methods were called
            mock_get_spectrum.assert_called()
            mock_allocate.assert_called()
            mock_update_stats.assert_called_with(bandwidth='50', remaining='0')
            mock_find_path_len.assert_called_with(path_list=['A', 'B', 'C'],
                                                  topology=self.controller.engine_props['topology'])

            # TODO: Forced segments isn't used in dynamic slicing?
            self.assertEqual(self.controller.sdn_props.num_trans, 2)
            self.assertTrue(self.controller.sdn_props.is_sliced)

    def test_handle_dynamic_slicing_congestion(self):
        """
        Test _handle_dynamic_slicing for a congestion scenario.
        """
        self.controller.sdn_props.bandwidth = '100'
        self.controller.sdn_props.remaining_bw = 100
        self.controller.engine_props['mod_per_bw'] = {
            '50': {'QPSK': {'max_length': 100}},
            '100': {'16QAM': {'max_length': 200}}
        }
        self.controller.engine_props['fixed_grid'] = True
        self.controller.engine_props['topology'] = {
            ('A', 'B'): {'length': 50},
            ('B', 'C'): {'length': 50}
        }
        self.engine_props['can_partially_serve'] = False
        self.controller.sdn_props.path_list = ['A', 'B', 'C']
        self.controller.sdn_props.num_trans = 0
        self.controller.sdn_props.path_index = 0
        self.controller.sdn_props.was_partially_routed = False
        self.controller.sdn_props.is_sliced = False
        
        self.controller.sdn_props.was_new_lp_established = list()

        with patch('src.spectrum_assignment.SpectrumAssignment.get_spectrum_dynamic_slicing',
                   return_value=('16QAM', 50)) as mock_get_spectrum, \
                patch.object(self.controller, 'allocate') as mock_allocate, \
                patch('src.sdn_controller.find_path_len', return_value=100) as mock_find_path_len:
            # Use the real _handle_congestion function
            with patch.object(self.controller, '_handle_congestion',
                              wraps=self.controller._handle_congestion) as mock_handle_congestion:
                # Simulate no free spectrum
                self.controller.spectrum_obj.spectrum_props.is_free = False

                self.controller._handle_dynamic_slicing(path_list=['A', 'B', 'C'], forced_segments=1)

                # Verify methods were called
                mock_get_spectrum.assert_called()
                mock_handle_congestion.assert_called_with(100)
                mock_allocate.assert_not_called()
                mock_find_path_len.assert_called_with(path_list=['A', 'B', 'C'],
                                                      topology=self.controller.engine_props['topology'])

                # Verify properties were updated
                self.assertFalse(self.controller.sdn_props.was_routed)
                self.assertEqual(self.controller.sdn_props.block_reason, 'congestion')
                self.assertFalse(self.controller.sdn_props.is_sliced)

    def test_usage_count_increment(self):
        """Test that usage_count increments after allocation."""
        self.controller.spectrum_obj.spectrum_props.start_slot = 0
        self.controller.spectrum_obj.spectrum_props.end_slot = 3
        self.controller.spectrum_obj.spectrum_props.core_num = 0
        self.controller.spectrum_obj.spectrum_props.curr_band = 'c'
        self.controller.engine_props['guard_slots'] = 1
        self.controller.spectrum_obj.spectrum_props.lightpath_id = 10

        for key in self.controller.sdn_props.net_spec_dict:
            self.controller.sdn_props.net_spec_dict[key]['usage_count'] = 0

        self.controller.allocate()

        for link in zip(self.controller.sdn_props.path_list, self.controller.sdn_props.path_list[1:]):
            self.assertEqual(self.controller.sdn_props.net_spec_dict[link]['usage_count'], 1)
            reverse_link = (link[1], link[0])
            self.assertEqual(self.controller.sdn_props.net_spec_dict[reverse_link]['usage_count'], 1)

    def test_release_throughput_tracking(self):
        """Test that throughput is correctly tracked on release."""
        for key in self.controller.sdn_props.net_spec_dict:
            self.controller.sdn_props.net_spec_dict[key]['throughput'] = 0

        self.controller.sdn_props.arrive = 0
        self.controller.sdn_props.depart = 5  # 5 seconds duration
        self.controller.sdn_props.bandwidth = '100'  # Gbps
        self.controller.sdn_props.net_spec_dict[('A', 'B')]['cores_matrix']['c'][0][:3] = 10
        self.controller.sdn_props.net_spec_dict[('A', 'B')]['cores_matrix']['c'][0][3] = -10
        self.controller.sdn_props.transponder_usage_dict['A']['total_transponder'] = 1
        self.controller.sdn_props.transponder_usage_dict['B']['total_transponder'] = 1
        self.controller.sdn_props.source = 'A'
        self.controller.sdn_props.destination = 'B'
        self.controller.sdn_props.path_list = ['A', 'B']
        self.controller.sdn_props.lightpath_status_dict[('A','B')] = {10:{"time_bw_usage": {0: 100}}}
        self.controller.sdn_props.lightpath_status_dict[('A','B')][10]['band'] = 'c'
        self.controller.sdn_props.lightpath_status_dict[('A','B')][10]['core'] = 0
        self.controller.sdn_props.lightpath_status_dict[('A','B')][10]['lightpath_bandwidth'] = 100
        self.controller.sdn_props.lightpath_status_dict[('A','B')][10]['requests_dict'] = [2]

        self.controller.release(lightpath_id = 10)

        expected_throughput = 500  # 100 Gbps * 5 seconds
        for link in zip(self.controller.sdn_props.path_list, self.controller.sdn_props.path_list[1:]):
            self.assertEqual(self.controller.sdn_props.net_spec_dict[link]['throughput'], expected_throughput)
            reverse_link = (link[1], link[0])
            self.assertEqual(self.controller.sdn_props.net_spec_dict[reverse_link]['throughput'],
                             expected_throughput)

    def test_release_missing_timing_fields(self):
        """Test that release handles missing timing fields gracefully."""
        for key in self.controller.sdn_props.net_spec_dict:
            self.controller.sdn_props.net_spec_dict[key]['throughput'] = 0

        self.controller.sdn_props.arrive = None
        self.controller.sdn_props.depart = None
        self.controller.sdn_props.bandwidth = None
        self.controller.sdn_props.net_spec_dict[('A', 'B')]['cores_matrix']['c'][0][:3] = 10
        self.controller.sdn_props.net_spec_dict[('A', 'B')]['cores_matrix']['c'][0][3] = -10
        self.controller.sdn_props.transponder_usage_dict['A']['total_transponder'] = 1
        self.controller.sdn_props.transponder_usage_dict['B']['total_transponder'] = 1
        self.controller.sdn_props.source = 'A'
        self.controller.sdn_props.destination = 'B'
        self.controller.sdn_props.path_list = ['A', 'B']
        self.controller.sdn_props.lightpath_status_dict[('A','B')] = {10:{"time_bw_usage": {0: 100}}}
        self.controller.sdn_props.lightpath_status_dict[('A','B')][10]['band'] = 'c'
        self.controller.sdn_props.lightpath_status_dict[('A','B')][10]['core'] = 0
        self.controller.sdn_props.lightpath_status_dict[('A','B')][10]['lightpath_bandwidth'] = 100
        self.controller.sdn_props.lightpath_status_dict[('A','B')][10]['requests_dict'] = [2]
        
        # Should not raise exception
        self.controller.release(lightpath_id = 10)

        for key in self.controller.sdn_props.net_spec_dict:
            self.assertEqual(self.controller.sdn_props.net_spec_dict[key]['throughput'], 0)

    @patch.object(SnrMeasurements, "reevaluate_after_new_lightpath", return_value=[(10, 15.0)])
    def test_handle_event_triggers_snr_recheck(self, mock_reeval):

        # Provide a simple topology
        G = nx.Graph()
        G.add_edge("A", "B", length=10)
        self.controller.sdn_props.topology = G
        self.controller.sdn_props.source = "A"
        self.controller.sdn_props.destination = "B"
        self.controller.route_obj.sdn_props = self.controller.sdn_props
        self.controller.sdn_props.mod_formats = {
            "QPSK": {"max_length": 2000},
            "16-QAM": {"max_length": 1000},
            "64-QAM": {"max_length": 1000}
        }

        # Set dummy spectrum properties so allocate() works
        self.controller.spectrum_obj.spectrum_props.start_slot = 0
        self.controller.spectrum_obj.spectrum_props.end_slot = 3
        self.controller.spectrum_obj.spectrum_props.core_num = 0
        self.controller.spectrum_obj.spectrum_props.curr_band = "c"
        self.controller.spectrum_obj.spectrum_props.lightpath_id = 99
        self.controller.spectrum_obj.spectrum_props.is_free = True

        # Patch get_spectrum to skip actual spectrum assignment
        with patch("src.spectrum_assignment.SpectrumAssignment.get_spectrum", return_value=None), \
                patch.object(self.controller, "_update_req_stats"):
            self.controller.handle_event(req_dict={}, request_type="arrival")

            mock_reeval.assert_called_once()
            args, kwargs = mock_reeval.call_args
            new_lp_info = args[0]

            # Ensure correct LP details are passed
            assert new_lp_info["path"] == ["A", "B"]
            assert new_lp_info["spectrum"] == (0, 3)
            assert new_lp_info["core"] == 0
            assert new_lp_info["band"] == "c"
            assert isinstance(new_lp_info["id"], int)  # ID should be a valid integer




if __name__ == '__main__':
    unittest.main()
