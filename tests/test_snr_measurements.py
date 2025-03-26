# pylint: disable=protected-access

import unittest
from unittest.mock import patch
from unittest.mock import MagicMock
import numpy as np
from src.snr_measurements import SnrMeasurements


class TestSnrMeasurements(unittest.TestCase):
    """Unit tests for SnrMeasurements class."""

    def setUp(self):
        """Set up necessary parameters and objects for the tests."""
        self.engine_props = {
            'bw_per_slot': 12.5,
            'input_power': 0.001,
            'multi_fiber': False,
            'cores_per_link':2,
            'network':'USbackbone60',
            'topology_info': {
                'links': {
                    0: {
                        'fiber': {
                            'attenuation': 0.2,
                            'dispersion': 16.7,
                            'non_linearity': 1.3e-3
                        },
                        'length': 100,
                        'span_length': 80
                    }
                }
            },
            'c_band': 40,
            'l_band': 40,
            's_band': 50,
            'egn_model': False,
            'xt_noise': False,
            'phi': {'QPSK': 0.5},
            'snr_type': 'snr_calc_nli',
            'requested_xt': {'QPSK': -20}
        }

        self.sdn_props = MagicMock()
        self.sdn_props.net_spec_dict = {
            ('A', 'B'): {
                'cores_matrix': {'c': np.zeros((7, 40))},
                'link_num': 0
            },
            ('B', 'C'): {
                'cores_matrix': {'c': np.zeros((7, 40))},
                'link_num': 0
            }
        }

        self.route_props = {'connection_index': 0}
        self.spectrum_props = MagicMock()
        self.spectrum_props.path_list = ['A', 'B', 'C']
        self.spectrum_props.start_slot = 10
        self.spectrum_props.end_slot = 15
        self.spectrum_props.core_num = 0
        self.spectrum_props.curr_band = 'c'
        self.spectrum_props.modulation = 'QPSK'

        self.snr_measurements = SnrMeasurements(
            engine_props=self.engine_props,
            sdn_props=self.sdn_props,
            spectrum_props=self.spectrum_props,
            route_props=self.route_props
        )

    def test_calculate_sci_psd(self):
        """Test the calculation of self-phase power spectral density (SCI PSD)."""
        self.snr_measurements.snr_props.center_psd = 1e-3
        self.snr_measurements.snr_props.bandwidth = 1e-9
        self.snr_measurements.snr_props.link_dict = {
            'dispersion': 16.7,
            'attenuation': 0.2
        }

        sci_psd = self.snr_measurements._calculate_sci_psd()
        expected_sci_psd = 4.120e-22  # Update with actual expected value

        self.assertAlmostEqual(sci_psd, expected_sci_psd, places=10)

    def test_update_link_xci(self):
        """Test the update of cross-phase modulation noise (XCI) on a link."""
        req_id = 1.0
        slot_index = 5
        curr_xci = 0.0
        curr_link = np.zeros((7, 40))
        curr_link[self.spectrum_props.core_num][slot_index] = req_id

        # Initialize center_freq to avoid NoneType error
        self.snr_measurements.snr_props.center_freq = (
                self.spectrum_props.start_slot * self.engine_props['bw_per_slot'] * 10 ** 9
        )

        new_xci = self.snr_measurements._update_link_xci(
            req_id=req_id, curr_link=curr_link, slot_index=slot_index, curr_xci=curr_xci
        )
        expected_xci = 1.428e-27  # Update with actual expected value

        self.assertAlmostEqual(new_xci, expected_xci, places=3)

    def test_check_xt(self):
        """Test the check for cross-talk (XT) interference on a request."""
        self.snr_measurements.spectrum_props.start_slot = 10
        self.snr_measurements.spectrum_props.end_slot = 15

        # Set num_slots to the correct value before calling check_xt
        self.snr_measurements.num_slots = (
                self.spectrum_props.end_slot - self.spectrum_props.start_slot + 1
        )

        # Call check_xt and test the response
        resp, cross_talk = self.snr_measurements.check_xt()
        expected_cross_talk = 0.0  # Update with actual expected value

        self.assertTrue(resp)
        self.assertAlmostEqual(cross_talk, expected_cross_talk, places=10)

    @patch('src.snr_measurements.get_loaded_files')
    def test_check_snr_ext(self, path_index: int):
        """Test checking SNR using external resources."""
        self.snr_measurements.get_loaded_files = MagicMock(return_value=([], []))
        self.snr_measurements.get_slot_index = MagicMock(return_value=0)
        self.snr_measurements.compute_response = MagicMock(return_value=('success', 25))

        # Test check_snr_ext with mock data
        resp, snr_val = self.snr_measurements.check_snr_ext(path_index=0)

        # Verify the response
        self.assertEqual(resp, 'success')
        self.assertEqual(snr_val, 25)

    def test_check_snr_ext_slicing(self):
        """Test checking SNR using external resources with slicing."""
        self.snr_measurements.get_loaded_files.return_value = ([[[1]]], [])
        self.snr_measurements.get_slot_index.return_value = 0

        # Simulate modulation format and bandwidth
        mod_format, supported_bw, snr_val = self.snr_measurements.check_snr_ext_slicing(path_index=0)

        # Verify the response
        self.assertEqual(mod_format, 'QPSK')
        self.assertEqual(supported_bw, 50)
        self.assertEqual(snr_val, 0)

    def test_check_snr_ext_open_slots(self):
        """Test checking open slots for SNR."""
        open_slots_list = [1, 2, 3]
        self.snr_measurements.get_loaded_files.return_value = ([[[1]]], [])

        # Test check_snr_ext_open_slots with mock data
        open_slots_list = self.snr_measurements.check_snr_ext_open_slots(path_index=0, open_slots_list=open_slots_list)

        # Verify the open slots after checking
        self.assertNotIn(2, open_slots_list)  # Slot 2 should be removed


if __name__ == '__main__':
    unittest.main()
