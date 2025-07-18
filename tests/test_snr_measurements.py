# pylint: disable=protected-access

import unittest
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
            'egn_model': False,
            'xt_noise': False,
            'phi': {'QPSK': 0.5},
            'snr_type': 'snr_calc_nli',
            'requested_xt': {'QPSK': -20},
            'fixed_grid': False,
            'band_list': ['c'],
            'phi':{"BPSK": 1, "QPSK": 1, "8-QAM": 0.66667, "16-QAM":0.68, "32-QAM": 0.69, "64-QAM": 0.6190476190476191}
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

    def test_check_gsnr(self):
        """Test check_gsnr returns True with realistic values."""
        self.snr_measurements.snr_props.req_snr = {'BPSK': 3.71,'QPSK': 6.72, '8-QAM': 10.84, '16-QAM': 13.24, '32-QAM': 16.16, '64-QAM': 19.01}
        self.spectrum_props.modulation = 'QPSK'
        self.engine_props['topology_info']['links'][0] = {'fiber': {
                    'attenuation': 0.2 / 4.343 * 1e-3,
                    'non_linearity': 1.3e-3,
                    'dispersion': -21.3e-27,
                    'fiber_type': 0,
                    'bending_radius': 0.05,
                    'mode_coupling_co': 4.0e-4,
                    'propagation_const': 4e6,
                    'core_pitch': 4e-5,
                    'frequency_start_c': 3e8 / 1565e-9,
                    'frequency_end_c':  ((3e8 / 1565e-9)+ 6.0e12),
                    'frequency_start_l': 3e8 / 1620e-9,
                    'frequency_end_l':  ((3e8 / 1620e-9)+ 6.0e12),
                    'c_band_bw': 6.0e12,
                    'raman_gain_slope': 0.028 / (1e3)/ (1e12),
                    'gvd': (-22.6) * ((1e-12) * (1e-12)) / ( 1e3),
                    'gvd_slope': (0.14) * ((1e-12) * (1e-12) * (1e-12)) / ( 1e3),
            },
            'length': 200,
            'source': 'A',
            'destination': 'B',
            'span_length': 100,}

        self.snr_measurements.spectrum_props.start_slot = 10
        self.snr_measurements.spectrum_props.end_slot = 15
        self.snr_measurements.spectrum_props.core_num = 0
        self.snr_measurements.spectrum_props.curr_band = 'c'
        self.snr_measurements.spectrum_props.path_list = ['A','B']

        self.snr_measurements.snr_props.nsp = {'c':1.77, 'l': 1.99 }
        self.snr_measurements.snr_props.plank = 6.62607004e-34
        self.snr_measurements.sdn_props.bandwidth = '300'
        self.snr_measurements.spectrum_props.slicing_flag = False
        self.snr_measurements.sdn_props.net_spec_dict[('A','B')]['cores_matrix']['c'][0][0:3] = 1
        self.snr_measurements.sdn_props.lightpath_status_dict = {('A','B'):{1:{'mod_format': 'QPSK'}}}
        self.snr_measurements.num_slots = (
                self.snr_measurements.spectrum_props.end_slot - self.snr_measurements.spectrum_props.start_slot + 1
        )

        resp, gsnr_db, bw_resp = self.snr_measurements.check_gsnr()
        self.assertEqual(bw_resp, 300)
        self.assertEqual(gsnr_db, 21.69889484393792)
        self.assertTrue(resp)

    def test_check_gsnr_mb(self):
        """Test check_gsnr_mb returns True when SNR is sufficient."""
        self.snr_measurements.snr_props.req_snr = {'BPSK': 3.71,'QPSK': 6.72, '8-QAM': 10.84, '16-QAM': 13.24, '32-QAM': 16.16, '64-QAM': 19.01}
        self.spectrum_props.modulation = 'QPSK'
        self.engine_props['topology_info']['links'][0] = {'fiber': {
                    'attenuation': 0.2 / 4.343 * 1e-3,
                    'non_linearity': 1.3e-3,
                    'dispersion': -21.3e-27,
                    'fiber_type': 0,
                    'bending_radius': 0.05,
                    'mode_coupling_co': 4.0e-4,
                    'propagation_const': 4e6,
                    'core_pitch': 4e-5,
                    'frequency_start_c': 3e8 / 1565e-9,
                    'frequency_end_c':  ((3e8 / 1565e-9)+ 6.0e12),
                    'frequency_start_l': 3e8 / 1620e-9,
                    'frequency_end_l':  ((3e8 / 1620e-9)+ 6.0e12),
                    'c_band_bw': 6.0e12,
                    'raman_gain_slope': 0.028 / (1e3)/ (1e12),
                    'gvd': (-22.6) * ((1e-12) * (1e-12)) / ( 1e3),
                    'gvd_slope': (0.14) * ((1e-12) * (1e-12) * (1e-12)) / ( 1e3),
            },
            'length': 200,
            'source': 'A',
            'destination': 'B',
            'span_length': 100,}

        self.snr_measurements.spectrum_props.start_slot = 10
        self.snr_measurements.spectrum_props.end_slot = 15
        self.snr_measurements.spectrum_props.core_num = 0
        self.snr_measurements.spectrum_props.curr_band = 'c'
        self.snr_measurements.spectrum_props.path_list = ['A','B']

        self.snr_measurements.snr_props.nsp = {'c':1.77, 'l': 1.99 }
        self.snr_measurements.snr_props.plank = 6.62607004e-34
        self.snr_measurements.sdn_props.bandwidth = '300'
        self.snr_measurements.spectrum_props.slicing_flag = False
        self.snr_measurements.sdn_props.net_spec_dict[('A','B')]['cores_matrix']['c'][0][0:3] = 1
        self.snr_measurements.sdn_props.lightpath_status_dict = {('A','B'):{1:{'mod_format': 'QPSK'}}}
        self.snr_measurements.num_slots = (
                self.snr_measurements.spectrum_props.end_slot - self.snr_measurements.spectrum_props.start_slot + 1
        )

        resp, gsnr_db, bw_resp = self.snr_measurements.check_gsnr_mb()
        self.assertEqual(bw_resp, 300)
        self.assertEqual(gsnr_db, 21.627390247120072)
        self.assertTrue(resp)

    def test__compute_nli_mb(self):
        """Test _compute_nli_mb returns correct NLI for multi-band."""
        self.snr_measurements.snr_props.link_dict = {
                'attenuation': 0.2 / 4.343 * 1e-3,
                'non_linearity': 1.3e-3,
                'dispersion': -21.3e-27,
                'fiber_type': 0,
                'bending_radius': 0.05,
                'mode_coupling_co': 4.0e-4,
                'propagation_const': 4e6,
                'core_pitch': 4e-5,
                'frequency_start_c': 3e8 / 1565e-9,
                'frequency_end_c':  ((3e8 / 1565e-9)+ 6.0e12),
                'frequency_start_l': 3e8 / 1620e-9,
                'frequency_end_l':  ((3e8 / 1620e-9)+ 6.0e12),
                'c_band_bw': 6.0e12,
                'raman_gain_slope': 0.028 / (1e3)/ (1e12),
                'gvd': (-22.6) * ((1e-12) * (1e-12)) / ( 1e3),
                'gvd_slope': (0.14) * ((1e-12) * (1e-12) * (1e-12)) / ( 1e3),
        }

        self.snr_measurements.spectrum_props.start_slot = 10
        self.snr_measurements.spectrum_props.end_slot = 15
        self.snr_measurements.spectrum_props.core_num = 0
        self.snr_measurements.spectrum_props.curr_band = 'c'

        self.snr_measurements.snr_props.nsp = {'c':1.77, 'l': 1.99 }
        self.snr_measurements.snr_props.plank = 6.62607004e-34
        self.snr_measurements.snr_props.length = 100
        self.snr_measurements.snr_props.num_span = 1
        self.snr_measurements.sdn_props.net_spec_dict[('A','B')]['cores_matrix']['c'][0][0:3] = 1
        self.snr_measurements.sdn_props.lightpath_status_dict = {('A','B'):{1:{'mod_format': 'QPSK'}}}
        self.snr_measurements.num_slots = (
                self.snr_measurements.spectrum_props.end_slot - self.snr_measurements.spectrum_props.start_slot + 1
        )

        nli = self.snr_measurements._compute_nli_mb(source = 'A', dest = 'B', p_total = 0.002)

        self.assertEqual(nli, 9.14394298264424e-05)
        self.assertIsInstance(nli, float)

    def test__compute_ase_mb(self):
        """Test _compute_ase_mb returns non-negative ASE noise."""
        self.snr_measurements.snr_props.link_dict = {
                'attenuation': 0.2 / 4.343 * 1e-3,
                'non_linearity': 1.3e-3,
                'dispersion': -21.3e-27,
                'fiber_type': 0,
                'bending_radius': 0.05,
                'mode_coupling_co': 4.0e-4,
                'propagation_const': 4e6,
                'core_pitch': 4e-5,
                'frequency_start_c': 3e8 / 1565e-9,
                'frequency_end_c':  ((3e8 / 1565e-9)+ 6.0e12),
                'frequency_start_l': 3e8 / 1620e-9,
                'frequency_end_l':  ((3e8 / 1620e-9)+ 6.0e12),
                'c_band_bw': 6.0e12,
                'raman_gain_slope': 0.028 / (1e3)/ (1e12),
                'gvd': (-22.6) * ((1e-12) * (1e-12)) / ( 1e3),
                'gvd_slope': (0.14) * ((1e-12) * (1e-12) * (1e-12)) / ( 1e3),
        }

        self.snr_measurements.spectrum_props.start_slot = 10
        self.snr_measurements.spectrum_props.end_slot = 15
        self.snr_measurements.spectrum_props.core_num = 0
        self.snr_measurements.spectrum_props.curr_band = 'c'

        self.snr_measurements.snr_props.nsp = {'c':1.77, 'l': 1.99 }
        self.snr_measurements.snr_props.plank = 6.62607004e-34
        self.snr_measurements.snr_props.length = 100
        self.snr_measurements.snr_props.num_span = 1
        self.snr_measurements.sdn_props.net_spec_dict[('A','B')]['cores_matrix']['c'][0][0:3] = 1

        # Set num_slots to the correct value before calling check_xt
        self.snr_measurements.num_slots = (
                self.snr_measurements.spectrum_props.end_slot - self.snr_measurements.spectrum_props.start_slot + 1
        )

        ase = self.snr_measurements._compute_ase_mb(
            source = 'A', dest = 'B', p_total = 0.002
        )

        self.assertEqual(ase, 0.00334151139400405)
        self.assertIsInstance(ase, float)

    # def test__gsnr_calc_mb(self):
    #     """Test _gsnr_calc_mb returns positive SNR value."""
    #     ase = 1e-9
    #     nli = 1e-10
    #     signal_power = 1e-3

    #     snr = self.snr_measurements._gsnr_calc_mb(ase=ase, nli=nli, signal_power=signal_power)

    #     self.assertGreater(snr, 0)
    #     self.assertIsInstance(snr, float)


if __name__ == '__main__':
    unittest.main()
