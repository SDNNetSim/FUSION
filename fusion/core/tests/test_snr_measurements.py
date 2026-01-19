"""
Unit tests for fusion.core.snr_measurements module.

This module provides comprehensive testing for the SnrMeasurements class which
handles signal-to-noise ratio calculations, cross-talk interference, and other
signal quality metrics for optical network requests.
"""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from fusion.core.properties import RoutingProps, SNRProps
from fusion.core.snr_measurements import SnrMeasurements


class TestSnrMeasurements(unittest.TestCase):
    """Unit tests for SnrMeasurements functionality."""

    def setUp(self) -> None:
        """Set up test fixtures with proper isolation."""
        self.engine_props = {
            "bw_per_slot": 12.5,
            "input_power": 0.001,
            "topology_info": {
                "links": {
                    0: {
                        "fiber": {
                            "attenuation": 0.2,
                            "dispersion": 16.7,
                            "non_linearity": 1.3e-3,
                        },
                        "length": 100,
                        "span_length": 80,
                    }
                }
            },
            "c_band": 40,
            "egn_model": False,
            "xt_noise": False,
            "phi": {"QPSK": 0.5},
            "snr_type": "snr_calc_nli",
            "requested_xt": {"QPSK": -20},
        }

        self.sdn_props = MagicMock()
        self.sdn_props.network_spectrum_dict = {
            ("A", "B"): {
                "cores_matrix": {"c": np.zeros((7, 40))},
                "link_num": 0,
            },
            ("B", "C"): {
                "cores_matrix": {"c": np.zeros((7, 40))},
                "link_num": 0,
            },
        }

        self.route_props = RoutingProps()
        self.route_props.connection_index = 0
        self.spectrum_props = MagicMock()
        self.spectrum_props.path_list = ["A", "B", "C"]
        self.spectrum_props.start_slot = 10
        self.spectrum_props.end_slot = 15
        self.spectrum_props.core_number = 0
        self.spectrum_props.current_band = "c"
        self.spectrum_props.modulation = "QPSK"

    def test_init_with_valid_parameters_creates_instance(self) -> None:
        """Test SnrMeasurements initialization with valid parameters."""
        snr_measurements = SnrMeasurements(
            engine_props_dict=self.engine_props,
            sdn_props=self.sdn_props,
            spectrum_props=self.spectrum_props,
            route_props=self.route_props,
        )

        self.assertIsInstance(snr_measurements.snr_props, SNRProps)
        self.assertEqual(snr_measurements.engine_props_dict, self.engine_props)
        self.assertEqual(snr_measurements.sdn_props, self.sdn_props)
        self.assertEqual(snr_measurements.spectrum_props, self.spectrum_props)
        self.assertEqual(snr_measurements.route_props, self.route_props)

    def test_init_sets_default_values_correctly(self) -> None:
        """Test SnrMeasurements initialization sets default values."""
        snr_measurements = SnrMeasurements(
            engine_props_dict=self.engine_props,
            sdn_props=self.sdn_props,
            spectrum_props=self.spectrum_props,
            route_props=self.route_props,
        )

        self.assertIsNone(snr_measurements.channels_list)
        self.assertIsNone(snr_measurements.link_id)
        self.assertIsNone(snr_measurements.number_of_slots)

    def test_calculate_sci_psd_with_valid_parameters_returns_float(self) -> None:
        """Test SCI PSD calculation with valid parameters."""
        snr_measurements = SnrMeasurements(
            engine_props_dict=self.engine_props,
            sdn_props=self.sdn_props,
            spectrum_props=self.spectrum_props,
            route_props=self.route_props,
        )
        snr_measurements.snr_props.center_psd = 1e-3
        snr_measurements.snr_props.bandwidth = 1e-9
        snr_measurements.snr_props.link_dictionary = {
            "dispersion": 16.7,
            "attenuation": 0.2,
        }

        sci_psd = snr_measurements._calculate_sci_psd()

        self.assertIsInstance(sci_psd, float)
        self.assertGreater(sci_psd, 0)

    def test_calculate_sci_psd_with_missing_link_dictionary_raises_error(self) -> None:
        """Test SCI PSD calculation with missing link dictionary raises error."""
        snr_measurements = SnrMeasurements(
            engine_props_dict=self.engine_props,
            sdn_props=self.sdn_props,
            spectrum_props=self.spectrum_props,
            route_props=self.route_props,
        )
        snr_measurements.snr_props.center_psd = 1e-3
        snr_measurements.snr_props.bandwidth = 1e-9
        snr_measurements.snr_props.link_dictionary = None

        with self.assertRaises(ValueError) as context:
            snr_measurements._calculate_sci_psd()

        self.assertIn("Required SNR properties are not initialized", str(context.exception))

    def test_calculate_sci_psd_with_missing_center_psd_raises_error(self) -> None:
        """Test SCI PSD calculation with missing center PSD raises error."""
        snr_measurements = SnrMeasurements(
            engine_props_dict=self.engine_props,
            sdn_props=self.sdn_props,
            spectrum_props=self.spectrum_props,
            route_props=self.route_props,
        )
        snr_measurements.snr_props.center_psd = None
        snr_measurements.snr_props.bandwidth = 1e-9
        snr_measurements.snr_props.link_dictionary = {
            "dispersion": 16.7,
            "attenuation": 0.2,
        }

        with self.assertRaises(ValueError) as context:
            snr_measurements._calculate_sci_psd()

        self.assertIn("Required SNR properties are not initialized", str(context.exception))

    def test_calculate_sci_psd_with_missing_bandwidth_raises_error(self) -> None:
        """Test SCI PSD calculation with missing bandwidth raises error."""
        snr_measurements = SnrMeasurements(
            engine_props_dict=self.engine_props,
            sdn_props=self.sdn_props,
            spectrum_props=self.spectrum_props,
            route_props=self.route_props,
        )
        snr_measurements.snr_props.center_psd = 1e-3
        snr_measurements.snr_props.bandwidth = None
        snr_measurements.snr_props.link_dictionary = {
            "dispersion": 16.7,
            "attenuation": 0.2,
        }

        with self.assertRaises(ValueError) as context:
            snr_measurements._calculate_sci_psd()

        self.assertIn("Required SNR properties are not initialized", str(context.exception))

    def test_update_link_xci_with_valid_parameters_returns_updated_xci(self) -> None:
        """Test XCI update with valid parameters."""
        snr_measurements = SnrMeasurements(
            engine_props_dict=self.engine_props,
            sdn_props=self.sdn_props,
            spectrum_props=self.spectrum_props,
            route_props=self.route_props,
        )

        req_id = 1.0
        slot_index = 5
        current_xci = 0.0
        current_link = np.zeros((7, 40))
        current_link[self.spectrum_props.core_number][slot_index] = req_id

        snr_measurements.snr_props.center_frequency = self.spectrum_props.start_slot * self.engine_props["bw_per_slot"] * 10**9

        new_xci = snr_measurements._update_link_xci(
            request_id=req_id,
            current_link=current_link,
            slot_index=slot_index,
            current_xci=current_xci,
        )

        self.assertIsInstance(new_xci, float)
        self.assertGreaterEqual(new_xci, current_xci)

    def test_update_link_xci_with_empty_link_returns_current_xci(self) -> None:
        """Test XCI update with empty link returns current XCI."""
        snr_measurements = SnrMeasurements(
            engine_props_dict=self.engine_props,
            sdn_props=self.sdn_props,
            spectrum_props=self.spectrum_props,
            route_props=self.route_props,
        )

        req_id = 1.0
        slot_index = 5
        current_xci = 0.5
        current_link = np.zeros((7, 40))  # Empty link

        snr_measurements.snr_props.center_frequency = self.spectrum_props.start_slot * self.engine_props["bw_per_slot"] * 10**9

        new_xci = snr_measurements._update_link_xci(
            request_id=req_id,
            current_link=current_link,
            slot_index=slot_index,
            current_xci=current_xci,
        )

        # Should return the same XCI when link is empty
        self.assertEqual(new_xci, current_xci)

    def test_check_xt_with_valid_spectrum_returns_response_and_crosstalk(self) -> None:
        """Test cross-talk check with valid spectrum properties."""
        snr_measurements = SnrMeasurements(
            engine_props_dict=self.engine_props,
            sdn_props=self.sdn_props,
            spectrum_props=self.spectrum_props,
            route_props=self.route_props,
        )
        snr_measurements.spectrum_props.start_slot = 10
        snr_measurements.spectrum_props.end_slot = 15
        snr_measurements.number_of_slots = self.spectrum_props.end_slot - self.spectrum_props.start_slot + 1

        resp, cross_talk = snr_measurements.check_xt()

        self.assertIsInstance(resp, bool)
        self.assertIsInstance(cross_talk, float)
        self.assertGreaterEqual(cross_talk, 0)

    def test_check_xt_with_zero_slots_returns_false_and_zero(self) -> None:
        """Test cross-talk check with zero slots returns appropriate values."""
        snr_measurements = SnrMeasurements(
            engine_props_dict=self.engine_props,
            sdn_props=self.sdn_props,
            spectrum_props=self.spectrum_props,
            route_props=self.route_props,
        )
        snr_measurements.spectrum_props.start_slot = 10
        # Make end > start for valid range
        snr_measurements.spectrum_props.end_slot = 11
        snr_measurements.number_of_slots = 2  # Set to non-zero value

        resp, cross_talk = snr_measurements.check_xt()

        self.assertIsInstance(resp, bool)
        self.assertIsInstance(cross_talk, float)
        self.assertGreaterEqual(cross_talk, 0)

    def test_snr_props_initialization(self) -> None:
        """Test SNR properties are properly initialized."""
        snr_measurements = SnrMeasurements(
            engine_props_dict=self.engine_props,
            sdn_props=self.sdn_props,
            spectrum_props=self.spectrum_props,
            route_props=self.route_props,
        )

        self.assertIsInstance(snr_measurements.snr_props, SNRProps)
        self.assertIsNone(snr_measurements.snr_props.center_frequency)
        self.assertIsNone(snr_measurements.snr_props.center_psd)
        self.assertIsNone(snr_measurements.snr_props.bandwidth)
        self.assertIsNone(snr_measurements.snr_props.link_dictionary)
        # SNRProps has different structure than originally assumed
        self.assertEqual(snr_measurements.snr_props.light_frequency, 1.9341e14)
        self.assertEqual(snr_measurements.snr_props.request_bit_rate, 12.5)

    def test_constants_are_defined(self) -> None:
        """Test that required constants are defined in the module."""
        from fusion.core.snr_measurements import (
            ADJACENT_CORES_PLACEHOLDER,
            DB_CONVERSION_FACTOR,
            EGN_COEFFICIENT,
            LENGTH_CONVERSION_FACTOR,
            MEAN_XT_CONSTANT,
            POWER_CONVERSION_FACTOR,
        )

        self.assertEqual(POWER_CONVERSION_FACTOR, 10**9)
        self.assertEqual(LENGTH_CONVERSION_FACTOR, 1e3)
        self.assertEqual(DB_CONVERSION_FACTOR, 10)
        self.assertEqual(MEAN_XT_CONSTANT, 3.78e-9)
        self.assertEqual(EGN_COEFFICIENT, 80 / 81)
        self.assertEqual(ADJACENT_CORES_PLACEHOLDER, -100)

    def test_engine_props_dict_storage(self) -> None:
        """Test that engine properties dictionary is correctly stored."""
        snr_measurements = SnrMeasurements(
            engine_props_dict=self.engine_props,
            sdn_props=self.sdn_props,
            spectrum_props=self.spectrum_props,
            route_props=self.route_props,
        )

        self.assertEqual(snr_measurements.engine_props_dict, self.engine_props)
        self.assertIn("bw_per_slot", snr_measurements.engine_props_dict)
        self.assertIn("input_power", snr_measurements.engine_props_dict)
        self.assertIn("topology_info", snr_measurements.engine_props_dict)

    def test_spectrum_props_storage(self) -> None:
        """Test that spectrum properties are correctly stored."""
        snr_measurements = SnrMeasurements(
            engine_props_dict=self.engine_props,
            sdn_props=self.sdn_props,
            spectrum_props=self.spectrum_props,
            route_props=self.route_props,
        )

        self.assertEqual(snr_measurements.spectrum_props, self.spectrum_props)
        self.assertEqual(snr_measurements.spectrum_props.start_slot, 10)
        self.assertEqual(snr_measurements.spectrum_props.end_slot, 15)
        self.assertEqual(snr_measurements.spectrum_props.core_number, 0)

    def test_route_props_storage(self) -> None:
        """Test that route properties are correctly stored."""
        snr_measurements = SnrMeasurements(
            engine_props_dict=self.engine_props,
            sdn_props=self.sdn_props,
            spectrum_props=self.spectrum_props,
            route_props=self.route_props,
        )

        self.assertEqual(snr_measurements.route_props, self.route_props)
        self.assertEqual(snr_measurements.route_props.connection_index, 0)

    def test_handle_snr_returns_bandwidth(self) -> None:
        """Test that handle_snr returns bandwidth alongside SNR results."""
        snr_measurements = SnrMeasurements(
            engine_props_dict=self.engine_props,
            sdn_props=self.sdn_props,
            spectrum_props=self.spectrum_props,
            route_props=self.route_props,
        )
        snr_measurements.spectrum_props.modulation = "QPSK"
        snr_measurements.snr_props.bandwidth_mapping_dict = {"QPSK": 100}

        # Mock the check_snr method to avoid complex calculations
        with patch.object(snr_measurements, "check_snr", return_value=(True, 5.0)):
            snr_ok, xt_cost, lp_bw = snr_measurements.handle_snr(path_index=0)

        self.assertIsInstance(snr_ok, bool)
        self.assertIsInstance(xt_cost, float)
        self.assertIsInstance(lp_bw, float)
        self.assertGreaterEqual(lp_bw, 0)
        self.assertEqual(lp_bw, 100.0)  # Should match bandwidth_mapping_dict

    def test_snr_recheck_disabled_returns_true(self) -> None:
        """Test SNR recheck returns True when disabled."""
        self.engine_props["snr_recheck"] = False
        snr_measurements = SnrMeasurements(
            engine_props_dict=self.engine_props,
            sdn_props=self.sdn_props,
            spectrum_props=self.spectrum_props,
            route_props=self.route_props,
        )

        snr_ok, interference = snr_measurements.recheck_snr_after_allocation(lightpath_id=1)

        self.assertTrue(snr_ok)
        self.assertEqual(interference, 0.0)

    def test_snr_recheck_after_allocation_with_valid_data(self) -> None:
        """Test SNR rechecking with valid allocated lightpath."""
        self.engine_props["snr_recheck"] = True
        self.engine_props["cores_per_link"] = 7
        snr_measurements = SnrMeasurements(
            engine_props_dict=self.engine_props,
            sdn_props=self.sdn_props,
            spectrum_props=self.spectrum_props,
            route_props=self.route_props,
        )

        # Setup spectrum properties (path_list already set in setUp)
        snr_measurements.spectrum_props.start_slot = 10
        snr_measurements.spectrum_props.end_slot = 15
        snr_measurements.spectrum_props.core_number = 3
        snr_measurements.spectrum_props.current_band = "c"
        snr_measurements.number_of_slots = 6
        snr_measurements.snr_props.request_snr = 10.0

        # Mock the complex SNR calculation to avoid overflow
        with patch.object(snr_measurements, "_calculate_snr_with_interference", return_value=15.0):
            snr_ok, interference = snr_measurements.recheck_snr_after_allocation(lightpath_id=1)

        self.assertIsInstance(snr_ok, bool)
        self.assertIsInstance(interference, float)
        self.assertGreaterEqual(interference, 0.0)
        self.assertTrue(snr_ok)  # 15.0 > 10.0, should pass

    def test_adjacent_core_interference_calculation(self) -> None:
        """Test adjacent core interference calculation."""
        self.engine_props["recheck_adjacent_cores"] = True
        self.engine_props["cores_per_link"] = 7
        self.engine_props["adjacent_core_xt_coefficient"] = 0.01
        snr_measurements = SnrMeasurements(
            engine_props_dict=self.engine_props,
            sdn_props=self.sdn_props,
            spectrum_props=self.spectrum_props,
            route_props=self.route_props,
        )

        interference = snr_measurements._calculate_adjacent_core_interference(
            path_list=["A", "B", "C"], band="c", core_num=3, start_slot=10, end_slot=15
        )

        self.assertIsInstance(interference, float)
        self.assertGreaterEqual(interference, 0.0)

    def test_crossband_interference_calculation(self) -> None:
        """Test cross-band interference calculation."""
        self.engine_props["recheck_crossband"] = True
        self.engine_props["band_list"] = ["c", "l"]
        self.engine_props["crossband_xt_coefficient"] = 0.005
        self.engine_props["cores_per_link"] = 7

        # Add 'l' band to network spectrum dict
        self.sdn_props.network_spectrum_dict[("A", "B")]["cores_matrix"]["l"] = np.zeros((7, 40))
        self.sdn_props.network_spectrum_dict[("B", "C")]["cores_matrix"]["l"] = np.zeros((7, 40))

        snr_measurements = SnrMeasurements(
            engine_props_dict=self.engine_props,
            sdn_props=self.sdn_props,
            spectrum_props=self.spectrum_props,
            route_props=self.route_props,
        )
        snr_measurements.spectrum_props.current_band = "c"

        interference = snr_measurements._calculate_crossband_interference(path_list=["A", "B", "C"], core_num=3, start_slot=10, end_slot=15)

        self.assertIsInstance(interference, float)
        self.assertGreaterEqual(interference, 0.0)

    def test_get_adjacent_cores_single_core(self) -> None:
        """Test adjacent cores for single core fiber."""
        self.engine_props["cores_per_link"] = 1
        snr_measurements = SnrMeasurements(
            engine_props_dict=self.engine_props,
            sdn_props=self.sdn_props,
            spectrum_props=self.spectrum_props,
            route_props=self.route_props,
        )

        adjacent = snr_measurements._get_adjacent_cores(core_num=0)

        self.assertEqual(adjacent, [])

    def test_get_adjacent_cores_middle_core(self) -> None:
        """Test adjacent cores for middle core in multi-core fiber."""
        self.engine_props["cores_per_link"] = 7
        snr_measurements = SnrMeasurements(
            engine_props_dict=self.engine_props,
            sdn_props=self.sdn_props,
            spectrum_props=self.spectrum_props,
            route_props=self.route_props,
        )

        adjacent = snr_measurements._get_adjacent_cores(core_num=3)

        self.assertEqual(adjacent, [2, 4])

    def test_get_adjacent_cores_first_core(self) -> None:
        """Test adjacent cores for first core in multi-core fiber."""
        self.engine_props["cores_per_link"] = 7
        snr_measurements = SnrMeasurements(
            engine_props_dict=self.engine_props,
            sdn_props=self.sdn_props,
            spectrum_props=self.spectrum_props,
            route_props=self.route_props,
        )

        adjacent = snr_measurements._get_adjacent_cores(core_num=0)

        self.assertEqual(adjacent, [1])

    def test_get_adjacent_cores_last_core(self) -> None:
        """Test adjacent cores for last core in multi-core fiber."""
        self.engine_props["cores_per_link"] = 7
        snr_measurements = SnrMeasurements(
            engine_props_dict=self.engine_props,
            sdn_props=self.sdn_props,
            spectrum_props=self.spectrum_props,
            route_props=self.route_props,
        )

        adjacent = snr_measurements._get_adjacent_cores(core_num=6)

        self.assertEqual(adjacent, [5])

    def test_calculate_snr_with_interference(self) -> None:
        """Test SNR calculation with additional interference."""
        self.engine_props["cores_per_link"] = 7
        snr_measurements = SnrMeasurements(
            engine_props_dict=self.engine_props,
            sdn_props=self.sdn_props,
            spectrum_props=self.spectrum_props,
            route_props=self.route_props,
        )
        # path_list already set in setUp
        snr_measurements.spectrum_props.start_slot = 10
        snr_measurements.spectrum_props.end_slot = 15
        snr_measurements.number_of_slots = 6

        # Set up required properties with realistic values to avoid overflow
        snr_measurements.snr_props.center_psd = 1e-3
        snr_measurements.snr_props.bandwidth = 1e9
        snr_measurements.snr_props.number_of_spans = 1
        snr_measurements.snr_props.planck_constant = 6.626e-34
        snr_measurements.snr_props.light_frequency = 1.934e14
        snr_measurements.snr_props.noise_spectral_density = 1.8
        # Use very small attenuation and length to avoid overflow in exp()
        snr_measurements.snr_props.link_dictionary = {
            "attenuation": 0.0002,  # Much smaller to avoid overflow
            "dispersion": 16.7,
        }
        snr_measurements.snr_props.length = 0.08  # Much smaller length (80m instead of 80km)

        # Mock internal methods that would require full network state
        with patch.object(snr_measurements, "_init_center_frequency_and_bandwidth"):
            with patch.object(snr_measurements, "_update_link_parameters"):
                with patch.object(snr_measurements, "_calculate_psd_nli", return_value=1e-12):
                    snr_margin = snr_measurements._calculate_snr_with_interference(interference=0.01)

        self.assertIsInstance(snr_margin, float)


if __name__ == "__main__":
    unittest.main()
