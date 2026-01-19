"""
Unit tests for fusion.core.properties module.

This module provides comprehensive testing for all core property classes including
RoutingProps, SpectrumProps, SDNProps, StatsProps, and SNRProps that define
the data structures used throughout the simulation.
"""

import unittest

from fusion.core.properties import (
    DEFAULT_FREQUENCY_SPACING,
    DEFAULT_INPUT_POWER,
    DEFAULT_SPAN_LENGTH,
    LIGHT_FREQUENCY_CENTER,
    PLANCK_CONSTANT,
    SNAP_KEYS_LIST,
    WORST_CASE_MCI,
    GroomingProps,
    RoutingProps,
    SDNProps,
    SNRProps,
    SpectrumProps,
    StatsProps,
)


class TestRoutingProps(unittest.TestCase):
    """Unit tests for RoutingProps class."""

    def test_init_creates_instance_with_defaults(self) -> None:
        """Test RoutingProps initialization with default values."""
        props = RoutingProps()

        self.assertEqual(props.paths_matrix, [])
        self.assertEqual(props.modulation_formats_matrix, [])
        self.assertEqual(props.weights_list, [])
        self.assertEqual(props.path_index_list, [])
        self.assertEqual(props.input_power, DEFAULT_INPUT_POWER)
        self.assertEqual(props.frequency_spacing, DEFAULT_FREQUENCY_SPACING)
        self.assertEqual(props.mci_worst, WORST_CASE_MCI)
        self.assertEqual(props.span_length, DEFAULT_SPAN_LENGTH)
        self.assertIsNone(props.max_link_length)
        self.assertIsNone(props.max_span)
        self.assertIsNone(props.connection_index)
        self.assertIsNone(props.path_index)

    def test_init_with_default_constants(self) -> None:
        """Test RoutingProps uses correct default constants."""
        props = RoutingProps()

        self.assertEqual(props.input_power, 1e-3)
        self.assertEqual(props.frequency_spacing, 12.5e9)
        self.assertEqual(props.span_length, 100.0)
        self.assertEqual(props.mci_worst, 6.3349755556585961e-27)

    def test_repr_returns_string_representation(self) -> None:
        """Test RoutingProps __repr__ returns string with all properties."""
        props = RoutingProps()
        repr_str = repr(props)

        self.assertIn("RoutingProps", repr_str)
        self.assertIn("paths_matrix", repr_str)
        self.assertIn("input_power", repr_str)
        self.assertIn("frequency_spacing", repr_str)

    def test_attributes_are_mutable(self) -> None:
        """Test RoutingProps attributes can be modified."""
        props = RoutingProps()

        props.paths_matrix = [["A", "B", "C"]]
        props.input_power = 0.002
        props.max_link_length = 1000.0
        props.connection_index = 5

        self.assertEqual(props.paths_matrix, [["A", "B", "C"]])
        self.assertEqual(props.input_power, 0.002)
        self.assertEqual(props.max_link_length, 1000.0)
        self.assertEqual(props.connection_index, 5)


class TestSpectrumProps(unittest.TestCase):
    """Unit tests for SpectrumProps class."""

    def test_init_creates_instance_with_defaults(self) -> None:
        """Test SpectrumProps initialization with default values."""
        props = SpectrumProps()

        self.assertIsNone(props.path_list)
        self.assertIsNone(props.slots_needed)
        self.assertIsNone(props.modulation)
        self.assertIsNone(props.forced_core)
        self.assertIsNone(props.core_number)
        self.assertIsNone(props.cores_matrix)
        self.assertIsNone(props.reverse_cores_matrix)
        self.assertFalse(props.is_free)
        self.assertIsNone(props.forced_index)
        self.assertIsNone(props.start_slot)
        self.assertIsNone(props.end_slot)
        self.assertIsNone(props.forced_band)
        self.assertIsNone(props.current_band)
        self.assertIsNone(props.crosstalk_cost)

    def test_repr_returns_string_representation(self) -> None:
        """Test SpectrumProps __repr__ returns string with all properties."""
        props = SpectrumProps()
        repr_str = repr(props)

        self.assertIn("SpectrumProps", repr_str)
        self.assertIn("path_list", repr_str)
        self.assertIn("is_free", repr_str)
        self.assertIn("start_slot", repr_str)

    def test_attributes_are_mutable(self) -> None:
        """Test SpectrumProps attributes can be modified."""
        props = SpectrumProps()

        props.path_list = [1, 2, 3]
        props.slots_needed = 5
        props.modulation = "QPSK"
        props.is_free = True
        props.start_slot = 10
        props.end_slot = 15

        self.assertEqual(props.path_list, [1, 2, 3])
        self.assertEqual(props.slots_needed, 5)
        self.assertEqual(props.modulation, "QPSK")
        self.assertTrue(props.is_free)
        self.assertEqual(props.start_slot, 10)
        self.assertEqual(props.end_slot, 15)


class TestSDNProps(unittest.TestCase):
    """Unit tests for SDNProps class."""

    def test_init_creates_instance_with_defaults(self) -> None:
        """Test SDNProps initialization with default values."""
        props = SDNProps()

        self.assertIsNone(props.request_id)
        self.assertIsNone(props.source)
        self.assertIsNone(props.destination)
        self.assertIsNone(props.bandwidth)
        self.assertIsNone(props.path_list)
        self.assertIsNone(props.was_routed)
        self.assertIsNone(props.block_reason)
        # stat_key_list has default values in actual implementation
        self.assertEqual(
            props.stat_key_list,
            [
                "modulation_list",
                "crosstalk_list",
                "core_list",
                "band_list",
                "start_slot_list",
                "end_slot_list",
                "lightpath_bandwidth_list",
                "lightpath_id_list",
            ],
        )
        self.assertEqual(props.bandwidth_list, [])

    def test_get_data_method_exists(self) -> None:
        """Test SDNProps has get_data method for accessing dynamic data."""
        props = SDNProps()

        # get_data should exist as a method
        self.assertTrue(hasattr(props, "get_data"))
        self.assertTrue(callable(props.get_data))

    def test_repr_returns_string_representation(self) -> None:
        """Test SDNProps __repr__ returns string with all properties."""
        props = SDNProps()
        repr_str = repr(props)

        self.assertIn("SDNProps", repr_str)
        self.assertIn("request_id", repr_str)
        self.assertIn("was_routed", repr_str)

    def test_attributes_are_mutable(self) -> None:
        """Test SDNProps attributes can be modified."""
        props = SDNProps()

        props.request_id = 123
        props.source = "A"
        props.destination = "B"
        props.bandwidth = 100.0
        props.was_routed = True
        props.block_reason = "congestion"

        self.assertEqual(props.request_id, 123)
        self.assertEqual(props.source, "A")
        self.assertEqual(props.destination, "B")
        self.assertEqual(props.bandwidth, 100.0)
        self.assertTrue(props.was_routed)
        self.assertEqual(props.block_reason, "congestion")


class TestStatsProps(unittest.TestCase):
    """Unit tests for StatsProps class."""

    def test_init_creates_instance_with_empty_collections(self) -> None:
        """Test StatsProps initialization with empty collections."""
        props = StatsProps()

        self.assertEqual(props.simulation_blocking_list, [])
        self.assertEqual(props.simulation_bitrate_blocking_list, [])
        self.assertEqual(props.transponders_list, [])
        self.assertEqual(props.hops_list, [])
        self.assertEqual(props.lengths_list, [])
        self.assertEqual(props.route_times_list, [])
        self.assertEqual(props.cores_dict, {})
        # block_reasons_dict has default None values in actual implementation
        self.assertEqual(
            props.block_reasons_dict,
            {"distance": None, "congestion": None, "xt_threshold": None},
        )
        self.assertEqual(props.snapshots_dict, {})
        self.assertEqual(props.weights_dict, {})
        self.assertEqual(props.modulations_used_dict, {})
        self.assertEqual(props.bandwidth_blocking_dict, {})
        self.assertEqual(props.link_usage_dict, {})
        self.assertEqual(props.path_index_list, [])

    def test_repr_returns_string_representation(self) -> None:
        """Test StatsProps __repr__ returns string with all properties."""
        props = StatsProps()
        repr_str = repr(props)

        self.assertIn("StatsProps", repr_str)
        self.assertIn("simulation_blocking_list", repr_str)

    def test_collections_are_mutable(self) -> None:
        """Test StatsProps collections can be modified."""
        props = StatsProps()

        props.simulation_blocking_list.append(0.1)
        props.transponders_list.extend([1, 2, 3])
        props.cores_dict[1] = 42
        props.block_reasons_dict["congestion"] = 10

        self.assertEqual(props.simulation_blocking_list, [0.1])
        self.assertEqual(props.transponders_list, [1, 2, 3])
        self.assertEqual(props.cores_dict[1], 42)
        self.assertEqual(props.block_reasons_dict["congestion"], 10)

    def test_nested_dict_structure_support(self) -> None:
        """Test StatsProps supports nested dictionary structures."""
        props = StatsProps()

        # Test nested weights_dict structure
        props.weights_dict["50GHz"] = {"QPSK": [1.0, 2.0, 3.0]}
        props.modulations_used_dict["QPSK"] = {
            "C": 5,
            "length": {"C": [100, 200], "overall": [300]},
        }

        self.assertEqual(props.weights_dict["50GHz"]["QPSK"], [1.0, 2.0, 3.0])
        self.assertEqual(props.modulations_used_dict["QPSK"]["C"], 5)
        self.assertEqual(props.modulations_used_dict["QPSK"]["length"]["C"], [100, 200])


class TestSNRProps(unittest.TestCase):
    """Unit tests for SNRProps class."""

    def test_init_creates_instance_with_defaults(self) -> None:
        """Test SNRProps initialization with default values."""
        props = SNRProps()

        self.assertIsNone(props.center_frequency)
        self.assertIsNone(props.center_psd)
        self.assertIsNone(props.bandwidth)
        self.assertIsNone(props.link_dictionary)
        # SNRProps doesn't have adjacent_cores, it has other attributes
        self.assertEqual(props.light_frequency, 1.9341e14)
        self.assertEqual(props.planck_constant, 6.62607004e-34)
        self.assertEqual(props.request_bit_rate, 12.5)
        self.assertEqual(props.request_snr, 8.5)

    def test_repr_returns_string_representation(self) -> None:
        """Test SNRProps __repr__ returns string with all properties."""
        props = SNRProps()
        repr_str = repr(props)

        self.assertIn("SNRProps", repr_str)
        self.assertIn("center_frequency", repr_str)
        self.assertIn("light_frequency", repr_str)

    def test_attributes_are_mutable(self) -> None:
        """Test SNRProps attributes can be modified."""
        props = SNRProps()

        props.center_frequency = 193.1e12
        props.center_psd = 1e-3
        props.bandwidth = 12.5e9
        props.link_dictionary = {"attenuation": 0.2, "dispersion": 16.7}
        props.request_bit_rate = 25.0

        self.assertEqual(props.center_frequency, 193.1e12)
        self.assertEqual(props.center_psd, 1e-3)
        self.assertEqual(props.bandwidth, 12.5e9)
        self.assertEqual(props.link_dictionary["attenuation"], 0.2)
        self.assertEqual(props.request_bit_rate, 25.0)


class TestPropertiesValidation(unittest.TestCase):
    """Unit tests for properties validation and edge cases."""

    def test_routing_props_with_edge_case_values(self) -> None:
        """Test RoutingProps handles edge case values."""
        props = RoutingProps()

        # Test negative values (invalid but should be maintained)
        props.input_power = -1.0
        props.frequency_spacing = -12.5e9
        props.span_length = -100.0

        self.assertEqual(props.input_power, -1.0)
        self.assertEqual(props.frequency_spacing, -12.5e9)
        self.assertEqual(props.span_length, -100.0)

    def test_spectrum_props_with_negative_slots(self) -> None:
        """Test SpectrumProps handles negative slot values."""
        props = SpectrumProps()

        props.slots_needed = -5
        props.start_slot = -10
        props.end_slot = -1
        props.core_number = -1

        self.assertEqual(props.slots_needed, -5)
        self.assertEqual(props.start_slot, -10)
        self.assertEqual(props.end_slot, -1)
        self.assertEqual(props.core_number, -1)

    def test_sdn_props_with_empty_collections(self) -> None:
        """Test SDNProps maintains empty collections correctly."""
        props = SDNProps()

        props.path_list = []
        props.stat_key_list = []
        props.bandwidth_list = []

        self.assertEqual(props.path_list, [])
        self.assertEqual(props.stat_key_list, [])
        self.assertEqual(props.bandwidth_list, [])
        self.assertIsInstance(props.path_list, list)

    def test_stats_props_with_none_values(self) -> None:
        """Test StatsProps handles None values in collections."""
        props = StatsProps()

        props.simulation_blocking_list = [0.0, 0.1, 0.0]
        props.cores_dict = {1: 0, 2: 5}

        self.assertEqual(props.simulation_blocking_list, [0.0, 0.1, 0.0])
        self.assertEqual(props.cores_dict[1], 0)
        self.assertEqual(props.cores_dict[2], 5)

    def test_snr_props_with_zero_values(self) -> None:
        """Test SNRProps handles zero values correctly."""
        props = SNRProps()

        props.center_frequency = 0.0
        props.center_psd = 0.0
        props.bandwidth = 0.0

        self.assertEqual(props.center_frequency, 0.0)
        self.assertEqual(props.center_psd, 0.0)
        self.assertEqual(props.bandwidth, 0.0)


class TestGroomingProps(unittest.TestCase):
    """Unit tests for GroomingProps class."""

    def test_init_creates_instance_with_defaults(self) -> None:
        """Test GroomingProps initialization with default values."""
        props = GroomingProps()

        self.assertIsNone(props.grooming_type)
        self.assertIsNone(props.lightpath_status_dict)

    def test_repr_returns_string_representation(self) -> None:
        """Test GroomingProps __repr__ returns string with all properties."""
        props = GroomingProps()
        repr_str = repr(props)

        self.assertIn("GroomingProps", repr_str)
        self.assertIn("grooming_type", repr_str)
        self.assertIn("lightpath_status_dict", repr_str)

    def test_attributes_are_mutable(self) -> None:
        """Test GroomingProps attributes can be modified."""
        props = GroomingProps()

        props.grooming_type = "end_to_end"
        props.lightpath_status_dict = {("A", "B"): {1: {"bandwidth": 100}}}

        self.assertEqual(props.grooming_type, "end_to_end")
        self.assertEqual(
            props.lightpath_status_dict, {("A", "B"): {1: {"bandwidth": 100}}}
        )


class TestSDNPropsGrooming(unittest.TestCase):
    """Unit tests for SDNProps grooming-related attributes."""

    def test_sdn_props_has_grooming_attributes(self) -> None:
        """Test SDNProps has all grooming-related attributes."""
        props = SDNProps()

        # Lightpath tracking dictionaries
        self.assertTrue(hasattr(props, "lightpath_status_dict"))
        self.assertTrue(hasattr(props, "transponder_usage_dict"))
        self.assertTrue(hasattr(props, "lp_bw_utilization_dict"))

        # Grooming state flags
        self.assertTrue(hasattr(props, "was_groomed"))
        self.assertTrue(hasattr(props, "was_partially_groomed"))
        self.assertTrue(hasattr(props, "was_partially_routed"))
        self.assertTrue(hasattr(props, "was_new_lp_established"))

        # Lightpath resource tracking
        self.assertTrue(hasattr(props, "lightpath_id_list"))
        self.assertTrue(hasattr(props, "lightpath_bandwidth_list"))
        self.assertTrue(hasattr(props, "remaining_bw"))

        # Lightpath ID counter
        self.assertTrue(hasattr(props, "lightpath_counter"))

    def test_get_lightpath_id_increments_counter(self) -> None:
        """Test get_lightpath_id() increments counter and returns unique IDs."""
        props = SDNProps()

        lp_id_1 = props.get_lightpath_id()
        lp_id_2 = props.get_lightpath_id()
        lp_id_3 = props.get_lightpath_id()

        self.assertEqual(lp_id_1, 1)
        self.assertEqual(lp_id_2, 2)
        self.assertEqual(lp_id_3, 3)
        self.assertEqual(props.lightpath_counter, 3)

    def test_reset_lightpath_id_counter_resets_to_zero(self) -> None:
        """Test reset_lightpath_id_counter() resets counter to zero."""
        props = SDNProps()

        # Generate some IDs
        props.get_lightpath_id()
        props.get_lightpath_id()
        props.get_lightpath_id()
        self.assertEqual(props.lightpath_counter, 3)

        # Reset
        props.reset_lightpath_id_counter()
        self.assertEqual(props.lightpath_counter, 0)

        # Next ID should be 1
        lp_id = props.get_lightpath_id()
        self.assertEqual(lp_id, 1)

    def test_reset_params_clears_grooming_fields(self) -> None:
        """Test reset_params() clears grooming-related fields."""
        props = SDNProps()

        # Set some grooming state
        props.lightpath_id_list = [1, 2, 3]
        props.lightpath_bandwidth_list = [100, 200, 150]
        props.was_groomed = True
        props.was_partially_groomed = True
        props.was_new_lp_established = [1, 2]
        props.remaining_bw = 50

        # Reset
        props.reset_params()

        # Verify grooming fields are cleared
        self.assertEqual(props.lightpath_id_list, [])
        self.assertEqual(props.lightpath_bandwidth_list, [])
        self.assertIsNone(props.was_groomed)
        self.assertFalse(props.was_partially_groomed)
        self.assertEqual(props.was_new_lp_established, [])
        self.assertIsNone(props.remaining_bw)

    def test_stat_key_list_includes_grooming_keys(self) -> None:
        """Test stat_key_list includes grooming-related keys."""
        props = SDNProps()

        self.assertIn("lightpath_bandwidth_list", props.stat_key_list)
        self.assertIn("lightpath_id_list", props.stat_key_list)
        # Note: remaining_bw is a scalar, not a list, so it's not in stat_key_list


class TestSpectrumPropsGrooming(unittest.TestCase):
    """Unit tests for SpectrumProps grooming-related attributes."""

    def test_spectrum_props_has_lightpath_tracking(self) -> None:
        """Test SpectrumProps has lightpath ID and bandwidth attributes."""
        props = SpectrumProps()

        self.assertTrue(hasattr(props, "lightpath_id"))
        self.assertTrue(hasattr(props, "lightpath_bandwidth"))

    def test_lightpath_id_can_be_set(self) -> None:
        """Test lightpath_id can be set and retrieved."""
        props = SpectrumProps()

        props.lightpath_id = 42
        self.assertEqual(props.lightpath_id, 42)

    def test_lightpath_bandwidth_can_be_set(self) -> None:
        """Test lightpath_bandwidth can be set and retrieved."""
        props = SpectrumProps()

        props.lightpath_bandwidth = 200.0
        self.assertEqual(props.lightpath_bandwidth, 200.0)


class TestModuleConstants(unittest.TestCase):
    """Unit tests for module-level constants."""

    def test_physical_constants_are_defined(self) -> None:
        """Test physical constants are properly defined."""
        self.assertEqual(PLANCK_CONSTANT, 6.62607004e-34)
        self.assertEqual(LIGHT_FREQUENCY_CENTER, 1.9341e14)

    def test_network_constants_are_defined(self) -> None:
        """Test network constants are properly defined."""
        self.assertEqual(DEFAULT_INPUT_POWER, 1e-3)
        self.assertEqual(DEFAULT_FREQUENCY_SPACING, 12.5e9)
        self.assertEqual(DEFAULT_SPAN_LENGTH, 100.0)
        self.assertEqual(WORST_CASE_MCI, 6.3349755556585961e-27)

    def test_snap_keys_list_contains_expected_keys(self) -> None:
        """Test SNAP_KEYS_LIST contains expected snapshot keys."""
        expected_keys = [
            "occupied_slots",
            "guard_slots",
            "active_requests",
            "blocking_prob",
            "num_segments",
        ]

        self.assertEqual(SNAP_KEYS_LIST, expected_keys)
        self.assertEqual(len(SNAP_KEYS_LIST), 5)

    def test_constants_are_correct_types(self) -> None:
        """Test constants have correct types."""
        self.assertIsInstance(PLANCK_CONSTANT, float)
        self.assertIsInstance(LIGHT_FREQUENCY_CENTER, float)
        self.assertIsInstance(DEFAULT_INPUT_POWER, float)
        self.assertIsInstance(DEFAULT_FREQUENCY_SPACING, float)
        self.assertIsInstance(DEFAULT_SPAN_LENGTH, float)
        self.assertIsInstance(WORST_CASE_MCI, float)
        self.assertIsInstance(SNAP_KEYS_LIST, list)

    def test_constants_have_positive_values(self) -> None:
        """Test physical constants have positive values."""
        self.assertGreater(PLANCK_CONSTANT, 0)
        self.assertGreater(LIGHT_FREQUENCY_CENTER, 0)
        self.assertGreater(DEFAULT_INPUT_POWER, 0)
        self.assertGreater(DEFAULT_FREQUENCY_SPACING, 0)
        self.assertGreater(DEFAULT_SPAN_LENGTH, 0)
        self.assertGreater(WORST_CASE_MCI, 0)


if __name__ == "__main__":
    unittest.main()
