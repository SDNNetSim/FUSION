"""
Unit tests for fusion.core.spectrum_assignment module.

This module provides comprehensive testing for the SpectrumAssignment class which
handles spectrum allocation using various strategies including best-fit, first-fit,
last-fit, and priority-based allocation.
"""

import unittest
from typing import Any, cast
from unittest.mock import MagicMock, patch

import numpy as np

from fusion.core.properties import RoutingProps, SpectrumProps
from fusion.core.spectrum_assignment import SpectrumAssignment


class TestSpectrumAssignment(unittest.TestCase):
    """Unit tests for SpectrumAssignment functionality."""

    def setUp(self) -> None:
        """Set up test fixtures with proper isolation."""
        cores_matrix = {
            "c": np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 2, -2],
                    [1, -1, 0, 0, 0, 0, 3, 3, 3, -3],
                    [0, 4, 4, 4, -4, 0, 0, 1, 1, 0],
                    [0, 4, 4, 4, -4, 0, 0, 1, 1, 0],
                    [0, 4, 4, 4, -4, 0, 0, 1, 1, 0],
                    [0, 4, 4, 4, -4, 0, 0, 1, 1, 0],
                    [0, 4, 4, 4, -4, 0, 0, 1, 1, 0],
                ]
            )
        }
        self.engine_props = {
            "cores_per_link": 7,
            "guard_slots": 1,
            "snr_type": "None",
            "band_list": ["c"],
            "allocation_method": "first_fit",
            "fixed_grid": False,
            "spectrum_priority": "BSC",
            "multi_fiber": False,
            "network": "USbackbone60",
            "l_band": 2,
            "c_band": 1,
        }
        self.sdn_props = MagicMock()
        # Explicitly set backup_path to None to prevent MagicMock from creating it
        self.sdn_props.backup_path = None
        self.sdn_props.network_spectrum_dict = {
            ("source", "dest"): {"cores_matrix": cores_matrix},
            ("dest", "source"): {"cores_matrix": cores_matrix},
            (0, 1): {"cores_matrix": cores_matrix},
            (1, 0): {"cores_matrix": cores_matrix},
        }
        self.sdn_props.modulation_formats_dict = {
            "16QAM": {"slots_needed": 2},
            "QPSK": {"slots_needed": 3},
            "64QAM": {"slots_needed": 4},
        }

        self.route_props = RoutingProps()
        self.spec_assign = SpectrumAssignment(
            engine_props=self.engine_props,
            sdn_props=self.sdn_props,
            route_props=self.route_props,
        )
        self.spec_assign.spectrum_props.slots_needed = 2
        self.spec_assign.spectrum_props.core_number = 0
        self.spec_assign.spectrum_props.path_list = [0, 1]  # Node indices as integers
        self.spec_assign.spectrum_props.cores_matrix = {
            "c": np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 2, -2],
                    [1, -1, 0, 0, 0, 0, 3, 3, 3, -3],
                    [0, 4, 4, 4, -4, 0, 0, 1, 1, 0],
                    [0, 4, 4, 4, -4, 0, 0, 1, 1, 0],
                    [0, 4, 4, 4, -4, 0, 0, 1, 1, 0],
                    [0, 4, 4, 4, -4, 0, 0, 1, 1, 0],
                    [0, 4, 4, 4, -4, 0, 0, 1, 1, 0],
                ]
            )
        }

    def test_init_with_valid_parameters_creates_instance(self) -> None:
        """Test SpectrumAssignment initialization with valid parameters."""
        self.assertIsInstance(self.spec_assign.spectrum_props, SpectrumProps)
        self.assertEqual(self.spec_assign.engine_props_dict, self.engine_props)
        self.assertEqual(self.spec_assign.sdn_props, self.sdn_props)
        self.assertEqual(self.spec_assign.route_props, self.route_props)
        self.assertIsNotNone(self.spec_assign.snr_measurements)
        self.assertIsNotNone(self.spec_assign.spectrum_helpers)

    def test_init_creates_snr_measurements_with_correct_parameters(self) -> None:
        """Test SpectrumAssignment creates SNR measurements with correct parameters."""
        self.assertEqual(
            self.spec_assign.snr_measurements.engine_props_dict, self.engine_props
        )
        self.assertEqual(self.spec_assign.snr_measurements.sdn_props, self.sdn_props)
        self.assertEqual(
            self.spec_assign.snr_measurements.spectrum_props,
            self.spec_assign.spectrum_props,
        )
        self.assertEqual(
            self.spec_assign.snr_measurements.route_props, self.route_props
        )

    def test_init_creates_spectrum_helpers_with_correct_parameters(self) -> None:
        """Test SpectrumAssignment creates spectrum helpers with correct parameters."""
        self.assertEqual(
            self.spec_assign.spectrum_helpers.engine_props, self.engine_props
        )
        self.assertEqual(self.spec_assign.spectrum_helpers.sdn_props, self.sdn_props)
        self.assertEqual(
            self.spec_assign.spectrum_helpers.spectrum_props,
            self.spec_assign.spectrum_props,
        )

    def test_allocate_best_fit_spectrum_with_valid_channels_allocates_correctly(
        self,
    ) -> None:
        """Test best-fit spectrum allocation with valid channel list."""
        channels_list = [
            {
                "link": ("source", "dest"),
                "core": 0,
                "channel": [1, 2, 3, 4, 5],
                "band": "c",
            },
            {
                "link": ("source", "dest"),
                "core": 1,
                "channel": [5, 6, 7, 8],
                "band": "l",
            },
        ]
        channels_list = sorted(
            channels_list, key=lambda d: len(cast(list, d.get("channel", [])))
        )
        with patch.object(self.spec_assign.spectrum_helpers, "check_other_links"):
            self.spec_assign._allocate_best_fit_spectrum(channels_list)

        self.assertEqual(self.spec_assign.spectrum_props.start_slot, 5)
        self.assertEqual(self.spec_assign.spectrum_props.end_slot, 7)
        self.assertEqual(self.spec_assign.spectrum_props.core_number, 1)
        self.assertEqual(self.spec_assign.spectrum_props.current_band, "l")

    def test_allocate_best_fit_spectrum_with_short_path_bypasses_check(self) -> None:
        """Test best-fit allocation with short path bypasses other link checks."""
        channels_list = [
            {
                "link": ("source", "dest"),
                "core": 0,
                "channel": [1, 2, 3],
                "band": "c",
            },
        ]
        self.spec_assign.spectrum_props.path_list = [0, 1]  # Short path

        self.spec_assign._allocate_best_fit_spectrum(channels_list)

        self.assertEqual(self.spec_assign.spectrum_props.start_slot, 1)
        self.assertTrue(self.spec_assign.spectrum_props.is_free)

    def test_allocate_best_fit_spectrum_with_incomplete_channel_skips(self) -> None:
        """Test best-fit allocation skips channels that don't have enough slots."""
        channels_list = [
            {
                "link": ("source", "dest"),
                "core": 0,
                "channel": [1, 2],  # Not enough slots for slots_needed + guard
                "band": "c",
            },
            {
                "link": ("source", "dest"),
                "core": 1,
                "channel": [5, 6, 7, 8, 9],  # Enough slots
                "band": "c",
            },
        ]
        self.spec_assign.spectrum_props.path_list = [0, 1]  # Short path

        self.spec_assign._allocate_best_fit_spectrum(channels_list)

        self.assertEqual(self.spec_assign.spectrum_props.start_slot, 5)
        self.assertEqual(self.spec_assign.spectrum_props.core_number, 1)

    def test_setup_first_last_allocation_with_forced_core_sets_correctly(self) -> None:
        """Test first/last allocation setup with forced core."""
        self.spec_assign.spectrum_props.forced_core = 2

        core_matrix, core_list, _ = self.spec_assign._setup_first_last_allocation()

        self.assertEqual(core_list, [2])
        self.assertTrue(
            np.array_equal(core_matrix[0], [[0, 4, 4, 4, -4, 0, 0, 1, 1, 0]])
        )

    def test_setup_first_last_allocation_with_priority_first_sets_order(self) -> None:
        """Test first/last allocation setup with priority first method."""
        self.spec_assign.spectrum_props.forced_core = None
        self.spec_assign.engine_props_dict["allocation_method"] = "priority_first"

        _, core_list, _ = self.spec_assign._setup_first_last_allocation()

        # Priority order: center core first, then alternating outward
        expected_order = [0, 2, 4, 1, 3, 5, 6]
        self.assertEqual(core_list, expected_order)

    def test_setup_first_last_allocation_with_default_method_uses_range(self) -> None:
        """Test first/last allocation setup with default method."""
        self.spec_assign.spectrum_props.forced_core = None
        self.spec_assign.engine_props_dict["allocation_method"] = "default"

        _, core_list, _ = self.spec_assign._setup_first_last_allocation()

        expected_list = list(
            range(0, self.spec_assign.engine_props_dict["cores_per_link"])
        )
        self.assertEqual(core_list, expected_list)

    def test_handle_first_last_allocation_with_first_fit_allocates(self) -> None:
        """Test first/last allocation handling with first fit method."""
        self.spec_assign.engine_props_dict["allocation_method"] = "first_fit"

        self.spec_assign.handle_first_last_allocation("first_fit")

        self.assertEqual(self.spec_assign.spectrum_props.start_slot, 0)
        self.assertEqual(self.spec_assign.spectrum_props.end_slot, 3)

    def test_handle_first_last_allocation_with_last_fit_allocates(self) -> None:
        """Test first/last allocation handling with last fit method."""
        self.spec_assign.engine_props_dict["allocation_method"] = "last_fit"

        self.spec_assign.handle_first_last_allocation("last_fit")

        self.assertEqual(self.spec_assign.spectrum_props.start_slot, 5)
        self.assertEqual(self.spec_assign.spectrum_props.end_slot, 8)

    def test_get_spectrum_with_valid_modulation_allocates_successfully(self) -> None:
        """Test spectrum assignment with valid modulation format."""

        def mock_get_spectrum_side_effect() -> None:
            self.spec_assign.spectrum_props.is_free = True
            self.spec_assign.spectrum_props.start_slot = 0
            self.spec_assign.spectrum_props.end_slot = 3
            self.spec_assign.spectrum_props.core_number = 0
            self.spec_assign.spectrum_props.current_band = "c"

        with (
            patch.object(
                self.spec_assign,
                "_determine_spectrum_allocation",
                side_effect=mock_get_spectrum_side_effect,
            ) as mock_get_spectrum,
            patch.object(
                self.spec_assign.snr_measurements,
                "handle_snr",
                return_value=(True, 0.5),
            ),
        ):
            self.spec_assign.spectrum_props.is_free = False
            mod_format_list = ["QPSK"]

            self.spec_assign.get_spectrum(mod_format_list, slice_bandwidth=None)

            self.assertTrue(mock_get_spectrum.called)
            self.assertTrue(self.spec_assign.spectrum_props.is_free)
            self.assertEqual(self.spec_assign.spectrum_props.modulation, "QPSK")

    def test_get_spectrum_with_failed_allocation_remains_not_free(self) -> None:
        """Test spectrum assignment with failed allocation."""
        with (
            patch.object(
                self.spec_assign, "_determine_spectrum_allocation"
            ) as mock_get_spectrum,
            patch.object(
                self.spec_assign.snr_measurements,
                "handle_snr",
                return_value=(True, 0.5),
            ),
        ):
            # Mock allocation failure - spectrum remains not free
            self.spec_assign.spectrum_props.is_free = False
            mod_format_list = ["QPSK"]

            self.spec_assign.get_spectrum(mod_format_list, slice_bandwidth=None)

            self.assertTrue(mock_get_spectrum.called)
            self.assertFalse(self.spec_assign.spectrum_props.is_free)

    def test_first_fit_with_zero_guard_slots_allocates_correctly(self) -> None:
        """Test first fit allocation when guard slots are zero."""
        self.spec_assign.engine_props_dict["guard_slots"] = 0
        self.spec_assign.spectrum_props.slots_needed = 2

        self.spec_assign.handle_first_last_allocation("first_fit")

        self.assertEqual(self.spec_assign.spectrum_props.start_slot, 0)
        self.assertEqual(self.spec_assign.spectrum_props.end_slot, 1)

    def test_first_fit_with_single_slot_no_guard_allocates_correctly(self) -> None:
        """Test first fit allocation with single slot and no guard."""
        self.spec_assign.engine_props_dict["guard_slots"] = 0
        self.spec_assign.spectrum_props.slots_needed = 1

        self.spec_assign.handle_first_last_allocation("first_fit")

        self.assertEqual(self.spec_assign.spectrum_props.start_slot, 0)
        self.assertEqual(self.spec_assign.spectrum_props.end_slot, 0)

    def test_first_fit_with_single_slot_and_guard_allocates_correctly(self) -> None:
        """Test first fit allocation with single slot and guard."""
        self.spec_assign.engine_props_dict["guard_slots"] = 1
        self.spec_assign.spectrum_props.slots_needed = 1

        self.spec_assign.handle_first_last_allocation("first_fit")

        self.assertEqual(self.spec_assign.spectrum_props.start_slot, 0)
        self.assertEqual(self.spec_assign.spectrum_props.end_slot, 2)

    def test_handle_first_last_priority_bsc_allocates_correctly(self) -> None:
        """Test first/last priority BSC allocation."""
        self.spec_assign.engine_props_dict["spectrum_priority"] = "BSC"
        self.spec_assign.engine_props_dict["allocation_method"] = "priority_first"

        self.spec_assign.handle_first_last_priority_bsc("priority_first")

        self.assertEqual(self.spec_assign.spectrum_props.start_slot, 0)
        self.assertEqual(self.spec_assign.spectrum_props.end_slot, 3)
        self.assertEqual(self.spec_assign.spectrum_props.current_band, "c")

    def test_handle_first_last_priority_band_allocates_correctly(self) -> None:
        """Test first/last priority band allocation for non-BSC."""
        self.spec_assign.engine_props_dict["spectrum_priority"] = "non-BSC"
        self.spec_assign.engine_props_dict["allocation_method"] = "priority_last"
        self.spec_assign.spectrum_props.slots_needed = 2

        def mock_check_super_channels_effect(*args: Any, **kwargs: Any) -> bool:
            self.spec_assign.spectrum_props.start_slot = 5
            self.spec_assign.spectrum_props.end_slot = 8
            self.spec_assign.spectrum_props.current_band = "c"
            return True

        with patch.object(
            self.spec_assign.spectrum_helpers,
            "check_super_channels",
            side_effect=mock_check_super_channels_effect,
        ):
            self.spec_assign.handle_first_last_priority_band("priority_last")

            self.assertEqual(self.spec_assign.spectrum_props.start_slot, 5)
            self.assertEqual(self.spec_assign.spectrum_props.end_slot, 8)
            self.assertEqual(self.spec_assign.spectrum_props.current_band, "c")

    def test_handle_first_last_priority_bsc_with_snr_external(self) -> None:
        """Test first/last priority BSC with SNR external resources."""
        self.spec_assign.engine_props_dict["spectrum_priority"] = "BSC"
        self.spec_assign.engine_props_dict["allocation_method"] = "priority_first"
        self.spec_assign.engine_props_dict["cores_per_link"] = 13
        self.spec_assign.engine_props_dict["snr_type"] = "snr_e2e_external_resources"
        self.route_props.connection_index = 0
        self.sdn_props.path_index = 0

        # Mock function that sets is_free when check_super_channels is called
        def mock_check_super_channels(*args: Any, **kwargs: Any) -> bool:
            self.spec_assign.spectrum_props.is_free = True
            self.spec_assign.spectrum_props.start_slot = 0
            self.spec_assign.spectrum_props.end_slot = 3
            self.spec_assign.spectrum_props.core_number = 0
            self.spec_assign.spectrum_props.current_band = "c"
            return True

        # Mock both SNR external check methods to avoid file loading
        with patch.object(
            self.spec_assign.snr_measurements,
            "check_snr_ext",
            return_value=(True, 25.0),  # Return valid SNR response
        ):
            with patch.object(
                self.spec_assign.snr_measurements,
                "check_snr_ext_open_slots",
                return_value=[0, 1, 2, 3, 4],  # Return valid slots
            ):
                with patch.object(
                    self.spec_assign.spectrum_helpers,
                    "check_super_channels",
                    side_effect=mock_check_super_channels,
                ) as mock_check:
                    self.spec_assign.handle_first_last_priority_bsc("priority_first")

                    mock_check.assert_called()
                    self.assertTrue(self.spec_assign.spectrum_props.is_free)

    def test_handle_first_last_priority_band_with_snr_external(self) -> None:
        """Test first/last priority band with SNR external resources."""
        self.spec_assign.engine_props_dict["spectrum_priority"] = "non-BSC"
        self.spec_assign.engine_props_dict["allocation_method"] = "priority_last"
        self.spec_assign.engine_props_dict["cores_per_link"] = 13
        self.spec_assign.engine_props_dict["snr_type"] = "snr_e2e_external_resources"
        self.route_props.connection_index = 0
        self.sdn_props.path_index = 0

        def mock_check_super_channels_effect(*args: Any, **kwargs: Any) -> bool:
            self.spec_assign.spectrum_props.is_free = True
            self.spec_assign.spectrum_props.start_slot = 5
            self.spec_assign.spectrum_props.end_slot = 8
            self.spec_assign.spectrum_props.current_band = "c"
            return True

        # Mock the SNR external check to avoid file loading
        with patch.object(
            self.spec_assign.snr_measurements,
            "check_snr_ext_open_slots",
            return_value=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],  # Return valid slots list
        ):
            with patch.object(
                self.spec_assign.spectrum_helpers,
                "check_super_channels",
                side_effect=mock_check_super_channels_effect,
            ):
                self.spec_assign.handle_first_last_priority_band("priority_last")

                self.assertTrue(self.spec_assign.spectrum_props.is_free)
                self.assertEqual(self.spec_assign.spectrum_props.start_slot, 5)
                self.assertEqual(self.spec_assign.spectrum_props.end_slot, 8)
                self.assertEqual(self.spec_assign.spectrum_props.current_band, "c")

    def test_find_best_fit_processes_channels_correctly(self) -> None:
        """Test find_best_fit processes candidate channels correctly."""
        # Setup spectrum properties with path
        self.spec_assign.spectrum_props.path_list = [0, 1]
        self.spec_assign.spectrum_props.slots_needed = 2

        with patch.object(
            self.spec_assign, "_allocate_best_fit_spectrum"
        ) as mock_allocate:
            self.spec_assign.find_best_fit()

            # Check that _allocate_best_fit_spectrum was called
            mock_allocate.assert_called_once()

            # Get the channels that were passed to _allocate_best_fit_spectrum
            called_channels = mock_allocate.call_args[1]["candidate_channels_list"]

            # Verify the channels are sorted by length (shortest first)
            for i in range(1, len(called_channels)):
                self.assertLessEqual(
                    len(called_channels[i - 1]["channel"]),
                    len(called_channels[i]["channel"]),
                )


if __name__ == "__main__":
    unittest.main()
