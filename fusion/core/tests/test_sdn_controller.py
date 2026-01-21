"""
Unit tests for fusion.core.sdn_controller module.

This module provides comprehensive testing for the SDNController class which manages
network requests, routing, spectrum allocation, and resource management in
software-defined optical networks.
"""

import unittest
from typing import Any
from unittest.mock import patch

import networkx as nx
import numpy as np

from fusion.core.properties import SDNProps
from fusion.core.sdn_controller import SDNController


class TestSDNController(unittest.TestCase):
    """Unit tests for SDNController functionality."""

    def setUp(self) -> None:
        """Set up test fixtures with proper isolation."""
        # Create a simple test topology
        topology = nx.Graph()
        topology.add_edge("A", "B", weight=1, length=10)
        topology.add_edge("B", "C", weight=1, length=10)
        topology.add_edge("A", "C", weight=2, length=20)

        self.engine_props = {
            "cores_per_link": 7,
            "guard_slots": 1,
            "route_method": "shortest_path",
            "max_segments": 1,
            "band_list": ["c"],
            "topology": topology,
        }
        self.controller = SDNController(engine_props=self.engine_props)

        # Mock SDNProps properly
        self.controller.sdn_props = SDNProps()
        self.controller.slicing_manager.sdn_props = self.controller.sdn_props
        self.controller.sdn_props.path_list = [0, 1, 2]
        self.controller.sdn_props.request_id = 1
        self.controller.sdn_props.network_spectrum_dict = {
            ("A", "B"): {
                "cores_matrix": {"c": np.zeros((7, 10))},
                "usage_count": 0,
                "throughput": 0,
            },
            ("B", "A"): {
                "cores_matrix": {"c": np.zeros((7, 10))},
                "usage_count": 0,
                "throughput": 0,
            },
            ("B", "C"): {
                "cores_matrix": {"c": np.zeros((7, 10))},
                "usage_count": 0,
                "throughput": 0,
            },
            ("C", "B"): {
                "cores_matrix": {"c": np.zeros((7, 10))},
                "usage_count": 0,
                "throughput": 0,
            },
            ("A", "C"): {
                "cores_matrix": {"c": np.zeros((7, 10))},
                "usage_count": 0,
                "throughput": 0,
            },
            ("C", "A"): {
                "cores_matrix": {"c": np.zeros((7, 10))},
                "usage_count": 0,
                "throughput": 0,
            },
            # Add numeric keys for backward compatibility with some tests
            (0, 1): {
                "cores_matrix": {"c": np.zeros((7, 10))},
                "usage_count": 0,
                "throughput": 0,
            },
            (1, 0): {
                "cores_matrix": {"c": np.zeros((7, 10))},
                "usage_count": 0,
                "throughput": 0,
            },
            (1, 2): {
                "cores_matrix": {"c": np.zeros((7, 10))},
                "usage_count": 0,
                "throughput": 0,
            },
            (2, 1): {
                "cores_matrix": {"c": np.zeros((7, 10))},
                "usage_count": 0,
                "throughput": 0,
            },
            (0, 2): {
                "cores_matrix": {"c": np.zeros((7, 10))},
                "usage_count": 0,
                "throughput": 0,
            },
            (2, 0): {
                "cores_matrix": {"c": np.zeros((7, 10))},
                "usage_count": 0,
                "throughput": 0,
            },
        }

    def test_init_with_valid_parameters_creates_instance(self) -> None:
        """Test SDNController initialization with valid parameters."""
        controller = SDNController(engine_props=self.engine_props)

        self.assertEqual(controller.engine_props, self.engine_props)
        self.assertIsInstance(controller.sdn_props, SDNProps)
        self.assertIsNotNone(controller.route_obj)
        self.assertIsNotNone(controller.spectrum_obj)
        self.assertIsNotNone(controller.slicing_manager)
        self.assertIsNotNone(controller.grooming_obj)
        self.assertIsNone(controller.ai_obj)

    def test_init_creates_proper_component_relationships(self) -> None:
        """Test SDNController initialization creates proper component relationships."""
        controller = SDNController(engine_props=self.engine_props)

        # Check that spectrum assignment gets route props
        self.assertEqual(controller.spectrum_obj.route_props, controller.route_obj.route_props)
        # Check that slicing manager gets proper references
        self.assertEqual(controller.slicing_manager.sdn_props, controller.sdn_props)
        self.assertEqual(controller.slicing_manager.spectrum_obj, controller.spectrum_obj)

    def test_allocate_with_valid_spectrum_allocates_correctly(self) -> None:
        """Test spectrum allocation with valid spectrum properties."""
        self.controller.spectrum_obj.spectrum_props.start_slot = 0
        self.controller.spectrum_obj.spectrum_props.end_slot = 3
        self.controller.spectrum_obj.spectrum_props.core_number = 0
        self.controller.spectrum_obj.spectrum_props.current_band = "c"
        self.controller.spectrum_obj.spectrum_props.lightpath_id = 1

        self.controller.allocate()

        assert self.controller.sdn_props.path_list is not None
        assert self.controller.sdn_props.network_spectrum_dict is not None
        assert self.controller.sdn_props.request_id is not None
        path_list = self.controller.sdn_props.path_list
        network_dict = self.controller.sdn_props.network_spectrum_dict

        for link in zip(path_list, path_list[1:], strict=False):
            link_tuple = (link[0], link[1])
            core_matrix = network_dict[link_tuple]["cores_matrix"]["c"][0]
            self.assertTrue(np.all(core_matrix[:2] == self.controller.sdn_props.request_id))
            self.assertEqual(core_matrix[2], self.controller.sdn_props.request_id * -1)

    def test_allocate_without_guard_slots_allocates_correctly(self) -> None:
        """Test spectrum allocation without guard slots."""
        self.controller.spectrum_obj.spectrum_props.start_slot = 0
        self.controller.spectrum_obj.spectrum_props.end_slot = 2
        self.controller.spectrum_obj.spectrum_props.core_number = 0
        self.controller.spectrum_obj.spectrum_props.current_band = "c"
        self.controller.spectrum_obj.spectrum_props.lightpath_id = 1
        self.controller.engine_props["guard_slots"] = 0

        self.controller.allocate()

        assert self.controller.sdn_props.path_list is not None
        assert self.controller.sdn_props.network_spectrum_dict is not None
        assert self.controller.sdn_props.request_id is not None
        path_list = self.controller.sdn_props.path_list
        network_dict = self.controller.sdn_props.network_spectrum_dict

        for link in zip(path_list, path_list[1:], strict=False):
            link_tuple = (link[0], link[1])
            core_matrix = network_dict[link_tuple]["cores_matrix"]["c"][0]
            # Check allocation without guard band
            self.assertTrue(np.all(core_matrix[:3] == self.controller.sdn_props.request_id))
            # Ensure no guard band is allocated
            self.assertNotIn(self.controller.sdn_props.request_id * -1, core_matrix)

    def test_allocate_updates_usage_count(self) -> None:
        """Test that allocation increments usage count."""
        self.controller.spectrum_obj.spectrum_props.start_slot = 0
        self.controller.spectrum_obj.spectrum_props.end_slot = 3
        self.controller.spectrum_obj.spectrum_props.core_number = 0
        self.controller.spectrum_obj.spectrum_props.current_band = "c"
        self.controller.spectrum_obj.spectrum_props.lightpath_id = 1

        # Initialize usage counts
        assert self.controller.sdn_props.network_spectrum_dict is not None
        network_dict = self.controller.sdn_props.network_spectrum_dict
        for key in network_dict:
            network_dict[key]["usage_count"] = 0

        self.controller.allocate()

        assert self.controller.sdn_props.path_list is not None
        path_list = self.controller.sdn_props.path_list

        for link in zip(path_list, path_list[1:], strict=False):
            link_tuple = (link[0], link[1])
            self.assertEqual(network_dict[link_tuple]["usage_count"], 1)
            reverse_link = (link_tuple[1], link_tuple[0])
            self.assertEqual(network_dict[reverse_link]["usage_count"], 1)

    def test_release_with_allocated_spectrum_deallocates_correctly(self) -> None:
        """Test spectrum release with previously allocated spectrum."""
        # First allocate spectrum
        assert self.controller.sdn_props.network_spectrum_dict is not None
        spectrum_dict = self.controller.sdn_props.network_spectrum_dict
        spectrum_dict[(0, 1)]["cores_matrix"]["c"][0][:3] = 1
        spectrum_dict[(0, 1)]["cores_matrix"]["c"][0][3] = -1

        self.controller.release(lightpath_id=1)

        assert self.controller.sdn_props.path_list is not None
        path_list = self.controller.sdn_props.path_list

        for link in zip(path_list, path_list[1:], strict=False):
            link_tuple = (link[0], link[1])
            for core_num in range(self.engine_props["cores_per_link"]):
                core_arr = spectrum_dict[link_tuple]["cores_matrix"]["c"][core_num]
                self.assertTrue(np.all(core_arr[:4] == 0))

    def test_release_with_empty_path_list_handles_gracefully(self) -> None:
        """Test release handles empty path list gracefully."""
        self.controller.sdn_props.path_list = []  # Empty list, not None

        # Should not raise exception
        self.controller.release(lightpath_id=1)

    def test_release_updates_throughput_tracking(self) -> None:
        """Test that _update_throughput correctly tracks throughput."""
        assert self.controller.sdn_props.network_spectrum_dict is not None
        network_dict = self.controller.sdn_props.network_spectrum_dict
        for key in network_dict:
            network_dict[key]["throughput"] = 0

        self.controller.sdn_props.arrive = 0
        self.controller.sdn_props.depart = 5  # 5 seconds duration
        self.controller.sdn_props.bandwidth = 100.0  # Gbps

        # _update_throughput is called by _handle_release, not release()
        self.controller._update_throughput()
        self.controller.release(lightpath_id=1)

        assert self.controller.sdn_props.path_list is not None
        path_list = self.controller.sdn_props.path_list
        expected_throughput = 500  # 100 Gbps * 5 seconds
        for link in zip(path_list, path_list[1:], strict=False):
            link_tuple = (link[0], link[1])
            self.assertEqual(network_dict[link_tuple]["throughput"], expected_throughput)
            reverse_link = (link_tuple[1], link_tuple[0])
            self.assertEqual(network_dict[reverse_link]["throughput"], expected_throughput)

    def test_release_with_missing_timing_fields_handles_gracefully(self) -> None:
        """Test that release handles missing timing fields gracefully."""
        assert self.controller.sdn_props.network_spectrum_dict is not None
        network_dict = self.controller.sdn_props.network_spectrum_dict
        for key in network_dict:
            network_dict[key]["throughput"] = 0

        self.controller.sdn_props.arrive = None
        self.controller.sdn_props.depart = None
        self.controller.sdn_props.bandwidth = None

        # Should not raise exception
        self.controller.release(lightpath_id=1)

        for key in network_dict:
            self.assertEqual(network_dict[key]["throughput"], 0)

    def test_update_req_stats_with_valid_data_updates_correctly(self) -> None:
        """Test request statistics update with valid data."""
        self.controller.sdn_props.crosstalk_list = []
        self.controller.sdn_props.core_list = []
        self.controller.spectrum_obj.spectrum_props.crosstalk_cost = 10
        self.controller.spectrum_obj.spectrum_props.core_number = 2
        self.controller.sdn_props.stat_key_list = ["crosstalk_list", "core_list"]

        self.controller._update_req_stats(bandwidth=100.0)

        self.assertIn(100.0, self.controller.sdn_props.bandwidth_list)
        self.assertIn(10, self.controller.sdn_props.crosstalk_list)
        self.assertIn(2, self.controller.sdn_props.core_list)

    def test_update_req_stats_with_missing_stat_keys_handles_gracefully(self) -> None:
        """Test request statistics update with missing stat keys."""
        self.controller.sdn_props.stat_key_list = []

        # Should not raise exception
        self.controller._update_req_stats(bandwidth=100.0)

        self.assertIn(100.0, self.controller.sdn_props.bandwidth_list)

    @patch("fusion.core.sdn_controller.SDNController.allocate")
    @patch("fusion.core.sdn_controller.SDNController._update_req_stats")
    @patch("fusion.core.spectrum_assignment.SpectrumAssignment.get_spectrum")
    def test_allocate_slicing_with_free_spectrum_allocates_successfully(
        self, mock_get_spectrum: Any, mock_update_req_stats: Any, mock_allocate: Any
    ) -> None:
        """Test slicing allocation with available spectrum."""
        self.controller.spectrum_obj.spectrum_props.is_free = True
        self.controller.sdn_props.bandwidth = 100  # Required for remaining_bw calculation
        mock_get_spectrum.return_value = None

        self.controller._allocate_slicing(num_segments=2, mod_format="QPSK", path_list=[0, 1], bandwidth="50.0")

        mock_get_spectrum.assert_called()
        mock_allocate.assert_called()
        # Slicing calls _update_req_stats with both bandwidth and remaining params
        mock_update_req_stats.assert_called_with(bandwidth="50.0", remaining="0")

    @patch("fusion.core.sdn_controller.SDNController.allocate")
    @patch("fusion.core.sdn_controller.SDNController._update_req_stats")
    @patch("fusion.core.routing.Routing.get_route")
    @patch("fusion.core.spectrum_assignment.SpectrumAssignment.get_spectrum")
    def test_handle_event_arrival_with_successful_allocation(
        self,
        mock_get_spectrum: Any,
        mock_get_route: Any,
        mock_update_stats: Any,
        mock_allocate: Any,
    ) -> None:
        """Test handle event with arrival request and successful allocation."""
        mock_get_route.return_value = None
        self.controller.route_obj.route_props.paths_matrix = [[0, 1, 2]]
        self.controller.route_obj.route_props.modulation_formats_matrix = [["QPSK"]]
        self.controller.route_obj.route_props.weights_list = [10]

        self.controller.spectrum_obj.spectrum_props.is_free = True
        mock_get_spectrum.return_value = None

        self.controller.handle_event({}, request_type="arrival")

        mock_allocate.assert_called_once()
        self.assertTrue(self.controller.sdn_props.was_routed)

    def test_handle_dynamic_slicing_with_valid_parameters(self) -> None:
        """Test dynamic slicing with valid parameters."""
        self.controller.sdn_props.bandwidth = 100.0
        self.controller.engine_props["mod_per_bw"] = {
            "50": {"QPSK": {"max_length": 100}},
            "100": {"16QAM": {"max_length": 200}},
        }
        self.controller.engine_props["fixed_grid"] = True
        self.controller.engine_props["topology"] = {
            (0, 1): {"length": 50},
            (1, 2): {"length": 50},
        }
        self.controller.sdn_props.path_list = [0, 1, 2]
        self.controller.sdn_props.number_of_transponders = 0
        self.controller.spectrum_obj.spectrum_props.is_free = True

        with (
            patch(
                "fusion.core.spectrum_assignment.SpectrumAssignment.get_spectrum_dynamic_slicing",
                return_value=("16QAM", 50),
            ) as mock_get_spectrum,
            patch.object(self.controller, "allocate") as mock_allocate,
            patch.object(self.controller, "_update_req_stats") as mock_update_stats,
            patch(
                "fusion.modules.spectrum.light_path_slicing.find_path_len",
                return_value=100,
            ) as mock_find_path_len,
        ):
            self.controller._handle_dynamic_slicing(path_list=["A", "B", "C"], path_index=0, forced_segments=1)

            mock_get_spectrum.assert_called()
            mock_allocate.assert_called()
            mock_update_stats.assert_called_with(bandwidth="50")
            mock_find_path_len.assert_called_with(
                path_list=["A", "B", "C"],
                topology=self.controller.engine_props["topology"],
            )

            self.assertEqual(self.controller.sdn_props.number_of_transponders, 2)
            self.assertTrue(self.controller.sdn_props.is_sliced)

    def test_handle_dynamic_slicing_with_congestion(self) -> None:
        """Test dynamic slicing handles congestion correctly."""
        self.controller.sdn_props.bandwidth = 100.0
        self.controller.engine_props["mod_per_bw"] = {
            "50": {"QPSK": {"max_length": 100}},
            "100": {"16QAM": {"max_length": 200}},
        }
        self.controller.engine_props["fixed_grid"] = True
        self.controller.engine_props["topology"] = {
            (0, 1): {"length": 50},
            (1, 2): {"length": 50},
        }
        self.controller.sdn_props.path_list = [0, 1, 2]
        self.controller.sdn_props.number_of_transponders = 0

        with (
            patch(
                "fusion.core.spectrum_assignment.SpectrumAssignment.get_spectrum_dynamic_slicing",
                return_value=("16QAM", 50),
            ) as mock_get_spectrum,
            patch.object(self.controller, "allocate") as mock_allocate,
            patch(
                "fusion.modules.spectrum.light_path_slicing.find_path_len",
                return_value=100,
            ) as mock_find_path_len,
        ):
            with patch.object(
                self.controller,
                "_handle_congestion",
                wraps=self.controller._handle_congestion,
            ) as mock_handle_congestion:
                # Simulate no free spectrum
                self.controller.spectrum_obj.spectrum_props.is_free = False

                self.controller._handle_dynamic_slicing(path_list=["A", "B", "C"], path_index=0, forced_segments=1)

                mock_get_spectrum.assert_called()
                mock_handle_congestion.assert_called_with(remaining_bw=100)
                mock_allocate.assert_not_called()
                mock_find_path_len.assert_called_with(
                    path_list=["A", "B", "C"],
                    topology=self.controller.engine_props["topology"],
                )

                self.assertFalse(self.controller.sdn_props.was_routed)
                self.assertEqual(self.controller.sdn_props.block_reason, "congestion")
                self.assertFalse(self.controller.sdn_props.is_sliced)

    def test_sdn_controller_grooming_init(self) -> None:
        """Test SDN controller has grooming object."""
        controller = SDNController(engine_props=self.engine_props)
        self.assertIsNotNone(controller.grooming_obj)
        self.assertTrue(hasattr(controller, "grooming_obj"))

    def test_release_with_lightpath_id(self) -> None:
        """Test release with specific lightpath ID."""
        # Setup network with lightpath ID allocation
        assert self.controller.sdn_props.network_spectrum_dict is not None
        spectrum_dict = self.controller.sdn_props.network_spectrum_dict
        lightpath_id = 100
        spectrum_dict[(0, 1)]["cores_matrix"]["c"][0][:3] = lightpath_id
        spectrum_dict[(1, 0)]["cores_matrix"]["c"][0][:3] = lightpath_id
        spectrum_dict[(1, 2)]["cores_matrix"]["c"][0][:3] = lightpath_id
        spectrum_dict[(2, 1)]["cores_matrix"]["c"][0][:3] = lightpath_id

        self.controller.release(lightpath_id=lightpath_id)

        # Verify lightpath ID was released
        assert self.controller.sdn_props.path_list is not None
        path_list = self.controller.sdn_props.path_list
        for link in zip(path_list, path_list[1:], strict=False):
            link_tuple = (link[0], link[1])
            core_arr = spectrum_dict[link_tuple]["cores_matrix"]["c"][0]
            self.assertTrue(np.all(core_arr[:3] == 0))

    def test_release_with_slicing_flag(self) -> None:
        """Test release with slicing_flag (spectrum only)."""
        # Setup network with lightpath ID allocation
        assert self.controller.sdn_props.network_spectrum_dict is not None
        spectrum_dict = self.controller.sdn_props.network_spectrum_dict
        lightpath_id = 200
        spectrum_dict[(0, 1)]["cores_matrix"]["c"][0][:3] = lightpath_id
        spectrum_dict[(1, 0)]["cores_matrix"]["c"][0][:3] = lightpath_id

        # Mock transponder dict and lightpath status
        self.controller.sdn_props.source = "0"
        self.controller.sdn_props.destination = "2"
        self.controller.sdn_props.transponder_usage_dict = {
            "0": {"available_transponder": 5},
            "2": {"available_transponder": 5},
        }
        self.controller.sdn_props.lightpath_status_dict = {}

        # Release with slicing flag (should not call _release_lightpath_resources)
        self.controller.release(lightpath_id=lightpath_id, slicing_flag=True)

        # Verify spectrum was released but transponders unchanged
        self.assertEqual(
            self.controller.sdn_props.transponder_usage_dict["0"]["available_transponder"],
            5,
        )
        self.assertEqual(
            self.controller.sdn_props.transponder_usage_dict["2"]["available_transponder"],
            5,
        )

    def test_allocate_with_lightpath_id(self) -> None:
        """Test allocate uses lightpath_id from spectrum props."""
        lightpath_id = 300
        self.controller.spectrum_obj.spectrum_props.start_slot = 0
        self.controller.spectrum_obj.spectrum_props.end_slot = 3
        self.controller.spectrum_obj.spectrum_props.core_number = 0
        self.controller.spectrum_obj.spectrum_props.current_band = "c"
        self.controller.spectrum_obj.spectrum_props.lightpath_id = lightpath_id

        self.controller.allocate()

        assert self.controller.sdn_props.path_list is not None
        assert self.controller.sdn_props.network_spectrum_dict is not None
        path_list = self.controller.sdn_props.path_list
        network_dict = self.controller.sdn_props.network_spectrum_dict

        for link in zip(path_list, path_list[1:], strict=False):
            link_tuple = (link[0], link[1])
            core_matrix = network_dict[link_tuple]["cores_matrix"]["c"][0]
            # Verify lightpath_id is used instead of request_id
            self.assertTrue(np.all(core_matrix[:2] == lightpath_id))
            self.assertEqual(core_matrix[2], lightpath_id * -1)

    def test_allocate_guard_band_with_lightpath_id(self) -> None:
        """Test _allocate_guard_band uses lightpath_id parameter."""
        lightpath_id = 400
        band = "c"
        core_num = 0
        end_slot = 5

        assert self.controller.sdn_props.network_spectrum_dict is not None
        core_matrix = self.controller.sdn_props.network_spectrum_dict[(0, 1)]["cores_matrix"]
        reverse_core_matrix = self.controller.sdn_props.network_spectrum_dict[(1, 0)]["cores_matrix"]

        self.controller._allocate_guard_band(
            band=band,
            core_matrix=core_matrix,
            reverse_core_matrix=reverse_core_matrix,
            core_num=core_num,
            end_slot=end_slot,
            lightpath_id=lightpath_id,
        )

        # Verify guard band uses lightpath_id
        self.assertEqual(core_matrix[band][core_num][end_slot], lightpath_id * -1)
        self.assertEqual(reverse_core_matrix[band][core_num][end_slot], lightpath_id * -1)

    def test_check_snr_after_allocation_disabled(self) -> None:
        """Test _check_snr_after_allocation when disabled."""
        self.controller.engine_props["snr_recheck"] = False
        result = self.controller._check_snr_after_allocation(lightpath_id=500)
        self.assertTrue(result)

    def test_handle_congestion_with_grooming(self) -> None:
        """Test _handle_congestion_with_grooming rollback logic."""
        # Setup grooming state
        self.controller.sdn_props.bandwidth = 100.0
        self.controller.sdn_props.was_new_lp_established = [1000, 1001]
        self.controller.sdn_props.lightpath_id_list = [1000, 1001]
        self.controller.sdn_props.lightpath_bandwidth_list = [50, 50]
        self.controller.sdn_props.start_slot_list = [0, 5]
        self.controller.sdn_props.band_list = ["c", "c"]
        self.controller.sdn_props.core_list = [0, 0]
        self.controller.sdn_props.end_slot_list = [5, 10]
        self.controller.sdn_props.crosstalk_list = [0.0, 0.0]
        self.controller.sdn_props.snr_list = [10.0, 10.0]
        self.controller.sdn_props.xt_list = [0.0, 0.0]
        self.controller.sdn_props.bandwidth_list = [50, 50]
        self.controller.sdn_props.modulation_list = ["QPSK", "QPSK"]
        self.controller.sdn_props.was_partially_groomed = False

        with patch.object(self.controller, "release") as mock_release:
            self.controller._handle_congestion_with_grooming(remaining_bw=50)

            # Verify release was called for each new lightpath
            self.assertEqual(mock_release.call_count, 2)
            mock_release.assert_any_call(lightpath_id=1000, slicing_flag=True, skip_validation=True)
            mock_release.assert_any_call(lightpath_id=1001, slicing_flag=True, skip_validation=True)

            # Verify lists were cleared
            self.assertEqual(len(self.controller.sdn_props.was_new_lp_established), 0)
            self.assertEqual(len(self.controller.sdn_props.lightpath_id_list), 0)


if __name__ == "__main__":
    unittest.main()
