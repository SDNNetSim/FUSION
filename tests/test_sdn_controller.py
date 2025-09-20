# pylint: disable=protected-access

import unittest
from unittest.mock import patch

import numpy as np

from fusion.core.properties import SDNProps  # Class import for sdn_props
from fusion.core.sdn_controller import SDNController


class TestSDNController(unittest.TestCase):
    """
    Tests the SDNController class.
    """

    def setUp(self):
        self.engine_props = {
            "cores_per_link": 7,
            "guard_slots": 1,
            "route_method": "shortest_path",
            "max_segments": 1,
            "band_list": ["c"],
        }
        self.controller = SDNController(engine_props=self.engine_props)

        # Mock SDNProps to ensure it's treated as a class object
        self.controller.sdn_props = SDNProps()
        # Update slicing manager's reference to the new sdn_props
        self.controller.slicing_manager.sdn_props = self.controller.sdn_props
        self.controller.sdn_props.path_list = ["A", "B", "C"]
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
        }

    def test_release(self):
        """
        Test the release method.
        """
        self.controller.sdn_props.network_spectrum_dict[("A", "B")]["cores_matrix"][
            "c"
        ][0][:3] = 1
        self.controller.sdn_props.network_spectrum_dict[("A", "B")]["cores_matrix"][
            "c"
        ][0][3] = -1
        self.controller.release()

        for link in zip(
            self.controller.sdn_props.path_list,
            self.controller.sdn_props.path_list[1:],
            strict=False,
        ):
            link_tuple = tuple(link)  # Convert to tuple for dictionary key
            for core_num in range(self.engine_props["cores_per_link"]):
                core_arr = self.controller.sdn_props.network_spectrum_dict[link_tuple][
                    "cores_matrix"
                ]["c"][core_num]
                self.assertTrue(
                    np.all(core_arr[:4] == 0),
                    "Request and guard band not properly cleared",
                )

    def test_allocate(self):
        """
        Test the allocate method.
        """
        self.controller.spectrum_obj.spectrum_props.start_slot = 0
        self.controller.spectrum_obj.spectrum_props.end_slot = 3
        self.controller.spectrum_obj.spectrum_props.core_number = 0
        self.controller.spectrum_obj.spectrum_props.current_band = "c"
        self.controller.engine_props["guard_slots"] = 1
        self.controller.allocate()

        for link in zip(
            self.controller.sdn_props.path_list,
            self.controller.sdn_props.path_list[1:],
            strict=False,
        ):
            core_matrix = self.controller.sdn_props.network_spectrum_dict[tuple(link)][
                "cores_matrix"
            ]["c"][0]
            self.assertTrue(
                np.all(core_matrix[:2] == self.controller.sdn_props.request_id),
                "Request not properly allocated",
            )

            self.assertEqual(
                core_matrix[2],
                self.controller.sdn_props.request_id * -1,
                msg="Guard band not properly allocated.",
            )

    def test_update_req_stats(self):
        """
        Test the update request statistics method.
        """
        # Properly initialize sdn_props with necessary attributes
        self.controller.sdn_props.crosstalk_list = []
        self.controller.sdn_props.core_list = []
        self.controller.spectrum_obj.spectrum_props.crosstalk_cost = 10
        self.controller.spectrum_obj.spectrum_props.core_number = 2

        # Set sdn_props.stat_key_list to ensure the relevant keys are present
        self.controller.sdn_props.stat_key_list = ["crosstalk_list", "core_list"]

        # Call the method to update request statistics
        self.controller._update_req_stats(bandwidth="100G")

        # Verify that the bandwidth and other stats are updated correctly
        self.assertIn("100G", self.controller.sdn_props.bandwidth_list)
        self.assertIn(
            10, self.controller.sdn_props.crosstalk_list
        )  # Check if crosstalk_cost was updated correctly
        self.assertIn(
            2, self.controller.sdn_props.core_list
        )  # Check if core_number was updated correctly

    @patch("fusion.core.sdn_controller.SDNController.allocate")
    @patch("fusion.core.sdn_controller.SDNController._update_req_stats")
    @patch("fusion.core.spectrum_assignment.SpectrumAssignment.get_spectrum")
    def test_allocate_slicing(
        self, mock_get_spectrum, mock_update_req_stats, mock_allocate
    ):
        """
        Tests the allocate slicing method.
        """
        self.controller.spectrum_obj.spectrum_props.is_free = (
            True  # Ensure spectrum is free
        )
        mock_get_spectrum.return_value = (
            None  # Ensure that the method is mocked correctly
        )

        # Call the allocate slicing method
        self.controller._allocate_slicing(
            num_segments=2, mod_format="QPSK", path_list=["A", "B"], bandwidth="50G"
        )

        # Verify that the relevant methods were called
        mock_get_spectrum.assert_called()
        mock_allocate.assert_called()
        mock_update_req_stats.assert_called_with(bandwidth="50G")

    @patch("fusion.core.sdn_controller.SDNController.allocate")
    @patch("fusion.core.sdn_controller.SDNController._update_req_stats")
    @patch("fusion.core.routing.Routing.get_route")
    @patch("fusion.core.spectrum_assignment.SpectrumAssignment.get_spectrum")
    def test_handle_event_arrival(
        self, mock_allocate, mock_stats, mock_route, mock_spectrum
    ):  # pylint: disable=unused-argument
        """
        Tests the handle event with an arrival request.
        """
        mock_route.return_value = None
        self.controller.route_obj.route_props.paths_matrix = [["A", "B", "C"]]
        self.controller.route_obj.route_props.modulation_formats_matrix = [["QPSK"]]
        self.controller.route_obj.route_props.weights_list = [10]

        self.controller.spectrum_obj.spectrum_props.is_free = True
        mock_spectrum.return_value = None

        self.controller.handle_event({}, request_type="arrival")

        mock_allocate.assert_called_once()
        self.assertTrue(self.controller.sdn_props.was_routed)

    def test_allocate_without_guard_band(self):
        """
        Test the allocate method when no guard band is present.
        """
        self.controller.spectrum_obj.spectrum_props.start_slot = 0
        self.controller.spectrum_obj.spectrum_props.end_slot = 2
        self.controller.spectrum_obj.spectrum_props.core_number = 0
        self.controller.spectrum_obj.spectrum_props.current_band = "c"
        self.controller.engine_props["guard_slots"] = 0

        # Perform allocation
        self.controller.allocate()

        for link in zip(
            self.controller.sdn_props.path_list,
            self.controller.sdn_props.path_list[1:],
            strict=False,
        ):
            core_matrix = self.controller.sdn_props.network_spectrum_dict[tuple(link)][
                "cores_matrix"
            ]["c"][0]

            # Check allocation
            self.assertTrue(
                np.all(core_matrix[:3] == self.controller.sdn_props.request_id),
                "Request not properly allocated",
            )

            # Ensure no guard band is allocated
            self.assertNotIn(
                self.controller.sdn_props.request_id * -1,
                core_matrix,
                "Guard band should not be allocated.",
            )

    def test_handle_dynamic_slicing_success(self):
        """
        Test _handle_dynamic_slicing for a successful allocation scenario.
        """
        self.controller.sdn_props.bandwidth = "100"
        self.controller.engine_props["mod_per_bw"] = {
            "50": {"QPSK": {"max_length": 100}},
            "100": {"16QAM": {"max_length": 200}},
        }
        self.controller.engine_props["fixed_grid"] = True
        self.controller.engine_props["topology"] = {
            ("A", "B"): {"length": 50},
            ("B", "C"): {"length": 50},
        }
        self.controller.sdn_props.path_list = ["A", "B", "C"]
        self.controller.sdn_props.number_of_transponders = 0

        # Set spectrum to free
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
            self.controller._handle_dynamic_slicing(
                path_list=["A", "B", "C"], path_index=0, forced_segments=1
            )

            # Verify methods were called
            mock_get_spectrum.assert_called()
            mock_allocate.assert_called()
            mock_update_stats.assert_called_with(bandwidth="50")
            mock_find_path_len.assert_called_with(
                path_list=["A", "B", "C"],
                topology=self.controller.engine_props["topology"],
            )

            # TODO: Forced segments isn't used in dynamic slicing?
            self.assertEqual(self.controller.sdn_props.number_of_transponders, 2)
            self.assertTrue(self.controller.sdn_props.is_sliced)

    def test_handle_dynamic_slicing_congestion(self):
        """
        Test _handle_dynamic_slicing for a congestion scenario.
        """
        self.controller.sdn_props.bandwidth = "100"
        self.controller.engine_props["mod_per_bw"] = {
            "50": {"QPSK": {"max_length": 100}},
            "100": {"16QAM": {"max_length": 200}},
        }
        self.controller.engine_props["fixed_grid"] = True
        self.controller.engine_props["topology"] = {
            ("A", "B"): {"length": 50},
            ("B", "C"): {"length": 50},
        }
        self.controller.sdn_props.path_list = ["A", "B", "C"]
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
            # Use the real _handle_congestion function
            with patch.object(
                self.controller,
                "_handle_congestion",
                wraps=self.controller._handle_congestion,
            ) as mock_handle_congestion:
                # Simulate no free spectrum
                self.controller.spectrum_obj.spectrum_props.is_free = False

                self.controller._handle_dynamic_slicing(
                    path_list=["A", "B", "C"], path_index=0, forced_segments=1
                )

                # Verify methods were called
                mock_get_spectrum.assert_called()
                mock_handle_congestion.assert_called_with(remaining_bw=100)
                mock_allocate.assert_not_called()
                mock_find_path_len.assert_called_with(
                    path_list=["A", "B", "C"],
                    topology=self.controller.engine_props["topology"],
                )

                # Verify properties were updated
                self.assertFalse(self.controller.sdn_props.was_routed)
                self.assertEqual(self.controller.sdn_props.block_reason, "congestion")
                self.assertFalse(self.controller.sdn_props.is_sliced)

    def test_usage_count_increment(self):
        """Test that usage_count increments after allocation."""
        self.controller.spectrum_obj.spectrum_props.start_slot = 0
        self.controller.spectrum_obj.spectrum_props.end_slot = 3
        self.controller.spectrum_obj.spectrum_props.core_number = 0
        self.controller.spectrum_obj.spectrum_props.current_band = "c"
        self.controller.engine_props["guard_slots"] = 1

        for key in self.controller.sdn_props.network_spectrum_dict:
            self.controller.sdn_props.network_spectrum_dict[key]["usage_count"] = 0

        self.controller.allocate()

        for link in zip(
            self.controller.sdn_props.path_list,
            self.controller.sdn_props.path_list[1:],
            strict=False,
        ):
            link_tuple = tuple(link)
            self.assertEqual(
                self.controller.sdn_props.network_spectrum_dict[link_tuple][
                    "usage_count"
                ],
                1,
            )
            reverse_link = (link_tuple[1], link_tuple[0])
            self.assertEqual(
                self.controller.sdn_props.network_spectrum_dict[reverse_link][
                    "usage_count"
                ],
                1,
            )

    def test_release_throughput_tracking(self):
        """Test that throughput is correctly tracked on release."""
        for key in self.controller.sdn_props.network_spectrum_dict:
            self.controller.sdn_props.network_spectrum_dict[key]["throughput"] = 0

        self.controller.sdn_props.arrive = 0
        self.controller.sdn_props.depart = 5  # 5 seconds duration
        self.controller.sdn_props.bandwidth = "100"  # Gbps

        self.controller.release()

        expected_throughput = 500  # 100 Gbps * 5 seconds
        for link in zip(
            self.controller.sdn_props.path_list,
            self.controller.sdn_props.path_list[1:],
            strict=False,
        ):
            link_tuple = tuple(link)
            self.assertEqual(
                self.controller.sdn_props.network_spectrum_dict[link_tuple][
                    "throughput"
                ],
                expected_throughput,
            )
            reverse_link = (link_tuple[1], link_tuple[0])
            self.assertEqual(
                self.controller.sdn_props.network_spectrum_dict[reverse_link][
                    "throughput"
                ],
                expected_throughput,
            )

    def test_release_missing_timing_fields(self):
        """Test that release handles missing timing fields gracefully."""
        for key in self.controller.sdn_props.network_spectrum_dict:
            self.controller.sdn_props.network_spectrum_dict[key]["throughput"] = 0

        self.controller.sdn_props.arrive = None
        self.controller.sdn_props.depart = None
        self.controller.sdn_props.bandwidth = None

        # Should not raise exception
        self.controller.release()

        for key in self.controller.sdn_props.network_spectrum_dict:
            self.assertEqual(
                self.controller.sdn_props.network_spectrum_dict[key]["throughput"], 0
            )


if __name__ == "__main__":
    unittest.main()
