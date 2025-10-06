"""
Unit tests for fusion.core.metrics module.

This module provides comprehensive testing for the SimStats class which handles
simulation statistics collection, including blocking metrics, snapshot management,
and confidence interval calculations.
"""

import unittest
from typing import Any
from unittest.mock import patch

import networkx as nx
import numpy as np

from fusion.core.metrics import SimStats
from fusion.core.properties import SNAP_KEYS_LIST, StatsProps


class TestSimStats(unittest.TestCase):
    """Unit tests for SimStats metrics collection functionality."""

    engine_props: dict[str, Any]
    sim_info: str
    stats_props: StatsProps
    topology: nx.Graph
    sim_stats: SimStats

    def setUp(self) -> None:
        """Set up test fixtures with proper isolation."""
        self.engine_props = {
            "num_requests": 100,
            "snapshot_step": 20,
            "cores_per_link": 4,
            "save_snapshots": True,
            "mod_per_bw": {"50GHz": {"QPSK": {}, "16QAM": {}}},
            "output_train_data": True,
            "band_list": ["C"],
            "save_start_end_slots": False,
            "k_paths": 3,
        }
        self.sim_info = "Test Simulation"
        self.stats_props = StatsProps()
        self.stats_props.block_reasons_dict = {
            "congestion": 0,
            "distance": 0,
            "xt_threshold": 0,
        }

        # Initialize topology for testing
        self.topology = nx.Graph()
        self.topology.add_edge("A", "B", length=10)
        self.topology.add_edge("B", "C", length=15)
        self.topology.add_edge("A", "C", length=20)

        self.sim_stats = SimStats(
            engine_props=self.engine_props,
            sim_info=self.sim_info,
            stats_props=self.stats_props,
        )
        self.sim_stats.topology = self.topology

    def test_init_with_valid_parameters_creates_instance(self) -> None:
        """Test SimStats initialization with valid parameters."""
        stats = SimStats(
            engine_props=self.engine_props,
            sim_info="Test",
            stats_props=None,
        )

        self.assertIsInstance(stats.stats_props, StatsProps)
        self.assertEqual(stats.engine_props, self.engine_props)
        self.assertEqual(stats.sim_info, "Test")
        self.assertEqual(stats.blocked_requests, 0)
        self.assertEqual(stats.total_transponders, 0)

    def test_init_with_none_stats_props_creates_default(self) -> None:
        """Test SimStats initialization creates default StatsProps when None."""
        stats = SimStats(
            engine_props=self.engine_props,
            sim_info="Test",
            stats_props=None,
        )

        self.assertIsInstance(stats.stats_props, StatsProps)
        self.assertIsNotNone(stats.stats_props.simulation_blocking_list)
        self.assertIsNotNone(stats.stats_props.block_reasons_dict)

    def test_get_snapshot_info_with_valid_data_returns_correct_metrics(self) -> None:
        """Test snapshot info calculation with valid network data."""
        network_spectrum_dict = {
            (0, 1): {
                "cores_matrix": [
                    np.array([0, 1, 0, -1]),
                    np.array([1, 0, -1, 0]),
                ]
            },
            (1, 0): {
                "cores_matrix": [
                    np.array([0, -1, 1, 0]),
                    np.array([-1, 1, 0, 0]),
                ]
            },
        }
        path_list = [(0, 1)]

        occupied_slots, guard_slots, active_reqs = SimStats._get_snapshot_info(
            network_spectrum_dict=network_spectrum_dict, path_list=path_list
        )

        self.assertEqual(occupied_slots, 4)
        self.assertEqual(guard_slots, 2)
        self.assertEqual(active_reqs, 1)

    def test_get_snapshot_info_with_empty_network_returns_zeros(self) -> None:
        """Test snapshot info with empty network returns zero values."""
        network_spectrum_dict: dict[tuple[int, int], dict[str, list[np.ndarray]]] = {}
        path_list = None

        occupied_slots, guard_slots, active_reqs = SimStats._get_snapshot_info(
            network_spectrum_dict=network_spectrum_dict, path_list=path_list
        )

        self.assertEqual(occupied_slots, 0)
        self.assertEqual(guard_slots, 0)
        self.assertEqual(active_reqs, 0)

    def test_update_snapshot_with_valid_request_updates_correctly(self) -> None:
        """Test snapshot update with valid request data."""
        self.sim_stats.stats_props.snapshots_dict = {
            2: {
                "occupied_slots": [],
                "guard_slots": [],
                "active_requests": [],
                "blocking_prob": [],
                "num_segments": [],
                "bit_rate_blocking_prob": [],
            }
        }

        with patch.object(
            self.sim_stats, "_get_snapshot_info", return_value=(3, 3, 1)
        ), patch(
            "fusion.analysis.network_analysis.NetworkAnalyzer.get_link_usage_summary",
            return_value={}
        ):
            self.sim_stats.blocked_requests = 1
            self.sim_stats.bit_rate_request = 100
            self.sim_stats.bit_rate_blocked = 50
            req_num = 2
            path_list = [(0, 1)]

            self.sim_stats.update_snapshot(
                network_spectrum_dict={}, request_number=req_num, path_list=path_list
            )

            snapshot = self.sim_stats.stats_props.snapshots_dict[req_num]
            self.assertEqual(snapshot["occupied_slots"][0], 3)
            self.assertEqual(snapshot["guard_slots"][0], 3)
            self.assertEqual(snapshot["active_requests"][0], 1)
            self.assertAlmostEqual(snapshot["blocking_prob"][0], 0.5)

    def test_init_snapshots_creates_correct_structure(self) -> None:
        """Test snapshot initialization creates correct data structure."""
        self.sim_stats._init_snapshots()

        expected_req_nums = list(range(0, 101, 20))
        for req_num in expected_req_nums:
            self.assertIn(req_num, self.sim_stats.stats_props.snapshots_dict)
            for key in SNAP_KEYS_LIST:
                self.assertIn(key, self.sim_stats.stats_props.snapshots_dict[req_num])
                self.assertEqual(
                    self.sim_stats.stats_props.snapshots_dict[req_num][key], []
                )

    def test_init_mods_weights_bws_creates_nested_structure(self) -> None:
        """Test modulation/weights/bandwidth initialization creates proper structure."""
        self.sim_stats._init_mods_weights_bws()

        # Check bandwidth level structure
        for bandwidth in self.engine_props["mod_per_bw"]:
            bandwidth_key = str(bandwidth)
            self.assertIn(bandwidth_key, self.sim_stats.stats_props.weights_dict)
            self.assertIn(
                bandwidth_key, self.sim_stats.stats_props.modulations_used_dict
            )
            self.assertIn(
                bandwidth_key, self.sim_stats.stats_props.bandwidth_blocking_dict
            )

            # Check modulation level structure
            for modulation in self.engine_props["mod_per_bw"][bandwidth]:
                weights_dict = self.sim_stats.stats_props.weights_dict[bandwidth_key]
                mod_used_dict = self.sim_stats.stats_props.modulations_used_dict[
                    bandwidth_key
                ]
                self.assertIn(modulation, weights_dict)
                self.assertIn(modulation, mod_used_dict)
                self.assertEqual(weights_dict[modulation], [])
                self.assertEqual(mod_used_dict[modulation], 0)

    def test_calculate_blocking_statistics_with_valid_data_calculates_correctly(
        self,
    ) -> None:
        """Test blocking statistics calculation with valid data."""
        self.sim_stats.blocked_requests = 20
        self.sim_stats.bit_rate_blocked = 100
        self.sim_stats.bit_rate_request = 500

        self.sim_stats.calculate_blocking_statistics()

        expected_blocking_prob = 20 / 100
        expected_bit_rate_blocking_prob = 100 / 500
        self.assertIn(
            expected_blocking_prob, self.sim_stats.stats_props.simulation_blocking_list
        )
        self.assertIn(
            expected_bit_rate_blocking_prob,
            self.sim_stats.stats_props.simulation_bitrate_blocking_list,
        )

    def test_calculate_blocking_statistics_with_zero_requests_handles_correctly(
        self,
    ) -> None:
        """Test blocking statistics calculation with zero requests."""
        self.sim_stats.engine_props["num_requests"] = 0

        self.sim_stats.calculate_blocking_statistics()

        self.assertIn(0.0, self.sim_stats.stats_props.simulation_blocking_list)
        self.assertIn(0.0, self.sim_stats.stats_props.simulation_bitrate_blocking_list)

    def test_finalize_iteration_statistics_normalizes_block_reasons(self) -> None:
        """Test iteration finalization normalizes blocking reasons correctly."""
        self.sim_stats.blocked_requests = 20
        self.sim_stats.total_transponders = 300
        self.stats_props.block_reasons_dict = {"congestion": 15, "distance": 5}
        self.stats_props.snapshots_dict = {}
        self.stats_props.transponders_list = []

        with patch.object(self.sim_stats, "_get_iter_means"):
            self.sim_stats.finalize_iteration_statistics()

        expected_trans_mean = 300 / (100 - 20)
        self.assertIn(expected_trans_mean, self.stats_props.transponders_list)
        self.assertEqual(self.stats_props.block_reasons_dict["congestion"], 15 / 20)
        self.assertEqual(self.stats_props.block_reasons_dict["distance"], 5 / 20)

    def test_finalize_iteration_statistics_with_all_blocked_requests(self) -> None:
        """Test iteration finalization when all requests are blocked."""
        self.sim_stats.blocked_requests = 100
        self.sim_stats.total_transponders = 0
        self.stats_props.transponders_list = []

        with patch.object(self.sim_stats, "_get_iter_means"):
            self.sim_stats.finalize_iteration_statistics()

        # Should append 0 when all requests are blocked
        self.assertIn(0, self.stats_props.transponders_list)

    def test_calculate_confidence_interval_with_sufficient_data_returns_boolean(
        self,
    ) -> None:
        """Test confidence interval calculation with sufficient data."""
        self.sim_stats.stats_props.simulation_blocking_list = [0.1, 0.2, 0.15, 0.25]
        self.sim_stats.stats_props.simulation_bitrate_blocking_list = [
            0.1, 0.2, 0.15, 0.25
        ]

        should_end = self.sim_stats.calculate_confidence_interval()

        self.assertIsNotNone(self.sim_stats.block_mean)
        self.assertIsNotNone(self.sim_stats.block_ci)
        self.assertIsNotNone(self.sim_stats.block_ci_percent)
        self.assertIsInstance(should_end, bool)

    def test_calculate_confidence_interval_with_zero_mean_returns_false(self) -> None:
        """Test confidence interval calculation with zero blocking mean."""
        self.sim_stats.stats_props.simulation_blocking_list = [0.0, 0.0, 0.0]
        self.sim_stats.stats_props.simulation_bitrate_blocking_list = [0.0, 0.0, 0.0]

        should_end = self.sim_stats.calculate_confidence_interval()

        self.assertEqual(self.sim_stats.block_mean, 0.0)
        self.assertFalse(should_end)

    def test_calculate_confidence_interval_with_insufficient_data_returns_false(
        self,
    ) -> None:
        """Test confidence interval calculation with insufficient data."""
        self.sim_stats.stats_props.simulation_blocking_list = [0.1]
        self.sim_stats.stats_props.simulation_bitrate_blocking_list = [0.1]

        should_end = self.sim_stats.calculate_confidence_interval()

        self.assertFalse(should_end)

    def test_get_blocking_statistics_returns_complete_dictionary(self) -> None:
        """Test get_blocking_statistics returns all required statistics."""
        self.sim_stats.block_mean = 0.2
        self.sim_stats.block_variance = 0.02
        self.sim_stats.block_ci = 0.05
        self.sim_stats.block_ci_percent = 5
        self.sim_stats.bit_rate_block_mean = 0.1
        self.sim_stats.bit_rate_block_variance = 0.01
        self.sim_stats.bit_rate_block_ci = 0.025
        self.sim_stats.bit_rate_block_ci_percent = 2.5
        self.sim_stats.iteration = 1

        blocking_stats = self.sim_stats.get_blocking_statistics()

        expected_keys = [
            "block_mean",
            "block_variance",
            "block_ci",
            "block_ci_percent",
            "bit_rate_block_mean",
            "bit_rate_block_variance",
            "bit_rate_block_ci",
            "bit_rate_block_ci_percent",
            "iteration",
        ]

        for key in expected_keys:
            self.assertIn(key, blocking_stats)

        self.assertEqual(blocking_stats["block_mean"], 0.2)
        self.assertEqual(blocking_stats["iteration"], 1)

    def test_end_iter_update_calls_finalize_iteration_statistics(self) -> None:
        """Test backward compatibility method calls correct implementation."""
        with patch.object(
            self.sim_stats, "finalize_iteration_statistics"
        ) as mock_finalize:
            self.sim_stats.end_iter_update()

            mock_finalize.assert_called_once()

    def test_save_stats_creates_persistence_and_saves(self) -> None:
        """Test backward compatibility save_stats method."""
        with patch("fusion.core.persistence.StatsPersistence") as mock_persistence:
            mock_instance = mock_persistence.return_value
            self.sim_stats.iteration = 1

            self.sim_stats.save_stats("test_path")

            mock_persistence.assert_called_once_with(
                engine_props=self.engine_props, sim_info=self.sim_info
            )
            mock_instance.save_stats.assert_called_once()

    def test_init_stat_dicts_initializes_all_required_dicts(self) -> None:
        """Test initialization of all required statistics dictionaries."""
        self.sim_stats._init_stat_dicts()

        # Check cores_dict initialization
        expected_cores = dict.fromkeys(range(self.engine_props["cores_per_link"]), 0)
        self.assertEqual(self.sim_stats.stats_props.cores_dict, expected_cores)

        # Check block_reasons_dict initialization
        expected_block_reasons = {"distance": 0, "congestion": 0, "xt_threshold": 0}
        self.assertEqual(
            self.sim_stats.stats_props.block_reasons_dict, expected_block_reasons
        )

        # Check link_usage_dict initialization
        self.assertEqual(self.sim_stats.stats_props.link_usage_dict, {})


if __name__ == "__main__":
    unittest.main()
