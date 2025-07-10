"""Unit tests for reinforcement_learning.utils.sim_data."""

# pylint: disable=protected-access

from unittest import TestCase, mock

import numpy as np
from reinforcement_learning.utils import sim_data as sd


# ------------------------------------------------------------------ #
# helpers                                                             #
# ------------------------------------------------------------------ #
def _patch_isdir(always=True):
    return mock.patch("reinforcement_learning.utils.sim_data.os.path.isdir",
                      return_value=always)


def _patch_exists(always=True):
    return mock.patch("reinforcement_learning.utils.sim_data.os.path.exists",
                      return_value=always)


# ------------------------------------------------------------------ #
class TestExtractTrafficLabel(TestCase):
    """_extract_traffic_label directory scan."""

    def test_returns_first_erlang_prefix(self):
        """Finds 'e400' part from nested file name."""
        with mock.patch("reinforcement_learning.utils.sim_data.os.listdir",
                        side_effect=[["run1"], ["e400_erlang.json"]]), \
                _patch_isdir():
            label = sd._extract_traffic_label("any/path")
        self.assertEqual(label, "e400")

    def test_returns_empty_when_none_found(self):
        """No matching file yields empty string."""
        with mock.patch("reinforcement_learning.utils.sim_data.os.listdir",
                        return_value=["run1"]), _patch_isdir():
            label = sd._extract_traffic_label("path")
        self.assertEqual(label, "")


class TestFilenameTrafficLabel(TestCase):
    """_extract_traffic_label_from_filename regex."""

    def test_parses_numeric_part(self):
        """
        Test parsing numeric part from filename.
        """
        self.assertEqual(sd._extract_traffic_label_from_filename(
            "state_vals_e123.5.json", "x"), "123.5")

    def test_fallback_when_no_match(self):
        """
        Test fallback when no match is found.
        """
        self.assertEqual(sd._extract_traffic_label_from_filename(
            "state_vals.json", "fallback"), "fallback")


class TestLoadMemoryUsage(TestCase):
    """load_memory_usage presence & missing file branches."""

    def setUp(self):
        self.sim_times = {"PPO": [["run1"]]}
        self.base_logs = "/logs"
        self.base_dir = "/base"
        self.arr = np.array([1, 2])

    @mock.patch("reinforcement_learning.utils.sim_data.np.load",
                return_value=np.array([1, 2]))
    @mock.patch("reinforcement_learning.utils.sim_data._extract_traffic_label",
                return_value="400")
    @_patch_exists(True)
    def test_file_found_loads_numpy(self, *_):
        """Dict entry created with loaded array."""
        data = sd.load_memory_usage(self.sim_times, self.base_logs,
                                    self.base_dir, "net", "d")
        self.assertTrue(
            np.array_equal(data["PPO"]["400"], self.arr)
        )

    @mock.patch("reinforcement_learning.utils.sim_data.os.listdir",
                return_value=[])  # ← NEW
    @_patch_exists(False)
    @mock.patch("builtins.print")
    def test_missing_file_logs_and_skips(
            self, mock_print, _mock_exists, _mock_listdir
    ):
        """Missing file prints warning; dict empty."""
        data = sd.load_memory_usage(
            self.sim_times, self.base_logs, self.base_dir, "net", "d"
        )
        self.assertEqual(data["PPO"], {})
        mock_print.assert_called()  # warning emitted


class TestLoadAllRewards(TestCase):
    """load_all_rewards_files regex & nesting."""

    def setUp(self):
        self.sim_times = {"A2C": [["run1"]]}
        self.base_logs = "/logs"
        self.base_dir = "/base"
        self.reward_arr = np.array([0.5])

    @mock.patch("reinforcement_learning.utils.sim_data.np.load",
                return_value=np.array([0.5]))
    @mock.patch("reinforcement_learning.utils.sim_data.os.listdir",
                return_value=["rewards_e400.0_routes_c2_t1_iter_3.npy"])
    @_patch_exists(True)
    @mock.patch("reinforcement_learning.utils.sim_data._extract_traffic_label",
                return_value="400")
    def test_regex_parses_indices_and_stores(self, *_):
        """Nested dict contains trial and episode keys."""
        data = sd.load_all_rewards_files(self.sim_times, self.base_logs,
                                         self.base_dir, "net", "d")
        self.assertEqual(data["A2C"]["400"][1][3].tolist(), [0.5])
