"""Unit tests for fusion.modules.rl.utils.sim_data module."""

# pylint: disable=protected-access

from typing import Any
from unittest import mock

import numpy as np
import pytest

from fusion.modules.rl.utils import sim_data as sd


# ------------------------------------------------------------------ #
# helpers                                                             #
# ------------------------------------------------------------------ #
def _patch_isdir(always: bool = True) -> Any:
    return mock.patch("fusion.modules.rl.utils.sim_data.os.path.isdir", return_value=always)


def _patch_exists(always: bool = True) -> Any:
    return mock.patch("fusion.modules.rl.utils.sim_data.os.path.exists", return_value=always)


# ------------------------------------------------------------------ #
class TestExtractTrafficLabel:
    """_extract_traffic_label directory scan."""

    def test_returns_first_erlang_prefix(self) -> None:
        """Finds 'e400' part from nested file name."""
        with (
            mock.patch(
                "fusion.modules.rl.utils.sim_data.os.listdir",
                side_effect=[["run1"], ["e400_erlang.json"]],
            ),
            _patch_isdir(),
        ):
            label = sd._extract_traffic_label("any/path")
        assert label == "e400"

    def test_returns_empty_when_none_found(self) -> None:
        """No matching file yields empty string."""
        with (
            mock.patch("fusion.modules.rl.utils.sim_data.os.listdir", return_value=["run1"]),
            _patch_isdir(),
        ):
            label = sd._extract_traffic_label("path")
        assert label == ""


class TestFilenameTrafficLabel:
    """_extract_traffic_label_from_filename regex."""

    def test_parses_numeric_part(self) -> None:
        """
        Test parsing numeric part from filename.
        """
        assert sd._extract_traffic_label_from_filename("state_vals_e123.5.json", "x") == "123.5"

    def test_fallback_when_no_match(self) -> None:
        """
        Test fallback when no match is found.
        """
        assert sd._extract_traffic_label_from_filename("state_vals.json", "fallback") == "fallback"


class TestLoadMemoryUsage:
    """load_memory_usage presence & missing file branches."""

    @pytest.fixture
    def setup_data(self) -> dict[str, Any]:
        """Setup common test data."""
        return {
            "sim_times": {"PPO": [["run1"]]},
            "base_logs": "/logs",
            "base_dir": "/base",
            "arr": np.array([1, 2]),
        }

    @mock.patch("fusion.modules.rl.utils.sim_data.np.load", return_value=np.array([1, 2]))
    @mock.patch("fusion.modules.rl.utils.sim_data._extract_traffic_label", return_value="400")
    @_patch_exists(True)
    def test_file_found_loads_numpy(
        self,
        mock_exists: mock.MagicMock,
        _mock_extract: mock.MagicMock,
        mock_load: mock.MagicMock,
        setup_data: dict[str, Any],
    ) -> None:
        """Dict entry created with loaded array."""
        data = sd.load_memory_usage(
            setup_data["sim_times"],
            setup_data["base_logs"],
            setup_data["base_dir"],
            "net",
            "d",
        )
        assert np.array_equal(data["PPO"]["400"], setup_data["arr"])

    @mock.patch("fusion.modules.rl.utils.sim_data.os.listdir", return_value=[])  # ← NEW
    @_patch_exists(False)
    @mock.patch("builtins.print")
    def test_missing_file_logs_and_skips(
        self,
        mock_print: mock.MagicMock,
        _mock_exists: mock.MagicMock,
        _mock_listdir: mock.MagicMock,
        setup_data: dict[str, Any],
    ) -> None:
        """Missing file prints warning; dict empty."""
        data = sd.load_memory_usage(
            setup_data["sim_times"],
            setup_data["base_logs"],
            setup_data["base_dir"],
            "net",
            "d",
        )
        assert data["PPO"] == {}
        mock_print.assert_called()  # warning emitted


class TestLoadAllRewards:
    """load_all_rewards_files regex & nesting."""

    @pytest.fixture
    def setup_data(self) -> dict[str, Any]:
        """Setup common test data."""
        return {
            "sim_times": {"A2C": [["run1"]]},
            "base_logs": "/logs",
            "base_dir": "/base",
            "reward_arr": np.array([0.5]),
        }

    @mock.patch("fusion.modules.rl.utils.sim_data.np.load", return_value=np.array([0.5]))
    @mock.patch(
        "fusion.modules.rl.utils.sim_data.os.listdir",
        return_value=["rewards_e400.0_routes_c2_t1_iter_3.npy"],
    )
    @_patch_exists(True)
    @mock.patch("fusion.modules.rl.utils.sim_data._extract_traffic_label", return_value="400")
    def test_regex_parses_indices_and_stores(
        self,
        _mock_extract: mock.MagicMock,
        mock_exists: mock.MagicMock,
        mock_listdir: mock.MagicMock,
        mock_load: mock.MagicMock,
        setup_data: dict[str, Any],
    ) -> None:
        """Nested dict contains trial and episode keys."""
        data = sd.load_all_rewards_files(
            setup_data["sim_times"],
            setup_data["base_logs"],
            setup_data["base_dir"],
            "net",
            "d",
        )
        assert data["A2C"]["400"][1][3].tolist() == [0.5]
