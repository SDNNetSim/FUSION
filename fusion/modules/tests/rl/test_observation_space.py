"""Unit tests for fusion.modules.rl.utils.observation_space module."""

from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from fusion.modules.rl.utils import observation_space as obs_mod


def _rl_props() -> SimpleNamespace:
    return SimpleNamespace(
        num_nodes=4,
        k_paths=3,
        arrival_list=[{"bandwidth": "100"}],  # numeric string
    )


def _engine_obj(key: str = "obs_1") -> SimpleNamespace:
    return SimpleNamespace(
        engine_props={
            "mod_per_bw": {
                "100": {"QPSK": {"slots_needed": 4}},
                "100G": {"QPSK": {"slots_needed": 4}},
            },
            "topology": "dummy_topo",
            "obs_space": key,
        }
    )


def _fake_topo(*_: object, **__: object) -> tuple[np.ndarray, np.ndarray, np.ndarray, None]:
    """ei(2,5) edge_idx, ea(5,1) edge_attr, xf(4,3) node_feat, _."""
    ei = np.zeros((2, 5), dtype=int)
    ea = np.zeros((5, 1), dtype=float)
    xf = np.zeros((4, 3), dtype=float)
    return ei, ea, xf, None


# ------------------------------------------------------------------ #
class TestGetObservationSpace:
    """get_observation_space builds correct keys."""

    @mock.patch.object(obs_mod, "OBS_DICT", {"obs_1": ["source", "destination"]})
    @mock.patch.object(obs_mod, "convert_networkx_topo", side_effect=_fake_topo)
    def test_without_graph_features(self, _: mock.MagicMock) -> None:
        """Dict includes only requested non-graph features."""
        with mock.patch.object(obs_mod, "spaces") as mock_spaces:
            mock_spaces.MultiBinary.return_value = "mb"
            space = obs_mod.get_observation_space(_rl_props(), _engine_obj())

        assert space == {"source": "mb", "destination": "mb"}
        mock_spaces.MultiBinary.assert_called()

    @mock.patch.object(
        obs_mod,
        "OBS_DICT",
        {"obs_1": ["source"], "obs_1_graph": ["source"]},  # keyed but ignored
    )
    @mock.patch.object(obs_mod, "convert_networkx_topo", side_effect=_fake_topo)
    def test_with_graph_features(self, _: mock.MagicMock) -> None:
        """Graph flag adds x, edge_index, edge_attr, path_masks."""
        with mock.patch.object(obs_mod, "spaces") as mock_spaces:
            mock_spaces.MultiBinary.return_value = "mb"
            mock_spaces.Box.return_value = "bx"
            space = obs_mod.get_observation_space(_rl_props(), _engine_obj("obs_1_graph"))

        graph_keys = {"x", "edge_index", "edge_attr", "path_masks"}
        assert graph_keys.issubset(space)
        assert space["source"] == "mb"
        assert space["x"] == "bx"  # one sample check


class TestFragmentationTracker:
    """FragmentationTracker update & compute."""

    @pytest.fixture
    def tracker(self) -> obs_mod.FragmentationTracker:
        """Create FragmentationTracker instance."""
        return obs_mod.FragmentationTracker(num_nodes=3, core_count=2, spectral_slots=4)

    def test_fragmentation_values(self, tracker: obs_mod.FragmentationTracker) -> None:
        """update then get_fragmentation returns expected fractions."""
        # allocate slots 1-2 on link 0→1, core 0
        tracker.update(0, 1, core_index=0, start_slot=1, end_slot=2)
        frag = tracker.get_fragmentation([0, 1], core_index=0)

        assert frag["fragmentation"][0] == pytest.approx(64.0)
        assert frag["path_frag"][0] == pytest.approx(32.0)

    def test_path_len_one_returns_zero(self, tracker: obs_mod.FragmentationTracker) -> None:
        """Single-node path yields zero fragmentation."""
        frag = tracker.get_fragmentation([0], core_index=0)
        assert frag["fragmentation"][0] == 0.0
        assert frag["path_frag"][0] == 0.0
