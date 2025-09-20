"""Unit tests for reinforcement_learning.utils.observation_space."""

from types import SimpleNamespace
from unittest import TestCase, mock

import numpy as np

from fusion.modules.rl.utils import observation_space as obs_mod


def _rl_props():
    return SimpleNamespace(
        num_nodes=4,
        k_paths=3,
        arrival_list=[{"bandwidth": "100"}],  # numeric string
    )


def _engine_obj(key="obs_1"):
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


def _fake_topo(*_, **__):
    """ei(2,5) edge_idx, ea(5,1) edge_attr, xf(4,3) node_feat, _."""
    ei = np.zeros((2, 5), dtype=int)
    ea = np.zeros((5, 1), dtype=float)
    xf = np.zeros((4, 3), dtype=float)
    return ei, ea, xf, None


# ------------------------------------------------------------------ #
class TestGetObservationSpace(TestCase):
    """get_observation_space builds correct keys."""

    @mock.patch.object(obs_mod, "OBS_DICT", {"obs_1": ["source", "destination"]})
    @mock.patch.object(obs_mod, "convert_networkx_topo", side_effect=_fake_topo)
    def test_without_graph_features(self, _):
        """Dict includes only requested non-graph features."""
        with mock.patch.object(obs_mod, "spaces") as mock_spaces:
            mock_spaces.MultiBinary.return_value = "mb"
            space = obs_mod.get_observation_space(_rl_props(), _engine_obj())

        self.assertDictEqual(space, {"source": "mb", "destination": "mb"})
        mock_spaces.MultiBinary.assert_called()

    @mock.patch.object(
        obs_mod,
        "OBS_DICT",
        {"obs_1": ["source"], "obs_1_graph": ["source"]},  # keyed but ignored
    )
    @mock.patch.object(obs_mod, "convert_networkx_topo", side_effect=_fake_topo)
    def test_with_graph_features(self, _):
        """Graph flag adds x, edge_index, edge_attr, path_masks."""
        with mock.patch.object(obs_mod, "spaces") as mock_spaces:
            mock_spaces.MultiBinary.return_value = "mb"
            mock_spaces.Box.return_value = "bx"
            space = obs_mod.get_observation_space(
                _rl_props(), _engine_obj("obs_1_graph")
            )

        graph_keys = {"x", "edge_index", "edge_attr", "path_masks"}
        self.assertTrue(graph_keys.issubset(space))
        self.assertEqual(space["source"], "mb")
        self.assertEqual(space["x"], "bx")  # one sample check


class TestFragmentationTracker(TestCase):
    """FragmentationTracker update & compute."""

    def setUp(self):
        self.tr = obs_mod.FragmentationTracker(
            num_nodes=3, core_count=2, spectral_slots=4
        )

    def test_fragmentation_values(self):
        """update then get_fragmentation returns expected fractions."""
        # allocate slots 1-2 on link 0→1, core 0
        self.tr.update(0, 1, core_index=0, start_slot=1, end_slot=2)
        frag = self.tr.get_fragmentation([0, 1], core_index=0)

        self.assertAlmostEqual(frag["fragmentation"][0], 64.0)
        self.assertAlmostEqual(frag["path_frag"][0], 32.0)

    def test_path_len_one_returns_zero(self):
        """Single-node path yields zero fragmentation."""
        frag = self.tr.get_fragmentation([0], core_index=0)
        self.assertEqual(frag["fragmentation"][0], 0.0)
        self.assertEqual(frag["path_frag"][0], 0.0)
