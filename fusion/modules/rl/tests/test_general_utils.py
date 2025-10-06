"""Unit tests for reinforcement_learning.utils.general_utils."""

from types import SimpleNamespace
from unittest import TestCase, mock

import numpy as np

from fusion.modules.rl.utils import general_utils as gu


# ------------------------------------------------------------------ #
#  Helper builders                                                    #
# ------------------------------------------------------------------ #
def _rl_props():
    return SimpleNamespace(
        arrival_count=0,
        super_channel_space=2,
        chosen_path_list=[[[0, 1]]],
        spectral_slots=[1550],
        forced_index=None,
        core_index=None,
        arrival_list=[{"arrive": 0, "bandwidth": "100G"}],
        depart_list=[],
    )


def _engine(snapshot=True):
    return SimpleNamespace(
        engine_props={
            "snapshot_step": 1,
            "save_snapshots": snapshot,
            "cong_cutoff": 0.5,
            "topology": "topo",
            "mod_per_bw": {"100G": "mods"},
        },
        stats_obj=SimpleNamespace(
            update_snapshot=mock.MagicMock(),
            blocked_reqs=0,
            stats_props=dict(
                block_reasons_dict=dict(congestion=0),
                block_bw_dict={"100G": 0},
            ),
        ),
        network_spectrum_dict="net",
        handle_release=mock.MagicMock(),
        handle_arrival=mock.MagicMock(),
        generate_requests=mock.MagicMock(),
        reqs_dict={},
    )


def _route():
    return SimpleNamespace(
        route_props=SimpleNamespace(
            paths_matrix=None,
            modulation_formats_matrix=None,
            weights_list=[],
        )
    )


# ------------------------------------------------------------------ #
class TestUpdateSnapshots(TestCase):
    """update_snapshots triggers stats update."""

    def test_calls_update_when_condition_met(self):
        """update_snapshot called at snapshot_step."""
        helper = gu.CoreUtilHelpers(_rl_props(), _engine(), _route())
        helper.update_snapshots()
        helper.engine_props.stats_obj.update_snapshot.assert_called_once()


class TestGetSuperChannels(TestCase):
    """get_super_channels fragmentation & padding."""

    @mock.patch(
        "fusion.modules.rl.utils.general_utils.get_shannon_entropy_fragmentation"
    )
    def test_returns_padded_matrix_and_flag(self, mock_hfrag):
        """Matrix padded with 100.0 and no_penalty flag false."""
        sc_mat = np.array([[0, 0]])
        hfrag = np.array([1.5, np.inf])
        mock_hfrag.return_value = (sc_mat, hfrag)

        helper = gu.CoreUtilHelpers(_rl_props(), _engine(), _route())
        frag, no_penalty = helper.get_super_channels(slots_needed=1, num_channels=1)

        self.assertFalse(no_penalty)
        # after padding to super_channel_space = 2
        np.testing.assert_array_equal(frag, np.array([1.5, 100.0]))


class TestClassifyPathsAndCores(TestCase):
    """classify_paths/cores delegate helpers."""

    @mock.patch(
        "fusion.modules.rl.utils.general_utils.classify_congestion", return_value=2
    )
    @mock.patch(
        "fusion.modules.rl.utils.general_utils.find_path_congestion",
        return_value=(0.4, None),
    )
    def test_classify_paths_returns_info(self, *_):
        """Returns list of tuples (idx,path,cong)."""
        helper = gu.CoreUtilHelpers(_rl_props(), _engine(), _route())
        paths = np.array([[[0, 1]], [[0, 2]]])
        info = helper.classify_paths(paths)

        self.assertEqual(info[0][0], 0)  # index
        self.assertEqual(info[0][2], 2)  # congestion level
        self.assertListEqual(info[0][1].tolist(), [0, 1])  # path list


class TestHandleReleases(TestCase):
    """handle_releases processes due departures."""

    def test_releases_until_future_time(self):
        """handle_release called once; index advanced."""
        rl = _rl_props()
        rl.depart_list = [{"depart": 0}, {"depart": 5}]
        eng = _engine()
        helper = gu.CoreUtilHelpers(rl, eng, _route())

        helper.handle_releases()

        eng.handle_release.assert_called_once_with(current_time=0)
        self.assertEqual(helper._last_processed_index, 1)  # pylint: disable=protected-access


class TestAllocateBlocking(TestCase):
    """allocate branch when chosen index unavailable."""

    def test_blocks_and_updates_stats(self):
        """Blocked counters increment when index OOB."""
        rl = _rl_props()
        rl.forced_index = 1  # super_channel_indexes will be empty
        eng = _engine()
        helper = gu.CoreUtilHelpers(rl, eng, _route())
        helper.super_channel_indexes = []  # none available

        helper.allocate()

        self.assertEqual(eng.stats_obj.blocked_reqs, 1)
        self.assertEqual(
            eng.stats_obj.stats_props["block_reasons_dict"]["congestion"], 1
        )
        self.assertEqual(eng.stats_obj.stats_props["block_bw_dict"]["100G"], 1)
        eng.handle_arrival.assert_not_called()


class TestDetermineModelType(TestCase):
    """determine_model_type returns correct key."""

    def test_returns_path_algorithm(self):
        """Detects path_algorithm key."""
        sim = {"path_algorithm": "ppo"}
        self.assertEqual(gu.determine_model_type(sim), "path_algorithm")

    def test_raises_when_missing(self):
        """Raises ValueError when no algo keys present."""
        with self.assertRaises(gu.ConfigurationError):
            gu.determine_model_type({})


class TestSaveArr(TestCase):
    """save_arr constructs path and calls np.save."""

    @mock.patch("fusion.modules.rl.utils.general_utils.np.save")
    @mock.patch(
        "fusion.modules.rl.utils.general_utils.os.path.join",
        return_value="joined/path.npy",
    )
    def test_save_arr_joins_path_and_saves(self, mock_join, mock_save):
        """np.save called with path from os.path.join."""
        sim = {
            "path_algorithm": "ppo",
            "network": "net",
            "date": "d",
            "sim_start": "t0",
        }
        arr = np.array([1, 2])
        gu.save_arr(arr, sim, "file.npy")

        mock_join.assert_called_once_with("logs", "ppo", "net", "d", "t0", "file.npy")
        mock_save.assert_called_once_with("joined/path.npy", arr)
