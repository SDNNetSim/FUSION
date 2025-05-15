# pylint: disable=protected-access

"""Unit tests for reinforcement_learning.algorithms.bandits."""

from types import SimpleNamespace
from unittest import TestCase, mock

import numpy as np
from reinforcement_learning.algorithms import bandits


# ----------------------------- helpers --------------------------------
def _mk_engine(**overrides):
    base = dict(
        k_paths=3,
        cores_per_link=3,
        conf_param=2.0,
        max_iters=10,
        save_step=1,
        num_requests=1,
        network="net",
        date="d",
        sim_start="t0",
        erlang=30,
    )
    base.update(overrides)
    return base


def _mk_rl(num_nodes=2):
    return SimpleNamespace(num_nodes=num_nodes)


# ------------------------------ tests ---------------------------------
class TestGetBaseFp(TestCase):
    """_get_base_fp behaviour."""

    def test_path_flag_true_returns_routes_str(self):
        """Route flag builds routes string."""
        res = bandits._get_base_fp(True, 30, 4)
        self.assertEqual(res, "e30_routes_c4")

    def test_path_flag_false_returns_cores_str(self):
        """Core flag builds cores string."""
        res = bandits._get_base_fp(False, 10, 2)
        self.assertEqual(res, "e10_cores_c2")


class TestSaveModelLowLevel(TestCase):
    """_save_model JSON handling."""

    @mock.patch("reinforcement_learning.algorithms.bandits.json.dump")
    @mock.patch("builtins.open", new_callable=mock.mock_open)
    @mock.patch("os.getcwd", return_value="/cwd")
    def test_save_model_writes_json_for_path(
            self, mock_cwd, mock_open_fn, mock_dump
    ):
        """_save_model converts arrays and dumps JSON."""
        data = {("a",): np.array([1, 2])}
        bandits._save_model(
            state_values_dict=data,
            erlang=20,
            cores_per_link=4,
            save_dir="save_dir",
            is_path=True,
            trial=0,
        )
        # tuples→str and ndarray→list
        expected = {"('a',)": [1, 2]}
        mock_dump.assert_called_once_with(expected, mock.ANY)

    def test_save_model_cores_not_implemented(self):
        """_save_model raises when is_path False."""
        with self.assertRaises(NotImplementedError):
            bandits._save_model({}, 10, 2, "d", False, 1)


class TestGetQTable(TestCase):
    """get_q_table key construction."""

    def test_path_mode_creates_pair_keys(self):
        """Keys omit path index in path mode."""
        dummy = SimpleNamespace(
            num_nodes=2,
            is_path=True,
            n_arms=2,
            engine_props={"k_paths": 2},
        )
        counts, values = bandits.get_q_table(dummy)
        self.assertIn((0, 1), counts)
        self.assertNotIn((0, 1, 0), counts)
        self.assertEqual(counts[(0, 1)].shape[0], 2)
        self.assertTrue(np.all(values[(0, 1)] == 0.0))


class TestUpdateBandit(TestCase):
    """_update_bandit value updates."""

    @mock.patch("reinforcement_learning.algorithms.bandits.save_model")
    def test_update_bandit_first_step_sets_value(
            self, mock_save_model
    ):
        """First update sets value to reward."""
        props = SimpleNamespace(rewards_matrix=[])
        self_obj = SimpleNamespace(
            counts={(0, 1): np.zeros(1)},
            values={(0, 1): np.zeros(1)},
            props=props,
            iteration=0,
            is_path=True,
            source=0,
            dest=1,
            path_index=None,
            engine_props=_mk_engine(),
        )
        bandits._update_bandit(
            self=self_obj,
            iteration=0,
            reward=5.0,
            arm=0,
            algorithm="epsilon_greedy_bandit",
            trial=0,
        )
        self.assertEqual(self_obj.counts[(0, 1)][0], 1)
        self.assertEqual(self_obj.values[(0, 1)][0], 5.0)
        self.assertEqual(props.rewards_matrix, [[5.0]])
        mock_save_model.assert_called_once()


class TestEpsilonGreedy(TestCase):
    """EpsilonGreedyBandit action selection."""

    @mock.patch("reinforcement_learning.algorithms.bandits.np.random.rand",
                return_value=0.9)
    def test_get_action_exploits_when_rand_gt_eps(self, _):
        """With rand>eps the greedy arm is chosen."""
        bandit = bandits.EpsilonGreedyBandit(
            rl_props=_mk_rl(), engine_props=_mk_engine(), is_path=True
        )
        bandit.epsilon = 0.1
        pair = (0, 1)
        bandit.values[pair] = np.array([2.0, 1.0, 0.0])
        arm = bandit.select_path_arm(*pair)
        self.assertEqual(arm, 0)

    @mock.patch("reinforcement_learning.algorithms.bandits.np.random.randint",
                return_value=2)
    @mock.patch("reinforcement_learning.algorithms.bandits.np.random.rand",
                return_value=0.0)
    def test_get_action_explores_when_rand_lt_eps(self, _, __):
        """With rand<eps a random arm is chosen."""
        bandit = bandits.EpsilonGreedyBandit(
            rl_props=_mk_rl(), engine_props=_mk_engine(), is_path=True
        )
        bandit.epsilon = 1.0
        arm = bandit.select_path_arm(0, 1)
        self.assertEqual(arm, 2)


class TestUCB(TestCase):
    """UCBBandit action logic."""

    def test_ucb_selects_uncounted_arm_first(self):
        """Zero-count arm is selected first."""
        bandit = bandits.UCBBandit(
            rl_props=_mk_rl(), engine_props=_mk_engine(), is_path=True
        )
        # All counts zero → argmin returns 0
        arm = bandit.select_path_arm(0, 1)
        self.assertEqual(arm, 0)

    def test_ucb_computes_confidence_bound(self):
        """UCB returns argmax of UCB values."""
        eng = _mk_engine(conf_param=1.0)
        bandit = bandits.UCBBandit(
            rl_props=_mk_rl(), engine_props=eng, is_path=True
        )
        pair = (0, 1)
        bandit.counts[pair] = np.array([5, 1, 1])
        bandit.values[pair] = np.array([0.2, 0.1, 0.0])
        arm = bandit.select_path_arm(*pair)
        self.assertEqual(arm, 1)
