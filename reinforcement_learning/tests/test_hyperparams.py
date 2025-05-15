"""Unit tests for reinforcement_learning.utils.hyperparams."""

from types import SimpleNamespace
from unittest import TestCase, mock

import numpy as np
from reinforcement_learning.utils import hyperparams as hp


# ---------------------------- stubs ----------------------------------
def _engine_props():
    return {
        "max_iters": 4,
        "k_paths": 2,
        "cores_per_link": 2,
        "alpha_start": 0.4,
        "alpha_end": 0.1,
        "epsilon_start": 0.8,
        "epsilon_end": 0.2,
        "alpha_update": "linear_decay",
        "epsilon_update": "linear_decay",
        "decay_rate": 0.5,
    }


def _rl_props(nodes=3):
    return SimpleNamespace(num_nodes=nodes)


def _mock_trial():
    """Return an optuna-like trial returning fixed values."""
    trial = SimpleNamespace()

    def _sf(name, low=None, high=None, **kw):  # suggest_float
        return {"gamma": 0.95, "clip_range": 0.2}.get(name, low)

    def _si(name, low=None, high=None, **kw):  # suggest_int
        return low

    def _sc(name, choices):  # suggest_categorical
        return choices[0]

    trial.suggest_float = _sf
    trial.suggest_int = _si
    trial.suggest_categorical = _sc
    return trial


# ----------------------------- tests ---------------------------------
class TestLinearDecay(TestCase):
    """Linear epsilon / alpha decay."""

    @mock.patch("reinforcement_learning.utils.hyperparams.get_q_table",
                return_value=(None, None))
    def test_linear_decay_updates_values(self, _):
        """_linear_eps / _linear_alpha compute expected value."""
        cfg = hp.HyperparamConfig(_engine_props(), _rl_props(), is_path=True)
        cfg.iteration = 2
        cfg._linear_eps()  # pylint: disable=protected-access
        cfg._linear_alpha()  # pylint: disable=protected-access

        self.assertAlmostEqual(cfg.curr_epsilon, 0.5)  # halfway
        self.assertAlmostEqual(cfg.curr_alpha, 0.25)


class TestExponentialDecay(TestCase):
    """Exponential epsilon / alpha decay."""

    @mock.patch("reinforcement_learning.utils.hyperparams.get_q_table",
                return_value=(None, None))
    def test_exp_decay(self, _):
        """exp decay = start * rate**iter."""
        props = _engine_props() | {"alpha_update": "exp_decay",
                                   "epsilon_update": "exp_decay"}
        cfg = hp.HyperparamConfig(props, _rl_props(), is_path=True)
        cfg.iteration = 3
        cfg._exp_eps()  # pylint: disable=protected-access
        cfg._exp_alpha()  # pylint: disable=protected-access

        rate = props["decay_rate"]
        self.assertAlmostEqual(cfg.curr_epsilon,
                               props["epsilon_start"] * rate ** 3)
        self.assertAlmostEqual(cfg.curr_alpha,
                               props["alpha_start"] * rate ** 3)


class TestRewardBased(TestCase):
    """Reward-based update reduces params when reward diff grows."""

    @mock.patch("reinforcement_learning.utils.hyperparams.get_q_table",
                return_value=(None, None))
    def test_reward_based_updates(self, _):
        """Greater diff → smaller epsilon/alpha."""
        cfg = hp.HyperparamConfig(_engine_props(), _rl_props(), True)
        cfg.reward_list = [0.5, 0.1]
        cfg.curr_reward = 1.0
        cfg._reward_based_eps()  # pylint: disable=protected-access
        cfg._reward_based_alpha()  # pylint: disable=protected-access

        self.assertLess(cfg.curr_epsilon, cfg.epsilon_start)
        self.assertLess(cfg.curr_alpha, cfg.alpha_start)


class TestStateBased(TestCase):
    """State visitation update depends on counts table."""

    @mock.patch("reinforcement_learning.utils.hyperparams.get_q_table")
    def test_state_based_increments_counts(self, mock_q):
        """Counts increment and params change."""
        counts = {(0, 1): np.zeros(2)}
        mock_q.return_value = (counts, None)

        cfg = hp.HyperparamConfig(_engine_props(), _rl_props(), True)
        cfg.update_timestep_data((0, 1), 1)
        cfg._state_based_eps()  # pylint: disable=protected-access
        cfg._state_based_alpha()  # pylint: disable=protected-access

        self.assertEqual(counts[(0, 1)][1], 2)  # incremented twice
        self.assertLess(cfg.curr_epsilon, cfg.epsilon_start)
        self.assertLess(cfg.curr_alpha, 1)


class TestGetActivation(TestCase):
    """_get_activation maps strings to nn modules."""

    def test_mapping_returns_correct_class(self):
        trial = _mock_trial()
        act = hp._get_activation(trial)  # pylint: disable=protected-access
        from torch.nn import ReLU  # import locally to avoid heavy torch
        self.assertIs(act, ReLU)


class TestHyperparamSuggest(TestCase):
    """get_optuna_hyperparams branch selection."""

    def test_bandit_ucb_path(self):
        """Conf_param suggested only for UCB."""
        trial = _mock_trial()
        sim = {"path_algorithm": "ucb_bandit",
               "epsilon_update": "linear_decay",
               "alpha_update": "linear_decay",
               "num_requests": 10, "max_iters": 3}
        hps = hp.get_optuna_hyperparams(sim, trial)
        self.assertIn("conf_param", hps)
        self.assertIsNone(hps["epsilon_start"])  # UCB sets epsilon None

    def test_q_learning_includes_discount(self):
        """discount_factor returned for q_learning."""
        trial = _mock_trial()
        sim = {"path_algorithm": "q_learning",
               "epsilon_update": "exp_decay",
               "alpha_update": "exp_decay",
               "num_requests": 5, "max_iters": 2}
        hps = hp.get_optuna_hyperparams(sim, trial)
        self.assertIn("discount_factor", hps)

    def test_unknown_drl_algo_raises(self):
        """_drl_hyperparams NotImplementedError for bad algo."""
        trial = _mock_trial()
        sim = {"path_algorithm": "badalgo",
               "num_requests": 2, "max_iters": 1}
        with self.assertRaises(NotImplementedError):
            hp._drl_hyperparams(sim, trial)  # pylint: disable=protected-access
