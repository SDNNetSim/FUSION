"""Unit tests for reinforcement_learning.utils.callbacks."""

from types import SimpleNamespace
from unittest import TestCase, mock

import numpy as np
from reinforcement_learning.utils import callbacks as cb


# ------------------------------------------------------------------ #
#  Helpers                                                            #
# ------------------------------------------------------------------ #
class _DummyPolicy:  # pylint: disable=too-few-public-methods
    """Lightweight policy exposing predict_values()."""

    def predict_values(self, obs):  # noqa: D401
        """
        Predict values.
        """
        # return shape (1,1) tensor-like array
        return np.array([[obs.sum()]])


def _dummy_model():
    """Return a minimal SB3-like model object."""
    model = SimpleNamespace()
    model.get_parameters = mock.MagicMock(return_value={"w": 1})
    model.policy = _DummyPolicy()
    model.ent_coef = 1.0
    model.learning_rate = 1e-3
    return model


def _mk_sim_dict(**extra):
    base = dict(
        max_iters=2,
        num_requests=3,
        save_step=1,
        erlang_start=1,
        cores_per_link=2,
        path_algorithm="ppo",
        network="net",
        date="d",
        sim_start="t0",
        epsilon_start=0.5,
        epsilon_end=0.1,
        decay_rate=0.5,
        alpha_start=1e-3,
        alpha_end=5e-4,
    )
    base.update(extra)
    return base


# ------------------------------------------------------------------ #
class TestGetModelParams(TestCase):
    """GetModelParams captures params and value on each step."""

    def test_on_step_sets_fields_and_returns_true(self):
        """_on_step stores get_parameters() & value_estimate."""
        alg = cb.GetModelParams()
        alg.model = _dummy_model()
        alg.locals = {"obs_tensor": np.array([1.0])}

        self.assertTrue(alg._on_step())  # pylint: disable=protected-access
        self.assertEqual(alg.model_params, {"w": 1})
        self.assertAlmostEqual(alg.value_estimate, 1.0)


# ------------------------------------------------------------------ #
class TestEpisodicRewardCallback(TestCase):
    """EpisodicRewardCallback reward bookkeeping."""

    def setUp(self):
        self.sim = _mk_sim_dict()
        self.cb = cb.EpisodicRewardCallback()
        self.cb.sim_dict = self.sim
        self.cb.max_iters = self.sim["max_iters"]

    @mock.patch("reinforcement_learning.utils.callbacks.create_dir")
    @mock.patch("reinforcement_learning.utils.callbacks.np.save")
    def test_first_call_creates_matrix_and_accumulates(
            self, mock_save, mock_dir
    ):
        """First step allocates rewards_matrix and records reward."""
        self.cb.locals = {"rewards": [2.0], "dones": [False]}
        self.assertTrue(self.cb._on_step())  # pylint: disable=protected-access

        self.assertEqual(self.cb.current_episode_reward, 2.0)
        self.assertEqual(self.cb.rewards_matrix.shape, (2, 3))
        mock_dir.assert_not_called()
        mock_save.assert_not_called()

    @mock.patch.object(cb.EpisodicRewardCallback, "_save_drl_trial_rewards")
    def test_done_saves_and_resets(self, mock_save):
        """Episode end saves trial rewards and resets counters."""
        self.cb.rewards_matrix = np.zeros((2, 3))
        self.cb.locals = {"rewards": [1.0], "dones": [True]}
        self.cb.iter = 0
        self.cb.curr_step = 0

        self.cb._on_step()  # pylint: disable=protected-access

        self.assertEqual(self.cb.episode_rewards.tolist(), [1.0])
        self.assertEqual(self.cb.iter, 1)
        self.assertEqual(self.cb.curr_step, 0)
        mock_save.assert_called_once()


# ------------------------------------------------------------------ #
class TestLearnRateEntCallback(TestCase):
    """LearnRateEntCallback parameter decay."""

    def setUp(self):
        self.sim = _mk_sim_dict()
        self.lr_cb = cb.LearnRateEntCallback(verbose=0)
        self.lr_cb.sim_dict = self.sim
        self.lr_cb.model = _dummy_model()

    def test_first_done_initialises_and_sets_params(self):
        """First done sets ent_coef and learning_rate."""
        self.lr_cb.locals = {"dones": [True]}
        self.assertTrue(self.lr_cb._on_step())  # pylint: disable=protected-access

        self.assertAlmostEqual(self.lr_cb.current_ent, 0.25)  # expected entropy
        self.assertAlmostEqual(self.lr_cb.current_lr, 0.00075)
        self.assertAlmostEqual(self.lr_cb.model.ent_coef, 0.25)
        self.assertAlmostEqual(self.lr_cb.model.learning_rate, 0.00075)

    def test_subsequent_done_decays_and_updates(self):
        """Later episodes decay ent_coef and adjust lr linearly."""
        # Pretend first episode already ran
        self.lr_cb.current_ent = 0.4
        self.lr_cb.current_lr = 0.0009
        self.lr_cb.iter = 1

        self.lr_cb.locals = {"dones": [True]}
        self.lr_cb._on_step()  # pylint: disable=protected-access

        self.assertAlmostEqual(self.lr_cb.model.ent_coef, 0.2)  # 0.4*0.5

        expected_lr = 5e-4  # alpha_end reached after second episode
        self.assertAlmostEqual(self.lr_cb.model.learning_rate, expected_lr)
