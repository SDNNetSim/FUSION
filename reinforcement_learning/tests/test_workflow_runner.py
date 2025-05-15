"""Unit tests for reinforcement_learning.workflow_runner."""

from types import SimpleNamespace
from unittest import TestCase, mock

import numpy as np
from reinforcement_learning import workflow_runner as wr


# ---------------------------------------------------------------------
class TestSetupCallbacks(TestCase):
    """_setup_callbacks propagates attrs to every callback."""

    def test_attrs_are_set(self):
        """Each callback gets max_iters and sim_dict."""
        cb1 = SimpleNamespace()
        cb2 = SimpleNamespace()
        cb_list = SimpleNamespace(callbacks=[cb1, cb2])

        my_sim = {"max_iters": 5}

        wr._setup_callbacks(cb_list, my_sim)  # pylint: disable=protected-access

        for cb in (cb1, cb2):
            self.assertEqual(cb.max_iters, 5)
            self.assertIs(cb.sim_dict, my_sim)


# ---------------------------------------------------------------------
class TestUpdateEpisodeStats(TestCase):
    """_update_episode_stats handles rewards and resets."""

    def setUp(self):
        self.sim = {"max_iters": 2, "n_trials": 1}
        self.env = SimpleNamespace(
            iteration=0,
            trial=0,
            reset=mock.MagicMock(return_value=("new_obs", {})),
        )

    def _call(self, **kw):
        """Helper that fills defaults."""
        defaults = dict(
            obs="obs",
            reward=1,
            terminated=False,
            truncated=False,
            episodic_reward=0,
            episodic_rew_arr=np.array([]),
            completed_episodes=0,
            completed_trials=0,
            env=self.env,
            sim_dict=self.sim,
            rewards_matrix=np.zeros((1, 2)),
            trial=None,
        )
        defaults.update(kw)
        return wr._update_episode_stats(**defaults)  # pylint: disable=protected-access

    def test_no_termination_accumulates_reward(self):
        """When not done, reward is accumulated and no reset occurs."""
        new_obs, ep_rew, arr, comp_ep, comp_tr = self._call()[:5]

        self.assertEqual(ep_rew, 1)
        self.assertTrue(np.array_equal(arr, np.array([])))
        self.assertEqual(comp_ep, 0)
        self.assertEqual(comp_tr, 0)
        self.assertEqual(new_obs, "obs")  # unchanged
        self.env.reset.assert_not_called()

    def test_episode_end_records_reward_and_resets(self):
        """Termination appends reward, resets env, and increments counters."""
        new_obs, ep_rew, arr, comp_ep, comp_tr = self._call(
            terminated=True, episodic_reward=3
        )[:5]

        self.assertEqual(ep_rew, 0)  # cleared
        self.assertEqual(arr.tolist(), [4])  # 3+1 recorded
        self.assertEqual(comp_ep, 1)
        self.assertEqual(comp_tr, 0)
        self.assertEqual(new_obs, "new_obs")
        self.env.reset.assert_called_once_with(seed=0)

    def test_trial_completion_resets_iteration_and_trial(self):
        """Completing max_iters episodes moves to next trial."""
        # First call ends episode 1
        _, _, _, comp_ep, _ = self._call(terminated=True)
        # Second call ends episode 2 → triggers trial increment
        _, _, _, comp_ep2, comp_tr2 = self._call(
            terminated=True, completed_episodes=comp_ep,
            episodic_rew_arr=np.array([1]), completed_trials=0
        )[:5]

        self.assertEqual(comp_ep2, 0)  # episode counter reset
        self.assertEqual(comp_tr2, 1)  # trial counter incremented
        self.assertEqual(self.env.trial, 1)
        self.env.reset.assert_called()  # called twice in total
