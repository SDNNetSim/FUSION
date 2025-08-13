# pylint: disable=protected-access

"""Unit tests for reinforcement_learning.agents.path_agent."""

from types import SimpleNamespace
from unittest import TestCase, mock

import numpy as np
from fusion.modules.rl.agents import path_agent


class TestPathAgent(TestCase):
    """Unit-tests for PathAgent behaviour."""

    # ------------------------------------------------------------------
    def _mk_agent(self, alg="q_learning"):
        """Return PathAgent with stubbed attrs."""
        rl_props = SimpleNamespace(
            source=0,
            destination=1,
            k_paths=2,
            chosen_path_index=None,
            chosen_path_list=None,
            paths_list=None,
        )
        agent = path_agent.PathAgent(
            path_algorithm=alg,
            rl_props=rl_props,
            rl_help_obj=mock.MagicMock(),
        )
        agent.engine_props = {
            "path_algorithm": alg,
            "max_iters": 10,
            "dynamic_reward": False,
        }
        agent.reward_penalty_list = [0.0] * 10
        agent.algorithm_obj = mock.MagicMock()
        agent.hyperparam_obj = mock.MagicMock(
            iteration=0,
            alpha_strategy="episodic",
            epsilon_strategy="episodic",
            fully_episodic=False,
            curr_epsilon=0.2,
            curr_alpha=0.5,
        )
        return agent

    # ---------------------------- end_iter ----------------------------
    @mock.patch.object(path_agent, "EPISODIC_STRATEGIES", ["episodic"])
    def test_end_iter_updates_hyperparams(self):
        """end_iter increments iteration and updates α, ε."""
        agent = self._mk_agent()
        agent.end_iter()
        self.assertEqual(agent.hyperparam_obj.iteration, 1)
        agent.hyperparam_obj.update_alpha.assert_called_once()
        agent.hyperparam_obj.update_eps.assert_called_once()

    @mock.patch.object(path_agent, "EPISODIC_STRATEGIES", ["episodic"])
    def test_end_iter_skips_updates_for_bandit_ucb(self):
        """end_iter skips α/ε update for bandit and ucb cases."""
        agent = self._mk_agent(alg="ucb_bandit")
        agent.end_iter()
        agent.hyperparam_obj.update_alpha.assert_not_called()
        agent.hyperparam_obj.update_eps.assert_not_called()

    # ----------------------- _handle_hyperparams ----------------------
    @mock.patch.object(path_agent, "EPISODIC_STRATEGIES", ["episodic"])
    def test_handle_hyperparams_updates_state_and_rates(self):
        """_handle_hyperparams sets state-action and updates α, ε."""
        agent = self._mk_agent()
        agent.hyperparam_obj.alpha_strategy = "non_episodic"
        agent.hyperparam_obj.epsilon_strategy = "non_episodic"
        agent._handle_hyperparams()
        self.assertEqual(
            agent.state_action_pair, (agent.rl_props.source,
                                      agent.rl_props.destination)
        )
        self.assertEqual(agent.action_index, agent.rl_props.chosen_path_index)
        agent.hyperparam_obj.update_timestep_data.assert_called_once()
        agent.hyperparam_obj.update_alpha.assert_called_once()
        agent.hyperparam_obj.update_eps.assert_called_once()

    # ----------------------------- update -----------------------------
    def test_update_raises_at_max_iters(self):
        """update raises when iteration exceeds max_iters."""
        agent = self._mk_agent()
        agent.hyperparam_obj.iteration = 10
        with self.assertRaises(ValueError):
            agent.update(True, {}, 0, 5, 0)

    @mock.patch.object(path_agent, "VALID_DRL_ALGORITHMS", ["ppo"])
    def test_update_returns_early_for_drl(self):
        """update returns early for DRL algorithms."""
        agent = self._mk_agent(alg="ppo")
        with mock.patch.object(agent, "get_reward") as mock_reward:
            agent.update(True, {}, 0, 5, 0)
            mock_reward.assert_not_called()

    # ------------------------- _bandit_route --------------------------
    def test_bandit_route_selects_arm_and_sets_attrs(self):
        """_bandit_route sets chosen indices for bandit agent."""
        agent = self._mk_agent(alg="epsilon_greedy_bandit")
        agent.hyperparam_obj.curr_epsilon = 0.3
        agent.algorithm_obj.select_path_arm.return_value = 1

        paths = np.array([[0, 1, 2], [0, 3, 1]])
        route_obj = SimpleNamespace(route_props=SimpleNamespace(
            paths_matrix=paths
        ))

        agent._bandit_route(route_obj)

        self.assertEqual(agent.algorithm_obj.epsilon, 0.3)
        self.assertEqual(agent.rl_props.chosen_path_index, 1)
        np.testing.assert_array_equal(
            agent.rl_props.chosen_path_list, paths[1]
        )

    # -------------------------- _drl_route ----------------------------
    def test_drl_route_sets_path(self):
        """_drl_route assigns path for DRL agent."""
        agent = self._mk_agent(alg="ppo")
        paths = np.array([[0, 1, 2], [0, 3, 1]])
        route_obj = SimpleNamespace(route_props=SimpleNamespace(
            paths_matrix=paths
        ))

        agent._drl_route(route_obj=route_obj, action=0)
        self.assertEqual(agent.rl_props.chosen_path_index, 0)
        np.testing.assert_array_equal(
            agent.rl_props.chosen_path_list, paths[0]
        )

    def test_drl_route_unsupported_algo_raises(self):
        """_drl_route raises for unsupported algorithm."""
        agent = self._mk_agent(alg="epsilon_greedy_bandit")
        with self.assertRaises(NotImplementedError):
            agent._drl_route(route_obj=mock.MagicMock(), action=0)
