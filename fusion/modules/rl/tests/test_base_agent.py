# reinforcement_learning/tests/test_base_agent.py
"""Unit-test suite for reinforcement_learning.agents.base_agent."""

from unittest import TestCase, mock

from fusion.modules.rl.agents import base_agent
from fusion.modules.rl.errors import AlgorithmNotFoundError


class TestBaseAgent(TestCase):
    """Unit tests for BaseAgent."""

    # ------- global patching ------------------------------------------
    def setUp(self):
        """Patch HyperparamConfig for all tests."""
        patcher = mock.patch(
            "fusion.modules.rl.agents.base_agent.HyperparamConfig"
        )
        self._mock_hpc = patcher.start()
        self.addCleanup(patcher.stop)

    # ------- helpers ---------------------------------------------------
    def _minimal_engine_props(self):
        """Return minimal engine-prop dict."""
        return {
            "max_iters": 10,
            "penalty": 2.0,
            "gamma": 0.5,
            "reward": 10.0,
            "decay_factor": 0.5,
            "num_requests": 100,
            "core_beta": 1.0,
        }

    def _new_agent(self, algorithm="q_learning"):
        """Return BaseAgent with stubbed props."""
        agent = base_agent.BaseAgent(
            algorithm=algorithm,
            rl_props={},  # not used in tested paths
            rl_help_obj=None,
        )
        agent.engine_props = self._minimal_engine_props()
        return agent

    # ------- calculate_* ----------------------------------------------
    def test_calculate_dynamic_penalty_value(self):
        """Dynamic penalty equals core_index × beta."""
        agent = self._new_agent()
        self.assertEqual(agent.calculate_dynamic_penalty(4, 2), 4.0)

    def test_calculate_dynamic_reward_value(self):
        """Dynamic reward decays per req_id."""
        agent = self._new_agent()
        self.assertAlmostEqual(agent.calculate_dynamic_reward(2, 10), 4.5)

    # ------- get_reward -----------------------------------------------
    def test_get_reward_static_allocated(self):
        """Allocated+static returns fixed reward."""
        agent = self._new_agent()
        self.assertEqual(
            agent.get_reward(True, False, 0, 0), agent.engine_props["reward"]
        )

    def test_get_reward_dynamic_allocated(self):
        """Allocated+dynamic delegates to dynamic reward."""
        agent = self._new_agent()
        with mock.patch.object(
                agent, "calculate_dynamic_reward", return_value=7.7
        ) as dyn_reward:
            self.assertEqual(agent.get_reward(True, True, 3, 4), 7.7)
            dyn_reward.assert_called_once_with(3, 4)

    def test_get_reward_static_not_allocated(self):
        """Blocked+static returns fixed penalty."""
        agent = self._new_agent()
        self.assertEqual(
            agent.get_reward(False, False, 0, 0), agent.engine_props["penalty"]
        )

    def test_get_reward_dynamic_not_allocated(self):
        """Blocked+dynamic delegates to dynamic penalty."""
        agent = self._new_agent()
        with mock.patch.object(
                agent, "calculate_dynamic_penalty", return_value=-9.9
        ) as dyn_pen:
            self.assertEqual(agent.get_reward(False, True, 5, 6), -9.9)
            dyn_pen.assert_called_once_with(5, 6)

    # ------- setup_env -------------------------------------------------
    @mock.patch("fusion.modules.rl.agents.base_agent.QLearning")
    def test_setup_env_chooses_q_learning(self, mock_qlearn):
        """setup_env instantiates QLearning."""
        mock_qlearn.return_value = mock.MagicMock()
        agent = self._new_agent("q_learning")
        agent.setup_env(is_path=False)
        mock_qlearn.assert_called_once_with(
            rl_props=agent.rl_props, engine_props=agent.engine_props
        )
        self.assertIs(agent.algorithm_obj, mock_qlearn.return_value)

    def test_setup_env_bad_algorithm_raises(self):
        """Unknown algorithm raises AlgorithmNotFoundError."""
        agent = self._new_agent("unknown_algo")
        with self.assertRaises(AlgorithmNotFoundError):
            agent.setup_env(is_path=False)

    # ------- load_model -----------------------------------------------
    @mock.patch("fusion.modules.rl.agents.base_agent.np.load",
                return_value="dummy_matrix")
    @mock.patch("fusion.modules.rl.agents.base_agent.os.path.join",
                return_value="joined/path.npy")
    @mock.patch("fusion.modules.rl.agents.base_agent.QLearning")
    def test_load_model_sets_matrix(
            self, mock_qlearn, mock_join, mock_npload
    ):
        """load_model reads file and assigns cores_matrix."""
        alg_instance = mock_qlearn.return_value
        alg_instance.props = mock.MagicMock()

        agent = self._new_agent("q_learning")
        agent.load_model(
            model_path="my_run",
            file_prefix="core",
            erlang=60,
            num_cores=4,
            is_path=False,
        )

        mock_join.assert_called_once_with(
            "logs", "my_run", "core_e60_c4.npy"
        )
        mock_npload.assert_called_once_with("joined/path.npy",
                                            allow_pickle=True)
        self.assertEqual(alg_instance.props.cores_matrix, "dummy_matrix")


if __name__ == "__main__":  # pragma: no cover
    import unittest

    unittest.main()
