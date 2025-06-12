# pylint: disable=duplicate-code
# TODO: (version 5.5-6) Address all duplicate code if you can

"""Unit tests for reinforcement_learning.algorithms.dqn."""

from types import SimpleNamespace
from unittest import TestCase, mock

from reinforcement_learning.algorithms import dqn


class TestDQN(TestCase):
    """Unit tests for DQN."""

    # --------------------------- helpers ------------------------------
    @staticmethod
    def _mk_engine(k_paths=3):
        """Return minimal stub for engine_obj."""
        return SimpleNamespace(engine_props={"k_paths": k_paths})

    @staticmethod
    def _mk_agent(k_paths=3):
        """Return DQN with stubbed props and engine."""
        rl_props = SimpleNamespace()  # contents unused in tests
        return dqn.DQN(rl_props, TestDQN._mk_engine(k_paths))

    # ---------------------- get_obs_space -----------------------------
    @mock.patch(
        "reinforcement_learning.algorithms.dqn.spaces.Dict"
    )
    @mock.patch(
        "reinforcement_learning.algorithms.dqn.get_observation_space",
        return_value={"a": 1},
    )
    def test_get_obs_space_wraps_dict(
            self, mock_get_obs, mock_dict_space
    ):
        """get_obs_space returns gym Dict from helper output."""
        agent = self._mk_agent()
        result = agent.get_obs_space()

        mock_get_obs.assert_called_once_with(
            rl_props=agent.rl_props, engine_obj=agent.engine_obj
        )
        mock_dict_space.assert_called_once_with({"a": 1})
        self.assertIs(result, mock_dict_space.return_value)

    # ------------------- get_action_space -----------------------------
    @mock.patch(
        "reinforcement_learning.algorithms.dqn.spaces.Discrete"
    )
    def test_get_action_space_uses_k_paths(self, mock_discrete):
        """get_action_space returns Discrete(k_paths)."""
        agent = self._mk_agent(k_paths=5)
        result = agent.get_action_space()

        mock_discrete.assert_called_once_with(5)
        self.assertIs(result, mock_discrete.return_value)

    def test_get_action_space_missing_k_paths_raises(self):
        """get_action_space raises KeyError if k_paths absent."""
        agent = self._mk_agent(k_paths=None)
        agent.engine_obj.engine_props.pop("k_paths")

        with self.assertRaises(KeyError):
            agent.get_action_space()
