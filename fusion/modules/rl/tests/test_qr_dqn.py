"""Unit tests for reinforcement_learning.algorithms.qr_dqn."""

from types import SimpleNamespace
from unittest import TestCase, mock

from fusion.modules.rl.algorithms import qr_dqn


class TestQrDQN(TestCase):
    """Unit tests for QrDQN."""

    # --------------------------- helpers ------------------------------
    @staticmethod
    def _mk_engine(k_paths=3):
        """Return engine stub with engine_props."""
        return SimpleNamespace(engine_props={"k_paths": k_paths})

    @staticmethod
    def _mk_agent(k_paths=3):
        """Return QrDQN with stubbed props and engine."""
        rl_props = SimpleNamespace()  # unused in these tests
        return qr_dqn.QrDQN(rl_props, TestQrDQN._mk_engine(k_paths))

    # ---------------------- get_obs_space -----------------------------
    @mock.patch(
        "fusion.modules.rl.algorithms.qr_dqn.spaces.Dict"
    )
    @mock.patch(
        "fusion.modules.rl.algorithms.base_drl.get_observation_space",
        return_value={"x": 1},
    )
    def test_get_obs_space_wraps_dict(
            self, mock_get_obs, mock_dict_space
    ):
        """get_obs_space returns gym Dict from helper output."""
        agent = self._mk_agent()
        result = agent.get_obs_space()

        mock_get_obs.assert_called_once_with(
            rl_props=agent.rl_props, engine_props=agent.engine_obj
        )
        mock_dict_space.assert_called_once_with({"x": 1})
        self.assertIs(result, mock_dict_space.return_value)

    # ------------------- get_action_space -----------------------------
    @mock.patch(
        "fusion.modules.rl.algorithms.qr_dqn.spaces.Discrete"
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
