# pylint: disable=duplicate-code
# TODO: (version 5.5-6) Address all duplicate code if you can

"""Unit tests for fusion.modules.rl.algorithms.ppo module."""

from types import SimpleNamespace
from unittest import mock

import pytest

from fusion.modules.rl.algorithms import ppo


class TestPPO:
    """Unit tests for PPO."""

    # --------------------------- helpers ------------------------------
    @staticmethod
    def _mk_engine(k_paths: int = 3) -> SimpleNamespace:
        """Return minimal stub for engine_obj."""
        return SimpleNamespace(engine_props={"k_paths": k_paths})

    @staticmethod
    def _mk_agent(k_paths: int = 3) -> ppo.PPO:
        """Return PPO with stubbed props and engine."""
        rl_props = SimpleNamespace()  # contents unused in these tests
        return ppo.PPO(rl_props, TestPPO._mk_engine(k_paths))

    # ---------------------- get_obs_space -----------------------------
    @mock.patch("fusion.modules.rl.algorithms.ppo.spaces.Dict")
    @mock.patch(
        "fusion.modules.rl.algorithms.base_drl.get_observation_space",
        return_value={"a": 1},
    )
    def test_get_obs_space_wraps_dict(
        self, mock_get_obs: mock.MagicMock, mock_dict_space: mock.MagicMock
    ) -> None:
        """get_obs_space returns gym Dict from helper output."""
        agent = self._mk_agent()
        result = agent.get_obs_space()

        mock_get_obs.assert_called_once_with(
            rl_props=agent.rl_props, engine_props=agent.engine_obj
        )
        mock_dict_space.assert_called_once_with({"a": 1})
        assert result is mock_dict_space.return_value

    # ------------------- get_action_space -----------------------------
    @mock.patch("fusion.modules.rl.algorithms.ppo.spaces.Discrete")
    def test_get_action_space_uses_k_paths(self, mock_discrete: mock.MagicMock) -> None:
        """get_action_space returns Discrete(k_paths)."""
        agent = self._mk_agent(k_paths=5)
        result = agent.get_action_space()

        mock_discrete.assert_called_once_with(5)
        assert result is mock_discrete.return_value

    def test_get_action_space_missing_k_paths_raises(self) -> None:
        """get_action_space raises KeyError if k_paths absent."""
        agent = self._mk_agent(k_paths=3)
        agent.engine_obj.engine_props.pop("k_paths")

        with pytest.raises(KeyError):
            agent.get_action_space()
