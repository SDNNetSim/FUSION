# pylint: disable=duplicate-code
# TODO: (version 5.5-6) Address all duplicate code if you can

# pylint: disable=protected-access

"""Unit tests for fusion.modules.rl.algorithms/bandits module."""

from types import SimpleNamespace
from typing import Any
from unittest import mock

import numpy as np
import pytest

from fusion.modules.rl.algorithms import bandits
from fusion.modules.rl.errors import AlgorithmNotFoundError


# ----------------------------- helpers --------------------------------
def _mk_engine(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "k_paths": 3,
        "cores_per_link": 3,
        "conf_param": 2.0,
        "max_iters": 10,
        "save_step": 1,
        "num_requests": 1,
        "network": "net",
        "date": "d",
        "sim_start": "t0",
        "erlang": 30,
    }
    base.update(overrides)
    return base


def _mk_rl(num_nodes: int = 2) -> SimpleNamespace:
    return SimpleNamespace(num_nodes=num_nodes)


# ------------------------------ tests ---------------------------------
class TestGetBaseFp:
    """_get_base_fp behaviour."""

    def test_path_flag_true_returns_routes_str(self) -> None:
        """Route flag builds routes string."""
        res = bandits._get_base_fp(True, 30, 4)
        assert res == "e30_routes_c4"

    def test_path_flag_false_returns_cores_str(self) -> None:
        """Core flag builds cores string."""
        res = bandits._get_base_fp(False, 10, 2)
        assert res == "e10_cores_c2"


class TestSaveModelLowLevel:
    """_save_model JSON handling."""

    @mock.patch("fusion.modules.rl.algorithms.bandits.json.dump")
    @mock.patch("builtins.open", new_callable=mock.mock_open)
    @mock.patch("os.getcwd", return_value="/cwd")
    def test_save_model_writes_json_for_path(
        self,
        mock_cwd: mock.MagicMock,  # pylint: disable=unused-argument
        mock_open_fn: mock.MagicMock,
        mock_dump: mock.MagicMock,
    ) -> None:
        """_save_model converts arrays and dumps JSON."""
        data: dict[tuple[str, ...], np.ndarray[Any, Any]] = {("a",): np.array([1, 2])}
        bandits._save_model(
            state_values_dict=data,  # type: ignore[arg-type]
            erlang=20,
            cores_per_link=4,
            save_dir="save_dir",
            is_path=True,
            trial=0,
        )
        # tuples→str and ndarray→list
        expected = {"('a',)": [1, 2]}
        mock_dump.assert_called_once_with(expected, mock.ANY)

    def test_save_model_cores_not_implemented(self) -> None:
        """_save_model raises when is_path False."""
        with pytest.raises(AlgorithmNotFoundError):
            bandits._save_model({}, 10, 2, "d", False, 1)


class TestGetQTable:
    """get_q_table key construction."""

    def test_path_mode_creates_pair_keys(self) -> None:
        """Keys omit path index in path mode."""
        dummy = SimpleNamespace(
            num_nodes=2,
            is_path=True,
            n_arms=2,
            engine_props={"k_paths": 2},
            props=SimpleNamespace(),
            values={},
            counts={},
            source=0,
            dest=1,
            iteration=0,
        )
        counts, values = bandits.get_q_table(dummy)
        assert (0, 1) in counts
        assert (0, 1, 0) not in counts
        assert counts[(0, 1)].shape[0] == 2
        assert np.all(values[(0, 1)] == 0.0)


class TestUpdateBandit:
    """_update_bandit value updates."""

    @mock.patch("fusion.modules.rl.algorithms.bandits.save_model")
    def test_update_bandit_first_step_sets_value(
        self, mock_save_model: mock.MagicMock
    ) -> None:
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
            num_nodes=2,
            n_arms=3,
        )
        bandits._update_bandit(
            self=self_obj,
            iteration=0,
            reward=5.0,
            arm=0,
            algorithm="epsilon_greedy_bandit",
            trial=0,
        )
        assert self_obj.counts[(0, 1)][0] == 1
        assert self_obj.values[(0, 1)][0] == 5.0
        assert props.rewards_matrix == [[5.0]]
        mock_save_model.assert_called_once()


class TestEpsilonGreedy:
    """EpsilonGreedyBandit action selection."""

    @mock.patch("fusion.modules.rl.algorithms.bandits.np.random.rand", return_value=0.9)
    def test_get_action_exploits_when_rand_gt_eps(self, _: mock.MagicMock) -> None:
        """With rand>eps the greedy arm is chosen."""
        bandit = bandits.EpsilonGreedyBandit(
            rl_props=_mk_rl(), engine_props=_mk_engine(), is_path=True
        )
        bandit.epsilon = 0.1
        pair = (0, 1)
        # Ensure values dict has the required key
        if pair not in bandit.values:
            bandit.values[pair] = np.array([2.0, 1.0, 0.0])
        else:
            bandit.values[pair] = np.array([2.0, 1.0, 0.0])
        arm = bandit.select_path_arm(*pair)
        assert arm == 0

    @mock.patch(
        "fusion.modules.rl.algorithms.bandits.np.random.randint", return_value=2
    )
    @mock.patch("fusion.modules.rl.algorithms.bandits.np.random.rand", return_value=0.0)
    def test_get_action_explores_when_rand_lt_eps(
        self, _: mock.MagicMock, __: mock.MagicMock
    ) -> None:
        """With rand<eps a random arm is chosen."""
        bandit = bandits.EpsilonGreedyBandit(
            rl_props=_mk_rl(), engine_props=_mk_engine(), is_path=True
        )
        bandit.epsilon = 1.0
        # Ensure values dict is initialized properly
        pair = (0, 1)
        if pair not in bandit.values:
            bandit.values[pair] = np.array([0.0, 0.0, 0.0])
        arm = bandit.select_path_arm(0, 1)
        assert arm == 2


class TestUCB:
    """UCBBandit action logic."""

    def test_ucb_selects_uncounted_arm_first(self) -> None:
        """Zero-count arm is selected first."""
        bandit = bandits.UCBBandit(
            rl_props=_mk_rl(), engine_props=_mk_engine(), is_path=True
        )
        # Ensure counts and values dicts are initialized properly
        pair = (0, 1)
        if pair not in bandit.counts:
            bandit.counts[pair] = np.array([0, 0, 0])
        if pair not in bandit.values:
            bandit.values[pair] = np.array([0.0, 0.0, 0.0])
        # All counts zero → argmin returns 0
        arm = bandit.select_path_arm(0, 1)
        assert arm == 0

    def test_ucb_computes_confidence_bound(self) -> None:
        """UCB returns argmax of UCB values."""
        eng = _mk_engine(conf_param=1.0)
        bandit = bandits.UCBBandit(rl_props=_mk_rl(), engine_props=eng, is_path=True)
        pair = (0, 1)
        # Ensure the dicts are initialized first
        if pair not in bandit.counts:
            bandit.counts[pair] = np.array([5, 1, 1])
        else:
            bandit.counts[pair] = np.array([5, 1, 1])
        if pair not in bandit.values:
            bandit.values[pair] = np.array([0.2, 0.1, 0.0])
        else:
            bandit.values[pair] = np.array([0.2, 0.1, 0.0])
        arm = bandit.select_path_arm(*pair)
        assert arm == 1
