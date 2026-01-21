"""Unit tests for fusion.modules.rl.utils.hyperparams module."""

# TODO: (version 5.5-6) All tests need to be double checked and looked at

# pylint: disable=unused-argument

from types import SimpleNamespace
from typing import Any
from unittest import mock

import numpy as np
import pytest

from fusion.modules.rl.utils import hyperparams as hp


# ---------------------------- stubs ----------------------------------
def _engine_props() -> dict[str, Any]:
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


def _rl_props(nodes: int = 3) -> SimpleNamespace:
    return SimpleNamespace(num_nodes=nodes)


def _mock_trial() -> SimpleNamespace:
    """Return an optuna-like trial returning fixed values."""
    trial = SimpleNamespace()

    # suggest_float
    def _sf(name: str, low: Any = None, _high: Any = None, **_kw: Any) -> Any:
        return {"gamma": 0.95, "clip_range": 0.2}.get(name, low)

    # suggest_int
    def _si(name: str, low: Any = None, _high: Any = None, **_kw: Any) -> Any:
        return low

    def _sc(name: str, choices: list[Any]) -> Any:  # suggest_categorical
        return choices[0]

    trial.suggest_float = _sf
    trial.suggest_int = _si
    trial.suggest_categorical = _sc
    return trial


# ----------------------------- tests ---------------------------------
class TestLinearDecay:
    """Linear epsilon / alpha decay."""

    @mock.patch("fusion.modules.rl.utils.hyperparams.get_q_table", return_value=(None, None))
    def test_linear_decay_updates_values(self, _: mock.MagicMock) -> None:
        """_linear_eps / _linear_alpha compute expected value."""
        cfg = hp.HyperparamConfig(_engine_props(), _rl_props(), is_path=True)
        cfg.iteration = 2
        cfg._linear_eps()  # pylint: disable=protected-access
        cfg._linear_alpha()  # pylint: disable=protected-access

        assert cfg.current_epsilon == pytest.approx(0.5)  # halfway
        assert cfg.current_alpha == pytest.approx(0.25)


class TestExponentialDecay:
    """Exponential epsilon / alpha decay."""

    @mock.patch("fusion.modules.rl.utils.hyperparams.get_q_table", return_value=(None, None))
    def test_exp_decay(self, _: mock.MagicMock) -> None:
        """exp decay = start * rate**iter."""
        props = _engine_props() | {
            "alpha_update": "exp_decay",
            "epsilon_update": "exp_decay",
        }
        cfg = hp.HyperparamConfig(props, _rl_props(), is_path=True)
        cfg.iteration = 3
        cfg._exp_eps()  # pylint: disable=protected-access
        cfg._exp_alpha()  # pylint: disable=protected-access

        rate = props["decay_rate"]
        assert cfg.current_epsilon == pytest.approx(props["epsilon_start"] * rate**3)
        assert cfg.current_alpha == pytest.approx(props["alpha_start"] * rate**3)


class TestRewardBased:
    """Reward-based update reduces params when reward diff grows."""

    @mock.patch("fusion.modules.rl.utils.hyperparams.get_q_table", return_value=(None, None))
    def test_reward_based_updates(self, _: mock.MagicMock) -> None:
        """Greater diff → smaller epsilon/alpha."""
        cfg = hp.HyperparamConfig(_engine_props(), _rl_props(), True)
        cfg.reward_list = [0.5, 0.1]
        cfg.current_reward = 1.0
        cfg._reward_based_eps()  # pylint: disable=protected-access
        cfg._reward_based_alpha()  # pylint: disable=protected-access

        assert cfg.current_epsilon < cfg.epsilon_start
        assert cfg.current_alpha < cfg.alpha_start


class TestStateBased:
    """State visitation update depends on counts table."""

    @mock.patch("fusion.modules.rl.utils.hyperparams.get_q_table")
    def test_state_based_increments_counts(self, mock_q: mock.MagicMock) -> None:
        """Counts increment and params change."""
        counts = {(0, 1): np.zeros(2)}
        mock_q.return_value = (counts, None)

        cfg = hp.HyperparamConfig(_engine_props(), _rl_props(), True)
        cfg.update_timestep_data((0, 1), 1)
        cfg._state_based_eps()  # pylint: disable=protected-access
        cfg._state_based_alpha()  # pylint: disable=protected-access

        assert counts[(0, 1)][1] == 2  # incremented twice
        assert cfg.current_epsilon < cfg.epsilon_start
        assert cfg.current_alpha < 1


class TestHyperparamSuggest:
    """get_optuna_hyperparams branch selection."""

    def test_bandit_ucb_path(self) -> None:
        """Conf_param suggested only for UCB."""
        trial = _mock_trial()
        sim = {
            "path_algorithm": "ucb_bandit",
            "epsilon_update": "linear_decay",
            "alpha_update": "linear_decay",
            "num_requests": 10,
            "max_iters": 3,
        }
        hps = hp.get_optuna_hyperparams(sim, trial)  # type: ignore[arg-type]
        assert "conf_param" in hps
        assert hps["epsilon_start"] is None  # UCB sets epsilon None

    def test_q_learning_includes_discount(self) -> None:
        """discount_factor returned for q_learning."""
        trial = _mock_trial()
        sim = {
            "path_algorithm": "q_learning",
            "epsilon_update": "exp_decay",
            "alpha_update": "exp_decay",
            "num_requests": 5,
            "max_iters": 2,
        }
        hps = hp.get_optuna_hyperparams(sim, trial)  # type: ignore[arg-type]
        assert "discount_factor" in hps

    def test_unknown_drl_algo_raises(self) -> None:
        """_drl_hyperparams NotImplementedError for bad algo."""
        trial = _mock_trial()
        sim = {"path_algorithm": "badalgo", "num_requests": 2, "max_iters": 1}
        with pytest.raises(hp.HyperparameterError):
            hp._drl_hyperparams(sim, trial)  # type: ignore[arg-type]  # pylint: disable=protected-access
