"""Unit tests for fusion.modules.rl.utils.callbacks module."""

from types import SimpleNamespace
from typing import Any
from unittest import mock

import numpy as np
import pytest

from fusion.modules.rl.utils import callbacks as cb


# ------------------------------------------------------------------ #
#  Helpers                                                            #
# ------------------------------------------------------------------ #
class _DummyPolicy:  # pylint: disable=too-few-public-methods
    """Lightweight policy exposing predict_values()."""

    def predict_values(self, obs: np.ndarray) -> np.ndarray:  # noqa: D401
        """
        Predict values.
        """
        # return shape (1,1) tensor-like array
        return np.array([[obs.sum()]])


def _dummy_model() -> Any:
    """Return a minimal SB3-like model object."""
    model = SimpleNamespace()
    model.get_parameters = mock.MagicMock(return_value={"w": 1})
    model.policy = _DummyPolicy()
    model.ent_coef = 1.0
    model.learning_rate = 1e-3
    return model


def _mk_sim_dict(**extra: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "max_iters": 2,
        "num_requests": 3,
        "save_step": 1,
        "erlang_start": 1,
        "cores_per_link": 2,
        "path_algorithm": "ppo",
        "network": "net",
        "date": "d",
        "sim_start": "t0",
        "epsilon_start": 0.5,
        "epsilon_end": 0.1,
        "decay_rate": 0.5,
        "alpha_start": 1e-3,
        "alpha_end": 5e-4,
    }
    base.update(extra)
    return base


# ------------------------------------------------------------------ #
class TestGetModelParams:
    """GetModelParams captures params and value on each step."""

    def test_on_step_sets_fields_and_returns_true(self) -> None:
        """_on_step stores get_parameters() & value_estimate."""
        alg = cb.GetModelParams()
        alg.model = _dummy_model()  # type: ignore[assignment]
        alg.locals = {"obs_tensor": np.array([1.0])}

        assert alg._on_step() is True  # pylint: disable=protected-access
        assert alg.model_params == {"w": 1}
        assert alg.value_estimate == pytest.approx(1.0)


# ------------------------------------------------------------------ #
class TestEpisodicRewardCallback:
    """EpisodicRewardCallback reward bookkeeping."""

    @pytest.fixture
    def callback(self) -> cb.EpisodicRewardCallback:
        """Create callback with sim_dict."""
        sim = _mk_sim_dict()
        cb_obj = cb.EpisodicRewardCallback()
        cb_obj.sim_dict = sim
        cb_obj.max_iters = sim["max_iters"]
        return cb_obj

    @mock.patch("fusion.modules.rl.utils.callbacks.create_directory")
    @mock.patch("fusion.modules.rl.utils.callbacks.np.save")
    def test_first_call_creates_matrix_and_accumulates(
        self,
        mock_save: mock.MagicMock,
        mock_dir: mock.MagicMock,
        callback: cb.EpisodicRewardCallback,
    ) -> None:
        """First step allocates rewards_matrix and records reward."""
        callback.locals = {"rewards": [2.0], "dones": [False]}
        assert callback._on_step() is True  # pylint: disable=protected-access

        assert callback.current_episode_reward == 2.0
        assert callback.rewards_matrix is not None
        assert callback.rewards_matrix.shape == (2, 3)
        mock_dir.assert_not_called()
        mock_save.assert_not_called()

    @mock.patch.object(cb.EpisodicRewardCallback, "_save_drl_trial_rewards")
    def test_done_saves_and_resets(self, mock_save: mock.MagicMock, callback: cb.EpisodicRewardCallback) -> None:
        """Episode end saves trial rewards and resets counters."""
        callback.rewards_matrix = np.zeros((2, 3))
        callback.locals = {"rewards": [1.0], "dones": [True]}
        callback.iteration = 0
        callback.current_step = 0
        callback.current_episode_reward = 0  # Initialize current episode reward

        callback._on_step()  # pylint: disable=protected-access

        assert callback.episode_rewards.tolist() == [1.0]
        assert callback.iteration == 1
        assert callback.current_step == 0
        mock_save.assert_called_once()


# ------------------------------------------------------------------ #
class TestLearnRateEntCallback:
    """LearnRateEntCallback parameter decay."""

    @pytest.fixture
    def lr_callback(self) -> cb.LearnRateEntCallback:
        """Create LearnRateEntCallback with model and sim_dict."""
        sim = _mk_sim_dict()
        lr_cb = cb.LearnRateEntCallback(verbose=0)
        lr_cb.sim_dict = sim
        lr_cb.model = _dummy_model()
        return lr_cb

    def test_first_done_initialises_and_sets_params(self, lr_callback: cb.LearnRateEntCallback) -> None:
        """First done sets ent_coef and learning_rate."""
        lr_callback.locals = {"dones": [True]}
        assert lr_callback._on_step() is True  # pylint: disable=protected-access

        assert lr_callback.current_entropy == pytest.approx(0.25)  # expected entropy
        assert lr_callback.current_learning_rate == pytest.approx(0.00075)
        assert lr_callback.model.ent_coef == pytest.approx(0.25)  # type: ignore[attr-defined]
        assert lr_callback.model.learning_rate == pytest.approx(0.00075)  # type: ignore[attr-defined]

    def test_subsequent_done_decays_and_updates(self, lr_callback: cb.LearnRateEntCallback) -> None:
        """Later episodes decay ent_coef and adjust lr linearly."""
        # Pretend first episode already ran
        lr_callback.current_entropy = 0.4
        lr_callback.current_learning_rate = 0.0009
        lr_callback.iteration = 1

        lr_callback.locals = {"dones": [True]}
        lr_callback._on_step()  # pylint: disable=protected-access

        assert lr_callback.model.ent_coef == pytest.approx(0.2)  # type: ignore[attr-defined]  # 0.4*0.5

        expected_lr = 5e-4  # alpha_end reached after second episode
        assert lr_callback.model.learning_rate == pytest.approx(expected_lr)  # type: ignore[attr-defined]
