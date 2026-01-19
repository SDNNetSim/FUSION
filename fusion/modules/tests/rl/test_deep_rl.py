"""Unit tests for fusion.modules.rl.utils.deep_rl module."""

from types import SimpleNamespace
from typing import Any
from unittest import mock

import pytest

from fusion.modules.rl.utils import deep_rl as drl


# ---------------------------- helpers ---------------------------------
class _DummyAlgo:  # pylint: disable=too-few-public-methods
    """Minimal algorithm with obs/action space helpers."""

    def __init__(self, rl_props: Any, engine_props: Any) -> None:
        self.rl_props = rl_props
        self.engine_obj = engine_props

    def get_obs_space(self) -> str:
        """
        Mocks an observation space.
        """
        return "obs_space"

    def get_action_space(self) -> str:
        """
        Mocks an action space.
        """
        return "act_space"


def _engine() -> SimpleNamespace:
    return SimpleNamespace(engine_props={})


def _patch_globals(valid_list: list[str] | None = None, registry: dict[str, Any] | None = None) -> tuple[Any, Any]:
    """Patch VALID_PATH_ALGORITHMS and get_algorithm_registry."""
    vp = mock.patch.object(drl, "VALID_PATH_ALGORITHMS", valid_list if valid_list is not None else [])
    rg = mock.patch.object(
        drl,
        "get_algorithm_registry",
        return_value=registry if registry is not None else {},
    )
    return vp, rg


# ------------------------------ tests ---------------------------------
class TestGetAlgorithmInstance:
    """get_algorithm_instance branching."""

    def test_missing_underscore_raises(self) -> None:
        """ValueError when model_type has no underscore."""
        with (
            mock.patch.object(drl, "determine_model_type", return_value="ppo"),
            pytest.raises(drl.ConfigurationError),
        ):
            drl.get_algorithm_instance({}, None, _engine())

    def test_non_drl_algorithm_returns_none_and_flags(self) -> None:
        """Non-DRL path returns None and sets flag false."""
        vp, rg = _patch_globals(valid_list=["ksp"], registry={})
        with (
            vp,
            rg,
            mock.patch.object(drl, "determine_model_type", return_value="ksp_path"),
        ):
            sim = {"ksp_path": "ksp"}
            eng = _engine()
            algo = drl.get_algorithm_instance(sim, None, eng)
        assert algo is None
        assert eng.engine_props["is_drl_agent"] is False

    def test_unregistered_algorithm_raises(self) -> None:
        """NotImplementedError when algo unknown."""
        vp, rg = _patch_globals(valid_list=[], registry={})
        with (
            vp,
            rg,
            mock.patch.object(drl, "determine_model_type", return_value="ppo_path"),
        ):
            sim = {"ppo_path": "ppo"}
            with pytest.raises(drl.ModelSetupError):
                drl.get_algorithm_instance(sim, None, _engine())

    def test_registered_algorithm_returns_instance(self) -> None:
        """Returns algo instance and sets flag true."""
        registry = {"ppo": {"class": _DummyAlgo}}
        vp, rg = _patch_globals(valid_list=[], registry=registry)
        with (
            vp,
            rg,
            mock.patch.object(drl, "determine_model_type", return_value="ppo_path"),
        ):
            sim = {"ppo_path": "ppo"}
            eng = _engine()
            algo = drl.get_algorithm_instance(sim, "rl", eng)
        assert isinstance(algo, _DummyAlgo)
        assert eng.engine_props["is_drl_agent"] is True


class TestObsActSpaces:
    """get_obs_space and get_action_space delegation."""

    @pytest.fixture
    def patches(self) -> tuple[Any, Any]:
        """Return patch context managers for registry."""
        registry = {"ppo": {"class": _DummyAlgo}}
        return _patch_globals(valid_list=[], registry=registry)

    def test_get_obs_space_delegates(self, patches: tuple[Any, Any]) -> None:
        """Returns obs_space from algorithm."""
        vp, rg = patches
        with (
            vp,
            rg,
            mock.patch.object(drl, "determine_model_type", return_value="ppo_path"),
        ):
            sim = {"ppo_path": "ppo"}
            obs = drl.get_obs_space(sim, "rl", _engine())
        assert obs == "obs_space"

    def test_get_action_space_delegates(self, patches: tuple[Any, Any]) -> None:
        """Returns act_space from algorithm."""
        vp, rg = patches
        with (
            vp,
            rg,
            mock.patch.object(drl, "determine_model_type", return_value="ppo_path"),
        ):
            sim = {"ppo_path": "ppo"}
            act = drl.get_action_space(sim, "rl", _engine())
        assert act == "act_space"

    def test_none_when_non_drl(self) -> None:
        """Both space funcs return None for non-DRL algo."""
        vp, rg = _patch_globals(valid_list=["ksp"], registry={})
        with (
            vp,
            rg,
            mock.patch.object(drl, "determine_model_type", return_value="ksp_path"),
        ):
            sim = {"ksp_path": "ksp"}
            assert drl.get_obs_space(sim, None, _engine()) is None
            assert drl.get_action_space(sim, None, _engine()) is None
