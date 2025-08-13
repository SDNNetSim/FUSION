"""Unit tests for reinforcement_learning.utils.deep_rl."""

from types import SimpleNamespace
from unittest import TestCase, mock

from fusion.modules.rl.utils import deep_rl as drl


# ---------------------------- helpers ---------------------------------
class _DummyAlgo:  # pylint: disable=too-few-public-methods
    """Minimal algorithm with obs/action space helpers."""

    def __init__(self, rl_props, engine_obj):
        self.rl_props = rl_props
        self.engine_obj = engine_obj

    def get_obs_space(self):
        """
        Mocks an observation space.
        """
        return "obs_space"

    def get_action_space(self):
        """
        Mocks an action space.
        """
        return "act_space"


def _engine():
    return SimpleNamespace(engine_props={})


def _patch_globals(valid_list=None, registry=None):
    """Patch VALID_PATH_ALGORITHMS and ALGORITHM_REGISTRY."""
    vp = mock.patch.object(drl, "VALID_PATH_ALGORITHMS",
                           valid_list if valid_list is not None else [])
    rg = mock.patch.object(drl, "ALGORITHM_REGISTRY",
                           registry if registry is not None else {})
    return vp, rg


# ------------------------------ tests ---------------------------------
class TestGetAlgorithmInstance(TestCase):
    """get_algorithm_instance branching."""

    def test_missing_underscore_raises(self):
        """ValueError when model_type has no underscore."""
        with mock.patch.object(drl, "determine_model_type",
                               return_value="ppo"), self.assertRaises(
            ValueError
        ):
            drl.get_algorithm_instance({}, None, _engine())

    def test_non_drl_algorithm_returns_none_and_flags(self):
        """Non-DRL path returns None and sets flag false."""
        vp, rg = _patch_globals(valid_list=["ksp"], registry={})
        with vp, rg, mock.patch.object(drl, "determine_model_type",
                                       return_value="ksp_path"):
            sim = {"ksp_path": "ksp"}
            eng = _engine()
            algo = drl.get_algorithm_instance(sim, None, eng)
        self.assertIsNone(algo)
        self.assertFalse(eng.engine_props["is_drl_agent"])

    def test_unregistered_algorithm_raises(self):
        """NotImplementedError when algo unknown."""
        vp, rg = _patch_globals(valid_list=[], registry={})
        with vp, rg, mock.patch.object(drl, "determine_model_type",
                                       return_value="ppo_path"):
            sim = {"ppo_path": "ppo"}
            with self.assertRaises(NotImplementedError):
                drl.get_algorithm_instance(sim, None, _engine())

    def test_registered_algorithm_returns_instance(self):
        """Returns algo instance and sets flag true."""
        registry = {"ppo": {"class": _DummyAlgo}}
        vp, rg = _patch_globals(valid_list=[], registry=registry)
        with vp, rg, mock.patch.object(drl, "determine_model_type",
                                       return_value="ppo_path"):
            sim = {"ppo_path": "ppo"}
            eng = _engine()
            algo = drl.get_algorithm_instance(sim, "rl", eng)
        self.assertIsInstance(algo, _DummyAlgo)
        self.assertTrue(eng.engine_props["is_drl_agent"])


class TestObsActSpaces(TestCase):
    """get_obs_space and get_action_space delegation."""

    def setUp(self):
        self.registry = {"ppo": {"class": _DummyAlgo}}
        self.patches = _patch_globals(valid_list=[], registry=self.registry)

    def test_get_obs_space_delegates(self):
        """Returns obs_space from algorithm."""
        vp, rg = self.patches
        with vp, rg, mock.patch.object(drl, "determine_model_type",
                                       return_value="ppo_path"):
            sim = {"ppo_path": "ppo"}
            obs = drl.get_obs_space(sim, "rl", _engine())
        self.assertEqual(obs, "obs_space")

    def test_get_action_space_delegates(self):
        """Returns act_space from algorithm."""
        vp, rg = self.patches
        with vp, rg, mock.patch.object(drl, "determine_model_type",
                                       return_value="ppo_path"):
            sim = {"ppo_path": "ppo"}
            act = drl.get_action_space(sim, "rl", _engine())
        self.assertEqual(act, "act_space")

    def test_none_when_non_drl(self):
        """Both space funcs return None for non-DRL algo."""
        vp, rg = _patch_globals(valid_list=["ksp"], registry={})
        with vp, rg, mock.patch.object(drl, "determine_model_type",
                                       return_value="ksp_path"):
            sim = {"ksp_path": "ksp"}
            self.assertIsNone(drl.get_obs_space(sim, None, _engine()))
            self.assertIsNone(drl.get_action_space(sim, None, _engine()))
