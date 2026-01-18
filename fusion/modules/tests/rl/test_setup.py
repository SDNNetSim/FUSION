"""Unit tests for fusion.modules.rl.utils.setup module."""

from collections.abc import Iterator
from types import SimpleNamespace as SNS
from typing import Any
from unittest import mock

import pytest

from fusion.modules.rl.utils import setup as su


# ------------------------------------------------------------------ #
#  stubs & fixtures                                                   #
# ------------------------------------------------------------------ #
class _DummyEnv:  # pylint: disable=too-few-public-methods
    """Minimal env exposing engine_props."""

    def __init__(self, obs_space: str = "obs_1") -> None:
        self.engine_obj = SNS(
            engine_props={
                "obs_space": obs_space,
                "feature_extractor": "path_gnn",
                "emb_dim": 32,
                "layers": 2,
                "gnn_type": "sage",
                "heads": 4,
            }
        )
        self.rl_props = SNS(mock_sdn_dict={})


def _yaml_dict() -> dict[str, Any]:
    return {
        "env": {
            "policy": "MlpPolicy",
            "learning_rate": 1e-3,
            "n_steps": 64,
            "batch_size": 32,
            "n_epochs": 4,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.0,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "policy_kwargs": "{'net_arch':[64]}",
        }
    }


# ------------------------------------------------------------------ #
class TestPrintInfo:
    """print_info branch handling."""

    @mock.patch("fusion.modules.rl.utils.setup.logger")
    def test_path_agent_string(self, mock_logger: mock.MagicMock) -> None:
        """Logs correct message for path algorithm."""
        sim = {
            "path_algorithm": "q_learning",
            "core_algorithm": "none",
            "spectrum_algorithm": None,
        }
        su.print_info(sim)
        mock_logger.info.assert_called_once_with(
            "Beginning training process for the PATH AGENT using the %s algorithm.",
            "Q_Learning",
        )

    def test_invalid_algorithms_raise(self) -> None:
        """No RL algorithms → ValueError."""
        sim = {
            "path_algorithm": "none",
            "core_algorithm": "none",
            "spectrum_algorithm": None,
        }
        with pytest.raises(su.ModelSetupError):
            su.print_info(sim)


# ------------------------------------------------------------------ #
class TestSetupHelper:
    """SetupHelper side-effects."""

    @pytest.fixture(autouse=True)
    def setup_patches(self) -> Iterator[None]:
        """Patch heavy deps once for all tests in this class."""
        with (
            mock.patch.object(
                su,
                "SimulationEngine",
                return_value=SNS(engine_props={}),
            ),
            mock.patch.object(
                su,
                "Routing",
                return_value="routing",
            ),
            mock.patch.object(
                su,
                "create_input",
                return_value={"props": 1},
            ),
            mock.patch.object(su, "save_input"),
            mock.patch.object(su, "get_start_time"),
        ):
            yield

    @pytest.fixture
    def sim_env(self) -> SNS:
        """Create sim_env object for testing."""
        sim = {
            "path_algorithm": "q_learning",
            "core_algorithm": "none",
            "is_training": True,
            "network": "net",
        }
        return SNS(
            sim_dict=sim,
            rl_props=SNS(mock_sdn_dict={}),
            path_agent=SNS(setup_env=mock.MagicMock()),
        )

    def test_create_input_sets_attributes_and_calls_helpers(self, sim_env: SNS) -> None:
        """create_input populates engine_obj, route_obj, sim_props."""
        helper = su.SetupHelper(sim_env)
        helper.create_input()

        # Ensure engine_obj is the stub namespace
        assert isinstance(sim_env.engine_obj, SNS)
        assert sim_env.engine_obj.engine_props == {}
        assert sim_env.route_obj == "routing"
        assert sim_env.sim_props == {"props": 1}
        su.save_input.assert_called_once()  # type: ignore[attr-defined]  # pylint: disable=no-member

    def test_init_envs_invokes_path_agent_setup(self, sim_env: SNS) -> None:
        """init_envs calls path_agent.setup_env when training."""
        sim_env.engine_obj = SNS(engine_props={})
        helper = su.SetupHelper(sim_env)
        helper.init_envs()

        sim_env.path_agent.setup_env.assert_called_once_with(is_path=True)
