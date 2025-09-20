"""Unit tests for reinforcement_learning.utils.setup."""

from types import SimpleNamespace as SNS
from unittest import TestCase, mock

from fusion.modules.rl.utils import setup as su


# ------------------------------------------------------------------ #
#  stubs & fixtures                                                   #
# ------------------------------------------------------------------ #
class _DummyEnv:  # pylint: disable=too-few-public-methods
    """Minimal env exposing engine_props."""

    def __init__(self, obs_space="obs_1"):
        self.engine_obj = SNS(
            engine_props=dict(
                obs_space=obs_space,
                feature_extractor="path_gnn",
                emb_dim=32,
                layers=2,
                gnn_type="sage",
                heads=4,
            )
        )
        self.rl_props = SNS(mock_sdn_dict={})


def _yaml_dict():
    return {
        "env": dict(
            policy="MlpPolicy",
            learning_rate=1e-3,
            n_steps=64,
            batch_size=32,
            n_epochs=4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs="{'net_arch':[64]}",
        )
    }


# ------------------------------------------------------------------ #
class TestPrintInfo(TestCase):
    """print_info branch handling."""

    @mock.patch("builtins.print")
    def test_path_agent_string(self, mock_print):
        """Prints correct message for path algorithm."""
        sim = dict(
            path_algorithm="q_learning",
            core_algorithm="none",
            spectrum_algorithm=None,
        )
        su.print_info(sim)
        mock_print.assert_called_once_with(
            "Beginning training process for the PATH AGENT using the "
            "Q_Learning algorithm."
        )

    def test_invalid_algorithms_raise(self):
        """No RL algorithms → ValueError."""
        sim = dict(
            path_algorithm="none", core_algorithm="none", spectrum_algorithm=None
        )
        with self.assertRaises(su.ModelSetupError):
            su.print_info(sim)


# ------------------------------------------------------------------ #
class TestSetupHelper(TestCase):
    """SetupHelper side-effects."""

    def setUp(self):
        # patch heavy deps once for all tests in this class
        self.engine_patcher = mock.patch.object(
            su,
            "SimulationEngine",
            return_value=SNS(engine_props={}),
        )
        self.routing_patcher = mock.patch.object(
            su,
            "Routing",
            return_value="routing",
        )
        self.create_patcher = mock.patch.object(
            su,
            "create_input",
            return_value={"props": 1},
        )
        self.save_patcher = mock.patch.object(su, "save_input")
        self.start_patcher = mock.patch.object(su, "get_start_time")
        for p in (
            self.engine_patcher,
            self.routing_patcher,
            self.create_patcher,
            self.save_patcher,
            self.start_patcher,
        ):
            p.start()
            self.addCleanup(p.stop)

        self.sim = dict(
            path_algorithm="q_learning",
            core_algorithm="none",
            is_training=True,
            network="net",
        )
        self.sim_env = SNS(
            sim_dict=self.sim,
            rl_props=SNS(mock_sdn_dict={}),
            path_agent=SNS(setup_env=mock.MagicMock()),
        )

    def test_create_input_sets_attributes_and_calls_helpers(self):
        """create_input populates engine_obj, route_obj, sim_props."""
        helper = su.SetupHelper(self.sim_env)
        helper.create_input()

        # Ensure engine_obj is the stub namespace
        self.assertIsInstance(self.sim_env.engine_obj, SNS)
        self.assertEqual(self.sim_env.engine_obj.engine_props, {})
        self.assertEqual(self.sim_env.route_obj, "routing")
        self.assertEqual(self.sim_env.sim_props, {"props": 1})
        su.save_input.assert_called_once()  # pylint: disable=no-member

    def test_init_envs_invokes_path_agent_setup(self):
        """init_envs calls path_agent.setup_env when training."""
        self.sim_env.engine_obj = SNS(engine_props={})
        helper = su.SetupHelper(self.sim_env)
        helper.init_envs()

        self.sim_env.path_agent.setup_env.assert_called_once_with(is_path=True)
