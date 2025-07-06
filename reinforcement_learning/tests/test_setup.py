"""Unit tests for reinforcement_learning.utils.setup."""

from types import SimpleNamespace as SNS
from unittest import TestCase, mock

from reinforcement_learning.utils import setup as su


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
class TestSetupFeatureExtractor(TestCase):
    """setup_feature_extractor mapping."""

    def test_path_gnn_returns_class_and_kwargs(self):
        """path_gnn returns PathGNN and gnn_type kwarg present."""
        env = _DummyEnv()
        cls, kw = su.setup_feature_extractor(env)
        from reinforcement_learning.feat_extrs.path_gnn import PathGNN  # pylint: disable=import-outside-toplevel
        self.assertIs(cls, PathGNN)
        self.assertEqual(kw["gnn_type"], "sage")

    def test_graphormer_returns_with_heads(self):
        """graphormer returns GraphTransformerExtractor."""
        env = _DummyEnv()
        env.engine_obj.engine_props["feature_extractor"] = "graphormer"
        cls, kw = su.setup_feature_extractor(env)
        from reinforcement_learning.feat_extrs.graphormer import (  # pylint: disable=import-outside-toplevel
            GraphTransformerExtractor,
        )
        self.assertIs(cls, GraphTransformerExtractor)
        self.assertIn("heads", kw)

    def test_unknown_extractor_raises(self):
        """Invalid extractor raises NotImplementedError."""
        env = _DummyEnv()
        env.engine_obj.engine_props["feature_extractor"] = "bad"
        with self.assertRaises(NotImplementedError):
            su.setup_feature_extractor(env)


# ------------------------------------------------------------------ #
class TestGetDrlDicts(TestCase):
    """get_drl_dicts parsing and kwargs injection."""

    @mock.patch("reinforcement_learning.utils.setup.parse_yaml_file",
                return_value=_yaml_dict())
    def test_graph_obs_adds_extractor_kwargs(self, mock_yaml):
        """When obs_space contains 'graph' kwargs include extractor."""
        env = _DummyEnv(obs_space="obs_1_graph")
        cls = su.PathGNN  # pylint: disable=no-member
        yml, kw, env_name = su.get_drl_dicts(env, "yml_path")

        self.assertEqual(env_name, "env")
        mock_yaml.assert_called_once_with("yml_path")
        self.assertIs(kw["features_extractor_class"], cls)
        self.assertIn("features_extractor_kwargs", kw)
        self.assertEqual(yml, _yaml_dict())


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
        sim = dict(path_algorithm="none", core_algorithm="none",
                   spectrum_algorithm=None)
        with self.assertRaises(ValueError):
            su.print_info(sim)


# ------------------------------------------------------------------ #
class TestSetupHelper(TestCase):
    """SetupHelper side-effects."""

    def setUp(self):
        # patch heavy deps once for all tests in this class
        self.engine_patcher = mock.patch.object(
            su,
            "Engine",
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
