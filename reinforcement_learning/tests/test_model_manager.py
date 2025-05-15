"""Unit tests for reinforcement_learning.model_manager."""

from types import SimpleNamespace
from unittest import TestCase, mock

from reinforcement_learning import model_manager as mm


# ---------------------------- helpers ---------------------------------
def _patch_registry(algo="ppo"):
    """Return registry patch containing dummy setup/load callables."""
    setup_fn = mock.MagicMock(name="setup")
    load_fn = mock.MagicMock(name="load")
    return {algo: {"setup": setup_fn, "load": load_fn}}, setup_fn, load_fn


def _patch_determine(model_type):
    return mock.patch(
        "reinforcement_learning.model_manager.determine_model_type",
        return_value=model_type,
    )


# ------------------------------ tests ---------------------------------
class TestGetModel(TestCase):
    """get_model behaviour."""

    @mock.patch.object(mm, "ALGORITHM_REGISTRY", {}, create=True)
    def test_unknown_algorithm_raises(self):
        """NotImplementedError for unregistered algorithm."""
        with _patch_determine("algo_agent"):
            sim_dict = {"algo_agent": "unknown", "network": "net"}
            with self.assertRaises(NotImplementedError):
                mm.get_model(sim_dict, "cpu", env=None, yaml_dict={})

    def test_yaml_loaded_when_none(self):
        """Loads YAML and returns param_dict when yaml_dict is None."""
        registry, setup_fn, _ = _patch_registry()
        with mock.patch.object(mm, "ALGORITHM_REGISTRY", registry, create=True):
            with _patch_determine("algo_agent"):
                # Patch parse_yaml_file and os.path.join
                with mock.patch(
                        "reinforcement_learning.model_manager.parse_yaml_file",
                        return_value={"Env": {"a": 1}},
                ) as mock_yaml, mock.patch(
                    "os.path.join", return_value="yml_path"
                ) as mock_join:
                    sim = {"algo_agent": "ppo", "network": "net"}
                    model, params = mm.get_model(sim, "cpu", env="env", yaml_dict=None)

        mock_join.assert_called_once_with("sb3_scripts", "yml", "ppo_net.yml")
        mock_yaml.assert_called_once_with(yaml_file="yml_path")
        setup_fn.assert_called_once_with(env="env", device="cpu")
        self.assertIs(model, setup_fn.return_value)
        self.assertEqual(params, {"a": 1})


class TestGetTrainedModel(TestCase):
    """get_trained_model logic."""

    def setUp(self):
        self.registry, _, self.load_fn = _patch_registry("ppo")

    @mock.patch.object(mm, "ALGORITHM_REGISTRY", {}, create=True)
    def test_missing_underscore_raises_value_error(self):
        """ValueError when algorithm_info lacks agent suffix."""
        with _patch_determine("algo"), self.assertRaises(ValueError):
            mm.get_trained_model(env=None, sim_dict={"algo": "ppo"})

    @mock.patch.object(mm, "ALGORITHM_REGISTRY", {}, create=True)
    def test_unregistered_algorithm_raises(self):
        """NotImplementedError for unknown algorithm."""
        sim = {"algo_agent": "abc_def"}
        with _patch_determine("algo_agent"), self.assertRaises(
                NotImplementedError
        ):
            mm.get_trained_model(env=None, sim_dict=sim)

    def test_load_called_with_correct_path(self):
        """load() invoked with constructed path."""
        with mock.patch.object(mm, "ALGORITHM_REGISTRY", self.registry, create=True):
            with _patch_determine("algo_agent"), mock.patch(
                    "os.path.join", side_effect=lambda *p: "/".join(p)
            ) as mock_join:
                sim = {
                    "algo_agent": "ppo_path",
                    "path_model": "run123",
                }
                mm.get_trained_model(env="env", sim_dict=sim)

        mock_join.assert_called_once_with(
            "logs", "run123", "ppo_path_model.zip"
        )
        self.load_fn.assert_called_once_with(
            "logs/run123/ppo_path_model.zip", env="env"
        )


class TestSaveModel(TestCase):
    """save_model creates correct path and calls model.save."""

    def setUp(self):
        self.model = mock.MagicMock()

    def _env(self):
        props = {
            "network": "net",
            "date": "2025-05-15",
            "sim_start": "t0",
        }
        return SimpleNamespace(modified_props=props)

    def test_missing_underscore_raises_value_error(self):
        """ValueError when model_type lacks underscore."""
        sim = {"algo": "ppo"}
        with _patch_determine("algo"), self.assertRaises(ValueError):
            mm.save_model(sim_dict=sim, env=self._env(), model=self.model)

    def test_save_called_with_joined_path(self):
        """save() called with expected filepath."""
        with _patch_determine("algo_agent"), mock.patch(
                "os.path.join", side_effect=lambda *p: "/".join(p)
        ) as mock_join:
            sim = {"algo_agent": "ppo"}
            mm.save_model(sim_dict=sim, env=self._env(), model=self.model)

        mock_join.assert_called_once_with(
            "logs",
            "ppo",
            "net",
            "2025-05-15",
            "t0",
            "ppo_model.zip",
        )
        self.model.save.assert_called_once_with(
            "logs/ppo/net/2025-05-15/t0/ppo_model.zip"
        )
