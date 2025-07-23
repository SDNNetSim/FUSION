"""Unit tests for reinforcement_learning.gymnasium_envs.general_sim_env."""

# pylint: disable=too-few-public-methods

from types import ModuleType, SimpleNamespace
from unittest import TestCase, mock
import sys

gym_mod = ModuleType("gymnasium")
spaces_mod = ModuleType("gymnasium.spaces")


class _DummySpace:  # pylint: disable=too-few-public-methods
    """Minimal placeholder for Box / Discrete."""

    def __init__(self, *_, **__):
        pass


class _StubGymEnv:  # pylint: disable=too-few-public-methods
    """Lightweight stand-in for gymnasium.Env."""

    def reset(self, *_, **__):
        """
        Mock Gym reset.
        """
        return None, {}

    def step(self, *_, **__):
        """
        Mock Environment/Gym step.
        """
        return None, None, False, False, {}


spaces_mod.Box = _DummySpace
spaces_mod.Discrete = _DummySpace
gym_mod.Env = _StubGymEnv
gym_mod.spaces = spaces_mod
sys.modules.update({
    "gymnasium": gym_mod,
    "gymnasium.spaces": spaces_mod,
})

torch_mod = ModuleType("torch")
torch_nn_mod = ModuleType("torch.nn")


class _NNModule:
    """Lightweight torch.nn.Module replacement."""

    def forward(self, *_, **__):
        """
        Mock NN forward.
        """
        return None

    __call__ = forward


class _Linear(_NNModule):
    """Dummy Linear layer."""

    def __init__(self, in_features, out_features, bias=True):  # noqa: D401
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias


def _randn(*shape, **__):  # noqa: D401
    """Return zeros list with requested shape."""
    return [[0] * (shape[-1] or 1)]


torch_nn_mod.Module = _NNModule
torch_nn_mod.Linear = _Linear
torch_nn_mod.ReLU = _NNModule
torch_mod.nn = torch_nn_mod
torch_mod.randn = _randn
sys.modules.update({
    "torch": torch_mod,
    "torch.nn": torch_nn_mod,
})

# torch_geometric.nn with dummy convs
tg_mod = ModuleType("torch_geometric")
tg_nn_mod = ModuleType("torch_geometric.nn")
for _name in ("GraphConv", "SAGEConv", "GATv2Conv", "TransformerConv"):
    setattr(tg_nn_mod, _name, _NNModule)
tg_mod.nn = tg_nn_mod
sys.modules.update({
    "torch_geometric": tg_mod,
    "torch_geometric.nn": tg_nn_mod,
})

sb3_root = ModuleType("stable_baselines3")


class _BaseAlgo:  # pylint: disable=too-few-public-methods
    """Placeholder SB3 BaseAlgorithm."""

    def __init__(self, *_, **__):
        pass


for _alg in ("PPO", "A2C", "DQN"):
    setattr(sb3_root, _alg, type(_alg, (), {}))

sb3_common = ModuleType("stable_baselines3.common")
sb3_base_class = ModuleType("stable_baselines3.common.base_class")
sb3_torch_layers = ModuleType("stable_baselines3.common.torch_layers")
sb3_base_class.BaseAlgorithm = _BaseAlgo
sb3_torch_layers.BaseFeaturesExtractor = type(
    "BaseFeaturesExtractor", (), {}
)
sb3_common.base_class = sb3_base_class
sb3_common.torch_layers = sb3_torch_layers

sys.modules.update({
    "stable_baselines3": sb3_root,
    "stable_baselines3.common": sb3_common,
    "stable_baselines3.common.base_class": sb3_base_class,
    "stable_baselines3.common.torch_layers": sb3_torch_layers,
})

sb3_contrib = ModuleType("sb3_contrib")
for _name in ("ARS", "QRDQN"):
    setattr(sb3_contrib, _name, type(_name, (), {}))
sys.modules["sb3_contrib"] = sb3_contrib

from reinforcement_learning.gymnasium_envs import (  # pylint: disable=wrong-import-position
    general_sim_env as gen_env,
)


# ------------------------- lightweight stubs -------------------------
class _DummyEngine:  # pylint: disable=too-few-public-methods
    """Stub for engine_obj with minimal surface."""

    def __init__(self):
        self.engine_props = {
            "is_drl_agent": True,
            "reward": 10,
            "penalty": -5,
            "cores_per_link": 2,
            "holding_time": 10,
        }
        self.reqs_status_dict = {}
        self.topology = SimpleNamespace(nodes=[0, 1])

    def init_iter(self, *_, **__):
        """No-op."""

    def create_topology(self):
        """No-op."""


class _DummyRoute:
    """Stub for route_obj."""

    def __init__(self):
        self.route_props = SimpleNamespace(weights_list=[1])


class _DummySimEnvUtils:  # noqa: D401
    """Stub replacing SimEnvUtils."""

    def __init__(self, sim_env):
        self.sim_env = sim_env

    def handle_step(self, *_, **__):
        """No-op."""

    def get_obs(self, *_, **__):
        """
        Mock get obs.
        """
        return "obs"

    def check_terminated(self):
        """
        Mock check terminated.
        """
        return True

    def handle_test_train_step(self, *_, **__):
        """No-op."""


class _DummySimEnvObs:
    """Stub replacing SimEnvObs."""

    def __init__(self, sim_env):
        self.sim_env = sim_env

    def update_helper_obj(self, *_, **__):
        """No-op."""


class _DummyCoreUtilHelpers:
    """Stub CoreUtilHelpers."""

    def __init__(self, rl_props, *_, **__):
        self.rl_props = rl_props

    def reset_reqs_dict(self, *_, **__):
        """
        Mocking reset reqs dict.
        """
        self.rl_props.arrival_list.append(
            {"req_id": 0, "bandwidth": 10, "depart": 20, "arrive": 0}
        )

    def allocate(self):
        """No-op."""

    def update_snapshots(self):
        """No-op."""


class _DummySetupHelper:
    """Stub replacing SetupHelper."""

    def __init__(self, sim_env):
        self.sim_env = sim_env

    def init_envs(self):
        """No-op."""

    def create_input(self):
        """
        Mock create input.
        """
        self.sim_env.engine_obj = _DummyEngine()
        self.sim_env.route_obj = _DummyRoute()

    def load_models(self):
        """No-op."""


class _DummyPathAgent:
    """Stub for PathAgent."""

    def __init__(self, *_, **__):
        pass


_SIM_DICT = {
    "super_channel_space": 1,
    "is_training": True,
    "k_paths": 2,
    "cores_per_link": 2,
    "c_band": [1550],
    "optimize": False,
    "optimize_hyperparameters": False,
    "path_algorithm": "dummy",
    "erlang_start": 1,
    "erlang_stop": 1,
    "erlang_step": 1,
}

_PATCHES = {
    "SimEnvUtils": _DummySimEnvUtils,
    "SimEnvObs": _DummySimEnvObs,
    "CoreUtilHelpers": _DummyCoreUtilHelpers,
    "SetupHelper": _DummySetupHelper,
    "PathAgent": _DummyPathAgent,
    "setup_rl_sim": lambda: {"s1": _SIM_DICT},
    "get_obs_space": lambda *_, **__: "dummy_space",
    "get_action_space": lambda *_, **__: "dummy_action",  # NEW
}


def _apply_patches():
    """Apply monkey-patches to the target module."""
    patchers = []
    for name, repl in _PATCHES.items():
        patcher = mock.patch.object(gen_env, name, repl)
        patchers.append(patcher)
        patcher.start()
    return patchers


class TestSimEnv(TestCase):
    """SimEnv reset/step behaviour."""

    def setUp(self):
        self._patchers = _apply_patches()
        self.env = gen_env.SimEnv(sim_dict={"s1": _SIM_DICT})

    def tearDown(self):
        for patcher in self._patchers:
            patcher.stop()

    def test_reset_returns_obs_and_info(self):
        """reset yields 'obs' and empty info dict."""
        obs, info = self.env.reset(seed=123)
        self.assertEqual(obs, "obs")
        self.assertEqual(info, {})
        self.assertEqual(self.env.trial, 123)

    def _ensure_arrivals(self):
        """Guarantee an arrival exists before step()."""
        self.env.rl_props.arrival_count = 0
        if not self.env.rl_props.arrival_list:
            self.env.rl_props.arrival_list.append(
                {"req_id": 0, "bandwidth": 10, "depart": 20, "arrive": 0}
            )

    def test_step_reward_when_allocated(self):
        """step returns +reward for allocated request."""
        self._ensure_arrivals()
        self.env.engine_obj.reqs_status_dict = {0: True}
        _, reward, terminated, truncated, info = self.env.step(action=0)
        self.assertEqual(reward, self.env.engine_obj.engine_props["reward"])
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info, {})

    def test_step_reward_when_blocked(self):
        """step returns penalty for blocked request."""
        self._ensure_arrivals()
        self.env.engine_obj.reqs_status_dict = {}
        _, reward, terminated, truncated, _ = self.env.step(action=0)
        self.assertEqual(reward, self.env.engine_obj.engine_props["penalty"])
        self.assertTrue(terminated)
        self.assertFalse(truncated)
