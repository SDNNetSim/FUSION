"""Unit tests for fusion.modules.rl.gymnasium_envs/general_sim_env module."""

# pylint: disable=too-few-public-methods, unused-argument

import sys
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest import mock

import pytest

# Save original modules before mocking
_ORIGINAL_MODULES = {
    key: sys.modules.get(key)
    for key in [
        "gymnasium",
        "gymnasium.spaces",
        "torch",
        "torch.nn",
        "torch_geometric",
        "torch_geometric.nn",
        "stable_baselines3",
        "stable_baselines3.common",
        "stable_baselines3.common.base_class",
        "stable_baselines3.common.torch_layers",
        "sb3_contrib",
    ]
    if key in sys.modules
}

gym_mod = ModuleType("gymnasium")
spaces_mod = ModuleType("gymnasium.spaces")


class _DummySpace:  # pylint: disable=too-few-public-methods
    """Minimal placeholder for Box / Discrete."""

    def __init__(self, *_: Any, **__: Any) -> None:
        pass


class _StubGymEnv:  # pylint: disable=too-few-public-methods
    """Lightweight stand-in for gymnasium.Env."""

    def reset(self, *_: Any, **__: Any) -> tuple[None, dict[str, Any]]:
        """
        Mock Gym reset.
        """
        return None, {}

    def step(self, *_: Any, **__: Any) -> tuple[None, None, bool, bool, dict[str, Any]]:
        """
        Mock Environment/Gym step.
        """
        return None, None, False, False, {}


spaces_mod.Box = _DummySpace  # type: ignore[attr-defined]
spaces_mod.Discrete = _DummySpace  # type: ignore[attr-defined]
gym_mod.Env = _StubGymEnv  # type: ignore[attr-defined]
gym_mod.spaces = spaces_mod  # type: ignore[attr-defined]
sys.modules.update({
    "gymnasium": gym_mod,
    "gymnasium.spaces": spaces_mod,
})

torch_mod = ModuleType("torch")
torch_nn_mod = ModuleType("torch.nn")

class _Tensor:
    """Minimal placeholder for torch.Tensor."""
    def numel(self) -> int:
        """Return number of elements in tensor."""
        return 1

    def unsqueeze(self, dim: int) -> "_Tensor":
        """Return tensor with dimension `dim`."""
        return self

    def repeat(self, *args: Any, **kwargs: Any) -> "_Tensor":
        """Return tensor with dimension `dim`."""
        return self

torch_mod.Tensor = _Tensor  # type: ignore[attr-defined]

class _NNModule:
    """Lightweight torch.nn.Module replacement."""

    def forward(self, *_: Any, **__: Any) -> None:
        """
        Mock NN forward.
        """
        return None

    __call__ = forward


class _Linear(_NNModule):
    """Dummy Linear layer."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:  # noqa: D401
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias


def _randn(*shape: int, **__: Any) -> list[list[int]]:  # noqa: D401
    """Return zeros list with requested shape."""
    return [[0] * (shape[-1] or 1)]


torch_nn_mod.Module = _NNModule  # type: ignore[attr-defined]
torch_nn_mod.Linear = _Linear  # type: ignore[attr-defined]
torch_nn_mod.ReLU = _NNModule  # type: ignore[attr-defined]
torch_mod.nn = torch_nn_mod  # type: ignore[attr-defined]
torch_mod.randn = _randn  # type: ignore[attr-defined]
sys.modules.update({
    "torch": torch_mod,
    "torch.nn": torch_nn_mod,
})

# torch_geometric.nn with dummy convs
tg_mod = ModuleType("torch_geometric")
tg_nn_mod = ModuleType("torch_geometric.nn")
for _name in ("GraphConv", "SAGEConv", "GATv2Conv", "TransformerConv"):
    setattr(tg_nn_mod, _name, _NNModule)
tg_mod.nn = tg_nn_mod  # type: ignore[attr-defined]
sys.modules.update({
    "torch_geometric": tg_mod,
    "torch_geometric.nn": tg_nn_mod,
})

sb3_root = ModuleType("stable_baselines3")


class _BaseAlgo:  # pylint: disable=too-few-public-methods
    """Placeholder SB3 BaseAlgorithm."""

    def __init__(self, *_: Any, **__: Any) -> None:
        pass


for _alg in ("PPO", "A2C", "DQN"):
    setattr(sb3_root, _alg, type(_alg, (), {}))

sb3_common = ModuleType("stable_baselines3.common")
sb3_base_class = ModuleType("stable_baselines3.common.base_class")
sb3_torch_layers = ModuleType("stable_baselines3.common.torch_layers")
sb3_base_class.BaseAlgorithm = _BaseAlgo  # type: ignore[attr-defined]
sb3_torch_layers.BaseFeaturesExtractor = type(  # type: ignore[attr-defined]
    "BaseFeaturesExtractor", (), {}
)
sb3_common.base_class = sb3_base_class  # type: ignore[attr-defined]
sb3_common.torch_layers = sb3_torch_layers  # type: ignore[attr-defined]

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

from fusion.modules.rl.gymnasium_envs import (  # noqa: E402  # pylint: disable=wrong-import-position
    general_sim_env as gen_env,
)


# ------------------------- lightweight stubs -------------------------
class _DummyEngine:  # pylint: disable=too-few-public-methods
    """Stub for engine_obj with minimal surface."""

    def __init__(self) -> None:
        self.engine_props = {
            "is_drl_agent": True,
            "reward": 10,
            "penalty": -5,
            "cores_per_link": 2,
            "holding_time": 10,
        }
        self.reqs_status_dict: dict[int, bool] = {}
        self.topology = SimpleNamespace(nodes=[0, 1])

    def init_iter(self, *_: Any, **__: Any) -> None:
        """No-op."""

    def create_topology(self) -> None:
        """No-op."""


class _DummyRoute:
    """Stub for route_obj."""

    def __init__(self) -> None:
        self.route_props = SimpleNamespace(weights_list=[1])


class _DummySimEnvUtils:  # noqa: D401
    """Stub replacing SimEnvUtils."""

    def __init__(self, sim_env: Any) -> None:
        self.sim_env = sim_env

    def handle_step(self, *_: Any, **__: Any) -> None:
        """No-op."""

    def get_obs(self, *_: Any, **__: Any) -> str:
        """
        Mock get obs.
        """
        return "obs"

    def check_terminated(self) -> bool:
        """
        Mock check terminated.
        """
        return True

    def handle_test_train_step(self, *_: Any, **__: Any) -> None:
        """No-op."""


class _DummySimEnvObs:
    """Stub replacing SimEnvObs."""

    def __init__(self, sim_env: Any) -> None:
        self.sim_env = sim_env

    def update_helper_obj(self, *_: Any, **__: Any) -> None:
        """No-op."""


class _DummyCoreUtilHelpers:
    """Stub CoreUtilHelpers."""

    def __init__(self, rl_props: Any, *_: Any, **__: Any) -> None:
        self.rl_props = rl_props

    def reset_reqs_dict(self, *_: Any, **__: Any) -> None:
        """
        Mocking reset reqs dict.
        """
        self.rl_props.arrival_list.append(
            {"req_id": 0, "bandwidth": 10, "depart": 20, "arrive": 0}
        )

    def allocate(self) -> None:
        """No-op."""

    def update_snapshots(self) -> None:
        """No-op."""


class _DummySetupHelper:
    """Stub replacing SetupHelper."""

    def __init__(self, sim_env: Any) -> None:
        self.sim_env = sim_env

    def init_envs(self) -> None:
        """No-op."""

    def create_input(self) -> None:
        """
        Mock create input.
        """
        self.sim_env.engine_obj = _DummyEngine()
        self.sim_env.route_obj = _DummyRoute()

    def load_models(self) -> None:
        """No-op."""


class _DummyPathAgent:
    """Stub for PathAgent."""

    def __init__(self, *_: Any, **__: Any) -> None:
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


def _apply_patches() -> list[Any]:
    """Apply monkey-patches to the target module."""
    patchers: list[Any] = []
    for name, repl in _PATCHES.items():
        patcher = mock.patch.object(gen_env, name, repl)
        patchers.append(patcher)
        patcher.start()
    return patchers


@pytest.fixture(scope="module", autouse=True)
def _cleanup_sys_modules() -> Any:
    """Restore original sys.modules after all tests in this module complete."""
    yield
    # Restore original modules
    for key in [
        "gymnasium",
        "gymnasium.spaces",
        "torch",
        "torch.nn",
        "torch_geometric",
        "torch_geometric.nn",
        "stable_baselines3",
        "stable_baselines3.common",
        "stable_baselines3.common.base_class",
        "stable_baselines3.common.torch_layers",
        "sb3_contrib",
    ]:
        if key in _ORIGINAL_MODULES:
            original_mod = _ORIGINAL_MODULES[key]
            if original_mod is not None:
                sys.modules[key] = original_mod
        elif key in sys.modules:
            del sys.modules[key]


class TestSimEnv:
    """SimEnv reset/step behaviour."""

    def setup_method(self) -> None:
        self._patchers = _apply_patches()
        self.env = gen_env.SimEnv(sim_dict={"s1": _SIM_DICT})

    def teardown_method(self) -> None:
        for patcher in self._patchers:
            patcher.stop()

    def test_reset_returns_obs_and_info(self) -> None:
        """reset yields 'obs' and empty info dict."""
        obs, info = self.env.reset(seed=123)
        assert obs == "obs"
        assert info == {}
        assert self.env.trial == 123

    def _ensure_arrivals(self) -> None:
        """Guarantee an arrival exists before step()."""
        self.env.rl_props.arrival_count = 0
        if not self.env.rl_props.arrival_list:
            arrival_dict: dict[str, Any] = {
                "req_id": 0, "bandwidth": 10, "depart": 20, "arrive": 0
            }
            self.env.rl_props.arrival_list.append(arrival_dict)  # type: ignore[arg-type]

    def test_step_reward_when_allocated(self) -> None:
        """step returns +reward for allocated request."""
        self._ensure_arrivals()
        self.env.engine_obj.reqs_status_dict = {0: True}
        _, reward, terminated, truncated, info = self.env.step(action=0)
        assert reward == self.env.engine_obj.engine_props["reward"]
        assert terminated is True
        assert truncated is False
        assert info == {}

    def test_step_reward_when_blocked(self) -> None:
        """step returns penalty for blocked request."""
        self._ensure_arrivals()
        self.env.engine_obj.reqs_status_dict = {}
        _, reward, terminated, truncated, _ = self.env.step(action=0)
        assert reward == self.env.engine_obj.engine_props["penalty"]
        assert terminated is True
        assert truncated is False
