# Phase 4: RL Integration Migration

## Overview

Phase 4 migrates reinforcement learning (RL) functionality to use the new V4 architecture. The key objective is eliminating duplicated simulation logic by having RL use the SAME pipelines and orchestrator as non-RL simulations.

## Prerequisites

- Phase 1 complete: Domain objects (`SimulationConfig`, `Request`, `Lightpath`, result types)
- Phase 2 complete: `NetworkState` with legacy compatibility, pipeline protocols
- Phase 3 complete: `SDNOrchestrator`, `PipelineFactory`, feature flag working

## Objectives

1. Create `RLSimulationAdapter` that uses existing pipelines
2. Create `UnifiedSimEnv` Gymnasium environment
3. Migrate existing RL experiments to new environment
4. Validate parity with legacy RL behavior
5. Deprecate `mock_handle_arrival()` and legacy RL environments

---

## Micro-Phases

### P4.1: RLSimulationAdapter Scaffolding

**Goal**: Introduce RL adapter interface without wiring to environment yet.

**Files Created**:
- `fusion/rl/__init__.py`
- `fusion/rl/adapter.py`
- `fusion/rl/observation.py`
- `fusion/rl/reward.py`

**Files Touched**:
- None (purely additive)

**Implementation**:

```python
# fusion/rl/adapter.py

from dataclasses import dataclass
from typing import Optional

from fusion.domain.config import SimulationConfig
from fusion.domain.request import Request
from fusion.domain.network_state import NetworkState
from fusion.domain.results import AllocationResult, RouteResult, SpectrumResult
from fusion.core.orchestrator import SDNOrchestrator


@dataclass
class PathOption:
    """Candidate path for RL action selection."""
    path_index: int
    path: list[str]
    weight_km: float
    modulation: str | None
    is_feasible: bool
    congestion: float
    spectrum_result: SpectrumResult | None
    num_hops: int = 0
    available_slots: int = 0


class RLSimulationAdapter:
    """RL-friendly interface using V4 pipelines."""

    def __init__(
        self,
        config: SimulationConfig,
        orchestrator: SDNOrchestrator,
    ):
        self.config = config
        self.orchestrator = orchestrator
        # Reuse SAME pipeline instances
        self.routing = orchestrator.routing
        self.spectrum = orchestrator.spectrum

        self._current_request: Request | None = None
        self._path_options: list[PathOption] = []

    def get_path_options(
        self,
        request: Request,
        network_state: NetworkState,
    ) -> list[PathOption]:
        """Get paths with feasibility for action masking."""
        # Uses REAL routing pipeline
        route_result = self.routing.find_routes(
            request.source,
            request.destination,
            request.bandwidth_gbps,
            network_state,
        )

        options = []
        for i, path in enumerate(route_result.paths):
            mods = route_result.modulations[i]

            # Uses REAL spectrum pipeline - no duplication
            spectrum_result = self.spectrum.find_spectrum(
                path, mods, request.bandwidth_gbps, network_state
            )

            options.append(PathOption(
                path_index=i,
                path=path,
                weight_km=route_result.weights_km[i],
                modulation=mods[0] if mods else None,
                is_feasible=spectrum_result.is_free,
                congestion=self._compute_path_congestion(path, network_state),
                spectrum_result=spectrum_result if spectrum_result.is_free else None,
                num_hops=len(path) - 1,
            ))

        self._path_options = options
        return options

    def apply_action(
        self,
        action: int,
        request: Request,
        network_state: NetworkState,
    ) -> AllocationResult:
        """Apply RL action by calling orchestrator."""
        if action >= len(self._path_options):
            # Invalid action - block
            return AllocationResult(success=False, block_reason=BlockReason.NO_ROUTE)

        option = self._path_options[action]

        # Uses SAME orchestrator as non-RL simulation
        return self.orchestrator.handle_arrival(
            request,
            network_state,
            forced_path=option.path,
        )

    def _compute_path_congestion(
        self,
        path: list[str],
        network_state: NetworkState,
    ) -> float:
        """Compute congestion metric for a path."""
        # Implementation delegates to NetworkState methods
        total_used = 0
        total_capacity = 0
        for i in range(len(path) - 1):
            link = (path[i], path[i+1])
            spectrum = network_state.get_link_spectrum(link)
            for band, matrix in spectrum.cores_matrix.items():
                total_capacity += matrix.size
                total_used += np.count_nonzero(matrix)
        return total_used / total_capacity if total_capacity > 0 else 0.0
```

**Legacy Path**: Unchanged. Adapter is purely additive.

**Verification**:
```bash
pytest fusion/tests/rl/test_adapter.py -v
ruff check fusion/rl/
mypy fusion/rl/
```

---

### P4.2: Wire RL Adapter to Pipelines

**Goal**: Connect RL adapter to use new Request + NetworkState + pipelines.

**Files Created**:
- `fusion/rl/environments/__init__.py`
- `fusion/rl/environments/unified_env.py`

**Files Touched**:
- `fusion/rl/adapter.py` (add observation/reward methods)

**Implementation**:

```python
# fusion/rl/environments/unified_env.py

import gymnasium
import numpy as np
from gymnasium import spaces

from fusion.domain.config import SimulationConfig
from fusion.core.simulation import SimulationEngine
from fusion.core.pipeline_factory import PipelineFactory
from fusion.rl.adapter import RLSimulationAdapter


class UnifiedSimEnv(gymnasium.Env):
    """
    Gymnasium environment using V4 architecture.

    Key difference from legacy: Uses same pipelines as simulation.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: SimulationConfig,
        seed: int | None = None,
    ):
        super().__init__()
        self.config = config
        self._seed = seed

        # Create simulation engine (owns NetworkState)
        self.engine = SimulationEngine(config)

        # Create orchestrator with pipelines
        self.orchestrator = PipelineFactory.create_orchestrator(config)

        # Create adapter (uses SAME pipelines)
        self.adapter = RLSimulationAdapter(config, self.orchestrator)

        # Define spaces
        self._setup_spaces()

        # Episode state
        self._current_request = None
        self._current_options = []
        self._requests_processed = 0

    def _setup_spaces(self):
        """Define observation and action spaces."""
        num_nodes = len(self.engine.network_state.topology.nodes())
        max_paths = self.config.k_paths

        self.observation_space = spaces.Dict({
            "source": spaces.Box(0, 1, shape=(num_nodes,), dtype=np.float32),
            "destination": spaces.Box(0, 1, shape=(num_nodes,), dtype=np.float32),
            "bandwidth": spaces.Box(0, 1, shape=(1,), dtype=np.float32),
            "path_lengths": spaces.Box(0, 10, shape=(max_paths,), dtype=np.float32),
            "feasibility": spaces.Box(0, 1, shape=(max_paths,), dtype=np.float32),
            "congestion": spaces.Box(0, 1, shape=(max_paths,), dtype=np.float32),
            "num_hops": spaces.Box(0, 20, shape=(max_paths,), dtype=np.float32),
            "network_utilization": spaces.Box(0, 1, shape=(1,), dtype=np.float32),
        })

        self.action_space = spaces.Discrete(max_paths)

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict, dict]:
        super().reset(seed=seed)

        if seed is not None:
            self._seed = seed

        # Reset simulation engine
        self.engine.reset(seed=self._seed)
        self._requests_processed = 0

        # Get first request
        self._current_request = self.engine.get_next_request()

        if self._current_request is None:
            # Empty episode
            return self._empty_observation(), {}

        # Get path options using REAL pipelines
        self._current_options = self.adapter.get_path_options(
            self._current_request,
            self.engine.network_state,
        )

        obs = self._build_observation()
        info = {"action_mask": self._get_action_mask()}

        return obs, info

    def step(self, action: int) -> tuple[dict, float, bool, bool, dict]:
        # Apply action through adapter (calls same orchestrator)
        result = self.adapter.apply_action(
            action,
            self._current_request,
            self.engine.network_state,
        )

        # Compute reward from AllocationResult
        reward = self._compute_reward(result)

        # Process releases
        self.engine.process_releases_until(self._current_request.arrive_time)

        # Get next request
        self._requests_processed += 1
        self._current_request = self.engine.get_next_request()

        terminated = self._current_request is None
        truncated = False

        if terminated:
            obs = self._empty_observation()
            info = {"final_stats": self.engine.get_stats()}
        else:
            self._current_options = self.adapter.get_path_options(
                self._current_request,
                self.engine.network_state,
            )
            obs = self._build_observation()
            info = {"action_mask": self._get_action_mask()}

        return obs, reward, terminated, truncated, info

    def _get_action_mask(self) -> np.ndarray:
        """Build action mask from path feasibility."""
        mask = np.zeros(self.config.k_paths, dtype=np.float32)
        for i, opt in enumerate(self._current_options):
            if i >= len(mask):
                break
            mask[i] = 1.0 if opt.is_feasible else 0.0
        return mask
```

**Before/After Call Graph**:

```
BEFORE (Legacy RL):
  GeneralSimEnv.step(action)
      |
      +-- mock_handle_arrival(request_dict)  # Duplicated logic!
      |       +-- find_routes() (inline)
      |       +-- check_spectrum() (inline)
      |       +-- allocate() (inline)
      |
      +-- compute_reward()

AFTER (V4 RL):
  UnifiedSimEnv.step(action)
      |
      +-- adapter.apply_action(action, request, network_state)
      |       |
      |       +-- orchestrator.handle_arrival(request, network_state, forced_path)
      |               |
      |               +-- routing.find_routes()      # SAME pipeline
      |               +-- spectrum.find_spectrum()   # SAME pipeline
      |               +-- network_state.create_lightpath()
      |
      +-- _compute_reward(AllocationResult)
```

**Legacy Path**: `GeneralSimEnv` continues to work unchanged. New environment is additive.

**Verification**:
```bash
pytest fusion/tests/rl/test_unified_env.py -v
```

---

### P4.3: Migrate Existing RL Experiments

**Goal**: Update existing RL training scripts and experiments to use `UnifiedSimEnv`.

**Files Touched**:
- `fusion/modules/rl/gymnasium_envs/__init__.py` (add re-export)
- `scripts/train_rl.py` or equivalent (update env creation)
- `experiments/rl/*.py` (update to new env)

**Implementation**:

```python
# fusion/modules/rl/gymnasium_envs/__init__.py

# Re-export both old and new for transition period
from fusion.modules.rl.gymnasium_envs.general_sim_env import GeneralSimEnv
from fusion.rl.environments.unified_env import UnifiedSimEnv

# Feature flag for migration
USE_UNIFIED_ENV = False  # Set True to use new env


def create_sim_env(config, **kwargs):
    """Factory for creating simulation environment."""
    if USE_UNIFIED_ENV:
        return UnifiedSimEnv(config, **kwargs)
    else:
        return GeneralSimEnv(config, **kwargs)
```

```python
# Example migration in training script

# OLD:
from fusion.modules.rl.gymnasium_envs import GeneralSimEnv
env = GeneralSimEnv(engine_props=engine_props)

# NEW:
from fusion.rl.environments import UnifiedSimEnv
from fusion.domain.config import SimulationConfig

config = SimulationConfig.from_engine_props(engine_props)
env = UnifiedSimEnv(config)
```

**SB3 Integration**:

```python
# With action masking wrapper for SB3
from sb3_contrib import MaskablePPO
from fusion.rl.environments import UnifiedSimEnv
from fusion.rl.environments.wrappers import ActionMaskWrapper

config = SimulationConfig.from_engine_props(engine_props)
env = UnifiedSimEnv(config)
env = ActionMaskWrapper(env)  # Handles action_mask in info dict

model = MaskablePPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=100000)
```

**Legacy Path**: Old `GeneralSimEnv` preserved. Migration is opt-in via factory.

**Verification**:
```bash
# Run training with both envs, compare learning curves
python scripts/train_rl.py --env=legacy --output=results_legacy/
python scripts/train_rl.py --env=unified --output=results_unified/
python scripts/compare_rl_results.py results_legacy/ results_unified/
```

---

### P4.4: Validate Parity and Document Differences

**Goal**: Ensure new RL environment produces comparable results to legacy.

**Files Created**:
- `tests/rl/test_rl_parity.py`
- `docs/migration/rl_differences.md` (if differences found)

**Parity Tests**:

```python
# tests/rl/test_rl_parity.py

import pytest
import numpy as np

from fusion.modules.rl.gymnasium_envs import GeneralSimEnv
from fusion.rl.environments import UnifiedSimEnv
from fusion.domain.config import SimulationConfig


class TestRLParity:
    """Compare legacy and unified RL environments."""

    @pytest.fixture
    def config(self):
        return SimulationConfig.from_engine_props({
            "network": "NSFNET",
            "erlang": 100,
            "num_requests": 100,
            "k_paths": 3,
        })

    def test_same_paths_found(self, config):
        """Both envs should find same candidate paths."""
        seed = 42

        legacy_env = GeneralSimEnv(config.to_engine_props())
        unified_env = UnifiedSimEnv(config)

        legacy_obs, _ = legacy_env.reset(seed=seed)
        unified_obs, _ = unified_env.reset(seed=seed)

        # Compare path feasibility
        legacy_mask = legacy_env.get_action_mask()
        unified_mask = unified_env._get_action_mask()

        np.testing.assert_array_equal(legacy_mask, unified_mask)

    def test_same_allocation_outcome(self, config):
        """Same action should produce same allocation result."""
        seed = 42

        legacy_env = GeneralSimEnv(config.to_engine_props())
        unified_env = UnifiedSimEnv(config)

        legacy_env.reset(seed=seed)
        unified_env.reset(seed=seed)

        # Take same action
        action = 0
        legacy_obs, legacy_reward, _, _, _ = legacy_env.step(action)
        unified_obs, unified_reward, _, _, _ = unified_env.step(action)

        # Rewards should match (may need tolerance for floating point)
        assert abs(legacy_reward - unified_reward) < 0.01

    def test_deterministic_with_same_seed(self, config):
        """Same seed should produce identical trajectories."""
        seed = 42

        env1 = UnifiedSimEnv(config)
        env2 = UnifiedSimEnv(config)

        env1.reset(seed=seed)
        env2.reset(seed=seed)

        # Run 10 steps with deterministic policy (always action 0)
        for _ in range(10):
            _, r1, done1, _, _ = env1.step(0)
            _, r2, done2, _, _ = env2.step(0)

            assert r1 == r2
            if done1 or done2:
                break

    def test_blocking_probability_similar(self, config):
        """Blocking probability should be similar over many episodes."""
        seed = 42
        num_episodes = 10

        legacy_blocks = []
        unified_blocks = []

        for ep in range(num_episodes):
            ep_seed = seed + ep

            # Legacy
            legacy_env = GeneralSimEnv(config.to_engine_props())
            legacy_env.reset(seed=ep_seed)
            legacy_blocked = 0
            while True:
                _, reward, done, _, _ = legacy_env.step(0)
                if reward < 0:
                    legacy_blocked += 1
                if done:
                    break
            legacy_blocks.append(legacy_blocked)

            # Unified
            unified_env = UnifiedSimEnv(config)
            unified_env.reset(seed=ep_seed)
            unified_blocked = 0
            while True:
                _, reward, done, _, _ = unified_env.step(0)
                if reward < 0:
                    unified_blocked += 1
                if done:
                    break
            unified_blocks.append(unified_blocked)

        # Should be within tolerance
        legacy_mean = np.mean(legacy_blocks)
        unified_mean = np.mean(unified_blocks)
        assert abs(legacy_mean - unified_mean) / max(legacy_mean, 1) < 0.05
```

**Expected Differences**:

| Aspect | Legacy | Unified | Impact |
|--------|--------|---------|--------|
| Spectrum check | `mock_handle_arrival` | Real pipeline | More accurate |
| SNR validation | May skip | Always runs if enabled | More correct |
| Grooming | Separate logic | Same as simulation | More consistent |
| Observation shape | May differ | Standardized | Retrain needed |

**Documentation of Differences**:

If significant differences are found, create `docs/migration/rl_differences.md` documenting:
1. What changed and why
2. Expected impact on trained policies
3. Migration guidance

**Verification**:
```bash
pytest tests/rl/test_rl_parity.py -v
```

---

## Keeping Legacy Path Available

During Phase 4, both paths remain available:

```python
# In fusion/modules/rl/gymnasium_envs/__init__.py

# Legacy (will be removed in Phase 6)
from fusion.modules.rl.gymnasium_envs.general_sim_env import GeneralSimEnv

# New (V4 architecture)
from fusion.rl.environments.unified_env import UnifiedSimEnv

# Backward compatibility alias
SimEnv = GeneralSimEnv  # Change to UnifiedSimEnv in Phase 6
```

### Deprecation Warnings

Add deprecation warnings to legacy RL code in Phase 4:

```python
# In GeneralSimEnv.__init__
import warnings
warnings.warn(
    "GeneralSimEnv is deprecated and will be removed in Phase 6. "
    "Use UnifiedSimEnv from fusion.rl.environments instead.",
    DeprecationWarning,
    stacklevel=2,
)
```

### Feature Flag for Testing

```python
# Environment variable for CI testing
import os

USE_V4_RL = os.environ.get("USE_V4_RL", "0") == "1"

if USE_V4_RL:
    from fusion.rl.environments import UnifiedSimEnv as SimEnv
else:
    from fusion.modules.rl.gymnasium_envs import GeneralSimEnv as SimEnv
```

---

## Modules Touched (Names Only)

### New Modules Created

| Module | Purpose |
|--------|---------|
| `fusion/rl/adapter.py` | RLSimulationAdapter, PathOption |
| `fusion/rl/observation.py` | Observation building utilities |
| `fusion/rl/reward.py` | Reward computation utilities |
| `fusion/rl/environments/unified_env.py` | UnifiedSimEnv |
| `fusion/rl/environments/wrappers.py` | Action masking wrappers |

### Existing Modules Modified

| Module | Changes |
|--------|---------|
| `fusion/modules/rl/gymnasium_envs/__init__.py` | Add re-exports, factory |
| `fusion/core/simulation.py` | Add `get_next_request()`, `process_releases_until()` |
| `fusion/core/orchestrator.py` | Add `forced_path` parameter handling |

### Modules NOT Modified

| Module | Reason |
|--------|--------|
| `fusion/core/sdn_controller.py` | Legacy, preserved for Phase 5 removal |
| `fusion/modules/rl/gymnasium_envs/general_sim_env.py` | Legacy, deprecated |
| `fusion/modules/rl/helpers/mock_sdn.py` | Legacy, deprecated |

---

## Rollback Plan

If Phase 4 causes issues:

1. Set `USE_V4_RL=0` environment variable
2. Old `GeneralSimEnv` continues to work unchanged
3. New code is isolated in `fusion/rl/` directory
4. No modifications to legacy RL code paths

---

## Exit Criteria

- [ ] `RLSimulationAdapter` uses same pipelines as orchestrator
- [ ] `UnifiedSimEnv` passes all observation/action space tests
- [ ] Action masks correctly reflect feasibility from real spectrum checks
- [ ] Parity tests pass (blocking probability within 5% of legacy)
- [ ] Existing RL training scripts can use new env via factory
- [ ] Deprecation warnings added to legacy RL code
- [ ] Documentation updated for new RL environment
- [ ] Code passes ruff, mypy, and tests

---

## File Summary

### New Files Created

| File | Purpose |
|------|---------|
| `fusion/rl/__init__.py` | RL package exports |
| `fusion/rl/adapter.py` | RLSimulationAdapter |
| `fusion/rl/observation.py` | Observation building |
| `fusion/rl/reward.py` | Reward computation |
| `fusion/rl/environments/__init__.py` | Environment exports |
| `fusion/rl/environments/unified_env.py` | UnifiedSimEnv |
| `fusion/rl/environments/wrappers.py` | SB3 wrappers |
| `tests/rl/test_adapter.py` | Adapter tests |
| `tests/rl/test_unified_env.py` | Environment tests |
| `tests/rl/test_rl_parity.py` | Parity tests |

### Existing Files Modified

| File | Changes |
|------|---------|
| `fusion/modules/rl/gymnasium_envs/__init__.py` | Add factory, deprecation |
| `fusion/core/simulation.py` | Add RL-friendly methods |

---

## Related Documentation

- `architecture/rl_integration.md` - RL architecture design
- `decisions/0009-rl-env-design.md` - Design rationale
- `testing/phase_4_testing.md` - Test plan
- `tutorials/using_rl_with_v4_simulator.md` - Practical guide
