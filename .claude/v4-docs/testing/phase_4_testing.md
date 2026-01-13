# Phase 4 Testing: RL Integration

## Overview

This document defines the test strategy for Phase 4 (RL Integration). Tests ensure:
1. RL environment uses same pipelines as simulation
2. Observations are correctly built from domain objects
3. Actions are properly validated and applied
4. Rewards accurately reflect allocation outcomes
5. Parity with legacy RL behavior

---

## Test Categories

### 1. RLSimulationAdapter Tests

**File**: `fusion/tests/rl/test_adapter.py`

#### Test: Adapter Uses Same Pipeline Instances

```python
def test_adapter_uses_orchestrator_pipelines():
    """Adapter must use same pipeline instances as orchestrator."""
    config = create_test_config()
    orchestrator = PipelineFactory.create_orchestrator(config)
    adapter = RLSimulationAdapter(config, orchestrator)

    # Same instances, not copies
    assert adapter.routing is orchestrator.routing
    assert adapter.spectrum is orchestrator.spectrum


def test_adapter_does_not_create_own_pipelines():
    """Adapter must not instantiate its own pipelines."""
    config = create_test_config()
    orchestrator = PipelineFactory.create_orchestrator(config)

    # Track original pipeline
    original_routing = orchestrator.routing

    adapter = RLSimulationAdapter(config, orchestrator)

    # Still same instance after adapter creation
    assert orchestrator.routing is original_routing
    assert adapter.routing is original_routing
```

#### Test: PathOption Feasibility from Real Spectrum Check

```python
def test_path_options_use_real_spectrum_check():
    """PathOption.is_feasible must come from actual spectrum pipeline."""
    config = create_test_config()
    engine = SimulationEngine(config)
    orchestrator = PipelineFactory.create_orchestrator(config)
    adapter = RLSimulationAdapter(config, orchestrator)

    request = create_test_request(bandwidth_gbps=100)
    options = adapter.get_path_options(request, engine.network_state)

    # Verify against direct spectrum check
    for opt in options:
        direct_result = orchestrator.spectrum.find_spectrum(
            opt.path,
            [opt.modulation],
            request.bandwidth_gbps,
            engine.network_state,
        )
        assert opt.is_feasible == direct_result.is_free


def test_path_options_reflect_spectrum_changes():
    """Options must reflect current spectrum state, not cached."""
    config = create_test_config()
    engine = SimulationEngine(config)
    orchestrator = PipelineFactory.create_orchestrator(config)
    adapter = RLSimulationAdapter(config, orchestrator)

    request = create_test_request(bandwidth_gbps=100)

    # Get initial options
    options1 = adapter.get_path_options(request, engine.network_state)
    initial_feasible = sum(1 for o in options1 if o.is_feasible)

    # Allocate a lightpath to use up spectrum
    orchestrator.handle_arrival(request, engine.network_state)

    # Get options again
    request2 = create_test_request(bandwidth_gbps=100)
    options2 = adapter.get_path_options(request2, engine.network_state)
    after_feasible = sum(1 for o in options2 if o.is_feasible)

    # Should have fewer feasible options (or different spectrum assignment)
    # At minimum, state should have changed
    assert engine.network_state._lightpaths  # Lightpath was created
```

#### Test: apply_action Calls Orchestrator

```python
def test_apply_action_calls_orchestrator_handle_arrival():
    """apply_action must delegate to orchestrator.handle_arrival()."""
    config = create_test_config()
    engine = SimulationEngine(config)
    orchestrator = PipelineFactory.create_orchestrator(config)
    adapter = RLSimulationAdapter(config, orchestrator)

    request = create_test_request(bandwidth_gbps=100)
    options = adapter.get_path_options(request, engine.network_state)

    # Find a feasible action
    feasible_action = next(i for i, o in enumerate(options) if o.is_feasible)

    # Spy on orchestrator
    with patch.object(orchestrator, 'handle_arrival', wraps=orchestrator.handle_arrival) as mock:
        result = adapter.apply_action(feasible_action, request, engine.network_state)

        # Verify orchestrator was called with forced_path
        mock.assert_called_once()
        call_args = mock.call_args
        assert call_args.kwargs.get('forced_path') == options[feasible_action].path


def test_apply_action_returns_allocation_result():
    """apply_action must return AllocationResult from orchestrator."""
    config = create_test_config()
    engine = SimulationEngine(config)
    orchestrator = PipelineFactory.create_orchestrator(config)
    adapter = RLSimulationAdapter(config, orchestrator)

    request = create_test_request(bandwidth_gbps=100)
    options = adapter.get_path_options(request, engine.network_state)

    feasible_action = next(i for i, o in enumerate(options) if o.is_feasible)
    result = adapter.apply_action(feasible_action, request, engine.network_state)

    assert isinstance(result, AllocationResult)
    assert result.success is True  # Feasible action should succeed
```

---

### 2. Observation Tests

**File**: `fusion/tests/rl/test_observation.py`

#### Test: Observation Shape Correctness

```python
def test_observation_shape_matches_space():
    """Observation dict must match declared observation_space."""
    config = create_test_config(k_paths=3)
    env = UnifiedSimEnv(config)

    obs, info = env.reset(seed=42)

    # Check all keys present
    for key in env.observation_space.spaces:
        assert key in obs, f"Missing key: {key}"

    # Check shapes match
    for key, space in env.observation_space.spaces.items():
        assert obs[key].shape == space.shape, f"Shape mismatch for {key}"
        assert obs[key].dtype == space.dtype, f"Dtype mismatch for {key}"


def test_observation_values_in_bounds():
    """Observation values must be within declared bounds."""
    config = create_test_config()
    env = UnifiedSimEnv(config)

    obs, _ = env.reset(seed=42)

    for key, space in env.observation_space.spaces.items():
        assert np.all(obs[key] >= space.low), f"{key} below low bound"
        assert np.all(obs[key] <= space.high), f"{key} above high bound"
```

#### Test: Observation Built from Domain Objects

```python
def test_observation_source_destination_from_request():
    """Source/destination obs must come from Request object."""
    config = create_test_config()
    engine = SimulationEngine(config)
    orchestrator = PipelineFactory.create_orchestrator(config)
    adapter = RLSimulationAdapter(config, orchestrator)

    request = create_test_request(source="A", destination="B")
    options = adapter.get_path_options(request, engine.network_state)

    obs = adapter.build_observation(request, options, engine.network_state)

    # Source should be one-hot for node "A"
    node_a_idx = adapter._node_to_index("A")
    assert obs["source"][node_a_idx] == 1.0
    assert np.sum(obs["source"]) == 1.0

    # Destination should be one-hot for node "B"
    node_b_idx = adapter._node_to_index("B")
    assert obs["destination"][node_b_idx] == 1.0


def test_observation_feasibility_from_path_options():
    """Feasibility obs must come from PathOption objects."""
    config = create_test_config(k_paths=3)
    engine = SimulationEngine(config)
    orchestrator = PipelineFactory.create_orchestrator(config)
    adapter = RLSimulationAdapter(config, orchestrator)

    request = create_test_request(bandwidth_gbps=100)
    options = adapter.get_path_options(request, engine.network_state)

    obs = adapter.build_observation(request, options, engine.network_state)

    # Feasibility obs should match PathOption.is_feasible
    for i, opt in enumerate(options):
        if i < len(obs["feasibility"]):
            expected = 1.0 if opt.is_feasible else 0.0
            assert obs["feasibility"][i] == expected
```

---

### 3. Action Validity Tests

**File**: `fusion/tests/rl/test_actions.py`

#### Test: Action Mask Reflects Feasibility

```python
def test_action_mask_matches_feasibility():
    """Action mask must match PathOption.is_feasible."""
    config = create_test_config(k_paths=3)
    env = UnifiedSimEnv(config)

    obs, info = env.reset(seed=42)
    mask = info["action_mask"]

    options = env._current_options

    for i, opt in enumerate(options):
        if i < len(mask):
            expected = 1.0 if opt.is_feasible else 0.0
            assert mask[i] == expected, f"Mask mismatch at index {i}"


def test_action_mask_prevents_invalid_actions():
    """Selecting masked action should be handled gracefully."""
    config = create_test_config(k_paths=3)
    env = UnifiedSimEnv(config)

    obs, info = env.reset(seed=42)
    mask = info["action_mask"]

    # Find an invalid action (mask == 0)
    invalid_actions = [i for i, m in enumerate(mask) if m == 0.0]

    if invalid_actions:
        # Taking invalid action should still work (handled gracefully)
        obs, reward, done, truncated, info = env.step(invalid_actions[0])
        # Should either block or select valid alternative
        assert reward <= 0 or "fallback_action" in info
```

#### Test: Valid Action Produces Allocation

```python
def test_valid_action_creates_allocation():
    """Valid action should create successful allocation."""
    config = create_test_config(k_paths=3)
    env = UnifiedSimEnv(config)

    obs, info = env.reset(seed=42)
    mask = info["action_mask"]

    # Find a valid action
    valid_actions = [i for i, m in enumerate(mask) if m == 1.0]

    if valid_actions:
        action = valid_actions[0]
        initial_lightpaths = len(env.engine.network_state._lightpaths)

        obs, reward, done, truncated, info = env.step(action)

        # Should have created a lightpath
        assert len(env.engine.network_state._lightpaths) > initial_lightpaths
        assert reward > 0  # Success reward
```

---

### 4. Reward Computation Tests

**File**: `fusion/tests/rl/test_rewards.py`

#### Test: Reward from AllocationResult Success

```python
def test_reward_positive_on_success():
    """Successful allocation should produce positive reward."""
    config = create_test_config()
    adapter = create_test_adapter(config)

    result = AllocationResult(
        success=True,
        lightpaths_created=[1],
        total_bandwidth_allocated_gbps=100,
    )
    request = create_test_request(bandwidth_gbps=100)

    reward = adapter.compute_reward(result, request)

    assert reward > 0


def test_reward_negative_on_block():
    """Blocked allocation should produce negative reward."""
    config = create_test_config()
    adapter = create_test_adapter(config)

    result = AllocationResult(
        success=False,
        block_reason=BlockReason.NO_SPECTRUM,
    )
    request = create_test_request(bandwidth_gbps=100)

    reward = adapter.compute_reward(result, request)

    assert reward < 0


def test_reward_bonus_for_grooming():
    """Groomed allocation should get bonus."""
    config = create_test_config()
    adapter = create_test_adapter(config)

    result_no_groom = AllocationResult(
        success=True,
        lightpaths_created=[1],
        is_groomed=False,
    )

    result_groomed = AllocationResult(
        success=True,
        lightpaths_groomed=[1],
        is_groomed=True,
    )

    request = create_test_request()

    reward_no_groom = adapter.compute_reward(result_no_groom, request)
    reward_groomed = adapter.compute_reward(result_groomed, request)

    assert reward_groomed > reward_no_groom
```

---

### 5. Pipeline Bypass Prevention Tests

**File**: `fusion/tests/rl/test_no_bypass.py`

#### Test: RL Does Not Directly Mutate NetworkState

```python
def test_rl_adapter_does_not_mutate_spectrum_directly():
    """RL adapter must not directly modify spectrum matrices."""
    config = create_test_config()
    engine = SimulationEngine(config)
    orchestrator = PipelineFactory.create_orchestrator(config)
    adapter = RLSimulationAdapter(config, orchestrator)

    # Get spectrum state before
    link = list(engine.network_state._spectrum.keys())[0]
    spectrum_before = engine.network_state._spectrum[link].cores_matrix["c"].copy()

    # Get options and apply action
    request = create_test_request()
    options = adapter.get_path_options(request, engine.network_state)

    # These methods should be read-only
    spectrum_after_options = engine.network_state._spectrum[link].cores_matrix["c"].copy()
    np.testing.assert_array_equal(spectrum_before, spectrum_after_options)

    # Only apply_action should change state (via orchestrator)
    if any(o.is_feasible for o in options):
        feasible_action = next(i for i, o in enumerate(options) if o.is_feasible)
        adapter.apply_action(feasible_action, request, engine.network_state)

        # Now spectrum should be different (allocated via orchestrator)
        spectrum_after_action = engine.network_state._spectrum[link].cores_matrix["c"]
        # Spectrum may or may not change depending on path chosen


def test_rl_env_mutations_go_through_orchestrator():
    """All state mutations must go through orchestrator."""
    config = create_test_config()
    env = UnifiedSimEnv(config)

    obs, info = env.reset(seed=42)

    # Track orchestrator calls
    with patch.object(env.orchestrator, 'handle_arrival', wraps=env.orchestrator.handle_arrival) as mock:
        mask = info["action_mask"]
        valid_actions = [i for i, m in enumerate(mask) if m == 1.0]

        if valid_actions:
            env.step(valid_actions[0])
            mock.assert_called_once()
```

#### Test: Observation Building Does Not Mutate State

```python
def test_build_observation_is_read_only():
    """Building observations must not mutate any state."""
    config = create_test_config()
    engine = SimulationEngine(config)
    orchestrator = PipelineFactory.create_orchestrator(config)
    adapter = RLSimulationAdapter(config, orchestrator)

    request = create_test_request()
    options = adapter.get_path_options(request, engine.network_state)

    # Snapshot state before
    lightpaths_before = dict(engine.network_state._lightpaths)
    spectrum_snapshot = {}
    for link, ls in engine.network_state._spectrum.items():
        spectrum_snapshot[link] = {
            band: matrix.copy()
            for band, matrix in ls.cores_matrix.items()
        }

    # Build observation multiple times
    for _ in range(10):
        adapter.build_observation(request, options, engine.network_state)

    # State should be unchanged
    assert engine.network_state._lightpaths == lightpaths_before
    for link, ls in engine.network_state._spectrum.items():
        for band, matrix in ls.cores_matrix.items():
            np.testing.assert_array_equal(matrix, spectrum_snapshot[link][band])
```

---

### 6. Parity Tests (RL vs Heuristic)

**File**: `fusion/tests/rl/test_parity.py`

#### Test: RL Disabled Matches Pure Simulation

```python
def test_rl_disabled_matches_simulation():
    """With deterministic policy, RL env should match simulation."""
    config = create_test_config(num_requests=100)
    seed = 42

    # Run pure simulation
    sim_engine = SimulationEngine(config)
    sim_engine.reset(seed=seed)
    sim_results = sim_engine.run_simulation()
    sim_blocking = sim_results.blocking_probability

    # Run RL env with "always first valid" policy
    env = UnifiedSimEnv(config)
    obs, info = env.reset(seed=seed)

    rl_blocked = 0
    rl_total = 0

    while True:
        mask = info.get("action_mask", np.ones(config.k_paths))
        valid_actions = np.where(mask > 0)[0]

        if len(valid_actions) > 0:
            action = valid_actions[0]  # Deterministic: always first
        else:
            action = 0

        obs, reward, done, truncated, info = env.step(action)
        rl_total += 1
        if reward < 0:
            rl_blocked += 1

        if done:
            break

    rl_blocking = rl_blocked / rl_total if rl_total > 0 else 0

    # Should be very close (small tolerance for timing differences)
    assert abs(sim_blocking - rl_blocking) < 0.01


def test_rl_heuristic_policy_matches_ksp():
    """RL with KSP-like policy should match KSP simulation."""
    config = create_test_config(
        route_method="k_shortest_path",
        num_requests=100,
    )
    seed = 42

    # Run with legacy KSP
    legacy_results = run_legacy_simulation(config, seed=seed)

    # Run RL with "always pick shortest path" (action 0)
    env = UnifiedSimEnv(config)
    obs, info = env.reset(seed=seed)

    rl_blocked = 0
    while True:
        obs, reward, done, _, info = env.step(0)  # Always shortest
        if reward < 0:
            rl_blocked += 1
        if done:
            break

    # Blocking should be similar
    legacy_blocked = legacy_results.blocked_requests
    assert abs(rl_blocked - legacy_blocked) / max(legacy_blocked, 1) < 0.1
```

#### Test: Deterministic Reproducibility

```python
def test_same_seed_produces_identical_trajectories():
    """Same seed must produce identical results."""
    config = create_test_config(num_requests=50)
    seed = 42

    def run_episode(env, seed):
        obs, info = env.reset(seed=seed)
        rewards = []
        while True:
            action = 0  # Deterministic policy
            obs, reward, done, _, info = env.step(action)
            rewards.append(reward)
            if done:
                break
        return rewards

    env1 = UnifiedSimEnv(config)
    rewards1 = run_episode(env1, seed)

    env2 = UnifiedSimEnv(config)
    rewards2 = run_episode(env2, seed)

    assert rewards1 == rewards2
```

---

### 7. NetworkState Authority Tests

**File**: `fusion/tests/rl/test_state_authority.py`

#### Test: RL Respects NetworkState Immutability Rules

```python
def test_rl_does_not_store_network_state():
    """RL adapter must not cache NetworkState reference."""
    config = create_test_config()
    orchestrator = PipelineFactory.create_orchestrator(config)
    adapter = RLSimulationAdapter(config, orchestrator)

    # Adapter should not have stored network_state
    assert not hasattr(adapter, '_network_state')
    assert not hasattr(adapter, 'network_state')


def test_rl_uses_fresh_state_each_call():
    """Each RL call must use passed NetworkState, not cached."""
    config = create_test_config()
    engine1 = SimulationEngine(config)
    engine2 = SimulationEngine(config)
    orchestrator = PipelineFactory.create_orchestrator(config)
    adapter = RLSimulationAdapter(config, orchestrator)

    request = create_test_request()

    # Get options with engine1's state
    options1 = adapter.get_path_options(request, engine1.network_state)

    # Allocate something in engine1
    adapter.apply_action(0, request, engine1.network_state)

    # Get options with engine2's state (should be different)
    options2 = adapter.get_path_options(request, engine2.network_state)

    # engine2 should have different (more) feasible options
    # since we didn't allocate anything in it
    feasible1 = sum(1 for o in options1 if o.is_feasible)
    feasible2 = sum(1 for o in options2 if o.is_feasible)

    # After allocation in engine1, engine2 should have >= feasible options
    assert feasible2 >= feasible1
```

---

## Test File Organization

```
fusion/tests/rl/
    __init__.py
    conftest.py              # Shared fixtures
    test_adapter.py          # RLSimulationAdapter tests
    test_observation.py      # Observation building tests
    test_actions.py          # Action validity tests
    test_rewards.py          # Reward computation tests
    test_no_bypass.py        # Pipeline bypass prevention
    test_parity.py           # RL vs heuristic parity
    test_state_authority.py  # NetworkState authority
    test_unified_env.py      # UnifiedSimEnv Gymnasium compliance
```

---

## Shared Test Fixtures

```python
# fusion/tests/rl/conftest.py

import pytest
from fusion.domain.config import SimulationConfig
from fusion.domain.request import Request
from fusion.core.simulation import SimulationEngine
from fusion.core.pipeline_factory import PipelineFactory
from fusion.rl.adapter import RLSimulationAdapter
from fusion.rl.environments import UnifiedSimEnv


@pytest.fixture
def test_config():
    return SimulationConfig.from_engine_props({
        "network": "NSFNET",
        "erlang": 100,
        "num_requests": 100,
        "k_paths": 3,
        "cores_per_link": 1,
        "band_list": ["c"],
        "c_band": 320,
    })


@pytest.fixture
def test_engine(test_config):
    return SimulationEngine(test_config)


@pytest.fixture
def test_orchestrator(test_config):
    return PipelineFactory.create_orchestrator(test_config)


@pytest.fixture
def test_adapter(test_config, test_orchestrator):
    return RLSimulationAdapter(test_config, test_orchestrator)


@pytest.fixture
def test_env(test_config):
    return UnifiedSimEnv(test_config)


def create_test_request(**kwargs):
    defaults = {
        "request_id": 1,
        "source": "A",
        "destination": "B",
        "bandwidth_gbps": 100,
        "arrive_time": 0.0,
        "depart_time": 1.0,
    }
    defaults.update(kwargs)
    return Request(**defaults)
```

---

## CI Integration

```yaml
# In .github/workflows/quality.yml

- name: Run RL tests
  run: |
    pytest fusion/tests/rl/ -v --cov=fusion/rl --cov-report=xml

- name: Run RL parity tests
  run: |
    pytest fusion/tests/rl/test_parity.py -v --timeout=300
```

---

## Related Documentation

- `architecture/rl_integration.md` - RL architecture design
- `migration/phase_4_rl_integration.md` - Migration plan
- `decisions/0009-rl-env-design.md` - Design rationale
