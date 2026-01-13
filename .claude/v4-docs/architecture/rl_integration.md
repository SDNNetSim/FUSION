# RL Integration Architecture

## Overview

This document describes how Reinforcement Learning (RL) integrates with the V4 architecture. The key design principle is **unified execution**: RL agents use the SAME pipelines and orchestrator as non-RL simulations, eliminating duplicated logic and ensuring consistent behavior.

## Design Principles

### 1. No Forked Simulator

The legacy architecture had separate code paths:
- `handle_event()` for simulation
- `mock_handle_arrival()` for RL feasibility checks

This led to:
- Duplicated allocation logic
- Divergent behavior between RL and simulation
- Maintenance burden

The V4 architecture eliminates this by having RL use the same pipelines:

```
Legacy:
  Simulation ──> handle_event() ──> SDNController (inline logic)
  RL ──────────> mock_handle_arrival() ──> (duplicated logic)

V4:
  Simulation ──> SDNOrchestrator ──> Pipelines
  RL ────────────────────────────────> Pipelines (same instances!)
```

### 2. Adapter Pattern

`RLSimulationAdapter` provides an RL-friendly interface over the same pipelines:

```
RLSimulationAdapter
    |
    +── orchestrator reference
    +── routing pipeline reference
    +── spectrum pipeline reference
    |
    +── get_path_options()     -> queries pipelines for feasibility
    +── build_observation()    -> constructs obs from domain objects
    +── apply_action()         -> calls orchestrator.handle_arrival()
    +── compute_reward()       -> derives from AllocationResult
```

### 3. Clean Observation/Action/Reward Interface

RL components interact through well-defined interfaces using domain objects:

| Component | Source | Data Type |
|-----------|--------|-----------|
| Observations | `Request`, `NetworkState`, `PathOption` | Structured arrays |
| Actions | Agent output | Path index (integer) |
| Rewards | `AllocationResult`, `StatsCollector` | Scalar float |
| Action Masks | `PathOption.is_feasible` | Boolean array |

---

## RLSimulationAdapter Design

### Class Structure

```python
class RLSimulationAdapter:
    """RL-friendly interface using real pipelines."""

    def __init__(
        self,
        config: SimulationConfig,
        orchestrator: SDNOrchestrator,
        network_state: NetworkState,
    ):
        self.config = config
        self.orchestrator = orchestrator
        self.network_state = network_state

        # Reuse SAME pipelines - no mock_handle_arrival
        self.routing = orchestrator.routing
        self.spectrum = orchestrator.spectrum

    def reset(self, seed: int | None = None) -> None:
        """Reset adapter state for new episode."""
        # NetworkState is owned by SimulationEngine
        # Adapter just resets its internal tracking
        self._current_request: Request | None = None
        self._path_options: list[PathOption] = []

        if seed is not None:
            # Coordinate with SimulationEngine for seeding
            pass
```

### Path Options for Action Masking

```python
@dataclass
class PathOption:
    """A candidate path option for RL action selection."""

    path_index: int
    path: list[str]
    weight_km: float
    modulation: str | None
    is_feasible: bool
    congestion: float
    spectrum_result: SpectrumResult | None

    # Computed features for observation
    num_hops: int = 0
    available_slots: int = 0

def get_path_options(
    self,
    request: Request,
    network_state: NetworkState,
) -> list[PathOption]:
    """Get paths with feasibility for action masking."""

    # Use REAL routing pipeline
    route_result = self.routing.find_routes(
        request.source,
        request.destination,
        request.bandwidth_gbps,
        network_state,
    )

    options = []
    for i, path in enumerate(route_result.paths):
        mods = route_result.modulations[i]

        # Use REAL spectrum check - no duplication
        spectrum_result = self.spectrum.find_spectrum(
            path, mods, request.bandwidth_gbps, network_state
        )

        # Compute congestion from NetworkState
        congestion = self._compute_path_congestion(path, network_state)

        options.append(PathOption(
            path_index=i,
            path=path,
            weight_km=route_result.weights_km[i],
            modulation=mods[0] if mods else None,
            is_feasible=spectrum_result.is_free,
            congestion=congestion,
            spectrum_result=spectrum_result if spectrum_result.is_free else None,
            num_hops=len(path) - 1,
            available_slots=self._count_available_slots(path, network_state),
        ))

    return options
```

---

## What RL Controls vs Fixed Policy

### RL-Controlled Decisions

| Decision | RL Action | Description |
|----------|-----------|-------------|
| Path Selection | `action: int` | Index into `PathOption` list |
| Pipeline Selection | Future | Choose between pipeline configurations |
| Lightpath Selection (grooming) | Future | Which existing lightpath to groom to |

### Fixed Policy (Not RL-Controlled)

| Component | Behavior | Why Fixed |
|-----------|----------|-----------|
| Spectrum Assignment | First-fit within selected path | Lower-level optimization |
| Modulation Selection | Best valid for path length | Physical constraint |
| SNR Validation | Accept/reject based on threshold | Physical constraint |
| Grooming Bandwidth | Greedy allocation | Separate optimization problem |
| Slicing Configuration | Equal splits | Separate optimization problem |

### Extending RL Control

To extend RL to control additional decisions:

```python
# Example: Multi-head action for path + modulation
@dataclass
class RLAction:
    path_index: int
    modulation_index: int | None = None  # None = auto-select
    grooming_preference: float = 0.0      # 0-1 preference for grooming

# Adapter interprets action
def apply_action(self, action: RLAction, request: Request) -> AllocationResult:
    option = self._path_options[action.path_index]

    if action.modulation_index is not None:
        forced_modulation = self.config.modulation_list[action.modulation_index]
    else:
        forced_modulation = None

    return self.orchestrator.handle_arrival(
        request,
        self.network_state,
        forced_path=option.path,
        forced_modulation=forced_modulation,
    )
```

---

## Observations

### Observation Structure

Observations are built from domain objects, ensuring consistency with simulation:

```python
def build_observation(
    self,
    request: Request,
    options: list[PathOption],
    network_state: NetworkState,
) -> dict[str, np.ndarray]:
    """Build observation from domain objects."""

    num_nodes = len(network_state.topology.nodes())
    max_paths = self.config.k_paths

    # Request features from Request object
    source_idx = self._node_to_index(request.source)
    dest_idx = self._node_to_index(request.destination)

    # Path features from PathOption (which used real pipelines)
    path_lengths = np.zeros(max_paths, dtype=np.float32)
    feasibility = np.zeros(max_paths, dtype=np.float32)
    congestion = np.zeros(max_paths, dtype=np.float32)
    num_hops = np.zeros(max_paths, dtype=np.float32)

    for i, opt in enumerate(options):
        if i >= max_paths:
            break
        path_lengths[i] = opt.weight_km / 1000.0  # Normalize
        feasibility[i] = 1.0 if opt.is_feasible else 0.0
        congestion[i] = opt.congestion
        num_hops[i] = opt.num_hops

    # Network state features
    utilization = self._compute_network_utilization(network_state)
    fragmentation = self._compute_fragmentation(network_state)

    return {
        "source": self._one_hot(source_idx, num_nodes),
        "destination": self._one_hot(dest_idx, num_nodes),
        "bandwidth": np.array([request.bandwidth_gbps / 400.0], dtype=np.float32),
        "path_lengths": path_lengths,
        "feasibility": feasibility,
        "congestion": congestion,
        "num_hops": num_hops,
        "network_utilization": np.array([utilization], dtype=np.float32),
        "fragmentation": np.array([fragmentation], dtype=np.float32),
    }
```

### Observation Sources

| Feature | Source | Domain Object |
|---------|--------|---------------|
| Source/destination | `Request.source`, `Request.destination` | `Request` |
| Bandwidth | `Request.bandwidth_gbps` | `Request` |
| Path lengths | `RouteResult.weights_km` | `RouteResult` |
| Path feasibility | `SpectrumResult.is_free` | `SpectrumResult` |
| Congestion | Computed from `NetworkState` | `NetworkState` |
| Utilization | `Lightpath.utilization` | `Lightpath` |
| Fragmentation | Computed from spectrum matrices | `NetworkState` |

---

## Action Masking

Action masks prevent invalid actions (selecting infeasible paths):

```python
def get_action_mask(self, options: list[PathOption]) -> np.ndarray:
    """Build action mask from PathOption feasibility."""
    mask = np.zeros(self.config.k_paths, dtype=np.float32)
    for i, opt in enumerate(options):
        if i >= len(mask):
            break
        mask[i] = 1.0 if opt.is_feasible else 0.0
    return mask
```

### Integration with SB3

```python
# In Gymnasium environment
def step(self, action: int) -> tuple[dict, float, bool, bool, dict]:
    # Validate action against mask
    mask = self.get_action_mask(self._current_options)
    if mask[action] == 0.0:
        # Invalid action - should not happen with proper masking
        logger.warning(f"RL selected invalid action {action}")
        # Handle gracefully - select first valid or return blocking
        valid_actions = np.where(mask > 0)[0]
        if len(valid_actions) > 0:
            action = valid_actions[0]
        else:
            return self._handle_no_valid_actions()

    result = self.adapter.apply_action(action, self._current_request)
    reward = self.adapter.compute_reward(result)
    # ...
```

---

## Rewards

### Reward Computation from AllocationResult

```python
def compute_reward(
    self,
    result: AllocationResult,
    request: Request,
) -> float:
    """Compute reward from AllocationResult."""

    if not result.success:
        return self.config.rl_block_penalty  # e.g., -1.0

    reward = self.config.rl_success_reward  # e.g., 1.0

    # Reward shaping from result flags
    if result.is_groomed:
        reward += self.config.rl_grooming_bonus  # e.g., 0.1

    if result.is_sliced:
        reward += self.config.rl_slicing_penalty  # e.g., -0.05

    # Bandwidth-weighted reward
    if self.config.rl_bandwidth_weighted:
        reward *= request.bandwidth_gbps / 100.0

    # Resource efficiency bonus
    if result.lightpaths_created:
        efficiency = self._compute_efficiency(result, request)
        reward += self.config.rl_efficiency_weight * efficiency

    return reward
```

### Reward Sources

| Signal | Source | Description |
|--------|--------|-------------|
| Success/block | `AllocationResult.success` | Primary reward signal |
| Grooming bonus | `AllocationResult.is_groomed` | Reusing existing lightpaths |
| Slicing penalty | `AllocationResult.is_sliced` | Fragmentation cost |
| Efficiency | `AllocationResult.total_bandwidth_allocated_gbps` | Resource utilization |
| Holding time | `Request.holding_time` | Long-term value |

---

## Episodes and Simulation Runs

### Episode Mapping

| RL Concept | Simulation Mapping |
|------------|-------------------|
| Episode | One simulation run (all requests in traffic matrix) |
| Step | One request arrival |
| Terminal | All requests processed or early stopping |

### Episode Control

```python
class UnifiedSimEnv(gymnasium.Env):
    """Gymnasium environment using V4 architecture."""

    def reset(self, seed: int | None = None) -> tuple[dict, dict]:
        super().reset(seed=seed)

        # Reset simulation engine (owns NetworkState)
        self.engine.reset(seed=seed)

        # Get first request
        self._current_request = self.engine.get_next_request()
        self._current_options = self.adapter.get_path_options(
            self._current_request, self.engine.network_state
        )

        obs = self.adapter.build_observation(
            self._current_request,
            self._current_options,
            self.engine.network_state,
        )
        info = {"action_mask": self.adapter.get_action_mask(self._current_options)}

        return obs, info

    def step(self, action: int) -> tuple[dict, float, bool, bool, dict]:
        # Apply action through adapter (uses same pipelines)
        result = self.adapter.apply_action(action, self._current_request)
        reward = self.adapter.compute_reward(result, self._current_request)

        # Process any releases that occurred
        self.engine.process_releases_until(self._current_request.arrive_time)

        # Get next request
        self._current_request = self.engine.get_next_request()

        terminated = self._current_request is None
        truncated = False

        if terminated:
            obs = self._empty_observation()
            info = {}
        else:
            self._current_options = self.adapter.get_path_options(
                self._current_request, self.engine.network_state
            )
            obs = self.adapter.build_observation(
                self._current_request,
                self._current_options,
                self.engine.network_state,
            )
            info = {"action_mask": self.adapter.get_action_mask(self._current_options)}

        return obs, reward, terminated, truncated, info
```

---

## RNG and Seeding

### Seed Flow

```
User provides seed
    |
    v
SimulationEngine.reset(seed)
    |
    +-- self._rng = np.random.default_rng(seed)
    +-- NetworkState reset (uses same seed)
    +-- Traffic generation (deterministic from seed)
    |
    v
RLSimulationAdapter
    |
    +-- Uses NetworkState from engine (same RNG state)
    +-- No independent randomness
```

### Reproducibility Rules

1. **All RNG goes through engine**: Adapter does not maintain separate RNG
2. **Same seed = same traffic**: Traffic matrix generated deterministically
3. **Same seed + same policy = same results**: For comparison testing

```python
# Reproducibility test
def test_rl_reproducibility():
    seed = 42

    # Run 1
    env1 = UnifiedSimEnv(config)
    env1.reset(seed=seed)
    results1 = run_episode(env1, policy)

    # Run 2 with same seed
    env2 = UnifiedSimEnv(config)
    env2.reset(seed=seed)
    results2 = run_episode(env2, policy)

    assert results1.blocking_probability == results2.blocking_probability
```

---

## Interaction with Domain Objects

### Request Flow

```
1. SimulationEngine generates Request
2. Request passed to RLSimulationAdapter
3. Adapter queries pipelines for PathOptions
4. Agent selects action (path index)
5. Adapter calls orchestrator.handle_arrival(request, network_state, forced_path)
6. Orchestrator coordinates pipelines (same as non-RL)
7. AllocationResult returned
8. Adapter computes reward
9. Request status updated (ROUTED or BLOCKED)
```

### NetworkState Access

| Component | Access Type | Methods Used |
|-----------|-------------|--------------|
| RL Adapter | Read | `topology`, `get_lightpaths_*`, `is_spectrum_available` |
| Pipelines | Read + Write | All methods (through orchestrator) |
| StatsCollector | Read | `get_lightpath`, snapshot methods |

**Critical**: RL adapter NEVER directly mutates NetworkState. All mutations go through orchestrator/pipelines.

---

## Result Object Usage

### RouteResult in RL

```python
# Adapter uses RouteResult for path options
route_result = self.routing.find_routes(src, dst, bw, network_state)

for i, path in enumerate(route_result.paths):
    # RouteResult provides everything needed
    weight = route_result.weights_km[i]
    mods = route_result.modulations[i]
    # ...
```

### SpectrumResult for Feasibility

```python
# Adapter uses SpectrumResult for action masking
spectrum_result = self.spectrum.find_spectrum(path, mods, bw, network_state)

option = PathOption(
    is_feasible=spectrum_result.is_free,
    # ...
)
```

### AllocationResult for Rewards

```python
# Adapter uses AllocationResult for reward computation
result = self.orchestrator.handle_arrival(request, network_state, forced_path=path)

reward = 0.0
if result.success:
    reward = 1.0
    if result.is_groomed:
        reward += 0.1
# ...
```

---

## File Layout

```
fusion/
    rl/
        __init__.py
        adapter.py           # RLSimulationAdapter, PathOption
        observation.py       # Observation building utilities
        reward.py            # Reward computation utilities
        environments/
            __init__.py
            unified_env.py   # UnifiedSimEnv (Gymnasium environment)
            wrappers.py      # Action masking wrappers
```

---

## Related Documentation

- `architecture/orchestration.md` - SDNOrchestrator design
- `architecture/pipelines.md` - Pipeline implementations
- `architecture/result_objects.md` - Result type specifications
- `decisions/0009-rl-env-design.md` - Design rationale
- `tutorials/using_rl_with_v4_simulator.md` - Practical guide
