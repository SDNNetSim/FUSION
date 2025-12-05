# ADR-0004: RL and ML Integration Decisions

## Status

Accepted (Phase 1 implications, full implementation in Phase 4)

## Context

FUSION supports reinforcement learning (RL) for path selection via Stable-Baselines3 integration. The current implementation has several issues:
- `mock_handle_arrival()` duplicates simulation logic for RL feasibility checks
- RL environments build their own observations from raw dicts
- No unified interface for RL, ML, and heuristic policies

Phase 1 domain objects must be designed to support clean RL integration in Phase 4.

## Decisions

### 1. Domain Objects Expose RL-Friendly Data

Domain objects provide properties useful for RL observations:

```python
@dataclass
class Lightpath:
    # ...
    @property
    def utilization(self) -> float:
        """Fraction of bandwidth in use (0.0 to 1.0)."""
        used = self.total_bandwidth_gbps - self.remaining_bandwidth_gbps
        return used / self.total_bandwidth_gbps if self.total_bandwidth_gbps > 0 else 0.0

    @property
    def num_slots(self) -> int:
        """Spectrum slots used."""
        return self.end_slot - self.start_slot
```

### 2. Result Objects Include Feasibility Information

`RouteResult` and `SpectrumResult` contain all information needed for action masking:

```python
@dataclass(frozen=True)
class RouteResult:
    paths: list[list[str]]
    modulations: list[list[str | None]]
    # ...

    def is_path_feasible(self, index: int) -> bool:
        """Check if path has at least one valid modulation."""
        mods = self.modulations[index]
        return any(m is not None for m in mods)
```

### 3. Request Tracks Feature Usage

Request tracks whether it was groomed, sliced, or protected for RL reward shaping:

```python
@dataclass
class Request:
    # Feature flags set during allocation
    is_sliced: bool = False
    is_groomed: bool = False
    is_partially_groomed: bool = False
    is_protected: bool = False
```

### 4. AllocationResult is RL-Complete

`AllocationResult` contains everything needed to compute RL rewards:

```python
@dataclass(frozen=True)
class AllocationResult:
    success: bool
    block_reason: BlockReason | None = None

    # For reward shaping
    is_groomed: bool = False
    is_sliced: bool = False
    is_protected: bool = False

    total_bandwidth_allocated_gbps: int = 0
```

## Phase 4 Design Preview

These Phase 1 decisions enable clean Phase 4 implementation:

### RLSimulationAdapter

```python
class RLSimulationAdapter:
    """RL-friendly interface using real pipelines."""

    def __init__(self, config: SimulationConfig, orchestrator: SDNOrchestrator):
        self.config = config
        self.orchestrator = orchestrator
        # Reuse SAME pipelines - no mock_handle_arrival
        self.routing = orchestrator.routing
        self.spectrum = orchestrator.spectrum

    def get_path_options(
        self,
        request: Request,
        network_state: NetworkState,
    ) -> list[PathOption]:
        """Get paths with feasibility for action masking."""
        route_result = self.routing.find_routes(
            request.source, request.destination,
            request.bandwidth_gbps, network_state
        )

        options = []
        for i, path in enumerate(route_result.paths):
            # Use REAL spectrum check - no duplication
            spectrum_result = self.spectrum.find_spectrum(
                path, route_result.modulations[i],
                request.bandwidth_gbps, network_state
            )
            options.append(PathOption(
                path_index=i,
                is_feasible=spectrum_result.is_free,
                # ... other fields
            ))
        return options
```

### Unified Observation Building

```python
def build_observation(
    request: Request,
    options: list[PathOption],
    network_state: NetworkState,
) -> dict[str, np.ndarray]:
    """Build observation from domain objects."""

    # Request features from Request object
    source_idx = node_index(request.source)
    dest_idx = node_index(request.destination)

    # Path features from PathOption (which used real pipelines)
    path_lengths = [opt.weight_km for opt in options]
    feasibility = [opt.is_feasible for opt in options]
    congestion = [opt.congestion for opt in options]

    return {
        "source": one_hot(source_idx, num_nodes),
        "destination": one_hot(dest_idx, num_nodes),
        "path_lengths": np.array(path_lengths),
        "feasibility": np.array(feasibility),
        "congestion": np.array(congestion),
    }
```

### Action Mask from Domain Objects

```python
def get_action_mask(options: list[PathOption]) -> np.ndarray:
    """Build action mask from PathOption feasibility."""
    return np.array([opt.is_feasible for opt in options])
```

### Reward Computation from AllocationResult

```python
def compute_reward(result: AllocationResult, config: SimulationConfig) -> float:
    """Compute reward from AllocationResult."""
    if not result.success:
        return config.rl_penalty  # e.g., -1.0

    reward = config.rl_reward  # e.g., 1.0

    # Reward shaping from result flags
    if result.is_groomed:
        reward *= 1.1  # Bonus for reusing lightpaths
    if result.is_sliced:
        reward *= 0.95  # Small penalty for fragmentation

    return reward
```

## Consequences

### Positive

1. **No duplication**: RL uses same pipelines as simulation
2. **Consistent behavior**: RL and simulation see same feasibility
3. **Easy rewards**: AllocationResult has all needed info
4. **Clean observations**: Domain objects expose useful properties

### Negative

1. **Phase 1 scope creep risk**: Must resist adding RL-specific code in Phase 1
2. **More properties**: Domain objects have more computed properties

### Mitigations

- Phase 1 adds only generally useful properties (utilization, num_slots)
- RL-specific code deferred to Phase 4
- Properties are simple computations, no RL dependencies

## Related Decisions

- ADR-0001: Frozen Dataclasses
- ADR-0003: Result Object Design
- Phase 4 documentation will expand on RL adapter design
