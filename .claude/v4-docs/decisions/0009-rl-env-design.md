# ADR-0009: RL Environment Design

## Status

Accepted

## Context

FUSION supports reinforcement learning (RL) for path selection via Stable-Baselines3 integration. The legacy implementation has significant architectural problems:

### Problem 1: Duplicated Simulation Logic

The legacy `GeneralSimEnv` uses `mock_handle_arrival()` which duplicates the simulation's allocation logic:

```python
# Legacy: Two separate code paths
class GeneralSimEnv:
    def step(self, action):
        # Uses mock_handle_arrival - separate implementation
        result = self.mock_sdn.mock_handle_arrival(request_dict)

class SimulationEngine:
    def handle_arrival(self):
        # Uses SDNController - different implementation
        self.sdn_obj.handle_event(request_dict, 'arrival')
```

This leads to:
- Behavior divergence between RL and simulation
- Double maintenance burden
- Bugs that affect only one code path
- RL agents learning policies that don't transfer to production

### Problem 2: Inconsistent State Management

Legacy RL builds observations from raw dictionaries, bypassing domain objects:

```python
# Legacy: Direct dict access
obs = {
    "paths": request_dict["paths"],
    "feasibility": self._check_feasibility_manually(request_dict),
}
```

This creates:
- Type safety issues
- Inconsistent observation construction
- Difficulty adding new features

### Problem 3: Tight Coupling

The legacy environment directly accesses internal simulation state:

```python
# Legacy: Accessing internals
class GeneralSimEnv:
    def __init__(self):
        self.sdn_props = SDNProps()  # Direct access
        self.network_spectrum = {}   # Own copy of state
```

This makes:
- Testing difficult
- Refactoring risky
- State synchronization error-prone

## Decision

We will create a new `UnifiedSimEnv` that uses the SAME pipelines and orchestrator as non-RL simulations, connected through an `RLSimulationAdapter`.

### Design Principles

1. **Same Pipelines**: RL uses identical pipeline instances as simulation
2. **Adapter Pattern**: Clean interface between RL and simulation
3. **Domain Objects**: Observations built from `Request`, `PathOption`, `NetworkState`
4. **Result-Based Rewards**: Rewards computed from `AllocationResult`

### Architecture

```
UnifiedSimEnv (Gymnasium)
    |
    +-- RLSimulationAdapter
    |       |
    |       +-- orchestrator reference (SHARED)
    |       +-- routing pipeline reference (SHARED)
    |       +-- spectrum pipeline reference (SHARED)
    |
    +-- SimulationEngine
            |
            +-- NetworkState (SINGLE INSTANCE)
            +-- SDNOrchestrator (uses same pipelines)
```

### Interface Design

```python
class RLSimulationAdapter:
    """RL-friendly interface using real pipelines."""

    def get_path_options(
        self,
        request: Request,
        network_state: NetworkState,
    ) -> list[PathOption]:
        """Get paths with feasibility for action masking.

        Uses REAL routing and spectrum pipelines - no duplication.
        """

    def apply_action(
        self,
        action: int,
        request: Request,
        network_state: NetworkState,
    ) -> AllocationResult:
        """Apply RL action through orchestrator.

        Uses SAME orchestrator as non-RL simulation.
        """

    def compute_reward(
        self,
        result: AllocationResult,
        request: Request,
    ) -> float:
        """Compute reward from AllocationResult."""
```

### Observation/Action/Reward Specification

**Observations** are built from domain objects:
- `Request`: source, destination, bandwidth
- `PathOption`: feasibility, weight, congestion (from real pipeline checks)
- `NetworkState`: utilization, fragmentation

**Actions** are path indices:
- Single discrete action: index into `PathOption` list
- Action mask from `PathOption.is_feasible`

**Rewards** are computed from `AllocationResult`:
- Success/failure from `result.success`
- Bonus/penalty from `result.is_groomed`, `result.is_sliced`

## Alternatives Considered

### Alternative 1: RL-Only Simulator

**Approach**: Maintain separate, optimized simulator for RL.

**Pros**:
- Can optimize for RL-specific needs
- No risk of affecting production simulation

**Cons**:
- Duplicated logic (the core problem)
- Policies may not transfer
- Double maintenance burden

**Rejected**: Does not solve the fundamental duplication problem.

### Alternative 2: Direct Environment Inside SDNOrchestrator

**Approach**: Embed RL environment logic directly in orchestrator.

**Pros**:
- Single code path
- No adapter overhead

**Cons**:
- Tight coupling between RL and core simulation
- Difficult to test RL independently
- Orchestrator becomes complex

**Rejected**: Violates single responsibility principle for orchestrator.

### Alternative 3: Event-Driven RL Interface

**Approach**: RL receives events and submits decisions asynchronously.

**Pros**:
- Clean separation
- Could support distributed RL

**Cons**:
- Significant complexity
- Overkill for current needs
- Major architecture change

**Rejected**: Too complex for current requirements.

### Alternative 4: Wrapper Around Legacy Environment

**Approach**: Keep legacy env, wrap it to provide better interface.

**Pros**:
- Minimal code changes
- Lower risk

**Cons**:
- Does not fix duplication
- Technical debt remains
- Future maintenance burden

**Rejected**: Does not address root cause.

## Preventing Tight Coupling

### Stable Adapter Interface

The `RLSimulationAdapter` interface is stable and versioned:

```python
class RLSimulationAdapter(Protocol):
    """Stable interface for RL integration.

    Changes to this interface require:
    1. ADR documenting the change
    2. Deprecation period
    3. Migration guide
    """

    def get_path_options(
        self,
        request: Request,
        network_state: NetworkState,
    ) -> list[PathOption]: ...

    def apply_action(
        self,
        action: int,
        request: Request,
        network_state: NetworkState,
    ) -> AllocationResult: ...

    def compute_reward(
        self,
        result: AllocationResult,
        request: Request,
    ) -> float: ...
```

### Test Coverage Requirements

Before any pipeline changes:
1. RL adapter tests must pass
2. Parity tests with heuristic policy must pass
3. Observation shape tests must pass

```python
# Required test coverage
class TestRLAdapterInterface:
    def test_get_path_options_returns_valid_options(self): ...
    def test_apply_action_calls_orchestrator(self): ...
    def test_reward_computed_from_allocation_result(self): ...

class TestRLParity:
    def test_same_blocking_as_simulation(self): ...
    def test_deterministic_with_seed(self): ...
```

### RL Does Not Bypass Pipelines

Enforcement rules:
1. RL adapter ONLY reads from `NetworkState` (never writes directly)
2. All mutations go through `orchestrator.handle_arrival()`
3. No direct numpy array access in RL code

```python
# FORBIDDEN in RL adapter
network_state._spectrum[link].cores_matrix[band][core][start:end] = lp_id

# REQUIRED - go through orchestrator
result = self.orchestrator.handle_arrival(request, network_state, forced_path=path)
```

## Consequences

### Positive

1. **No Duplication**: RL uses same allocation logic as simulation
2. **Consistent Behavior**: RL policies transfer to production
3. **Single Maintenance**: Fix once, applies everywhere
4. **Clean Interface**: Adapter pattern enables testing
5. **Type Safety**: Domain objects throughout RL code

### Negative

1. **Initial Migration Cost**: Must update existing RL experiments
2. **Performance**: Adapter adds small overhead (negligible in practice)
3. **Retraining**: Existing policies may need retraining due to observation changes

### Mitigations

1. **Migration Cost**: Provide factory function for gradual migration
2. **Performance**: Profile and optimize adapter if needed
3. **Retraining**: Document observation changes, provide migration guide

## Implications for Future RL Work

### Adding New RL Agents

New agents automatically benefit from:
- Real pipeline behavior
- Consistent observations
- Accurate feasibility checking

### Adding New Reward Functions

Reward functions have access to:
- Full `AllocationResult` with all flags
- `Request` with feature annotations
- `NetworkState` for custom metrics

```python
# Easy to add new reward function
def custom_reward(result: AllocationResult, request: Request, network_state: NetworkState) -> float:
    base = 1.0 if result.success else -1.0

    # Custom shaping using domain objects
    if result.is_groomed:
        base += 0.2
    if request.bandwidth_gbps > 200:
        base *= 1.1  # Bonus for high-bandwidth

    return base
```

### Adding New State Features

New observation features can be added by:
1. Adding computed property to domain object (if generally useful)
2. Adding feature extraction in adapter (if RL-specific)

```python
# Option 1: Add to Lightpath
@property
def efficiency(self) -> float:
    """Bandwidth efficiency of this lightpath."""
    return self.total_bandwidth_gbps / (self.num_slots * SLOT_BW)

# Option 2: Add to adapter
def _compute_fragmentation_feature(self, network_state: NetworkState) -> float:
    """RL-specific fragmentation metric."""
    # ... custom computation
```

## Related Decisions

- ADR-0003: Result Object Design
- ADR-0004: RL and ML Integration Decisions
- ADR-0007: Orchestrator Design

## References

- `architecture/rl_integration.md` - Full architecture documentation
- `migration/phase_4_rl_integration.md` - Migration plan
- `tutorials/using_rl_with_v4_simulator.md` - Practical guide
