# ADR-0010: ControlPolicy Protocol

## Status

Proposed

## Context

FUSION supports multiple approaches for path selection in resource allocation:

1. **Heuristic approaches**: First-fit, shortest-path, load-balanced
2. **Reinforcement Learning (RL)**: Online learning with SB3 (PPO, DQN, etc.)
3. **Machine Learning (ML)**: Pre-trained classifiers and neural networks

Prior to this decision, each approach had its own integration point:

- Heuristics were hardcoded in `SDNController`
- RL used `UnifiedSimEnv` with custom action selection
- ML had no clear integration path

This fragmentation caused several problems:

- **Code duplication**: Feasibility checking duplicated across implementations
- **Testing difficulty**: Each approach required separate test infrastructure
- **Extension friction**: Adding new policies required understanding multiple code paths
- **Comparison complexity**: Comparing approaches required custom benchmarking code

## Decision

We will introduce a `ControlPolicy` protocol that provides a unified interface for all path selection strategies.

### Protocol Definition

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class ControlPolicy(Protocol):
    """Protocol for control policies."""

    def select_action(
        self,
        request: Request,
        options: list[PathOption],
        network_state: NetworkState,
    ) -> int:
        """
        Select an action (path index) for the request.

        Returns path index (0 to len(options)-1) or -1 if no valid action.
        """
        ...

    def update(self, request: Request, action: int, reward: float) -> None:
        """Optional: Update policy based on experience."""
        ...
```

### Key Design Decisions

1. **Protocol over ABC**: Use Python `Protocol` for structural typing rather than an abstract base class, allowing any class with the right methods to be used as a policy.

2. **PathOption abstraction**: Policies receive pre-computed `PathOption` objects containing:
   - Path information (nodes, weight)
   - Feasibility status
   - Resource requirements (slots, modulation)
   - Network metrics (congestion)

3. **Optional update**: The `update()` method supports learning but is optional. Heuristic policies implement it as a no-op.

4. **Action as index**: Actions are path indices (integers), consistent with Gymnasium discrete action spaces. This enables direct RL integration.

5. **NetworkState read-only**: Policies receive `network_state` for context but should not mutate it. Mutations are handled by the orchestrator after action selection.

## Alternatives Considered

### Alternative 1: Abstract Base Class

```python
from abc import ABC, abstractmethod

class ControlPolicy(ABC):
    @abstractmethod
    def select_action(self, ...): ...
```

**Rejected because**:
- Requires explicit inheritance
- Less flexible for third-party extensions
- Protocol better matches Python's duck typing philosophy

### Alternative 2: Callable Interface

```python
Policy = Callable[[Request, list[PathOption], NetworkState], int]
```

**Rejected because**:
- No support for stateful policies
- Cannot include optional `update()` method
- Less clear documentation and type hints

### Alternative 3: Strategy Pattern with Enum

```python
class PolicyType(Enum):
    FIRST_FIT = "first_fit"
    SHORTEST = "shortest"

def get_policy(policy_type: PolicyType) -> Callable:
    ...
```

**Rejected because**:
- Doesn't support custom policies
- Requires central registry modification for new policies
- Less composable

### Alternative 4: Pass Raw Network State

Instead of `PathOption` objects, pass raw network state and let policies compute paths.

**Rejected because**:
- Duplicates routing pipeline logic
- Policies shouldn't know about routing algorithms
- Violates separation of concerns

## Consequences

### Positive

1. **Unified interface**: All policies (heuristic, RL, ML) use the same interface
2. **Easy comparison**: Switch between policies via configuration
3. **Clear extension point**: New policies just implement two methods
4. **Testing simplified**: Mock policies for orchestrator testing
5. **Documentation clarity**: Single interface to document
6. **Composition enabled**: Policies can wrap other policies (fallback, tiebreaking)

### Negative

1. **Abstraction overhead**: Simple heuristics now require class definition
2. **PathOption computation**: Must compute options even for simple policies
3. **Interface learning curve**: Contributors must understand PathOption structure

### Neutral

1. **Migration effort**: Existing code must be refactored to use new interface
2. **Configuration complexity**: New config section for policy selection

## Implementation Notes

### Policy Registration

```python
# fusion/policies/registry.py

POLICY_REGISTRY: dict[str, type[ControlPolicy]] = {
    "first_feasible": FirstFeasiblePolicy,
    "shortest": ShortestFeasiblePolicy,
    # ...
}

def register_policy(name: str, policy_class: type[ControlPolicy]) -> None:
    POLICY_REGISTRY[name] = policy_class
```

### Orchestrator Integration

```python
class SDNOrchestrator:
    def __init__(self, ..., policy: Optional[ControlPolicy] = None):
        self.policy = policy or FirstFeasiblePolicy()

    def handle_arrival(self, request, network_state):
        options = self._adapter.get_path_options(...)
        action = self.policy.select_action(request, options, network_state)
        result = self._execute_action(action, request, options, network_state)
        self.policy.update(request, action, result.reward)
        return result
```

### RL Policy Wrapper

```python
class RLPolicy(ControlPolicy):
    def __init__(self, model: BaseAlgorithm):
        self.model = model

    def select_action(self, request, options, network_state) -> int:
        obs = self._build_observation(request, options, network_state)
        mask = [opt.is_feasible for opt in options]
        action, _ = self.model.predict(obs, action_masks=mask)
        return int(action)

    def update(self, request, action, reward) -> None:
        pass  # Model is pre-trained
```

## Related Decisions

- [ADR-0004: RL/ML Integration](./0004-rl-ml-integration.md)
- [ADR-0007: Orchestrator Design](./0007-orchestrator-design.md)
- [ADR-0011: ML Control Policy](./0011-ml-control-policy.md)

## References

- [Python Protocols (PEP 544)](https://peps.python.org/pep-0544/)
- [Gymnasium Action Spaces](https://gymnasium.farama.org/api/spaces/)
- [Strategy Pattern](https://refactoring.guru/design-patterns/strategy)
