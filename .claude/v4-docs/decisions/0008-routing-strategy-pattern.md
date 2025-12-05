# ADR-0008: Routing Strategy Pattern

## Status

Accepted

## Context

The legacy FUSION codebase implements routing algorithms directly in the `Routing` class with conditional logic:

```python
class Routing:
    def get_route(self, source, dest, algorithm):
        if algorithm == "k_shortest_path":
            return self._k_shortest_path(source, dest)
        elif algorithm == "load_balanced":
            return self._load_balanced(source, dest)
        elif algorithm == "1plus1_protection":
            return self._protection_routing(source, dest)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
```

This approach has several problems:

### Problem 1: Monolithic Class

All routing logic lives in one large class (500+ lines). Adding a new algorithm requires modifying this class, increasing complexity and risk.

### Problem 2: Conditional Explosion

Each feature combination adds more conditionals:

```python
if algorithm == "ksp" and protection:
    ...
elif algorithm == "ksp" and not protection:
    ...
elif algorithm == "load_balanced" and protection:
    ...
```

### Problem 3: Difficult Testing

Testing requires instantiating the full `Routing` class with all its dependencies. Cannot test algorithms in isolation.

### Problem 4: Inconsistent Return Types

Different algorithms return slightly different structures:

```python
# K-shortest-path
{"paths_matrix": [...], "weights_list": [...]}

# Protection routing
{"paths_matrix": [...], "weights_list": [...], "backup_paths": [...]}
```

### Problem 5: Configuration Coupling

Algorithm selection is tightly coupled to configuration parsing:

```python
if config["route_method"] == "k_shortest_path":
    k = config.get("k_paths", 3)
elif config["route_method"] == "load_balanced":
    threshold = config.get("load_threshold", 0.8)
```

## Decision

We will implement routing algorithms using the **Strategy Pattern** with a formal **Protocol** definition.

### Core Design

```python
# Protocol definition
from typing import Protocol

class RoutingStrategy(Protocol):
    """Protocol for all routing strategies."""

    def find_routes(
        self,
        source: str,
        destination: str,
        bandwidth_gbps: int,
        network_state: NetworkState,
        forced_path: list[str] | None = None,
    ) -> RouteResult:
        """Find candidate routes between endpoints."""
        ...
```

### Strategy Implementations

Each algorithm is a separate class:

```python
# fusion/routing/strategies/ksp.py
class KShortestPathStrategy:
    def __init__(self, config: SimulationConfig):
        self.k = config.k_paths

    def find_routes(self, source, dest, bw, network_state, forced_path=None):
        # K-shortest-path implementation
        ...
        return RouteResult(paths=paths, weights_km=weights, modulations=mods)


# fusion/routing/strategies/load_balanced.py
class LoadBalancedStrategy:
    def __init__(self, config: SimulationConfig):
        self.threshold = config.load_threshold

    def find_routes(self, source, dest, bw, network_state, forced_path=None):
        # Load-balanced implementation
        ...
        return RouteResult(...)


# fusion/routing/strategies/protection.py
class OnePlusOneProtectionStrategy:
    def __init__(self, config: SimulationConfig):
        self.k = config.k_paths

    def find_routes(self, source, dest, bw, network_state, forced_path=None):
        # Protection routing with disjoint paths
        ...
        return RouteResult(
            paths=primary_paths,
            backup_paths=backup_paths,
            ...
        )
```

### Strategy Registry

A registry maps configuration values to strategy classes:

```python
# fusion/routing/strategies/__init__.py

ROUTING_STRATEGIES = {
    "k_shortest_path": KShortestPathStrategy,
    "load_balanced": LoadBalancedStrategy,
    "least_congested": LeastCongestedStrategy,
    "1plus1_protection": OnePlusOneProtectionStrategy,
}
```

### Factory Integration

`PipelineFactory` uses the registry to instantiate strategies:

```python
class PipelineFactory:
    @staticmethod
    def create_routing(config: SimulationConfig) -> RoutingStrategy:
        strategy_cls = ROUTING_STRATEGIES.get(config.route_method)
        if strategy_cls is None:
            raise ValueError(f"Unknown routing method: {config.route_method}")
        return strategy_cls(config)
```

### Uniform Result Type

All strategies return `RouteResult`:

```python
@dataclass(frozen=True)
class RouteResult:
    paths: list[list[str]]
    weights_km: list[float]
    modulations: list[list[str | None]]

    # Optional for protection
    backup_paths: list[list[str]] | None = None
    backup_weights_km: list[float] | None = None
    backup_modulations: list[list[str | None]] | None = None

    strategy_name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return len(self.paths) == 0

    @property
    def has_protection(self) -> bool:
        return self.backup_paths is not None and len(self.backup_paths) > 0
```

## Consequences

### Positive

1. **Single Responsibility**: Each strategy class handles one algorithm

2. **Open/Closed Principle**: Add new algorithms without modifying existing code

3. **Easy Testing**: Test each strategy in isolation with minimal dependencies

4. **Type Safety**: Protocol ensures all strategies have consistent interface

5. **Uniform Results**: `RouteResult` standardizes output format

6. **Clear Extension Point**: Adding a strategy requires:
   - Create class implementing `RoutingStrategy`
   - Add to `ROUTING_STRATEGIES` registry
   - Done

### Negative

1. **More Files**: Each strategy is a separate file

2. **Indirection**: Must look up strategy in registry

3. **Learning Curve**: New pattern to understand

### Neutral

1. **Migration Effort**: Existing algorithms must be extracted to strategies

2. **Performance**: Negligible overhead from strategy lookup

## Alternatives Considered

### Alternative 1: Keep Conditional Logic

Continue with `if/elif/else` chain in monolithic class.

**Rejected because**: Does not solve maintainability problems. Complexity grows with each algorithm.

### Alternative 2: Inheritance Hierarchy

```python
class BaseRouting:
    def find_routes(self): ...

class KSPRouting(BaseRouting):
    def find_routes(self): ...

class LoadBalancedRouting(BaseRouting):
    def find_routes(self): ...
```

**Rejected because**:
- Python's duck typing makes Protocol more idiomatic
- Inheritance creates tight coupling
- Harder to compose behaviors

### Alternative 3: Function-Based Strategies

```python
def ksp_routing(source, dest, bw, network_state, config):
    ...

def load_balanced_routing(source, dest, bw, network_state, config):
    ...

STRATEGIES = {
    "ksp": ksp_routing,
    "load_balanced": load_balanced_routing,
}
```

**Rejected because**:
- Functions cannot hold configuration state
- Harder to test (no `__init__` for setup)
- Less discoverable in IDE

### Alternative 4: Plugin System

Dynamic loading of strategy modules at runtime.

**Rejected because**: Over-engineering for current needs. Registry pattern is sufficient.

## Implementation Notes

### Adding a New Strategy

1. Create strategy file:
```python
# fusion/routing/strategies/my_strategy.py
class MyCustomStrategy:
    def __init__(self, config: SimulationConfig):
        self.param = config.my_param

    def find_routes(self, source, dest, bw, network_state, forced_path=None):
        # Implementation
        return RouteResult(...)
```

2. Register in `__init__.py`:
```python
ROUTING_STRATEGIES["my_custom"] = MyCustomStrategy
```

3. Use via configuration:
```ini
[routing]
route_method = my_custom
my_param = 42
```

### Strategy Composition

Strategies can compose:

```python
class FallbackRoutingStrategy:
    """Try primary strategy, fall back to secondary."""

    def __init__(self, primary: RoutingStrategy, fallback: RoutingStrategy):
        self._primary = primary
        self._fallback = fallback

    def find_routes(self, source, dest, bw, network_state, forced_path=None):
        result = self._primary.find_routes(source, dest, bw, network_state, forced_path)
        if result.is_empty:
            result = self._fallback.find_routes(source, dest, bw, network_state, forced_path)
        return result
```

### Testing Strategies

```python
def test_ksp_returns_k_paths():
    config = SimulationConfig(k_paths=3, ...)
    strategy = KShortestPathStrategy(config)
    topology = create_test_topology()
    network_state = NetworkState(topology, config)

    result = strategy.find_routes("A", "D", 100, network_state)

    assert len(result.paths) <= 3
    assert not result.is_empty
```

## Integration with Orchestrator

The orchestrator uses strategies through the pipeline interface:

```python
class SDNOrchestrator:
    def __init__(self, config: SimulationConfig, pipelines: PipelineSet):
        self.routing = pipelines.routing  # RoutingStrategy instance

    def handle_arrival(self, request, network_state):
        route_result = self.routing.find_routes(
            request.source,
            request.destination,
            request.bandwidth_gbps,
            network_state,
        )

        if route_result.is_empty:
            return AllocationResult(success=False, block_reason=BlockReason.NO_ROUTE)

        # Process paths...
```

## Related Decisions

- **ADR-0003**: Result object design (RouteResult structure)
- **ADR-0006**: NetworkState as single source (strategies receive NetworkState)
- **ADR-0007**: Orchestrator design (orchestrator uses strategies via pipelines)

## References

- [Strategy Pattern (Wikipedia)](https://en.wikipedia.org/wiki/Strategy_pattern)
- [Python Protocols (PEP 544)](https://peps.python.org/pep-0544/)
- [Tutorial: Adding a New Routing Strategy](../tutorials/adding_a_new_routing_strategy.md)
