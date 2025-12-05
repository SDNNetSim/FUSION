# ADR-0001: Frozen Dataclasses for Configuration and Results

## Status

Accepted

## Context

The V4 architecture introduces typed domain objects to replace dictionary-based data structures. A key design decision is whether these objects should be mutable or immutable.

Current state (V3/legacy):
- `engine_props` is a mutable dictionary that can be modified anywhere
- Result values are returned as mutable dicts that could be accidentally modified
- No enforcement of "configuration should not change during simulation"

## Decision

Use frozen dataclasses (`@dataclass(frozen=True)`) for:
1. `SimulationConfig` - Configuration is set once at simulation start
2. All result types (`RouteResult`, `SpectrumResult`, `GroomingResult`, `SlicingResult`, `SNRResult`, `AllocationResult`)

Use regular mutable dataclasses for:
1. `Request` - Status changes during lifecycle
2. `Lightpath` - Capacity changes during grooming
3. `NetworkState` - Mutable by design (spectrum allocation)

## Consequences

### Positive

1. **Thread safety**: Frozen objects can be safely shared without locks
2. **Hashable**: Frozen dataclasses are hashable, enabling use as dict keys or in sets
3. **Predictable**: No accidental mutation of configuration or results
4. **Clear intent**: Immutability signals "this should not change"
5. **Debugging**: State at any point is exactly as created, no hidden mutations

### Negative

1. **Memory**: Creating new objects instead of mutating may increase allocations
2. **Verbose updates**: Must create new instances for any changes
3. **Learning curve**: Developers accustomed to mutable dicts may find it unfamiliar

### Mitigations

For the negatives:
- Memory impact is minimal for our use case (configuration objects are small)
- Results are short-lived and processed immediately
- Use `dataclasses.replace()` for creating modified copies when needed

## Examples

```python
# Good: Frozen configuration
@dataclass(frozen=True)
class SimulationConfig:
    k_paths: int
    # ...

config = SimulationConfig(k_paths=3, ...)
config.k_paths = 5  # FrozenInstanceError - prevents accidental mutation

# Good: Frozen result
@dataclass(frozen=True)
class RouteResult:
    paths: list[list[str]]
    # ...

result = routing.find_routes(...)
result.paths.append(["X", "Y"])  # Still possible (list is mutable)
result.paths = []  # FrozenInstanceError - prevents reassignment

# Mutable when lifecycle changes are needed
@dataclass
class Request:
    status: RequestStatus = RequestStatus.PENDING
    # ...

request.status = RequestStatus.ROUTED  # Valid - status changes
```

## Related Decisions

- ADR-0002: Request Status Enum
- ADR-0003: Result Object Design
