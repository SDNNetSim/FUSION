# ADR 0006: NetworkState as Authoritative Data Source

## Status

Accepted

## Context

The current FUSION architecture distributes network state across multiple locations:

1. **`sdn_props.network_spectrum_dict`**: Spectrum allocation per link
2. **`sdn_props.lightpath_status_dict`**: Active lightpath information
3. **`engine_props`**: Configuration and topology
4. **`stats_props`**: Statistics and counters
5. **Various local variables**: Temporary state during request processing

This distribution causes several problems:

### Problem 1: Inconsistent State

When state is spread across multiple dictionaries, keeping them synchronized requires careful coordination. If one update is missed, the simulation can enter an inconsistent state.

Example: A lightpath is removed from `lightpath_status_dict` but its spectrum allocation remains in `network_spectrum_dict`.

### Problem 2: Duplicated Logic

Multiple components need to access and modify state, leading to duplicated code:

```python
# In sdn_controller.py
key = tuple(sorted([source, destination]))
if key not in self.sdn_props.lightpath_status_dict:
    self.sdn_props.lightpath_status_dict[key] = {}

# In grooming.py (same pattern)
key = tuple(sorted([source, destination]))
if key not in self.sdn_props.lightpath_status_dict:
    return None
```

### Problem 3: Hidden Dependencies

Components implicitly depend on the structure of nested dictionaries. Changes to the dictionary structure require updates across multiple files.

### Problem 4: Testing Difficulty

Testing requires constructing complex nested dictionaries. Mocking is error-prone because the exact structure must match.

### Problem 5: RL Integration Challenges

RL environments currently duplicate feasibility checking logic because they cannot easily reuse the same state access patterns:

```python
# In RL env (duplicated from sdn_controller)
def mock_handle_arrival(self):
    # Re-implements spectrum checking
    # Re-implements lightpath creation
    # Diverges from simulation logic over time
```

## Decision

**NetworkState will be the single authoritative source for all mutable network state.**

### Core Principles

1. **Single Instance**: Exactly one `NetworkState` instance exists per simulation
2. **Owned by Engine**: `SimulationEngine` owns the instance; all other components receive it by reference
3. **No Caching**: Pipelines do NOT store or cache `NetworkState` data
4. **Explicit Mutations**: All state changes go through `NetworkState` methods
5. **Typed Objects**: State is accessed via typed objects (`Lightpath`, `LinkSpectrum`), not raw dictionaries

### State Ownership Model

```
SimulationEngine (OWNER)
    |
    +-- _network_state: NetworkState
            |
            +-- _topology: nx.Graph
            +-- _spectrum: dict[link, LinkSpectrum]
            +-- _lightpaths: dict[int, Lightpath]
```

### Access Pattern

```python
# Components receive NetworkState per method call
def handle_arrival(self, request: Request, network_state: NetworkState) -> AllocationResult:
    # Use network_state for all queries
    if network_state.is_spectrum_available(path, start, end, core, band):
        lightpath = network_state.create_lightpath(...)
```

## Consequences

### Positive

1. **Single Source of Truth**: No more synchronization bugs between multiple dictionaries

2. **Type Safety**: `Lightpath` and `LinkSpectrum` objects provide compile-time checking and IDE support

3. **Encapsulated Logic**: Spectrum allocation, lightpath management, and consistency checks live in one place

4. **Simplified Testing**: Mock `NetworkState` instead of complex nested dictionaries

5. **RL Integration**: RL environments use the same `NetworkState`, eliminating duplicated logic

6. **Clear Contracts**: Pipeline protocols define exactly what each component needs from state

### Negative

1. **Migration Effort**: Existing code must be updated to use new API

2. **Legacy Compatibility**: Temporary properties needed for gradual migration

3. **Learning Curve**: Developers must understand new patterns

### Neutral

1. **Performance**: Similar performance; O(n) legacy property construction is explicit, not hidden

## Alternatives Considered

### Alternative 1: Keep Distributed State

Continue with `sdn_props`, `engine_props`, `stats_props` pattern.

**Rejected because**: Does not solve the core problems. State synchronization bugs will continue.

### Alternative 2: Shared State Dictionary

Create a single shared dictionary containing all state.

**Rejected because**: Dictionaries lack type safety. Access patterns would still be stringly-typed.

### Alternative 3: Event-Sourced State

Use event sourcing where state is derived from a log of events.

**Rejected because**: Over-engineering for simulation use case. Adds complexity without proportional benefit.

### Alternative 4: Database-Backed State

Store state in SQLite or similar.

**Rejected because**: Performance overhead not justified. Simulation is single-threaded and in-memory.

## Implementation Notes

### Legacy Compatibility Period

During migration, `NetworkState` provides legacy properties:

```python
@property
def network_spectrum_dict(self) -> dict:
    """DEPRECATED: Use get_link_spectrum() instead."""
    # Builds legacy format from internal state
    ...
```

These properties will be removed after all code migrates to new API.

### Enforcement

To prevent accidental caching:

```python
class NetworkState:
    def __init__(self, ...):
        self._creation_id = id(self)

    def is_spectrum_available(self, ...) -> bool:
        assert id(self) == self._creation_id, "NetworkState appears copied!"
        ...
```

### Validation at Boundaries

State validation occurs at write boundaries:

```python
def create_lightpath(self, ...):
    # Validate before mutation
    if not self.is_spectrum_available(path, start, end, core, band):
        raise ValueError("Spectrum not available")

    # Mutation
    self._allocate_spectrum(...)
```

## Related Decisions

- **ADR 0001**: Frozen dataclasses for result objects
- **ADR 0003**: Result object design
- **ADR 0005**: Legacy compatibility approach

## References

- [Martin Fowler: Domain Model](https://martinfowler.com/eaaCatalog/domainModel.html)
- [Single Source of Truth Pattern](https://en.wikipedia.org/wiki/Single_source_of_truth)
- ARCHITECTURE_REFACTOR_PLAN_V3.md Section 2: NetworkState Sharing
