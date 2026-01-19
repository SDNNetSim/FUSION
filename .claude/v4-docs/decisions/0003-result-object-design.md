# ADR-0003: Result Object Design

## Status

Accepted

## Context

Pipelines in V4 need to return structured data about their computations. The legacy code returns results by mutating shared Props objects or returning untyped dictionaries. This makes it difficult to:
- Know what a pipeline returns
- Validate return values
- Compose pipelines safely

## Decision

Define frozen dataclass result types for each pipeline stage:

| Result Type | Producer | Key Fields |
|-------------|----------|------------|
| `RouteResult` | RoutingPipeline | paths, modulations, weights_km |
| `SpectrumResult` | SpectrumPipeline | is_free, start_slot, end_slot, core, band |
| `GroomingResult` | GroomingPipeline | fully_groomed, partially_groomed, remaining_bw |
| `SlicingResult` | SlicingPipeline | success, num_slices, lightpath_ids |
| `SNRResult` | SNRPipeline | passed, snr_db, required_snr |
| `AllocationResult` | Orchestrator | success, lightpaths_created, block_reason |

## Design Principles

### 1. Single Responsibility

Each result type represents exactly one pipeline's output:

```python
# Good: RouteResult only contains routing information
@dataclass(frozen=True)
class RouteResult:
    paths: list[list[str]]
    weights_km: list[float]
    modulations: list[list[str | None]]

# Bad: Mixing concerns
@dataclass
class CombinedResult:
    paths: list[list[str]]
    spectrum_slots: list[int]  # Wrong - spectrum is separate pipeline
```

### 2. Immutability

All results are frozen to prevent accidental modification:

```python
@dataclass(frozen=True)
class SpectrumResult:
    is_free: bool
    start_slot: int
    # ...
```

### 3. Computed Properties for Convenience

Add properties that derive from fields:

```python
@dataclass(frozen=True)
class RouteResult:
    paths: list[list[str]]
    # ...

    @property
    def is_empty(self) -> bool:
        return len(self.paths) == 0

    @property
    def num_paths(self) -> int:
        return len(self.paths)
```

### 4. Factory Methods for Common Cases

Provide factory methods for frequently used patterns:

```python
@dataclass(frozen=True)
class AllocationResult:
    success: bool
    block_reason: BlockReason | None = None
    # ...

    @classmethod
    def blocked(cls, reason: BlockReason) -> "AllocationResult":
        return cls(success=False, block_reason=reason)

    @classmethod
    def success_new(cls, lightpath_ids: list[int]) -> "AllocationResult":
        return cls(success=True, lightpaths_created=lightpath_ids)
```

### 5. Optional Fields Have Defaults

Optional fields default to None or empty collections:

```python
@dataclass(frozen=True)
class RouteResult:
    paths: list[list[str]]
    weights_km: list[float]
    modulations: list[list[str | None]]

    # Optional backup paths (for protection)
    backup_paths: list[list[str]] | None = None
    backup_weights_km: list[float] | None = None

    # Metadata
    strategy_name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
```

## Consequences

### Positive

1. **Type safety**: Full typing for all pipeline outputs
2. **IDE support**: Autocomplete, refactoring support
3. **Documentation**: Field names document the API
4. **Testing**: Easy to construct test results
5. **Immutability**: Results cannot be accidentally modified

### Negative

1. **Object creation overhead**: Creating dataclasses vs returning dicts
2. **More code**: Each result type is a new class
3. **Learning curve**: Developers must learn result types

### Mitigations

- Overhead is minimal for our scale
- Result types are documented in architecture docs
- Factory methods reduce boilerplate

## Result Type Specifications

### RouteResult

```python
@dataclass(frozen=True)
class RouteResult:
    """Output of routing pipeline."""
    paths: list[list[str]]              # [[node1, node2], ...]
    weights_km: list[float]             # Path lengths
    modulations: list[list[str | None]] # Valid mods per path

    backup_paths: list[list[str]] | None = None
    backup_weights_km: list[float] | None = None
    backup_modulations: list[list[str | None]] | None = None

    strategy_name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
```

### SpectrumResult

```python
@dataclass(frozen=True)
class SpectrumResult:
    """Output of spectrum pipeline."""
    is_free: bool
    start_slot: int = 0
    end_slot: int = 0
    core: int = 0
    band: str = ""
    modulation: str = ""
    slots_needed: int = 0
```

### AllocationResult

```python
@dataclass(frozen=True)
class AllocationResult:
    """Final allocation outcome from orchestrator."""
    success: bool
    block_reason: BlockReason | None = None

    lightpaths_created: list[int] = field(default_factory=list)
    lightpaths_groomed: list[int] = field(default_factory=list)

    is_groomed: bool = False
    is_partially_groomed: bool = False
    is_sliced: bool = False
    is_protected: bool = False

    total_bandwidth_allocated_gbps: int = 0
```

## Related Decisions

- ADR-0001: Frozen Dataclasses
- ADR-0002: Request Status Enum
