# ADR-0002: Request Status Enum and Lifecycle

## Status

Accepted

## Context

Requests in FUSION go through a lifecycle from arrival to either successful allocation or blocking, and eventually release. The legacy code tracks this implicitly through various flags and dictionary entries, making it difficult to:
- Understand what state a request is in
- Ensure valid state transitions
- Collect statistics by state

## Decision

Introduce a `RequestStatus` enum with explicit states:

```python
class RequestStatus(Enum):
    PENDING = auto()    # Created, not yet processed
    ROUTED = auto()     # Successfully allocated
    BLOCKED = auto()    # Failed allocation
    RELEASED = auto()   # Departed, resources freed
```

And a `BlockReason` enum for blocking causes:

```python
class BlockReason(Enum):
    NO_ROUTE = "no_route"
    NO_SPECTRUM = "no_spectrum"
    SNR_FAILURE = "snr_failure"
    XT_FAILURE = "xt_failure"
    CONGESTION = "congestion"
    NO_MODULATION = "no_modulation"
    PROTECTION_UNAVAILABLE = "protection_unavailable"
    GROOMING_FAILED = "grooming_failed"
    SLICING_FAILED = "slicing_failed"
```

## State Machine

```
PENDING ──┬──> ROUTED ──> RELEASED
          │
          └──> BLOCKED
```

**Valid transitions:**
- PENDING -> ROUTED (allocation succeeded)
- PENDING -> BLOCKED (allocation failed)
- ROUTED -> RELEASED (departure event)

**Invalid transitions:**
- BLOCKED -> * (terminal state)
- RELEASED -> * (terminal state)
- ROUTED -> BLOCKED (cannot fail after success)

## Consequences

### Positive

1. **Explicit states**: No ambiguity about request status
2. **Type safety**: Enum prevents invalid status strings
3. **IDE support**: Autocomplete for status values
4. **Statistics**: Easy to count requests by status
5. **Validation**: Can enforce state transition rules

### Negative

1. **Migration**: Must map legacy patterns to enum values
2. **Serialization**: Enums need special handling for JSON/pickle

### Implementation Notes

```python
@dataclass
class Request:
    status: RequestStatus = RequestStatus.PENDING
    block_reason: str | None = None

    @property
    def is_arrival(self) -> bool:
        return self.status == RequestStatus.PENDING

    @property
    def is_successful(self) -> bool:
        return self.status == RequestStatus.ROUTED

    @property
    def is_blocked(self) -> bool:
        return self.status == RequestStatus.BLOCKED
```

### BlockReason String Values

The enum uses string values that match legacy constants for backward compatibility:

```python
# Legacy code checks:
if block_reason == "no_spectrum":
    ...

# New code can use:
if block_reason == BlockReason.NO_SPECTRUM.value:
    ...
# Or better:
if request.block_reason == BlockReason.NO_SPECTRUM:
    ...
```

## Related Decisions

- ADR-0001: Frozen Dataclasses
- ADR-0003: Result Object Design
