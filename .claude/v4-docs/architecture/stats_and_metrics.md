# Stats and Metrics Architecture

This document describes the statistics collection system for V4.

## Overview

The V4 architecture introduces `StatsCollector` as a dedicated component for aggregating simulation metrics. This replaces the scattered statistics tracking in the current `SimStats` class and `StatsProps`.

## Current State (Legacy)

### SimStats Class

- **Location**: `fusion/core/metrics.py` (1459 lines)
- **Issues**:
  - Mixes statistics collection with analysis and formatting
  - Direct access to `sdn_props` and `stats_props`
  - Difficult to test in isolation

### StatsProps Class

- **Location**: `fusion/core/properties.py`
- **Issues**:
  - Mutable state modified from multiple places
  - No clear ownership model
  - Mixed concerns (counters, lists, computed values)

---

## V4 StatsCollector

### Design Goals

1. **Single entry point**: All stats updates go through `record_*` methods
2. **Clear ownership**: Owned by `SimulationEngine`, receives results per-request
3. **Testable**: No dependencies on legacy props classes
4. **Export-friendly**: Easy conversion to legacy format for `run_comparison.py`

### Location

`fusion/stats/collector.py`

### Interface

```python
@dataclass
class StatsCollector:
    """Collects simulation statistics."""

    config: SimulationConfig

    # Request counters
    total_requests: int = 0
    successful_requests: int = 0
    blocked_requests: int = 0

    # Blocking breakdown
    block_reasons: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    # Feature tracking
    groomed_requests: int = 0
    sliced_requests: int = 0
    protected_requests: int = 0

    # SNR tracking
    snr_values: list[float] = field(default_factory=list)

    # Modulation tracking
    modulations_used: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    # Bandwidth tracking
    total_bandwidth_requested: int = 0
    total_bandwidth_allocated: int = 0

    # Per-erlang tracking (for multi-erlang runs)
    erlang_stats: dict[float, dict] = field(default_factory=dict)
```

### Recording Methods

```python
def record_arrival(
    self,
    request: Request,
    result: AllocationResult,
) -> None:
    """
    Record arrival event outcome.

    Called by SimulationEngine after each arrival is processed.
    """
    self.total_requests += 1
    self.total_bandwidth_requested += request.bandwidth_gbps

    if result.success:
        self._record_success(request, result)
    else:
        self._record_block(request, result)

def _record_success(self, request: Request, result: AllocationResult) -> None:
    """Record successful allocation."""
    self.successful_requests += 1
    self.total_bandwidth_allocated += result.total_bandwidth_allocated_gbps

    if result.is_groomed:
        self.groomed_requests += 1
    if result.is_sliced:
        self.sliced_requests += 1
    if result.is_protected:
        self.protected_requests += 1

    # Track modulation usage
    if result.spectrum_result and result.spectrum_result.modulation:
        self.modulations_used[result.spectrum_result.modulation] += 1

def _record_block(self, request: Request, result: AllocationResult) -> None:
    """Record blocked request."""
    self.blocked_requests += 1
    if result.block_reason:
        self.block_reasons[result.block_reason.value] += 1

def record_snr(self, snr_db: float) -> None:
    """Record SNR measurement."""
    self.snr_values.append(snr_db)

def record_release(self, request: Request) -> None:
    """Record request departure (optional, for detailed tracking)."""
    pass  # Placeholder for future departure tracking
```

### Computed Properties

```python
@property
def blocking_probability(self) -> float:
    """Calculate current blocking probability."""
    if self.total_requests == 0:
        return 0.0
    return self.blocked_requests / self.total_requests

@property
def bandwidth_utilization(self) -> float:
    """Calculate bandwidth utilization ratio."""
    if self.total_bandwidth_requested == 0:
        return 0.0
    return self.total_bandwidth_allocated / self.total_bandwidth_requested

@property
def average_snr(self) -> float:
    """Calculate average SNR across all measurements."""
    if not self.snr_values:
        return 0.0
    return sum(self.snr_values) / len(self.snr_values)

@property
def grooming_ratio(self) -> float:
    """Fraction of successful requests that used grooming."""
    if self.successful_requests == 0:
        return 0.0
    return self.groomed_requests / self.successful_requests
```

### Export Methods

```python
def to_comparison_format(self) -> dict:
    """
    Export in format expected by run_comparison.py.

    Maintains compatibility with existing comparison framework.
    """
    return {
        "blocking_probability": self.blocking_probability,
        "total_requests": self.total_requests,
        "successful_requests": self.successful_requests,
        "blocked_requests": self.blocked_requests,
        "block_reasons": dict(self.block_reasons),
        "groomed_requests": self.groomed_requests,
        "sliced_requests": self.sliced_requests,
        "protected_requests": self.protected_requests,
        "modulations_used": dict(self.modulations_used),
        "average_snr": self.average_snr,
    }

def to_legacy_stats_props(self) -> dict:
    """
    Convert to legacy StatsProps format.

    Used during migration for compatibility.
    """
    return {
        "simulation_blocking_list": [self.blocking_probability],
        "block_reasons_dict": dict(self.block_reasons),
        "modulations_used_dict": dict(self.modulations_used),
        "snr_list": self.snr_values.copy(),
    }
```

---

## Integration with SimulationEngine

```python
class SimulationEngine:
    def __init__(self, config: SimulationConfig):
        self._config = config
        self._network_state = NetworkState(...)
        self._orchestrator = SDNOrchestrator(...)
        self._stats = StatsCollector(config)  # Single stats instance

    def handle_request(self, request: Request) -> None:
        if request.is_arrival:
            result = self._orchestrator.handle_arrival(request, self._network_state)
            self._stats.record_arrival(request, result)
        else:
            self._orchestrator.handle_release(request, self._network_state)
            self._stats.record_release(request)

    def get_results(self) -> dict:
        """Get final simulation results."""
        return self._stats.to_comparison_format()
```

---

## Legacy Mapping

| Legacy (`StatsProps`) | V4 (`StatsCollector`) | Notes |
|-----------------------|----------------------|-------|
| `simulation_blocking_list` | `blocking_probability` | Computed property |
| `block_reasons_dict` | `block_reasons` | Direct mapping |
| `modulations_used_dict` | `modulations_used` | Direct mapping |
| `snr_list` | `snr_values` | Direct mapping |
| Various scattered counters | Centralized fields | Single source of truth |

---

## Testing Strategy

### Unit Tests

```python
# fusion/tests/stats/test_collector.py

def test_record_arrival_success():
    """Successful arrival updates counters correctly."""
    config = create_test_config()
    collector = StatsCollector(config)
    request = create_test_request(bandwidth_gbps=100)
    result = AllocationResult(success=True, total_bandwidth_allocated_gbps=100)

    collector.record_arrival(request, result)

    assert collector.total_requests == 1
    assert collector.successful_requests == 1
    assert collector.blocked_requests == 0
    assert collector.blocking_probability == 0.0

def test_record_arrival_blocked():
    """Blocked arrival updates counters and reasons."""
    config = create_test_config()
    collector = StatsCollector(config)
    request = create_test_request()
    result = AllocationResult(success=False, block_reason=BlockReason.NO_SPECTRUM)

    collector.record_arrival(request, result)

    assert collector.total_requests == 1
    assert collector.successful_requests == 0
    assert collector.blocked_requests == 1
    assert collector.block_reasons["no_spectrum"] == 1

def test_blocking_probability():
    """Blocking probability computed correctly."""
    config = create_test_config()
    collector = StatsCollector(config)

    # 3 successes, 2 blocks
    for _ in range(3):
        collector.record_arrival(
            create_test_request(),
            AllocationResult(success=True, total_bandwidth_allocated_gbps=100)
        )
    for _ in range(2):
        collector.record_arrival(
            create_test_request(),
            AllocationResult(success=False, block_reason=BlockReason.NO_SPECTRUM)
        )

    assert collector.blocking_probability == 0.4

def test_to_comparison_format():
    """Export format matches expected structure."""
    config = create_test_config()
    collector = StatsCollector(config)
    collector.record_arrival(
        create_test_request(),
        AllocationResult(success=True, total_bandwidth_allocated_gbps=100)
    )

    result = collector.to_comparison_format()

    assert "blocking_probability" in result
    assert "total_requests" in result
    assert "block_reasons" in result
```

---

## Migration Notes

### Phase 1 (Current)

- Create `StatsCollector` skeleton with basic counters
- No integration with engine yet
- Tests validate counter logic

### Phase 3

- Integrate `StatsCollector` with `SimulationEngine`
- Add `record_arrival()` calls in engine
- Validate against `run_comparison.py`

### Phase 6

- Remove legacy `StatsProps` usage
- Remove `to_legacy_stats_props()` method
- Clean up `SimStats` class
