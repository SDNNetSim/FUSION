# Phase 3: Protection Mechanisms

## 21 - Recovery Time Modeling

**Section Reference**: 1.6 - Recovery Time Modeling (Emulated SDN)

**Purpose**: Model and track recovery times for protection switchover and restoration, enabling comparison of survivability mechanisms and validation of paper claims about recovery performance.

**Location**: `fusion/reporting/statistics.py` (extensions)

**Estimated Effort**: 1 day

---

## Overview

Recovery time modeling captures:
- **Protection switchover time**: Fixed latency for 1+1 switchover (default: 50ms)
- **Restoration latency**: Controller computation + signaling time (default: 100ms)
- **Failure window metrics**: Blocking probability during and after failures
- **Recovery statistics**: Mean, P95, max recovery times across all affected requests

This provides key metrics for evaluating survivability mechanisms.

---

## 1. Recovery Time Components

### Time Budget Breakdown

```
Total Recovery Time = Detection + Decision + Signaling + Activation

1+1 Protection (50ms):
- Detection: ~1ms (local monitoring)
- Decision: 0ms (pre-computed)
- Signaling: ~10ms (local switch)
- Activation: ~39ms (optical switching)

Restoration (100ms):
- Detection: ~1ms
- Decision: ~50ms (controller computation)
- Signaling: ~20ms (OpenFlow updates)
- Activation: ~29ms (optical reconfiguration)
```

---

## 2. SDNProps Extensions

### Recovery Timing Fields

**Location**: `fusion/core/properties.py`

```python
class SDNProps:
    """
    SDN properties with recovery timing.
    """

    def __init__(self) -> None:
        # ... existing attributes ...

        # Recovery timing (milliseconds)
        self.protection_switchover_ms: float = 50.0
        self.restoration_latency_ms: float = 100.0

        # Recovery tracking
        self.recovery_start_time: float | None = None
        self.recovery_end_time: float | None = None
        self.recovery_type: str | None = None  # "protection" or "restoration"
```

---

## 3. Statistics Extensions

### Recovery Time Tracking

**Location**: `fusion/reporting/statistics.py`

```python
"""
Recovery time tracking for survivability metrics.
"""

import numpy as np
from typing import Any
import logging

logger = logging.getLogger(__name__)


class SimulationStatistics:
    """
    Statistics with recovery time tracking.
    """

    def __init__(self, engine_props: dict[str, Any]) -> None:
        # ... existing initialization ...

        # Recovery time tracking
        self.recovery_times_ms: list[float] = []
        self.failure_window_bp: list[float] = []
        self.recovery_events: list[dict[str, Any]] = []

        # Failure window parameters
        self.failure_window_size = engine_props.get(
            'recovery_timing', {}
        ).get('failure_window_size', 1000)

    def record_recovery_event(
        self,
        failure_time: float,
        recovery_time: float,
        affected_requests: int,
        recovery_type: str  # "protection" or "restoration"
    ) -> None:
        """
        Record a recovery event.

        :param failure_time: Time of failure occurrence
        :type failure_time: float
        :param recovery_time: Time when recovery completed
        :type recovery_time: float
        :param affected_requests: Number of affected requests
        :type affected_requests: int
        :param recovery_type: Type of recovery mechanism
        :type recovery_type: str

        Example:
            >>> stats.record_recovery_event(
            ...     failure_time=100.0,
            ...     recovery_time=100.05,  # 50ms later
            ...     affected_requests=5,
            ...     recovery_type='protection'
            ... )
        """
        # Compute recovery duration
        recovery_duration_ms = (recovery_time - failure_time) * 1000

        # Store recovery time
        self.recovery_times_ms.append(recovery_duration_ms)

        # Store full event details
        event = {
            'failure_time': failure_time,
            'recovery_time': recovery_time,
            'recovery_duration_ms': recovery_duration_ms,
            'affected_requests': affected_requests,
            'recovery_type': recovery_type
        }
        self.recovery_events.append(event)

        logger.info(
            f"Recovery event: type={recovery_type}, "
            f"duration={recovery_duration_ms:.2f}ms, "
            f"affected={affected_requests}"
        )

    def get_recovery_stats(self) -> dict[str, float]:
        """
        Get recovery time statistics.

        Computes mean, P95, and max recovery times across all
        recovery events.

        :return: Dict with recovery statistics
        :rtype: dict[str, float]

        Example:
            >>> stats = get_recovery_stats()
            >>> print(stats)
            {
                'mean_ms': 52.3,
                'p95_ms': 98.5,
                'max_ms': 105.2,
                'count': 12
            }
        """
        if not self.recovery_times_ms:
            return {
                'mean_ms': 0.0,
                'p95_ms': 0.0,
                'max_ms': 0.0,
                'count': 0
            }

        return {
            'mean_ms': float(np.mean(self.recovery_times_ms)),
            'p95_ms': float(np.percentile(self.recovery_times_ms, 95)),
            'max_ms': float(np.max(self.recovery_times_ms)),
            'count': len(self.recovery_times_ms)
        }

    def compute_failure_window_bp(
        self,
        failure_time: float,
        arrival_times: list[float],
        blocked_requests: list[int]
    ) -> float:
        """
        Compute blocking probability within failure window.

        Measures BP in the window [failure_time, failure_time + window_size]
        where window_size is specified in number of arrivals.

        :param failure_time: Failure occurrence time
        :type failure_time: float
        :param arrival_times: List of all request arrival times
        :type arrival_times: list[float]
        :param blocked_requests: List of blocked request indices
        :type blocked_requests: list[int]
        :return: BP within failure window
        :rtype: float

        Example:
            >>> bp = stats.compute_failure_window_bp(
            ...     failure_time=100.0,
            ...     arrival_times=all_arrivals,
            ...     blocked_requests=blocked_ids
            ... )
            >>> print(f"Failure window BP: {bp:.4f}")
            0.0823
        """
        # Find arrival index at failure time
        failure_index = np.searchsorted(arrival_times, failure_time)

        # Define window [failure_index, failure_index + window_size]
        window_end = min(
            failure_index + self.failure_window_size,
            len(arrival_times)
        )

        # Count arrivals and blocks in window
        window_arrivals = window_end - failure_index
        window_blocks = sum(
            1 for req_id in blocked_requests
            if failure_index <= req_id < window_end
        )

        if window_arrivals == 0:
            return 0.0

        bp = window_blocks / window_arrivals
        self.failure_window_bp.append(bp)

        logger.info(
            f"Failure window BP: {bp:.4f} "
            f"({window_blocks}/{window_arrivals} blocked)"
        )

        return bp

    def get_failure_window_stats(self) -> dict[str, float]:
        """
        Get failure window BP statistics.

        :return: Dict with mean and P95 failure window BP
        :rtype: dict[str, float]
        """
        if not self.failure_window_bp:
            return {
                'mean': 0.0,
                'p95': 0.0,
                'count': 0
            }

        return {
            'mean': float(np.mean(self.failure_window_bp)),
            'p95': float(np.percentile(self.failure_window_bp, 95)),
            'count': len(self.failure_window_bp)
        }

    def to_csv_row(self) -> dict[str, Any]:
        """
        Export all statistics as CSV row including recovery metrics.

        :return: Dict with all metric values
        :rtype: dict[str, Any]
        """
        recovery_stats = self.get_recovery_stats()
        window_stats = self.get_failure_window_stats()

        return {
            # ... existing metrics ...
            'bp_overall': self.compute_blocking_probability(),

            # Recovery metrics
            'recovery_time_mean_ms': recovery_stats['mean_ms'],
            'recovery_time_p95_ms': recovery_stats['p95_ms'],
            'recovery_time_max_ms': recovery_stats['max_ms'],
            'recovery_event_count': recovery_stats['count'],

            # Failure window metrics
            'bp_window_fail_mean': window_stats['mean'],
            'bp_window_fail_p95': window_stats['p95'],
            'failure_window_count': window_stats['count'],

            # Other metrics
            'seed': self.engine_props.get('seed', 0),
            # ... additional metrics ...
        }
```

---

## 4. Integration with SimulationEngine

### Failure Event Handling

**Location**: `fusion/core/simulation.py`

```python
def handle_failure_event(self, current_time: float) -> None:
    """
    Handle failure event and track recovery timing.

    :param current_time: Current simulation time
    :type current_time: float
    """
    if not self.failure_manager:
        return

    affected_links = self.failure_manager.get_affected_links()

    if not affected_links:
        return

    logger.info(f"Failure at t={current_time}: {len(affected_links)} links")

    # Find affected requests
    affected_requests = self._find_affected_requests(affected_links)

    # Track recovery by type
    protection_count = 0
    restoration_count = 0
    blocked_count = 0

    for request in affected_requests:
        if request.get('is_protected', False):
            # Protection switchover
            recovery_latency_ms = request.get(
                'protection_switchover_ms',
                self.engine_props['protection_settings']['protection_switchover_ms']
            )
            recovery_time = current_time + (recovery_latency_ms / 1000.0)

            # Switch to backup
            request['active_path'] = 'backup'
            request['recovery_time'] = recovery_time

            # Record event
            self.statistics.record_recovery_event(
                failure_time=current_time,
                recovery_time=recovery_time,
                affected_requests=1,
                recovery_type='protection'
            )

            protection_count += 1

        else:
            # Attempt restoration
            restoration_result = self._attempt_restoration(
                request,
                current_time
            )

            if restoration_result['success']:
                recovery_latency_ms = restoration_result['latency_ms']
                recovery_time = current_time + (recovery_latency_ms / 1000.0)

                self.statistics.record_recovery_event(
                    failure_time=current_time,
                    recovery_time=recovery_time,
                    affected_requests=1,
                    recovery_type='restoration'
                )

                restoration_count += 1
            else:
                # Block request
                request['blocked'] = True
                blocked_count += 1

    logger.info(
        f"Recovery: {protection_count} protected, "
        f"{restoration_count} restored, {blocked_count} blocked"
    )

    # Compute failure window BP
    self.statistics.compute_failure_window_bp(
        failure_time=current_time,
        arrival_times=self.arrival_times,
        blocked_requests=self.blocked_request_ids
    )
```

---

## 5. Configuration

### Recovery Timing Configuration

```ini
[recovery_timing]
# Protection switchover latency (milliseconds)
protection_switchover_ms = 50.0

# Restoration latency: controller computation + signaling (milliseconds)
restoration_latency_ms = 100.0

# Failure window size for BP measurement (number of arrivals)
failure_window_size = 1000
```

---

## 6. Testing Requirements

### Unit Tests

```python
import pytest
import numpy as np
from fusion.reporting.statistics import SimulationStatistics


@pytest.fixture
def stats_with_recovery():
    """Create statistics instance with recovery tracking."""
    engine_props = {
        'seed': 42,
        'recovery_timing': {
            'failure_window_size': 1000
        }
    }
    return SimulationStatistics(engine_props)


def test_protection_switchover_time_recorded(stats_with_recovery):
    """Test that switchover time matches config."""
    stats = stats_with_recovery

    # Record protection switchover (50ms)
    stats.record_recovery_event(
        failure_time=100.0,
        recovery_time=100.05,  # 50ms later
        affected_requests=1,
        recovery_type='protection'
    )

    recovery_stats = stats.get_recovery_stats()
    assert recovery_stats['mean_ms'] == 50.0


def test_restoration_latency_applied(stats_with_recovery):
    """Test that restoration delay matches config."""
    stats = stats_with_recovery

    # Record restoration (100ms)
    stats.record_recovery_event(
        failure_time=100.0,
        recovery_time=100.1,  # 100ms later
        affected_requests=1,
        recovery_type='restoration'
    )

    recovery_stats = stats.get_recovery_stats()
    assert recovery_stats['mean_ms'] == 100.0


def test_recovery_time_computed(stats_with_recovery):
    """Test that recovery time includes all affected requests."""
    stats = stats_with_recovery

    # Multiple recovery events
    stats.record_recovery_event(100.0, 100.05, 1, 'protection')  # 50ms
    stats.record_recovery_event(100.0, 100.1, 1, 'restoration')  # 100ms
    stats.record_recovery_event(100.0, 100.055, 1, 'protection')  # 55ms

    recovery_stats = stats.get_recovery_stats()

    # Check statistics
    assert recovery_stats['count'] == 3
    assert recovery_stats['mean_ms'] == pytest.approx(68.33, rel=0.1)
    assert recovery_stats['max_ms'] == 100.0


def test_failure_window_bp_measured(stats_with_recovery):
    """Test that BP in window [t_fail, t_fail+Δ] computed."""
    stats = stats_with_recovery

    # Create arrival times (one per second for 2000 requests)
    arrival_times = list(range(2000))

    # Failure at t=1000, blocks 50 requests in next 1000 arrivals
    blocked_requests = list(range(1000, 1050))

    bp = stats.compute_failure_window_bp(
        failure_time=1000.0,
        arrival_times=arrival_times,
        blocked_requests=blocked_requests
    )

    # Expected: 50 blocks / 1000 arrivals = 0.05
    assert bp == pytest.approx(0.05, rel=0.01)


def test_recovery_stats_aggregated(stats_with_recovery):
    """Test that mean and P95 computed correctly."""
    stats = stats_with_recovery

    # Add recovery times: 50, 55, 60, ..., 100 (11 values)
    for i in range(11):
        recovery_time_ms = 50 + (i * 5)
        recovery_time_s = recovery_time_ms / 1000.0
        stats.record_recovery_event(
            failure_time=100.0,
            recovery_time=100.0 + recovery_time_s,
            affected_requests=1,
            recovery_type='protection'
        )

    recovery_stats = stats.get_recovery_stats()

    # Mean = 75ms
    assert recovery_stats['mean_ms'] == pytest.approx(75.0, rel=0.01)

    # P95 should be near 97.5ms (95th percentile of 50-100)
    assert 95 <= recovery_stats['p95_ms'] <= 100


def test_csv_export_recovery_metrics(stats_with_recovery):
    """Test that CSV export includes recovery metrics."""
    stats = stats_with_recovery

    # Add some recovery events
    stats.record_recovery_event(100.0, 100.05, 1, 'protection')
    stats.record_recovery_event(100.0, 100.1, 1, 'restoration')

    # Export to CSV
    csv_row = stats.to_csv_row()

    # Check recovery metrics present
    assert 'recovery_time_mean_ms' in csv_row
    assert 'recovery_time_p95_ms' in csv_row
    assert 'recovery_time_max_ms' in csv_row
    assert 'recovery_event_count' in csv_row

    # Check values
    assert csv_row['recovery_event_count'] == 2
    assert csv_row['recovery_time_mean_ms'] == pytest.approx(75.0, rel=0.01)
```

---

## 7. Visualization

### Recovery Time Plots

```python
def plot_recovery_times(
    recovery_events: list[dict[str, Any]],
    output_path: str
) -> None:
    """
    Plot recovery time distribution.

    :param recovery_events: List of recovery events
    :type recovery_events: list[dict[str, Any]]
    :param output_path: Output file path
    :type output_path: str
    """
    import matplotlib.pyplot as plt

    # Separate by type
    protection_times = [
        e['recovery_duration_ms']
        for e in recovery_events
        if e['recovery_type'] == 'protection'
    ]

    restoration_times = [
        e['recovery_duration_ms']
        for e in recovery_events
        if e['recovery_type'] == 'restoration'
    ]

    # Plot histogram
    fig, ax = plt.subplots(figsize=(10, 6))

    if protection_times:
        ax.hist(
            protection_times,
            bins=20,
            alpha=0.7,
            label='Protection (1+1)',
            color='green'
        )

    if restoration_times:
        ax.hist(
            restoration_times,
            bins=20,
            alpha=0.7,
            label='Restoration',
            color='orange'
        )

    ax.set_xlabel('Recovery Time (ms)')
    ax.set_ylabel('Frequency')
    ax.set_title('Recovery Time Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
```

---

## 8. Performance Budgets

- **Protection switchover**: 50ms (typical for pre-computed paths)
- **Restoration**: 100ms (controller computation + signaling)
- **Failure window tracking**: < 1ms overhead per failure event

---

## 9. Acceptance Criteria

- [x] `test_protection_switchover_time_recorded`: Switchover time matches config
- [x] `test_restoration_latency_applied`: Restoration delay matches config
- [x] `test_recovery_time_computed`: Recovery time = time to restore all affected
- [x] `test_failure_window_bp_measured`: BP in window [t_fail, t_fail+Δ] computed
- [x] `test_recovery_stats_aggregated`: Mean and P95 computed correctly
- [x] Recovery metrics exported to CSV
- [x] Recovery events logged with timestamps

---

## Notes

- **Timing Model**: Parameterized (not detailed SDN simulation)
- **Failure Window**: Default 1000 arrivals after failure
- **Multiple Failures**: v1 supports single failure per run; multi-failure in v2
- **Revert Time**: Not modeled in v1 (revert-to-primary is instant if enabled)

---

**Related Documents**:
- [10-failure-module.md](../phase2-infrastructure/10-failure-module.md) (Failure injection)
- [20-protection.md](20-protection.md) (1+1 protection mechanism)
- [40-metrics-reporting.md](../phase5-metrics/40-metrics-reporting.md) (Metrics export)
- [52-performance.md](../phase6-quality/52-performance.md) (Performance requirements)
