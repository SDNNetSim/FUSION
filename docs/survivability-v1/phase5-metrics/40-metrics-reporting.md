# Phase 5: Metrics & Reporting

## 40 - Metrics & Reporting

**Section Reference**: 1.7 - Metrics & Reporting

**Purpose**: Extend FUSION's statistics and reporting to capture survivability-specific metrics including recovery times, failure window blocking probability, fragmentation, and multi-seed aggregation.

**Location**: `fusion/reporting/statistics.py` (extensions)

**Estimated Effort**: 1 day

---

## Overview

Survivability experiments require tracking additional metrics beyond standard BP:
- **Recovery metrics**: Mean, P95, max recovery times
- **Failure window BP**: Blocking probability during/after failures
- **Fragmentation proxy**: Spectrum utilization efficiency
- **Decision times**: Policy inference latency
- **Multi-seed aggregation**: Mean, std, CI95 across seeds

---

## 1. Extended Statistics Class

**Location**: `fusion/reporting/statistics.py`

```python
"""
Extended statistics for survivability experiments.
"""

import numpy as np
from typing import Any
import logging

logger = logging.getLogger(__name__)


class SimulationStatistics:
    """
    Statistics with survivability metrics.
    """

    def __init__(self, engine_props: dict[str, Any]) -> None:
        # ... existing initialization ...

        # Survivability metrics
        self.recovery_times_ms: list[float] = []
        self.fragmentation_scores: list[float] = []
        self.decision_times_ms: list[float] = []
        self.failure_window_bp_list: list[float] = []

    def compute_fragmentation_proxy(
        self,
        path: list[int],
        network_spectrum_dict: dict[tuple[Any, Any], dict[str, Any]]
    ) -> float:
        """
        Compute fragmentation proxy for a path.

        Fragmentation = 1 - (largest_contiguous_block / total_free_slots)

        Higher values indicate more fragmentation.

        :param path: Path node list
        :type path: list[int]
        :param network_spectrum_dict: Spectrum state
        :type network_spectrum_dict: dict[tuple[Any, Any], dict[str, Any]]
        :return: Fragmentation score [0, 1]
        :rtype: float

        Example:
            >>> frag = stats.compute_fragmentation_proxy(path, spectrum_dict)
            >>> print(f"Fragmentation: {frag:.3f}")
            0.347
        """
        total_free = 0
        largest_contig = 0

        for i in range(len(path) - 1):
            link = (path[i], path[i + 1])
            reverse_link = (path[i + 1], path[i])

            link_spectrum = network_spectrum_dict.get(
                link,
                network_spectrum_dict.get(reverse_link, {})
            )

            if not link_spectrum:
                continue

            slots = link_spectrum.get('slots', [])

            # Find free blocks
            free_blocks = self._find_free_blocks(slots)

            if free_blocks:
                link_total = sum(block[1] - block[0] for block in free_blocks)
                link_largest = max(block[1] - block[0] for block in free_blocks)

                total_free += link_total
                largest_contig = max(largest_contig, link_largest)

        if total_free == 0:
            return 1.0  # Fully fragmented

        frag = 1.0 - (largest_contig / total_free)
        return frag

    def _find_free_blocks(self, slots: list[int]) -> list[tuple[int, int]]:
        """Find contiguous free blocks."""
        blocks = []
        start = None

        for i, slot in enumerate(slots):
            if slot == 0:  # Free
                if start is None:
                    start = i
            else:  # Occupied
                if start is not None:
                    blocks.append((start, i))
                    start = None

        if start is not None:
            blocks.append((start, len(slots)))

        return blocks

    def to_csv_row(self) -> dict[str, Any]:
        """
        Export all statistics as CSV row.

        :return: Dict with all metric values
        :rtype: dict[str, Any]
        """
        return {
            # Experiment parameters
            'topology': self.engine_props.get('network', 'unknown'),
            'load': self.engine_props.get('erlang', 0),
            'failure_type': self.engine_props.get('failure_settings', {}).get('failure_type', 'none'),
            'k_paths': self.engine_props.get('routing_settings', {}).get('k_paths', 1),
            'policy': self.engine_props.get('offline_rl_settings', {}).get('policy_type', 'ksp_ff'),
            'seed': self.engine_props.get('seed', 0),

            # Standard metrics
            'bp_overall': self.compute_blocking_probability(),

            # Failure window metrics
            'bp_window_fail_mean': np.mean(self.failure_window_bp_list) if self.failure_window_bp_list else 0.0,
            'bp_window_fail_p95': np.percentile(self.failure_window_bp_list, 95) if self.failure_window_bp_list else 0.0,

            # Recovery metrics
            'recovery_time_mean_ms': np.mean(self.recovery_times_ms) if self.recovery_times_ms else 0.0,
            'recovery_time_p95_ms': np.percentile(self.recovery_times_ms, 95) if self.recovery_times_ms else 0.0,

            # Fragmentation
            'frag_proxy_mean': np.mean(self.fragmentation_scores) if self.fragmentation_scores else 0.0,

            # Decision times
            'decision_time_mean_ms': np.mean(self.decision_times_ms) if self.decision_times_ms else 0.0
        }
```

---

## 2. Multi-Seed Aggregation

**Location**: `fusion/reporting/aggregation.py` (new)

```python
"""
Multi-seed result aggregation.
"""

import numpy as np
from typing import Any
import pandas as pd


def aggregate_seed_results(
    results: list[dict[str, Any]],
    metric_keys: list[str] = ['bp_overall', 'recovery_time_mean_ms']
) -> dict[str, dict[str, float]]:
    """
    Aggregate results across multiple seeds.

    Computes mean, std, and 95% confidence intervals.

    :param results: List of result dicts (one per seed)
    :type results: list[dict[str, Any]]
    :param metric_keys: Metrics to aggregate
    :type metric_keys: list[str]
    :return: Aggregated stats with mean, std, ci95
    :rtype: dict[str, dict[str, float]]

    Example:
        >>> results = [{'bp_overall': 0.05, 'seed': 42}, ...]
        >>> agg = aggregate_seed_results(results, ['bp_overall'])
        >>> print(agg['bp_overall'])
        {'mean': 0.052, 'std': 0.003, 'ci95_lower': 0.049, 'ci95_upper': 0.055}
    """
    aggregated = {}

    for key in metric_keys:
        values = [r[key] for r in results if key in r]

        if not values:
            continue

        mean = np.mean(values)
        std = np.std(values)
        ci95 = 1.96 * std / np.sqrt(len(values))  # 95% CI

        aggregated[key] = {
            'mean': mean,
            'std': std,
            'ci95_lower': mean - ci95,
            'ci95_upper': mean + ci95,
            'n': len(values)
        }

    return aggregated


def create_comparison_table(
    baseline_results: list[dict[str, Any]],
    rl_results: list[dict[str, Any]],
    metrics: list[str]
) -> pd.DataFrame:
    """
    Create comparison table for baseline vs RL.

    :param baseline_results: Baseline results
    :type baseline_results: list[dict[str, Any]]
    :param rl_results: RL policy results
    :type rl_results: list[dict[str, Any]]
    :param metrics: Metrics to compare
    :type metrics: list[str]
    :return: Comparison dataframe
    :rtype: pd.DataFrame
    """
    baseline_agg = aggregate_seed_results(baseline_results, metrics)
    rl_agg = aggregate_seed_results(rl_results, metrics)

    rows = []
    for metric in metrics:
        if metric in baseline_agg and metric in rl_agg:
            baseline_val = baseline_agg[metric]['mean']
            rl_val = rl_agg[metric]['mean']
            improvement = ((baseline_val - rl_val) / baseline_val) * 100

            rows.append({
                'Metric': metric,
                'Baseline': f"{baseline_val:.4f} ± {baseline_agg[metric]['std']:.4f}",
                'RL': f"{rl_val:.4f} ± {rl_agg[metric]['std']:.4f}",
                'Improvement (%)': f"{improvement:.2f}"
            })

    return pd.DataFrame(rows)
```

---

## 3. CSV Export

**Location**: `fusion/reporting/csv_export.py` (new)

```python
"""
CSV export utilities.
"""

import csv
from pathlib import Path
from typing import Any
import logging

logger = logging.getLogger(__name__)


def export_results_to_csv(
    results: list[dict[str, Any]],
    output_path: str
) -> None:
    """
    Export results to CSV file.

    :param results: List of result dicts
    :type results: list[dict[str, Any]]
    :param output_path: Output CSV path
    :type output_path: str

    Example:
        >>> results = [stats.to_csv_row() for stats in all_stats]
        >>> export_results_to_csv(results, 'results/survivability.csv')
    """
    if not results:
        logger.warning("No results to export")
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = list(results[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    logger.info(f"Exported {len(results)} results to {output_path}")
```

---

## 4. Configuration

```ini
[reporting]
# CSV export settings
export_csv = true
csv_output_path = results/survivability_results.csv

# Multi-seed aggregation
aggregate_seeds = true
seed_list = [42, 43, 44, 45, 46]

# Metrics to export
metrics = [bp_overall, recovery_time_mean_ms, frag_proxy_mean]
```

---

## 5. Testing Requirements

```python
def test_fragmentation_proxy_computed():
    """Test that frag score in [0, 1] for all paths."""
    pass  # See implementation in phase2-infrastructure/21-recovery-timing.md


def test_decision_time_recorded():
    """Test that decision time logged for each request."""
    pass


def test_csv_export_all_metrics():
    """Test that CSV contains all required columns."""
    pass


def test_seed_aggregation():
    """Test that mean, std, CI95 computed correctly across seeds."""
    pass


def test_failure_window_bp_tracked():
    """Test that BP in failure window measured separately."""
    pass
```

---

## 6. Acceptance Criteria

- [x] All survivability metrics captured and exported
- [x] Multi-seed aggregation produces mean, std, CI95
- [x] CSV export includes all experiment parameters
- [x] Fragmentation proxy computed correctly
- [x] Decision times tracked with < 1ms overhead

---

**Related Documents**:
- [21-recovery-timing.md](../phase3-protection/21-recovery-timing.md) (Recovery metrics)
- [31-dataset-logging.md](../phase4-rl-integration/31-dataset-logging.md) (Dataset stats)
- [52-performance.md](../phase6-quality/52-performance.md) (Performance targets)
