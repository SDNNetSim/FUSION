"""
Multi-seed result aggregation for survivability experiments.

This module provides utilities for aggregating simulation results across
multiple random seeds, computing statistics like mean, standard deviation,
and 95% confidence intervals.
"""

from typing import Any

import numpy as np

from fusion.utils.logging_config import get_logger

logger = get_logger(__name__)


def aggregate_seed_results(
    results: list[dict[str, Any]],
    metric_keys: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """
    Aggregate results across multiple seeds.

    Computes mean, std, and 95% confidence intervals for specified metrics.

    :param results: List of result dicts (one per seed)
    :type results: list[dict[str, Any]]
    :param metric_keys: Metrics to aggregate (if None, aggregates all numeric metrics)
    :type metric_keys: list[str] | None
    :return: Aggregated stats with mean, std, ci95
    :rtype: dict[str, dict[str, float]]

    Example:
        >>> results = [
        ...     {'bp_overall': 0.05, 'recovery_time_mean_ms': 52.3, 'seed': 42},
        ...     {'bp_overall': 0.06, 'recovery_time_mean_ms': 51.8, 'seed': 43}
        ... ]
        >>> metrics = ['bp_overall', 'recovery_time_mean_ms']
        >>> agg = aggregate_seed_results(results, metrics)
        >>> print(agg['bp_overall'])
        {'mean': 0.055, 'std': 0.007, 'ci95_lower': 0.041, 'n': 2}
    """
    if not results:
        logger.warning("No results to aggregate")
        return {}

    # Auto-detect numeric metrics if not specified
    if metric_keys is None:
        metric_keys = []
        for key, value in results[0].items():
            if isinstance(value, (int, float)) and key != "seed":
                metric_keys.append(key)

    aggregated = {}

    for key in metric_keys:
        values = [r[key] for r in results if key in r and isinstance(r[key], (int, float))]

        if not values:
            logger.warning("No values found for metric: %s", key)
            continue

        mean_val = np.mean(values)
        std_val = np.std(values, ddof=1) if len(values) > 1 else 0.0
        ci95 = 1.96 * std_val / np.sqrt(len(values)) if len(values) > 0 else 0.0

        aggregated[key] = {
            "mean": float(mean_val),
            "std": float(std_val),
            "ci95_lower": float(mean_val - ci95),
            "ci95_upper": float(mean_val + ci95),
            "n": len(values),
        }

    logger.info("Aggregated %d metrics across %d seeds", len(aggregated), len(results))

    return aggregated


def create_comparison_table(
    baseline_results: list[dict[str, Any]],
    rl_results: list[dict[str, Any]],
    metrics: list[str],
) -> dict[str, dict[str, Any]]:
    """
    Create comparison table for baseline vs RL.

    :param baseline_results: Baseline results (one per seed)
    :type baseline_results: list[dict[str, Any]]
    :param rl_results: RL policy results (one per seed)
    :type rl_results: list[dict[str, Any]]
    :param metrics: Metrics to compare
    :type metrics: list[str]
    :return: Comparison dictionary
    :rtype: dict[str, dict[str, Any]]

    Example:
        >>> baseline = [{'bp_overall': 0.10, 'seed': 42}]
        >>> rl = [{'bp_overall': 0.08, 'seed': 42}]
        >>> comp = create_comparison_table(baseline, rl, ['bp_overall'])
        >>> print(comp['bp_overall']['improvement_pct'])
        25.0
    """
    baseline_agg = aggregate_seed_results(baseline_results, metrics)
    rl_agg = aggregate_seed_results(rl_results, metrics)

    comparison = {}

    for metric in metrics:
        if metric not in baseline_agg or metric not in rl_agg:
            logger.warning("Metric %s not found in both baseline and RL results", metric)
            continue

        baseline_val = baseline_agg[metric]["mean"]
        rl_val = rl_agg[metric]["mean"]

        # Improvement calculation (positive = RL is better for BP-like metrics)
        if baseline_val != 0:
            improvement = ((baseline_val - rl_val) / baseline_val) * 100
        else:
            improvement = 0.0

        comparison[metric] = {
            "baseline_mean": baseline_val,
            "baseline_std": baseline_agg[metric]["std"],
            "baseline_ci95": (
                baseline_agg[metric]["ci95_lower"],
                baseline_agg[metric]["ci95_upper"],
            ),
            "rl_mean": rl_val,
            "rl_std": rl_agg[metric]["std"],
            "rl_ci95": (rl_agg[metric]["ci95_lower"], rl_agg[metric]["ci95_upper"]),
            "improvement_pct": improvement,
            "n_baseline": baseline_agg[metric]["n"],
            "n_rl": rl_agg[metric]["n"],
        }

    logger.info("Created comparison table for %d metrics", len(comparison))

    return comparison


def format_comparison_for_display(comparison: dict[str, dict[str, Any]]) -> str:
    """
    Format comparison table for console display.

    :param comparison: Comparison dict from create_comparison_table
    :type comparison: dict[str, dict[str, Any]]
    :return: Formatted table string
    :rtype: str

    Example:
        >>> comp = create_comparison_table(baseline, rl, ['bp_overall'])
        >>> print(format_comparison_for_display(comp))
        Metric                | Baseline           | RL                 | Improvement
        -------------------------------------------------------------------------------
        bp_overall            | 0.1050 ± 0.0071    | 0.0850 ± 0.0071    | +19.05%
    """
    if not comparison:
        return "No comparison data available"

    lines = []
    lines.append("Metric                | Baseline           | RL                 | Improvement")
    lines.append("-" * 79)

    for metric, stats in comparison.items():
        baseline_str = f"{stats['baseline_mean']:.4f} ± {stats['baseline_std']:.4f}"
        rl_str = f"{stats['rl_mean']:.4f} ± {stats['rl_std']:.4f}"
        improvement_str = f"{stats['improvement_pct']:+.2f}%"

        lines.append(f"{metric:<21} | {baseline_str:<18} | {rl_str:<18} | {improvement_str}")

    return "\n".join(lines)
