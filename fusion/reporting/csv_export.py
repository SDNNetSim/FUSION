"""
CSV export utilities for survivability experiment results.

This module provides functions for exporting simulation results to CSV
format for easy analysis in spreadsheet tools and data processing.
"""

import csv
from pathlib import Path
from typing import Any

from fusion.utils.logging_config import get_logger

logger = get_logger(__name__)


def export_results_to_csv(results: list[dict[str, Any]], output_path: str) -> None:
    """
    Export results to CSV file.

    :param results: List of result dicts (each representing one simulation run)
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

    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    # Get all unique keys from all results
    all_keys: set[str] = set()
    for result in results:
        all_keys.update(result.keys())

    # Sort keys for consistent column ordering
    fieldnames = sorted(all_keys)

    # Move important columns to front
    priority_cols = ["seed", "topology", "load", "policy", "failure_type", "bp_overall"]
    for col in reversed(priority_cols):
        if col in fieldnames:
            fieldnames.remove(col)
            fieldnames.insert(0, col)

    with open(output_path_obj, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    logger.info("Exported %d results to %s", len(results), output_path)


def export_aggregated_results(
    aggregated: dict[str, dict[str, float]],
    output_path: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    """
    Export aggregated results (mean, std, CI95) to CSV.

    :param aggregated: Aggregated results from aggregate_seed_results
    :type aggregated: dict[str, dict[str, float]]
    :param output_path: Output CSV path
    :type output_path: str
    :param metadata: Optional metadata to include (topology, policy, etc.)
    :type metadata: dict[str, Any] | None

    Example:
        >>> agg = aggregate_seed_results(results, ['bp_overall'])
        >>> export_aggregated_results(agg, 'results/summary.csv', {'policy': 'bc'})
    """
    if not aggregated:
        logger.warning("No aggregated results to export")
        return

    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    # Build rows for CSV
    rows = []
    for metric, stats in aggregated.items():
        row: dict[str, Any] = {"metric": metric}
        if metadata:
            row.update(metadata)
        row.update(
            {
                "mean": stats["mean"],
                "std": stats["std"],
                "ci95_lower": stats["ci95_lower"],
                "ci95_upper": stats["ci95_upper"],
                "n": stats["n"],
            }
        )
        rows.append(row)

    if not rows:
        logger.warning("No rows to write")
        return

    with open(output_path_obj, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Exported %d aggregated metrics to %s", len(rows), output_path)


def export_comparison_table(
    comparison: dict[str, dict[str, Any]], output_path: str
) -> None:
    """
    Export comparison table (baseline vs RL) to CSV.

    :param comparison: Comparison dict from create_comparison_table
    :type comparison: dict[str, dict[str, Any]]
    :param output_path: Output CSV path
    :type output_path: str

    Example:
        >>> comp = create_comparison_table(baseline, rl, ['bp_overall'])
        >>> export_comparison_table(comp, 'results/comparison.csv')
    """
    if not comparison:
        logger.warning("No comparison data to export")
        return

    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for metric, stats in comparison.items():
        row = {
            "metric": metric,
            "baseline_mean": stats["baseline_mean"],
            "baseline_std": stats["baseline_std"],
            "baseline_ci95_lower": stats["baseline_ci95"][0],
            "baseline_ci95_upper": stats["baseline_ci95"][1],
            "rl_mean": stats["rl_mean"],
            "rl_std": stats["rl_std"],
            "rl_ci95_lower": stats["rl_ci95"][0],
            "rl_ci95_upper": stats["rl_ci95"][1],
            "improvement_pct": stats["improvement_pct"],
            "n_baseline": stats["n_baseline"],
            "n_rl": stats["n_rl"],
        }
        rows.append(row)

    with open(output_path_obj, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Exported comparison for %d metrics to %s", len(rows), output_path)


def append_result_to_csv(result: dict[str, Any], output_path: str) -> None:
    """
    Append a single result to CSV file.

    Useful for incremental logging during multi-seed experiments.

    :param result: Single result dict
    :type result: dict[str, Any]
    :param output_path: Output CSV path
    :type output_path: str

    Example:
        >>> for seed in range(10):
        ...     stats = run_simulation(seed)
        ...     append_result_to_csv(stats.to_csv_row(), 'results/live.csv')
    """
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    # Check if file exists to determine if we need to write header
    file_exists = output_path_obj.exists()

    with open(output_path_obj, "a", newline="", encoding="utf-8") as f:
        fieldnames = list(result.keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(result)

    logger.debug("Appended result to %s", output_path)
