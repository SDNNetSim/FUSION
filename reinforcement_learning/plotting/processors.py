"""
Convert the raw JSON dictionaries (keyed by run-id) into the tidy
algorithm-centred format each plotter expects.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Dict, Any
import numpy as np


# ─────────────────────────  helper  ─────────────────────────
def _mean_last(values: list[float | int], k: int = 10) -> float:
    """Mean of the last *k* entries (or whole list if shorter)."""
    if not values:
        return 0.0
    subset = values[-k:] if len(values) >= k else values
    return float(np.mean(subset))


# ─────────────────────────  main processors  ─────────────────
def process_blocking(raw_runs: Dict[str, Any]) -> dict:
    """
    Output:
        {algo: {traffic_volume: avg_blocking_prob}}
    """
    merged: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))

    for run_id, data in raw_runs.items():
        algo = run_id.split("_")[0]  # assumes run-id naming: "algo_variant"
        for tv, block_vector in data.items():
            if isinstance(block_vector, list):
                merged[algo][str(tv)].append(_mean_last(block_vector))
            elif isinstance(block_vector, (float, int)):
                merged[algo][str(tv)].append(float(block_vector))

    # final averaging over all seeds / runs
    final = {
        algo: {tv: float(np.mean(vals)) for tv, vals in tv_dict.items()}
        for algo, tv_dict in merged.items()
    }
    return final


def process_memory_usage(raw_runs: Dict[str, Any]) -> dict:
    """
    Output:
        {algo: {traffic_volume: avg_memory_mb}}
    """
    merged: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))

    for run_id, data in raw_runs.items():
        algo = run_id.split("_")[0]
        for tv, mem_list in data.items():
            if isinstance(mem_list, list):
                merged[algo][str(tv)].extend(mem_list)
            elif isinstance(mem_list, (float, int)):
                merged[algo][str(tv)].append(float(mem_list))

    return {
        algo: {tv: float(np.mean(vals)) for tv, vals in tv_dict.items()}
        for algo, tv_dict in merged.items()
    }


def process_sim_times(raw_runs: Dict[str, Any]) -> dict:
    """
    Expects each JSON to map traffic-volume → elapsed-seconds (float).
    Output:
        {algo: {traffic_volume: mean_seconds}}
    """
    merged: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))

    for run_id, data in raw_runs.items():
        algo = run_id.split("_")[0]
        for tv, secs in data.items():
            merged[algo][str(tv)].append(float(secs))

    return {
        algo: {tv: float(np.mean(vals)) for tv, vals in tv_dict.items()}
        for algo, tv_dict in merged.items()
    }


def process_rewards(raw_runs: Dict[str, Any]) -> dict:
    """
    A *light* aggregator, assuming each JSON already holds
    {traffic_label: {trial: {episode: [rewards]}}}.
    If multiple runs of same algo are supplied, merge trials together.
    """
    merged: Dict[str, Any] = defaultdict(lambda: defaultdict(dict))

    for run_id, data in raw_runs.items():
        algo = run_id.split("_")[0]
        for tv, trial_dict in data.items():
            dst_trials = merged[algo][tv]
            next_trial_idx = max(dst_trials.keys(), default=-1) + 1
            for _, episode_dict in trial_dict.items():
                dst_trials[next_trial_idx] = episode_dict
                next_trial_idx += 1

    return merged
