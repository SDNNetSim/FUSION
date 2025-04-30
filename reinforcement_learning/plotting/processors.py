# ✅ processors.py (updated to include full algo labels like k_shortest_path_2 vs _inf)
from collections import defaultdict
import numpy as np
from typing import Dict, Any


def _mean_last(values: list[float | int], k: int = 10) -> float:
    if not values:
        return 0.0
    subset = values[-k:] if len(values) >= k else values
    return float(np.mean(subset))


def process_blocking(raw_runs: Dict[str, Any], runid_to_algo: dict[str, str]) -> dict:
    merged = defaultdict(lambda: defaultdict(list))
    for run_id, data in raw_runs.items():
        algo = runid_to_algo.get(run_id, "unknown")
        for tv, info_vector in data.items():
            if isinstance(info_vector, dict):
                if info_vector.get('blocking_mean') is None and 'iter_stats' in info_vector:
                    last_key = next(reversed(info_vector['iter_stats']))
                    last_entry = info_vector['iter_stats'][last_key]
                    merged[algo][str(tv)].append(_mean_last(last_entry['sim_block_list']))
                else:
                    merged[algo][str(tv)].append(info_vector.get('blocking_mean', 0.0))
            elif isinstance(info_vector, (float, int)):
                merged[algo][str(tv)].append(float(info_vector))
    return {
        algo: {tv: float(np.mean(vals)) for tv, vals in tv_dict.items()}
        for algo, tv_dict in merged.items()
    }


def process_memory_usage(raw_runs: Dict[str, Any], runid_to_algo: dict[str, str]) -> dict:
    merged = defaultdict(lambda: defaultdict(list))
    for run_id, data in raw_runs.items():
        algo = runid_to_algo.get(run_id, "unknown")
        for tv, mem_list in data.items():
            if isinstance(mem_list, list):
                merged[algo][str(tv)].extend(mem_list)
            elif isinstance(mem_list, (float, int)):
                merged[algo][str(tv)].append(float(mem_list))
    return {
        algo: {tv: float(np.mean(vals)) for tv, vals in tv_dict.items()}
        for algo, tv_dict in merged.items()
    }


def process_sim_times(raw_runs: Dict[str, Any], runid_to_algo: dict[str, str]) -> dict:
    merged = defaultdict(lambda: defaultdict(list))
    for run_id, data in raw_runs.items():
        algo = runid_to_algo.get(run_id, "unknown")
        for tv, secs in data.items():
            merged[algo][str(tv)].append(float(secs))
    return {
        algo: {tv: float(np.mean(vals)) for tv, vals in tv_dict.items()}
        for algo, tv_dict in merged.items()
    }


def process_rewards(raw_runs: Dict[str, Any], runid_to_algo: dict[str, str]) -> dict:
    merged = defaultdict(lambda: defaultdict(dict))
    for run_id, data in raw_runs.items():
        algo = runid_to_algo.get(run_id, "unknown")
        for tv, trial_dict in data.items():
            dst_trials = merged[algo][tv]
            next_trial_idx = max(dst_trials.keys(), default=-1) + 1
            for trial_id, episode_dict in trial_dict.items():
                dst_trials[next_trial_idx] = episode_dict
                next_trial_idx += 1
    return merged
