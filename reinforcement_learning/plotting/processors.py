# ✅ processors.py (updated to include full algo labels like k_shortest_path_2 vs _inf)
from collections import defaultdict
from typing import Dict, Any
from datetime import datetime, timedelta
import numpy as np


def _mean_last(values: list[float | int], k: int = 20) -> float:
    if not values:
        return 0.0
    subset = values[-k:] if len(values) >= k else values
    return float(np.mean(subset))


def process_blocking(raw_runs: Dict[str, Any], runid_to_algo: dict[str, str]) -> dict:
    """
    Processes blocking runs.
    """
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

    processed = {}

    for algo, tv_dict in merged.items():
        processed[algo] = {}
        for tv, vals in tv_dict.items():
            mean_val = float(np.mean(vals))
            print(f"[SEED-DBG] {algo} Erlang={tv}  seeds={len(vals)}  vals={vals}  mean={mean_val:.4f}")
            processed[algo][tv] = mean_val

    return processed


def _add(collector: dict, algo: str, tv: str, val: Any) -> None:
    """Append one or many numeric values to collector[algo][tv]."""
    if isinstance(val, (list, tuple, np.ndarray)):
        collector[algo][tv].extend(map(float, val))
    else:
        collector[algo][tv].append(float(val))


def process_memory_usage(
        raw_runs: Dict[str, Any],
        runid_to_algo: dict[str, str]
) -> dict:
    """
    Aggregate memory usage (MB).

    * DRL runs → { 'overall': float }   from memory_usage.npy
    * Legacy runs → float / list / ndarray keyed by traffic volume

    Returns
    -------
    {algo: {traffic_volume_or_overall: mean_MB}}
    """
    merged = defaultdict(lambda: defaultdict(list))

    for run_id, data in raw_runs.items():
        algo = runid_to_algo.get(run_id, "unknown")
        traffic_volume = next(iter(data))
        merged[algo][traffic_volume] = {'overall': data.get('overall', -1.0)}

    return merged


def _stamp_to_dt(stamp: str) -> datetime:
    """
    Convert '0429_21_14_39_491949' to a datetime object.
    Assumes:
        - MMDD_HH_MM_SS_MSUS
        - MSUS is 3 digits milliseconds + 3 digits microseconds
    """
    mmdd, hh, mm, ss, msus = stamp.split("_")
    ms = int(msus[:3])  # first 3 digits = milliseconds
    us = int(msus[3:])  # last 3 digits = microseconds
    return datetime(
        year=datetime.now().year,
        month=int(mmdd[:2]),
        day=int(mmdd[2:]),
        hour=int(hh),
        minute=int(mm),
        second=int(ss),
        microsecond=ms * 1000 + us
    )


def process_sim_times(
        raw_runs: Dict[str, Any],
        runid_to_algo: dict[str, str],
        context: dict | None = None,
) -> dict:
    """
    If *context* supplies 'start_stamps', compute **wall-clock duration**
    (end-stamp − start-stamp). Otherwise fall back to treating the JSON
    values as already-computed seconds (old behaviour).
    """
    start_stamps = context.get("start_stamps") if context else None
    merged = defaultdict(lambda: defaultdict(list))

    for run_id, data in raw_runs.items():
        algo = runid_to_algo.get(run_id, "unknown")

        if start_stamps and run_id in start_stamps:
            t0 = _stamp_to_dt(start_stamps[run_id])

            for tv, info in data.items():
                if not isinstance(info, dict):
                    continue
                end_raw = info.get("sim_end_time")
                if not end_raw:
                    continue

                t1 = _stamp_to_dt(end_raw)
                if t1 < t0:  # crossed midnight
                    t1 += timedelta(days=1)

                merged[algo][str(tv)].append((t1 - t0).total_seconds())
        else:
            for tv, secs in data.items():
                merged[algo][str(tv)].append(float(secs))

    return {
        algo: {tv: float(np.mean(vals)) for tv, vals in tv_dict.items()}
        for algo, tv_dict in merged.items()
    }


def process_rewards(raw_runs: Dict[str, Any], runid_to_algo: dict[str, str]) -> dict:
    """
    Process rewards.
    """
    merged = defaultdict(lambda: defaultdict(dict))
    for run_id, data in raw_runs.items():
        algo = runid_to_algo.get(run_id, "unknown")
        for tv, trial_dict in data.items():
            dst_trials = merged[algo][tv]
            next_trial_idx = max(dst_trials.keys(), default=-1) + 1
            for _, episode_dict in trial_dict.items():
                dst_trials[next_trial_idx] = episode_dict
                next_trial_idx += 1
    return merged
