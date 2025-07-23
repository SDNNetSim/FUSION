from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, Any, Optional

import numpy as np
from scipy.stats import ttest_ind


def _mean_last(values: list[float | int], k: int = 5) -> float:
    if not values:
        return 0.0
    subset = values[-k:] if len(values) >= k else values
    return float(np.mean(subset))


def process_blocking(raw_runs: Dict[str, Any], runid_to_algo: dict[str, str]) -> dict:
    """
    Process blocking probability results into mean, std, CI and optionally effect sizes.
    """
    merged = defaultdict(lambda: defaultdict(list))
    baselines = ["k_shortest_path_1", "k_shortest_path_4", "cong_aware", "k_shortest_path_inf"]

    for run_id, data in raw_runs.items():
        algo = runid_to_algo.get(run_id, "unknown")
        for tv, info_vector in data.items():
            if isinstance(info_vector, dict):
                last_key = next(reversed(info_vector['iter_stats']))
                last_entry = info_vector['iter_stats'][last_key]
                if algo in baselines:
                    blocking_list = last_entry['sim_block_list']
                    merged[algo][str(tv)] = blocking_list
                elif info_vector.get('blocking_mean') is None and 'iter_stats' in info_vector:
                    merged[algo][str(tv)].append(_mean_last(last_entry['sim_block_list']))
                else:
                    raise NotImplementedError
            elif isinstance(info_vector, (float, int)):
                merged[algo][str(tv)].append(float(info_vector))

    processed = {}
    baseline_vals = defaultdict(dict)

    for base in baselines:
        for tv, vals in merged[base].items():
            baseline_vals[base][tv] = np.array(vals, dtype=float)

    for algo, tv_dict in merged.items():
        processed[algo] = {}
        for tv, vals in tv_dict.items():
            vals = np.array(vals, dtype=float)
            mean_val = np.mean(vals)
            std_val = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
            ci_val = 1.96 * (std_val / np.sqrt(len(vals))) if len(vals) > 1 else 0.0

            stats_block = {
                "mean": float(mean_val),
                "std": float(std_val),
                "ci": float(ci_val),
            }

            if algo not in baselines:
                for base in baselines:
                    base_vals = np.array(baseline_vals.get(base, {}).get(tv, []), dtype=float)
                    if len(base_vals) > 1 and len(vals) > 1:
                        _, p_val = ttest_ind(vals, base_vals, equal_var=False)
                        pooled_std = np.sqrt((np.var(vals, ddof=1) + np.var(base_vals, ddof=1)) / 2)
                        d = (np.mean(vals) - np.mean(base_vals)) / pooled_std if pooled_std else 0.0
                        mean_diff = np.mean(vals) - np.mean(base_vals)
                        stats_block[f"vs_{base}"] = {
                            "p": float(p_val),
                            "d": float(d),
                            "mean_diff": float(mean_diff),
                            "significant": p_val < 0.05
                        }

            processed[algo][tv] = stats_block

            print(f"[SEED-DBG] {algo} Erlang={tv} seeds={len(vals)} "
                  f"mean={mean_val:.4g} ±std={std_val:.4g} ±CI={ci_val:.4g}")

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
    """
    mmdd, hh, mm, ss, msus = stamp.split("_")
    ms = int(msus[:3])
    us = int(msus[3:])
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
        context: Optional[dict] = None,
) -> dict:
    """
    Compute wall-clock durations or fallback to reported simulation times.
    """
    start_stamps = context.get("start_stamps") if context else None
    merged = defaultdict(lambda: defaultdict(list))

    for run_id, data in raw_runs.items():
        algo = runid_to_algo.get(run_id, "unknown")

        if isinstance(start_stamps, dict) and run_id in start_stamps:  # pylint: disable=unsupported-membership-test
            t0 = _stamp_to_dt(start_stamps[run_id])

            for tv, info in data.items():
                if not isinstance(info, dict):
                    continue
                end_raw = info.get("sim_end_time")
                if not end_raw:
                    continue

                t1 = _stamp_to_dt(end_raw)
                if t1 < t0:
                    t1 += timedelta(days=1)

                merged[algo][str(tv)].append((t1 - t0).total_seconds())
        else:
            for tv, secs in data.items():
                merged[algo][str(tv)].append(float(secs))

    return {
        algo: {tv: float(np.mean(vals)) for tv, vals in tv_dict.items()}
        for algo, tv_dict in merged.items()
    }
