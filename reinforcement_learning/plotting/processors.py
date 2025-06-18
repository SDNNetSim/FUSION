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


def process_blocked_bandwidth(raw_runs, runid_to_algo):
    """
    Return {algo: {tv: {bw_gbps: mean_blocked_count}}}
    from *block_bw_dict* in the last iteration.
    """
    merged = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for run_id, data in raw_runs.items():
        algo = runid_to_algo.get(run_id, "unknown")
        for tv_raw, info in data.items():  # new
            tv = float(tv_raw)  # └─ store as float, not string
            if not isinstance(info, dict) or "iter_stats" not in info:
                continue
            last_iter = info["iter_stats"][next(reversed(info["iter_stats"]))]
            for bw, blocked in last_iter.get("block_bw_dict", {}).items():
                bw = float(bw)
                merged[algo][tv][bw].append(float(blocked))  # in bw‑block
    return {
        a: {tv: {bw: float(np.mean(vals))
                 for bw, vals in bw_dict.items()}
            for tv, bw_dict in tvs.items()}
        for a, tvs in merged.items()
    }


def process_rewards(raw_runs: Dict[str, Any],
                    runid_to_algo: dict[str, str]) -> dict:
    """
    Aggregate rewards across seeds and keep the **true episode indices**
    (0, 10, 20, …) that are encoded in the log‑file names.

    For every (algo, traffic‑volume) pair return:
      • 'mean'        : episode‑by‑episode mean across seeds
      • 'std'         : episode‑by‑episode std  across seeds  (raw, NOT /√n)
      • 'n'           : number of seeds
      • 'overall_ci'  : 95 % CI of the run‑mean reward across seeds
      • 'episodes'    : list of episode labels that go straight on the x‑axis
    """
    # by_algo_tv[algo][tv] ──► list[dict{ep_idx : mean_reward}]  (one dict per seed)
    by_algo_tv: dict[str, dict[str, list[dict[int, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    seen_seed: dict[str, set[tuple[str, int]]] = defaultdict(set)

    for run_id, data in raw_runs.items():
        algo = runid_to_algo.get(run_id, "unknown")

        for tv, trials in data.items():
            trial_dict = trials.get("rewards", {})
            if not trial_dict:
                continue

            for trial_idx, ep_dict in trial_dict.items():
                key = (tv, trial_idx)
                if key in seen_seed[algo]:
                    continue  # duplicate -> skip
                seen_seed[algo].add(key)

                ep_ids = sorted(ep_dict, key=float)
                ep_means = {int(ep): float(np.mean(ep_dict[ep])) for ep in ep_ids}
                by_algo_tv[algo][str(tv)].append(ep_means)

                if len(by_algo_tv[algo][str(tv)]) <= 2:
                    sample_vals = list(ep_means.values())[:5]
                    print(f"[SEED DBG] algo={algo:<15} tv={tv:<6} "
                          f"trial={trial_idx} sample={sample_vals}")

    processed: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)

    for algo, tv_dict in by_algo_tv.items():
        for tv, seed_dicts in tv_dict.items():
            n_seeds = len(seed_dicts)

            all_eps = sorted({ep for d in seed_dicts for ep in d})
            ep_to_pos = {ep: i for i, ep in enumerate(all_eps)}

            arr = np.full((n_seeds, len(all_eps)), np.nan, dtype=float)
            for row, ep_means in enumerate(seed_dicts):
                for ep, val in ep_means.items():
                    arr[row, ep_to_pos[ep]] = val

            mean_curve = np.nanmean(arr, axis=0)
            std_curve = np.nanstd(arr, axis=0, ddof=1)

            run_means = np.nanmean(arr, axis=1)  # one number per seed
            run_std = float(np.std(run_means, ddof=1)) if n_seeds > 1 else 0.0
            overall_ci = 1.96 * run_std / np.sqrt(n_seeds) if n_seeds > 1 else 0.0

            print(
                f"[DBG PROC] algo={algo:<15} tv={tv:<6} n_seeds={n_seeds} "
                f"run_means={run_means.tolist()} run_std={run_std:.4f} "
                f"overall_ci={overall_ci:.4f}"
            )

            processed[algo][tv] = {
                "mean": mean_curve.tolist(),
                "std": std_curve.tolist(),
                "n": n_seeds,
                "overall_ci": overall_ci,
                # Use the *real* episode numbers as x‑axis labels
                "episodes": [str(ep) for ep in all_eps],
            }

    return dict(processed)


def process_state_values(raw_runs: Dict[str, Any],
                         runid_to_algo: dict[str, str]) -> dict:
    """
    Average state-value tensors across *all* seeds and trials.

    Output
    ------
    {
      algo: {
        '50.0': { (src,dst): [avg_path_vals...] },
        ...
      }
    }
    """
    merged = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for run_id, run_dict in raw_runs.items():
        algo = runid_to_algo.get(run_id, "unknown")
        for erl, payload in run_dict.items():
            trials = payload.get("state_vals", {})
            for trial_dict in trials.values():  # each trial's (src,dst)->[vals]
                for pair, vec in trial_dict.items():
                    merged[algo][erl][pair].append(vec)

    averaged = defaultdict(dict)  # final structure

    for algo, erl_dict in merged.items():
        for erl, pair_dict in erl_dict.items():
            averaged_pairs = {}
            for pair, vecs in pair_dict.items():
                max_len = max(map(len, vecs))
                padded = [
                    vec + [np.nan] * (max_len - len(vec)) for vec in vecs
                ]
                mean_vec = np.nanmean(np.array(padded, dtype=float), axis=0)
                averaged_pairs[pair] = mean_vec.tolist()
            averaged[algo][erl] = averaged_pairs

    return dict(averaged)


def process_modulation_usage(raw_runs, runid_to_algo):
    """
    Extract *mods_used_dict* from the final iteration of every seed and
    average counts across seeds.

    Returns
    -------
    {algo: {tv (str): {bw (str): {mod: mean_count}}}}
    """

    merged = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(list)
            )
        )
    )

    for run_id, data in raw_runs.items():
        algo = runid_to_algo.get(run_id, "unknown")

        for tv, info in data.items():  # tv remains *string*
            if not isinstance(info, dict) or "iter_stats" not in info:
                continue

            last_iter = info["iter_stats"][next(reversed(info["iter_stats"]))]

            for bw, mod_dict in last_iter.get("mods_used_dict", {}).items():
                # skip entries that contain non‑integer counts
                if not all(isinstance(cnt, int) for cnt in mod_dict.values()):
                    continue

                for mod, cnt in mod_dict.items():
                    merged[algo][tv][bw][mod].append(float(cnt))

    # average across seeds
    processed = {}
    for algo, tv_dict in merged.items():
        processed[algo] = {}
        for tv, bw_dict in tv_dict.items():
            processed[algo][tv] = {}
            for bw, mod_dict in bw_dict.items():
                processed[algo][tv][bw] = {
                    mod: float(np.mean(vals)) for mod, vals in mod_dict.items()
                }

    return processed
