from pathlib import Path
import json
import re
import ast
from collections import defaultdict
from typing import Dict, Any, Iterable, Tuple, Optional

import numpy as np

ROOT_OUTPUT = Path("../../data/output")
ROOT_INPUT = Path("../../data/input")
ROOT_LOGS = Path("../../logs")


def _compose_algo_label(path_algo: str, obs: Optional[str]) -> str:
    """Compose a unique algorithm label from base name and observation space."""
    return f"{path_algo}_{obs}" if obs else path_algo


def _safe_load_json(fp: Path) -> Optional[Any]:
    """Read *fp* safely, returning None on error."""
    try:
        with fp.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[loaders] ❌ Could not load {fp}: {exc}")
        return None


def discover_all_run_ids(network: str, dates: list[str], drl: bool, obs_filter: Optional[list[str]]) -> Dict[
    str, list[str]]:
    """Discovers all run IDs for a given network, date list, and DRL flag."""
    algo_runs: Dict[str, list[str]] = defaultdict(list)

    for date in dates:
        run_root = ROOT_OUTPUT / network / date

        for run_dir in run_root.iterdir():
            if not run_dir.is_dir():
                continue

            s1_meta = run_dir / "s1" / "meta.json"
            is_drl = s1_meta.exists()

            if drl != is_drl:
                continue

            if is_drl:
                meta = _safe_load_json(s1_meta) or {}
                algo_base = meta.get("path_algorithm", "unknown")
                obs = meta.get('obs_space')

                if (obs_filter and obs not in obs_filter) and (algo_base in ('dqn', 'ppo')):
                    continue

                algo = _compose_algo_label(algo_base, obs)
                run_id = meta.get("run_id")
                timestamp = run_dir.name

                for seed_dir in run_dir.glob("s*"):
                    unique = f"{run_id}@{timestamp}_{seed_dir.name}"
                    algo_runs[algo].append(unique)
            else:
                timestamp = run_dir.name
                for seed_dir in run_dir.glob("s*"):
                    s_num = seed_dir.name
                    run_id = f"{s_num}_{date}"
                    inp_fp = ROOT_INPUT / network / date / timestamp / f"sim_input_{s_num}.json"

                    algo = "unknown"
                    if inp_fp.exists():
                        inp = _safe_load_json(inp_fp) or {}
                        method = inp.get("route_method")
                        k = inp.get("k_paths", 0)
                        algo = f"{method}_{'inf' if k > 4 else k}" if method == "k_shortest_path" else method

                    unique = f"{run_id}@{timestamp}"
                    algo_runs[algo].append(unique)

    for alg in tuple(algo_runs):
        if alg == "unknown":
            raise ValueError("Algorithm not found.")
        algo_runs[alg] = list(dict.fromkeys(algo_runs[alg]))

    print(f"[DEBUG] Discovered {'DRL' if drl else 'non-DRL'} runs: {dict(algo_runs)}")
    return dict(algo_runs)


def load_metric_for_runs(
        run_ids: Iterable[str],
        metric: str,
        drl: bool,
        network: str,
        dates: list[str],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str], Dict[str, str]]:
    """Load metric data for each run."""
    raw_runs: Dict[str, Dict[str, Any]] = {}
    runid_to_algo: Dict[str, str] = {}
    start_stamps: Dict[str, str] = {}

    for date in dates:
        for s_dir in (ROOT_OUTPUT / network / date).rglob("s*"):
            if not s_dir.is_dir():
                continue

            if drl:
                result = _handle_drl_run(s_dir, run_ids, metric, network, date)
            else:
                result = _handle_non_drl_run(s_dir, run_ids, metric, network, date)

            if result:
                unique_run_id, algo, metric_vals, stamp = result
                raw_runs[unique_run_id] = metric_vals
                runid_to_algo[unique_run_id] = algo
                if stamp:
                    start_stamps[unique_run_id] = stamp

    return raw_runs, runid_to_algo, start_stamps


def _handle_drl_run(s_dir: Path, run_ids: Iterable[str], metric: str, network: str, date: str):  # pylint: disable=unused-argument
    base_run_dir = s_dir.parent
    seed = s_dir.name

    if metric == "sim_times":
        max_seed = max(int(p.name.lstrip("s")) for p in s_dir.parent.glob("s*"))
        if int(seed.lstrip("s")) != max_seed:
            return None

    meta_fp = next(base_run_dir.glob("s*/meta.json"), None)
    meta_run_id: Optional[str] = None
    algo = "unknown"

    if meta_fp and meta_fp.exists():
        meta = _safe_load_json(meta_fp) or {}
        algo_base = meta.get("path_algorithm", "unknown")
        obs = meta.get("obs_space")
        algo = _compose_algo_label(algo_base, obs)
        meta_run_id = meta.get("run_id")

    timestamp = base_run_dir.name
    meta_run_id = meta_run_id or timestamp
    composite_run_id = f"{meta_run_id}@{timestamp}"
    unique_run_id = f"{composite_run_id}_{seed}"

    if run_ids and not any(rid in run_ids for rid in (unique_run_id, composite_run_id, meta_run_id)):
        return None

    metric_vals = _load_metric_jsons(s_dir, metric)
    _augment_with_logs(metric, drl=True, metric_vals=metric_vals, algo=algo, network=network, base_run_dir=base_run_dir)

    stamp = f"{s_dir.parent.parent.name}_{s_dir.parent.name}" if metric == "sim_times" else None
    return unique_run_id, algo, metric_vals, stamp


def _handle_non_drl_run(s_dir: Path, run_ids: Iterable[str], metric: str, network: str, date: str):  # pylint: disable=unused-argument
    parent_dir = s_dir.parent
    seed = s_dir.name
    timestamp = parent_dir.name
    date_str = parent_dir.parent.name
    unique_run_id = f"{seed}_{date_str}@{timestamp}"

    if run_ids and unique_run_id not in run_ids:
        return None

    inp_fp = ROOT_INPUT / network / date_str / parent_dir.name / f"sim_input_{seed}.json"
    algo = "unknown"
    if inp_fp.exists():
        inp = _safe_load_json(inp_fp) or {}
        method = inp.get("route_method")
        k = inp.get("k_paths", 0)
        algo = f"{method}_{'inf' if k > 4 else k}" if method == "k_shortest_path" else method

    metric_vals = _load_metric_jsons(s_dir, metric)
    return unique_run_id, algo, metric_vals, None


def _load_metric_jsons(s_dir: Path, _: str) -> Dict[str, Any]:
    metric_vals: Dict[str, Any] = {}
    for fp in s_dir.glob("*.json"):
        if fp.name == "meta.json":
            continue
        match = re.match(r"(\d+\.?\d*)_erlang\.json", fp.name)
        if match:
            metric_vals[match.group(1)] = _safe_load_json(fp)
    return metric_vals


def _augment_with_logs(metric: str, drl: bool, metric_vals: Dict[str, Any], algo: str,
                       network: str, base_run_dir: Path):
    if metric == "memory" and drl:
        logs_fp = ROOT_LOGS / algo / network / base_run_dir.parent.name / base_run_dir.name / "memory_usage.npy"
        if logs_fp.exists():
            try:
                arr = np.load(logs_fp)
                metric_vals["overall"] = float(arr.max() / (1024 ** 2))
            except (OSError, ValueError) as exc:
                print(f"[loaders] ❌ could not load {logs_fp}: {exc}")

    elif metric == "state_values" and drl:
        logs_dir = ROOT_LOGS / algo / network / base_run_dir.parent.name / base_run_dir.name
        if not logs_dir.is_dir():
            return

        sv_rx = re.compile(r"state_vals_e(?P<erl>\d+\.?\d*)_.*?_t(?P<trial>\d+)\.json")
        state_vals: Dict[str, Dict[int, dict]] = defaultdict(dict)

        for fp in logs_dir.glob("state_vals_*.json"):
            m = sv_rx.match(fp.name)
            if not m:
                continue
            erlang = m.group("erl")
            trial = int(m.group("trial"))
            try:
                with fp.open("r", encoding="utf-8") as f:
                    sv_data = json.load(f)
                    sv_data = {ast.literal_eval(k): v for k, v in sv_data.items()}
                    state_vals[erlang][trial] = sv_data
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                print(f"[loaders] ❌ could not load {fp}: {exc}")

        for erl, trials in state_vals.items():
            if erl not in metric_vals:
                metric_vals[erl] = {}
            metric_vals[erl]["state_vals"] = trials
