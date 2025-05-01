from pathlib import Path
import json
import re
from collections import defaultdict
from typing import Dict, Any, Iterable, Tuple

ROOT_OUTPUT = Path("../../data/output")
ROOT_INPUT = Path("../../data/input")
ROOT_LOGS = Path("../../logs")


def _safe_load_json(fp: Path) -> Any | None:
    """Read *fp* safely, returning ``None`` on error."""
    try:
        with fp.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        print(f"[loaders] ❌ Could not load {fp}: {exc}")
        return None


def discover_all_run_ids(network: str, dates: list[str], drl: bool) -> Dict[str, list[str]]:
    """Return ``{algorithm: [run_id, …]}`` for every run under *dates*."""
    algo_runs: Dict[str, list[str]] = defaultdict(list)

    for date in dates:
        for s_dir in (ROOT_OUTPUT / network / date).rglob("s*"):
            if not s_dir.is_dir():
                continue

            meta_fp = s_dir / "meta.json"

            if drl and meta_fp.exists():
                meta = _safe_load_json(meta_fp)
                if not meta or "run_id" not in meta:
                    continue
                algo = meta.get("path_algorithm", "unknown")
                algo_runs[algo].append(meta["run_id"])

            elif not drl and not meta_fp.exists():
                parent_dir = s_dir.parent  # …/RUN_ID/
                date_str = parent_dir.parent.name
                s_num = s_dir.name  # s1 …

                run_id = f"{s_num}_{date_str}"
                inp_fp = (ROOT_INPUT / network / date_str /
                          parent_dir.name / f"sim_input_{s_num}.json")
                algo = "unknown"
                if inp_fp.exists():
                    inp = _safe_load_json(inp_fp) or {}
                    method = inp.get("route_method")
                    k = inp.get("k_paths", 0)
                    if method == "k_shortest_path":
                        algo = f"{method}_{'inf' if k > 4 else k}"
                    else:
                        algo = method
                algo_runs[algo].append(run_id)

    # deduplicate while preserving order
    for alg in tuple(algo_runs):
        algo_runs[alg] = list(dict.fromkeys(algo_runs[alg]))

    print(f"[DEBUG] Discovered {'DRL' if drl else 'non‑DRL'} runs: {dict(algo_runs)}")
    return dict(algo_runs)


def load_metric_for_runs(
        run_ids: Iterable[str],
        metric: str,
        drl: bool,
        network: str,
        dates: list[str],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str], Dict[str, str]]:
    """Load *metric* JSONs for *run_ids*.

    Returns
    -------
    raw_runs : dict
        ``{unique_run_id_per_seed: {erlang: value, …}}``
    runid_to_algo : dict
        Map each *unique_run_id_per_seed* to the algorithm label.
    """
    raw_runs: Dict[str, Dict[str, Any]] = {}
    runid_to_algo: Dict[str, str] = {}
    start_stamps: Dict[str, str] = {}

    # TODO: Why doesn't this work top level?
    import re
    pattern = re.compile(r"(\d+\.?\d*)_erlang\.json")

    for date in dates:
        for s_dir in (ROOT_OUTPUT / network / date).rglob("s*"):
            if not s_dir.is_dir():
                continue

            if drl:
                base_run_dir = s_dir.parent
                seed = s_dir.name
                if metric == "sim_times":
                    max_seed = max(int(p.name.lstrip("s")) for p in s_dir.parent.glob("s*"))
                    if int(seed.lstrip("s")) != max_seed:
                        continue

                meta_fp = next(base_run_dir.glob("s*/meta.json"), None)
                meta_run_id: str | None = None
                algo = "unknown"

                if meta_fp and meta_fp.exists():
                    meta = _safe_load_json(meta_fp) or {}
                    meta_run_id = meta.get("run_id")
                    algo = meta.get("path_algorithm", algo)

                if meta_run_id is None:
                    meta_run_id = base_run_dir.name

                if meta_run_id not in run_ids:
                    if base_run_dir.name in run_ids:
                        meta_run_id = base_run_dir.name
                    else:
                        continue  # not requested → skip

                unique_run_id = f"{meta_run_id}_{seed}"

            else:
                parent_dir = s_dir.parent
                date_str = parent_dir.parent.name
                seed = s_dir.name  # s1 … (scenario)
                unique_run_id = f"{seed}_{date_str}"
                if unique_run_id not in run_ids:
                    continue

                inp_fp = (ROOT_INPUT / network / date_str / parent_dir.name /
                          f"sim_input_{seed}.json")
                algo = "unknown"
                if inp_fp.exists():
                    inp = _safe_load_json(inp_fp) or {}
                    method = inp.get("route_method")
                    k = inp.get("k_paths", 0)
                    if method == "k_shortest_path":
                        algo = f"{method}_{'inf' if k > 4 else k}"
                    else:
                        algo = method

            metric_vals: Dict[str, Any] = {}
            for fp in s_dir.glob("*.json"):
                if fp.name == "meta.json":
                    continue
                m = pattern.match(fp.name)
                if m:
                    metric_vals[m.group(1)] = _safe_load_json(fp)

            if metric_vals:
                raw_runs[unique_run_id] = metric_vals
                runid_to_algo[unique_run_id] = algo

                if metric == "sim_times":
                    date_part = s_dir.parent.parent.name
                    time_part = s_dir.parent.name
                    start_stamps[unique_run_id] = f"{date_part}_{time_part}"

                if metric == "memory" and drl:
                    logs_fp = (ROOT_LOGS / algo / network / date /
                               base_run_dir.name / "memory_usage.npy")
                    if logs_fp.exists():
                        import numpy as np
                        try:
                            arr = np.load(logs_fp)
                            if not metric_vals:
                                metric_vals = {}
                            metric_vals["overall"] = float(arr.max() / (1024 ** 2))
                        except Exception as exc:
                            print(f"[loaders] ❌ could not load {logs_fp}: {exc}")

                if metric == "rewards" and drl:
                    # base_run_dir = …/DATE/<time>
                    logs_dir = ROOT_LOGS / algo / network / date / base_run_dir.name
                    if logs_dir.is_dir():
                        import numpy as np, re, json
                        trial_rx = re.compile(r"rewards_.*?_t(\d+)_iter_(\d+)\.npy")
                        trials: dict[int, dict[int, list[float]]] = defaultdict(dict)

                        for fp in logs_dir.glob("rewards_*_iter_*.npy"):
                            m = trial_rx.match(fp.name)
                            if not m:
                                continue
                            trial_idx = int(m.group(1))
                            ep_idx = int(m.group(2))
                            try:
                                rewards = np.load(fp).astype(float).tolist()
                                # store list-of-step rewards for that episode
                                trials[trial_idx][ep_idx] = rewards
                            except Exception as exc:
                                print(f"[loaders] ❌ could not load {fp}: {exc}")

                        if trials:
                            for tv in metric_vals:
                                # TODO: This is rewards per seed
                                # TODO: We should make this consistent across all loaders
                                metric_vals[tv]['rewards']  = trials

                if metric == "state_values" and drl:
                    logs_dir = ROOT_LOGS / algo / network / date / base_run_dir.name
                    if logs_dir.is_dir():
                        import re, json
                        sv_rx = re.compile(
                            r"state_vals_e(?P<erl>\d+\.?\d*)_.*?_t(?P<trial>\d+)\.json"
                        )
                        state_vals: dict[str, dict[int, dict]] = defaultdict(dict)
                        # {erlang: {trial_idx: {(src,dst): [vals]}}}

                        for fp in logs_dir.glob("state_vals_*.json"):
                            m = sv_rx.match(fp.name)
                            if not m:
                                continue
                            erlang = m.group("erl")    # '50.0'
                            trial  = int(m.group("trial"))

                            try:
                                with fp.open("r", encoding="utf-8") as f:
                                    sv_data = json.load(f)
                                    # convert "(0, 1)" → (0,1) tuple keys now to save later conversions
                                    sv_data = {
                                        eval(k): v for k, v in sv_data.items()
                                    }
                                    state_vals[erlang][trial] = sv_data
                            except Exception as exc:
                                print(f"[loaders] ❌ could not load {fp}: {exc}")

                        # graft into metric_vals structure
                        if state_vals:
                            for erl, trials in state_vals.items():
                                if erl not in metric_vals:
                                    metric_vals[erl] = {}
                                metric_vals[erl]["state_vals"] = trials

    # TODO: Add start stamps to the raw runs data structure
    return raw_runs, runid_to_algo, start_stamps
