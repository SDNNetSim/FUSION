# ✅ Fully Working Version of loaders.py
from pathlib import Path
import json
import re
from collections import defaultdict
from typing import Iterable, Dict, Any, Tuple

ROOT_OUTPUT = Path("../../data/output")
ROOT_INPUT = Path("../../data/input")


def _safe_load_json(fp: Path) -> Any | None:
    try:
        with fp.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[loaders] ❌ Could not load {fp}: {e}")
        return None


def discover_all_run_ids(network: str, dates: list[str], drl: bool) -> dict[str, list[str]]:
    algo_runs = defaultdict(list)

    for date in dates:
        for s_dir in (ROOT_OUTPUT / network / str(date)).rglob("s*"):
            if not s_dir.is_dir():
                continue
            meta_fp = s_dir / "meta.json"

            if drl and meta_fp.exists():
                meta = _safe_load_json(meta_fp)
                if isinstance(meta, dict) and "run_id" in meta:
                    run_id = meta["run_id"]
                    algo = meta.get("path_algorithm", "unknown")
                    algo_runs[algo].append(run_id)
            elif not drl and not meta_fp.exists():
                parent_dir = s_dir.parent
                date_str = parent_dir.parent.name
                s_number = s_dir.name
                run_id = f"{s_number}_{date_str}"
                input_file = ROOT_INPUT / network / date_str / parent_dir.name / f"sim_input_{s_number}.json"

                algo = "unknown"
                if input_file.exists():
                    input_data = _safe_load_json(input_file)
                    if isinstance(input_data, dict):
                        method = input_data.get("route_method")
                        k_paths = input_data.get("k_paths", 0)
                        if method == "k_shortest_path":
                            suffix = "inf" if k_paths > 4 else str(k_paths)
                            algo = f"{method}_{suffix}"
                        else:
                            algo = method
                algo_runs[algo].append(run_id)
    print(f"[DEBUG] Discovered {'DRL' if drl else 'non-DRL'} runs: {dict(algo_runs)}")
    return dict(algo_runs)


def load_metric_for_runs(
        run_ids: Iterable[str],
        metric: str,
        drl: bool,
        network: str,
        dates: list[str],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str]]:
    out = {}
    runid_to_algo = {}

    for date in dates:
        for s_dir in (ROOT_OUTPUT / network / str(date)).rglob("s*"):
            if not s_dir.is_dir():
                continue
            meta_fp = s_dir / "meta.json"

            if drl and meta_fp.exists():
                meta = _safe_load_json(meta_fp)
                if not meta or "run_id" not in meta:
                    continue
                run_id = meta["run_id"]
                algo = meta.get("path_algorithm", "unknown")
            elif not drl and not meta_fp.exists():
                parent_dir = s_dir.parent
                date_str = parent_dir.parent.name
                s_number = s_dir.name
                run_id = f"{s_number}_{date_str}"
                input_file = ROOT_INPUT / network / date_str / parent_dir.name / f"sim_input_{s_number}.json"

                algo = "unknown"
                if input_file.exists():
                    input_data = _safe_load_json(input_file)
                    if isinstance(input_data, dict):
                        method = input_data.get("route_method")
                        k_paths = input_data.get("k_paths", 0)
                        if method == "k_shortest_path":
                            suffix = "inf" if k_paths > 4 else str(k_paths)
                            algo = f"{method}_{suffix}"
                        else:
                            algo = method
            else:
                continue

            if run_id not in run_ids:
                continue

            result = {}
            for file in s_dir.glob("*.json"):
                if file.name == "meta.json":
                    continue
                match = re.match(r"(\d+\.?\d*)_erlang\.json", file.name)
                if match:
                    erlang_val = match.group(1)
                    result[erlang_val] = _safe_load_json(file)

            if result:
                out[run_id] = result
                runid_to_algo[run_id] = algo

    return out, runid_to_algo
