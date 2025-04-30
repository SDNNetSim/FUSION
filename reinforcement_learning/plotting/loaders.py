# ✅ loaders.py (now returns data + runid→algorithm mapping)
from pathlib import Path
import json
import re
from collections import defaultdict
from typing import Iterable, Dict, Any, Tuple

ROOT_OUTPUT = Path("../../data/output")


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
                else:
                    continue
            elif not drl and not meta_fp.exists():
                run_id = s_dir.parent.name
                algo = run_id.split("_")[0] if "_" in run_id else "unknown"
                algo_runs[algo].append(run_id)
            else:
                continue

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
                run_id = s_dir.parent.name
                algo = run_id.split("_")[0] if "_" in run_id else "unknown"
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
