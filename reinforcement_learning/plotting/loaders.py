"""
Centralised helpers for retrieving *raw* JSON output from your
simulation folder hierarchy.

The processors take the raw dictionaries returned here and massage them
into the format each plotting script expects.
"""
from pathlib import Path
import json
from collections import defaultdict
from typing import Iterable, Dict, Any

# Root folders – adjust if your structure differs
ROOT_OUTPUT = Path(__file__).resolve().parent / "../../data/output"
ROOT_LOGS = Path(__file__).resolve().parent / "../../logs"


def _safe_load_json(fp: Path) -> Any | None:
    """Read a JSON file and swallow errors gracefully."""
    try:
        with fp.open("r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"[loaders] File not found → {fp}")
    except json.JSONDecodeError as exc:
        print(f"[loaders] JSON error in {fp}: {exc}")
    return None


def load_metric_for_runs(
        run_ids: Iterable[str],
        metric: str,
        drl: bool = True,
) -> Dict[str, Any]:
    """
    Aggregate raw JSON contents for *several* runs that belong to the same
    algorithm / variant.

    Returns
    -------
    dict
        {run_id: raw_json_content}
        (the *processors* later decide how to merge these per-traffic etc.)
    """
    base = ROOT_LOGS if drl else ROOT_OUTPUT
    out: Dict[str, Any] = {}

    for rid in run_ids:
        fp = base / rid / f"{metric}.json"
        data = _safe_load_json(fp)
        if data is not None:
            out[rid] = data

    return out
