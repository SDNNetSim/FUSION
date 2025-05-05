#!/usr/bin/env python3
"""
Parse Optuna/SLURM log files, emit tidy CSV/JSON per run, **and** pick a single
hyper‑parameter set that is robust across all traffic volumes for each topology.

Usage (IDE / notebook):
1. Set `IN_ROOT`  to the top‑level folder containing your experiment logs.
2. Set `OUT_ROOT` to where you want the parsed results & best‑params saved.
3. Hit Run.  The script will:
   • discover every `*.out` file recursively
   • parse and write `<OUT_ROOT>/<alg>/<net>/<date>/<run_id>/<erlang>_results.csv`
   • aggregate those CSVs and write one `best_params.json` per topology

The selection rule defaults to minimising *mean rank* across Erlang loads
(with tie‑breakers on worst‑rank, then mean return).  Swap the logic in
`_rank_aggregate` if you prefer a different robustness criterion.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

# ───────────────────────── USER CONFIG ────────────────────────── #
IN_ROOT = Path("../experiments/input/0502")  # top‑level experiments folder
OUT_ROOT = Path("../experiments/output/0502/")  # where parsed CSV/JSON go
GLOB_PATTERN = "**/*.out"  # override only if needed
# ──────────────────────────────────────────────────────────────── #

# ───────────────────────── REGEX DEFINITIONS ─────────────────── #
CSV_ROW_RE = re.compile(r"CSV Row \d+:\s*(.*)")
TRIAL_RE = re.compile(
    r"Trial\s+(?P<trial>\d+)\s+finished\s+with\s+value:\s+"
    r"(?P<value>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+and\s+parameters:\s+(?P<params>\{.*?\})"
)
DATE_DIR_RE = re.compile(r"experiments[\\/](\d{4})[\\/]", re.IGNORECASE)


# ────────────────────────── HELPERS ──────────────────────────── #

def _parse_csv_row(row_str: str, header_str: str) -> Dict[str, str]:
    """Return dict mapping header fields -> row values."""
    headers = [h.strip() for h in header_str.split(",")]
    values = [v.strip() for v in row_str.split(",")]
    return dict(zip(headers, values))


def _parse_one_out(path: Path) -> Tuple[Dict[str, str], pd.DataFrame]:
    """Parse a single SLURM .out file and return (meta, trials_df)."""
    meta: Dict[str, str] = {}
    trials: List[Dict] = []
    row_str = None
    header_str = None

    with path.open("r", errors="ignore") as fh:
        for line in fh:
            # Capture *any* CSV Row X line
            m_row = CSV_ROW_RE.search(line)
            if m_row:
                row_str = m_row.group(1).strip()
                continue

            # Header appears on the same line ("Header: run_id,...")
            if line.startswith("Header:"):
                header_str = line.split("Header:", 1)[1].strip()
                if row_str:
                    meta.update(_parse_csv_row(row_str, header_str))
                    row_str = header_str = None  # reset for safety
                continue

            # Trials
            m_trial = TRIAL_RE.search(line)
            if m_trial:
                trials.append({
                    "trial": int(m_trial.group("trial")),
                    "objective_value": float(m_trial.group("value")),
                    **ast.literal_eval(m_trial.group("params"))
                })

    if not trials:
        raise ValueError(f"No trial lines detected in {path}")

    keep = ["run_id", "path_algorithm", "network", "erlang_start"]
    meta_small = {k: meta.get(k) for k in keep}

    trials_df = (
        pd.DataFrame(trials)
        .sort_values("trial")
        .reset_index(drop=True)
    )

    return meta_small, trials_df


def _destination(meta: Dict[str, str], out_root: Path, orig_path: Path) -> Tuple[Path, str]:
    """Build destination directory & filename for a given (meta, source_path)."""
    alg = meta["path_algorithm"]
    net = meta["network"]
    run_id = meta["run_id"]
    erlang = meta["erlang_start"]

    # Extract date chunk (e.g. 0502) from experiments path if possible
    match = DATE_DIR_RE.search(str(orig_path))
    date_chunk = match.group(1) if match else datetime.today().strftime("%m%d")

    dest_dir = out_root / alg / net / date_chunk / run_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    return dest_dir, f"{erlang}_results.csv"


def collect(in_root: Path, out_root: Path, glob_pattern: str = "**/*.out") -> None:
    """Phase 1 – parse every .out file under `in_root` and write CSV/JSON."""
    files = sorted(in_root.glob(glob_pattern))
    print(f"[Phase 1] Found {len(files)} log file(s) under {in_root}")

    for fp in files:
        try:
            meta, df = _parse_one_out(fp)
        except Exception as e:
            print(f"   [skip] {fp.name}: {e}")
            continue

        dest_dir, csv_name = _destination(meta, out_root, fp)
        df.to_csv(dest_dir / csv_name, index=False)
        (dest_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        print(f"   ✓ {fp.relative_to(in_root)} → {dest_dir.relative_to(out_root)}/{csv_name}")


# ────────────────────  PHASE 2: robust parameter selection  ──────────────────── #

def _hash_params(row: pd.Series, ignore=("trial", "objective_value", "erlang_start")) -> Tuple[str, Dict]:
    """Return (md5‑hash, params_dict) for a trial row, ignoring bookkeeping cols."""
    items = sorted((k, row[k]) for k in row.index if k not in ignore)
    return hashlib.md5(str(items).encode()).hexdigest(), dict(items)


def _rank_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate trial rows into one row per unique param vector with rank stats."""
    df = df.copy()
    # Rank (1 = best) within each Erlang traffic load
    df["rank"] = df.groupby("erlang_start")["objective_value"].rank(ascending=False, method="min")

    grouped: Dict[str, Dict] = defaultdict(lambda: {"ranks": [], "returns": []})
    for _, row in df.iterrows():
        key, params = _hash_params(row)
        entry = grouped[key]
        entry.setdefault("params", params)
        entry["ranks"].append(row["rank"])
        entry["returns"].append(row["objective_value"])

    summary_rows = []
    for key, d in grouped.items():
        ranks = d["ranks"]
        returns = d["returns"]
        summary_rows.append({
            **d["params"],
            "mean_rank": sum(ranks) / len(ranks),
            "worst_rank": max(ranks),
            "mean_return": sum(returns) / len(returns),
            "samples": len(ranks),
            "_key": key,
        })

    out = pd.DataFrame(summary_rows)
    # Lower mean_rank → better; then lower worst_rank; then higher mean_return
    return out.sort_values(["mean_rank", "worst_rank", "mean_return"], ascending=[True, True, False])


def _gather_csvs(topo_dir: Path) -> List[Path]:
    return list(topo_dir.glob("**/*_results.csv"))


def find_best_params_for_topology(topo_dir: Path) -> None:
    """
    Finds the best parameters for a topology.
    """
    csv_files = _gather_csvs(topo_dir)
    if not csv_files:
        print(f"[Phase 2] No CSVs under {topo_dir.relative_to(OUT_ROOT)}; skipping.")
        return

    frames = []
    for f in csv_files:
        erlang = int(f.stem.split("_", 1)[0])  # "200_results.csv" → 200
        df = pd.read_csv(f)
        df["erlang_start"] = erlang
        frames.append(df)

    df_all = pd.concat(frames, ignore_index=True)
    leaderboard = _rank_aggregate(df_all)
    best = leaderboard.iloc[0]

    best_params_path = topo_dir / "best_params.json"
    best_dict = {
        str(k): v.item() if hasattr(v, "item") else v  # converts np.int64, np.float64, np.bool_ to int/float/bool
        for k, v in best.items()
        if k != "_key"
    }
    best_params_path.write_text(json.dumps(best_dict, indent=2))

    print(
        f"[Phase 2] 🏆  {topo_dir.relative_to(OUT_ROOT)} → best_params.json saved "
        f"(mean_rank={best['mean_rank']:.2f}, worst={best['worst_rank']:.0f})"
    )
    # Optional: print top‑3 summary
    print(leaderboard.head(3)[["mean_rank", "worst_rank", "mean_return"]])


def sweep_all_topologies(out_root: Path) -> None:
    """
    Sweeps all topologies.
    """
    for alg_dir in out_root.iterdir():
        if not alg_dir.is_dir():
            continue
        for net_dir in alg_dir.iterdir():
            if net_dir.is_dir():
                find_best_params_for_topology(net_dir)


# ─────────────────────────────  MAIN  ──────────────────────────── #
if __name__ == "__main__":
    collect(IN_ROOT, OUT_ROOT, GLOB_PATTERN)  # Phase 1
    sweep_all_topologies(OUT_ROOT)  # Phase 2
