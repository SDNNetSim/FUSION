#!/usr/bin/env python3
"""
Generate manifest.csv for simulation campaigns,
saving under experiments/<MMDD>/<NETWORK>/<TIMESTAMP>/...
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import itertools
import json
import pathlib
import sys
from typing import Any, Dict, List, Sequence

try:
    import yaml
except ModuleNotFoundError:
    yaml = None


def _timestamp() -> str:
    return _dt.datetime.now().strftime("%H%M%S")


def _bool_is_rl(algorithm: str) -> str:
    return "yes" if algorithm in ("ppo", "qr_dqn", "a2c") else "no"


def _write_csv(path: pathlib.Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def _parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", help="NSFNet, GEANT, etc.")
    parser.add_argument("--alg", nargs="*", help="algorithm list, e.g. ppo a2c")
    parser.add_argument("--traffic", nargs="*", type=int, help="traffic volumes")
    parser.add_argument(
        "--k-paths",
        nargs="*",
        type=int,
        dest="k_paths",
        help="k_paths list",
    )
    parser.add_argument("--er-step", type=int, default=50, help="erlang step")
    parser.add_argument(
        "--repeat", type=int, default=1, help="repeats per config"
    )
    parser.add_argument("--spec", help="YAML/JSON spec file")
    return parser.parse_args()


def _load_spec(path: pathlib.Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yml", ".yaml"):
        if yaml is None:
            sys.exit("PyYAML not installed; install or use JSON spec.")
        return yaml.safe_load(text)
    return json.loads(text)


def _expand_grid(spec):
    algs = spec["algorithm"]
    traf = spec["traffic"]
    kpts = spec["k_paths"]
    repeat = spec.get("repeat", 1)
    er_step = spec.get("er_step", 50)
    rows: List[Dict[str, Any]] = []
    run_id = 0
    for algorithm, t_start, k in itertools.product(algs, traf, kpts):
        for seed in range(repeat):
            rows.append({
                "run_id": f"{run_id:05}",
                "algorithm": algorithm,
                "traffic_start": t_start,
                "traffic_stop": t_start + er_step,
                "k_paths": k,
                "seed": seed,
                "is_rl": _bool_is_rl(algorithm),
            })
            run_id += 1
    return rows


def _explicit_jobs(jobs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for idx, job in enumerate(jobs):
        rows.append({
            "run_id": f"{idx:05}",
            "algorithm": job["algorithm"],
            "traffic_start": job["traffic"],
            "traffic_stop": job.get("traffic_stop", job["traffic"] + 50),
            "k_paths": job["k_paths"],
            "seed": job.get("seed", 0),
            "is_rl": _bool_is_rl(job["algorithm"]),
        })
    return rows


def main() -> None:
    """
    Controls the script.
    """
    args = _parse_cli()
    # --- figure out rows & network just like before ---
    if args.spec:
        spec_path = pathlib.Path(args.spec)
        if not spec_path.exists():
            spec_path = pathlib.Path("specs") / args.spec
        if not spec_path.exists():
            sys.exit(f"Spec file not found: {args.spec}")
        spec_data = _load_spec(spec_path)
        if "grid" in spec_data:
            rows = _expand_grid(spec_data["grid"])
        elif "jobs" in spec_data:
            rows = _explicit_jobs(spec_data["jobs"])
        else:
            sys.exit("Spec file must contain 'grid' or 'jobs'.")
        network = spec_data.get("network", args.network)
        if not network:
            sys.exit("Network must be specified in spec or via --network.")
    else:
        if not (args.network and args.alg and args.traffic and args.k_paths):
            sys.exit(
                "Specify --network, --alg, --traffic, --k-paths or use --spec."
            )
        rows = _expand_grid({
            "algorithm": args.alg,
            "traffic": args.traffic,
            "k_paths": args.k_paths,
            "repeat": args.repeat,
            "er_step": args.er_step,
        })
        network = args.network

    # --- NEW: build experiments/<MMDD>/<NETWORK>/<TIMESTAMP> directory ---
    now = _dt.datetime.now()
    date_dir = now.strftime("%m%d")  # e.g. "0424"
    time_dir = now.strftime("%H%M%S")  # e.g. "20250424_153045"
    exp_dir = pathlib.Path("experiments") / date_dir / network / time_dir

    # write out CSV + metadata
    _write_csv(exp_dir / "manifest.csv", rows)
    meta = {
        "generated": now.isoformat(timespec="seconds"),
        "source": args.spec or "cli",
        "num_rows": len(rows),
    }
    (exp_dir / "manifest_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    print(f"Manifest written to {exp_dir / 'manifest.csv'}")
    print(f"Experiment directory:  {exp_dir}")


if __name__ == "__main__":
    main()
