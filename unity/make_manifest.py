#!/usr/bin/env python3
"""Generate **manifest.csv** from a YAML/JSON spec.

Usage
-----
```
python make_manifest.py <spec_name_or_path>
```
* If `<spec_name_or_path>` is a bare filename, the script first searches a
  `specs/` folder next to this file (adding `.yml`, `.yaml`, or `.json` if
  omitted).
* **No CLI overrides** – every simulation parameter must appear in the spec
  and match a name in `COMMAND_LINE_PARAMS`.
* Deprecated keys **repeat** and **er_step** are rejected.
* In a **grid** block you may put constants inside `grid.common`. Scalars
  there (e.g. `k_paths: 3`) are treated as one-element lists so the Cartesian
  sweep still works. Multi-item lists inside `grid.common` are not allowed.
* One manifest row is produced for each
  `algorithm × traffic × k_paths` combination, with
  `traffic_stop = traffic + 50`.

Output directory:
```
experiments/<MMDD>/<NETWORK>/<HHMMSS>/manifest.csv
```
"""

from __future__ import annotations

import ast
import csv
import datetime as dt
import itertools
import json
import pathlib
import sys
from typing import Any, Dict, List

try:
    import yaml  # type: ignore
except ModuleNotFoundError:
    yaml = None  # pylint: disable=invalid-name

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from arg_scripts.config_args import COMMAND_LINE_PARAMS  # pylint: disable=wrong-import-position

_PARAM_TYPES: dict[str, type] = {name: typ for name, typ, _ in COMMAND_LINE_PARAMS}
_BOOL_STRS = {"true", "yes", "1"}
_BASE_KEYS = {"algorithm", "traffic", "k_paths", "traffic_stop"}


def _str_to_bool(value: str) -> bool:
    return value.lower() in _BOOL_STRS


def _parse_literal(val: str) -> Any:
    try:
        return ast.literal_eval(val)
    except Exception:  # pylint: disable=broad-exception-caught
        return val


def _cast(key: str, value: Any) -> Any:
    typ = _PARAM_TYPES.get(key)
    if typ is None:
        return value
    if typ is bool:
        return _str_to_bool(value) if isinstance(value, str) else bool(value)
    if typ in {list, dict} and isinstance(value, str):
        return _parse_literal(value)
    try:
        return typ(value)
    except Exception:  # pylint: disable=broad-exception-caught
        return value


def _encode(val: Any) -> str:
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, (list, dict)):
        return json.dumps(val, separators=(",", ":"))
    return str(val)


def _is_rl(alg: str) -> str:
    rl_algs = {"ppo", "qr_dqn", "a2c", "dqn", "epsilon_greedy_bandit", "ucb_bandit", "q_learning"}
    return "yes" if alg in rl_algs else "no"


def _validate_keys(mapping: Dict[str, Any], ctx: str) -> None:
    for key in mapping:
        if key in _BASE_KEYS or key in _PARAM_TYPES:
            continue
        sys.exit(f"Unknown parameter '{key}' in {ctx}. Must exist in COMMAND_LINE_PARAMS.")


def _read_spec(path: pathlib.Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yml", ".yaml"}:
        if yaml is None:
            sys.exit("PyYAML not installed; install it or use a JSON spec file")
        return yaml.safe_load(text)
    return json.loads(text)


def _to_list(v: Any, *, ctx: str) -> List[Any]:
    if isinstance(v, list):
        if ctx == "common" and len(v) > 1:
            sys.exit(f"Only single values allowed in grid.common but got list {v}")
        return v
    return [v]


def _fetch(grid: Dict[str, Any], common: Dict[str, Any], key: str) -> List[Any]:
    if key in grid:
        return _to_list(grid[key], ctx="grid")
    if key in common:
        return _to_list(common[key], ctx="common")
    sys.exit(f"Grid spec missing required key '{key}' (searched grid and grid.common)")


def _expand_grid(grid: Dict[str, Any]) -> List[Dict[str, Any]]:
    for bad in {"repeat", "er_step"} & grid.keys():
        sys.exit(f"Key '{bad}' is deprecated; remove it.")

    common = grid.get("common", {})
    _validate_keys(common, ctx="grid.common")

    algs = _fetch(grid, common, "algorithm")
    traf = _fetch(grid, common, "traffic")
    kps = _fetch(grid, common, "k_paths")

    rows: List[Dict[str, Any]] = []
    rid = 0
    for alg, t0, kp in itertools.product(algs, traf, kps):
        rows.append({
            "run_id": f"{rid:05}",
            "algorithm": alg,
            "traffic_start": t0,
            "traffic_stop": t0 + 50,
            "k_paths": kp,
            "is_rl": _is_rl(alg),
            **{k: _cast(k, v) for k, v in common.items() if k not in {"algorithm", "traffic", "k_paths"}},
        })
        rid += 1
    return rows


def _explicit(jobs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for idx, job in enumerate(jobs):
        _validate_keys(job, ctx=f"jobs[{idx}]")
        base = {
            "run_id": f"{idx:05}",
            "algorithm": job["algorithm"],
            "traffic_start": job["traffic"],
            "traffic_stop": job.get("traffic_stop", job["traffic"] + 50),
            "k_paths": job["k_paths"],
            "is_rl": _is_rl(job["algorithm"]),
        }
        for k, v in job.items():
            if k not in base:
                base[k] = _cast(k, v)
        rows.append(base)
    return rows


def _write_csv(path: pathlib.Path, rows: List[Dict[str, Any]]) -> None:
    cols: List[str] = []
    for row in rows:
        for k in row:
            if k not in cols:
                cols.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: _encode(row.get(c, "")) for c in cols})


def _resolve_spec_path(arg: str) -> pathlib.Path:
    p = pathlib.Path(arg)
    if p.exists():
        return p
    specs_dir = pathlib.Path(__file__).resolve().parent / "specs"
    for ext in ("", ".yml", ".yaml", ".json"):
        trial = specs_dir / (arg + ext)
        if trial.exists():
            return trial
    sys.exit(f"Spec file '{arg}' not found (searched cwd and specs/ directory)")


def main() -> None:
    """
    Controls the script.
    """
    if len(sys.argv) != 2:
        sys.exit("Usage: make_manifest.py <spec_name_or_path>")

    spec_path = _resolve_spec_path(sys.argv[1])
    spec = _read_spec(spec_path)

    if "grid" in spec and "jobs" in spec:
        sys.exit("Spec must contain either 'grid' or 'jobs', not both")

    if "grid" in spec:
        rows = _expand_grid(spec["grid"])
    elif "jobs" in spec:
        rows = _explicit(spec["jobs"])
    else:
        sys.exit("Spec must contain 'grid' or 'jobs'")

    network = spec.get("network")
    if not network and "grid" in spec:
        grid = spec["grid"]
        network = grid.get("network") or grid.get("common", {}).get("network")
    if not network and "jobs" in spec and spec["jobs"]:
        network = spec["jobs"][0].get("network")
    if not network:
        sys.exit("Missing 'network' key. Include it top-level, in grid.common, or per job")

    now = dt.datetime.now()
    exp_dir = pathlib.Path("experiments") / now.strftime("%m%d") / str(network) / now.strftime("%H%M%S")
    _write_csv(exp_dir / "manifest.csv", rows)

    meta = {
        "generated": now.isoformat(timespec="seconds"),
        "source": str(spec_path),
        "num_rows": len(rows),
    }
    (exp_dir / "manifest_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Manifest → {exp_dir / 'manifest.csv'}")
    print(f"Experiment dir → {exp_dir}")


if __name__ == "__main__":
    main()
