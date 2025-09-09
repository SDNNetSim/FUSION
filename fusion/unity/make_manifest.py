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
from fusion.configs.schema import SIM_REQUIRED_OPTIONS, OPTIONAL_OPTIONS  # pylint: disable=wrong-import-position

# Build parameter types from config setup
_PARAM_TYPES: dict[str, type] = {}
for options_dict in [SIM_REQUIRED_OPTIONS, OPTIONAL_OPTIONS]:
    for category, options in options_dict.items():
        for option_name, option_type in options.items():
            _PARAM_TYPES[option_name] = option_type
_BOOL_STRS = {"true", "yes", "1"}

_RESOURCE_KEYS = {
    "partition", "time", "mem", "cpus", "gpus", "nodes"
}


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
    if isinstance(val, float):
        return format(val, ".10f").rstrip("0").rstrip(".")  # e.g., 0.000057
    return str(val)


def _is_rl(alg: str) -> str:
    rl_algs = {"ppo", "qr_dqn", "a2c", "dqn", "epsilon_greedy_bandit",
               "ucb_bandit", "q_learning"}
    return "yes" if alg in rl_algs else "no"


# ------------------------------- new --------------------------------------- #
def _validate_resource_keys(resources: Dict[str, Any]) -> None:
    """Warn the user if they mistype a resource key."""
    for key in resources:
        if key not in _RESOURCE_KEYS:
            sys.exit(
                f"Unknown resource key '{key}'. Allowed keys: "
                f"{', '.join(sorted(_RESOURCE_KEYS))}"
            )


def _validate_keys(mapping: Dict[str, Any], ctx: str) -> None:
    for key in mapping:
        if key in _PARAM_TYPES or key in _RESOURCE_KEYS:
            continue
        sys.exit(f"Unknown parameter '{key}' in {ctx}. Must exist in config options.")


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


def _expand_grid(grid: Dict[str, Any], starting_rid: int) -> tuple[List[Dict[str, Any]], int]:
    for bad in {"repeat", "er_step"} & grid.keys():
        sys.exit(f"Key '{bad}' is deprecated; remove it.")

    common = grid.get("common", {})
    _validate_keys(common, ctx="grid.common")

    # TODO: These should apply to all parameters
    algs = _fetch(grid, common, "path_algorithm")
    traf = _fetch(grid, common, "erlang_start")
    kps = _fetch(grid, common, "k_paths")
    obs = _fetch(grid, common, "obs_space")

    rid = starting_rid
    rows: List[Dict[str, Any]] = []
    for alg, t0, kp, curr_obs in itertools.product(algs, traf, kps, obs):
        rows.append({
            "run_id": f"{rid:05}",
            "path_algorithm": alg,
            "erlang_start": t0,
            "erlang_stop": t0 + 50,
            "k_paths": kp,
            "obs_space": curr_obs,
            "is_rl": _is_rl(alg),
            **{k: _cast(k, v) for k, v in common.items()
               if k not in {"path_algorithm", "erlang_start", "k_paths"}},
        })
        rid += 1
    return rows, rid


def _explicit(jobs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for idx, job in enumerate(jobs):
        _validate_keys(job, ctx=f"jobs[{idx}]")
        base = {
            "run_id": f"{idx:05}",
            "path_algorithm": job["algorithm"],
            "erlang_start": job["traffic"],
            "erlang_stop": job.get("erlang_stop", job["erlang_start"] + 50),
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
    seen = set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                cols.append(k)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            # For missing fields, insert blank automatically
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


def main() -> None:  # noqa: C901  (cyclomatic – fine here)
    """
    Controls the script.
    """
    if len(sys.argv) != 2:
        sys.exit("Usage: make_manifest.py <spec_name_or_path>")

    spec_path = _resolve_spec_path(sys.argv[1])
    spec = _read_spec(spec_path)

    resources: Dict[str, Any] = spec.get("resources", {})
    _validate_resource_keys(resources)

    if sum(k in spec for k in ("grid", "grids", "jobs")) > 1:
        sys.exit("Spec must contain only one of 'grid', 'grids', or 'jobs', not multiple.")

    global_rid = 0
    rows = []
    if "grids" in spec:
        for grid in spec["grids"]:
            grid_rows, global_rid = _expand_grid(grid, global_rid)
            rows.extend(grid_rows)
    elif "grid" in spec:
        grid_rows, global_rid = _expand_grid(spec["grid"], global_rid)
        rows.extend(grid_rows)
    elif "jobs" in spec:
        rows = _explicit(spec["jobs"])
    else:
        sys.exit("Spec must contain 'grid', 'grids', or 'jobs'")

    # Apply resources uniformly to every row
    if resources:
        for r in rows:
            r.update(resources)

    now = dt.datetime.now()
    base_dir = pathlib.Path("experiments") / now.strftime("%m%d") / now.strftime("%H%M%S")

    # Group rows by network
    network_groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        net = row.get("network")
        if not net:
            sys.exit(f"Row {row['run_id']} is missing 'network' field!")
        network_groups.setdefault(net, []).append(row)

    for net, group_rows in network_groups.items():
        net_dir = base_dir / net
        _write_csv(net_dir / "manifest.csv", group_rows)

        meta = {
            "generated": now.isoformat(timespec="seconds"),
            "source": str(spec_path),
            "network": net,
            "num_rows": len(group_rows),
            "resources": resources,
        }
        (net_dir / "manifest_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Wrote {len(network_groups)} manifests (one per network).")
    print(f"Base experiments dir → {base_dir}")


if __name__ == "__main__":
    main()
