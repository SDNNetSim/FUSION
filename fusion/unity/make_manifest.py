"""
Manifest generation module for Unity cluster simulations.

This module generates job manifests from specification files, handling parameter
grids, explicit job definitions, and resource allocation for SLURM submission.
Supports both YAML and JSON specification formats.
"""

from __future__ import annotations

import ast
import csv
import datetime as dt
import itertools
import json
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:
    yaml = None  # type: ignore[assignment]

sys.path.append(str(Path(__file__).resolve().parents[1]))
from fusion.configs.schema import OPTIONAL_OPTIONS_DICT, SIM_REQUIRED_OPTIONS_DICT
from fusion.unity.constants import (
    BOOL_TRUE_VALUES,
    EXPERIMENTS_DIR,
    RESOURCE_KEYS,
    RL_ALGORITHMS,
)
from fusion.utils.logging_config import get_logger

logger = get_logger(__name__)

# Build parameter types from config setup
_PARAM_TYPES: dict[str, type] = {}
for options_dict in [SIM_REQUIRED_OPTIONS_DICT, OPTIONAL_OPTIONS_DICT]:
    for _category, options in options_dict.items():
        for option_name, option_type in options.items():
            _PARAM_TYPES[option_name] = option_type  # type: ignore[assignment]


def _str_to_bool(value: str) -> bool:
    """
    Convert string to boolean value.

    :param value: String value to convert
    :type value: str
    :return: Boolean representation of the string
    :rtype: bool
    """
    return value.lower() in BOOL_TRUE_VALUES


def _parse_literal(val: str) -> Any:
    """
    Parse a string as a Python literal.

    :param val: String to parse as literal
    :type val: str
    :return: Parsed value or original string if parsing fails
    :rtype: Any
    """
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return val


def _cast(key: str, value: Any) -> Any:
    """
    Cast a configuration value to its appropriate type.

    :param key: Configuration key name
    :type key: str
    :param value: Value to cast
    :type value: Any
    :return: Cast value or original value if casting fails
    :rtype: Any
    """
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
    """
    Encode a value to its string representation for CSV output.

    :param val: Value to encode
    :type val: Any
    :return: String representation of the value
    :rtype: str
    """
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, (list, dict)):
        return json.dumps(val, separators=(",", ":"))
    if isinstance(val, float):
        return format(val, ".10f").rstrip("0").rstrip(".")  # e.g., 0.000057
    return str(val)


def _is_rl(alg: str) -> str:
    """
    Check if an algorithm is a reinforcement learning algorithm.

    :param alg: Algorithm name to check
    :type alg: str
    :return: 'yes' if RL algorithm, 'no' otherwise
    :rtype: str
    """
    rl_algs = RL_ALGORITHMS
    return "yes" if alg in rl_algs else "no"


# ------------------------------- new --------------------------------------- #
def _validate_resource_keys(resources: dict[str, Any]) -> None:
    """
    Validate that all resource keys are recognized.

    :param resources: Dictionary of resource configurations
    :type resources: dict[str, Any]
    :raises SystemExit: If an unknown resource key is found
    """
    for key in resources:
        if key not in RESOURCE_KEYS:
            sys.exit(
                f"Unknown resource key '{key}'. Allowed keys: "
                f"{', '.join(sorted(RESOURCE_KEYS))}"
            )


def _validate_keys(mapping: dict[str, Any], ctx: str) -> None:
    """
    Validate that all keys in mapping are recognized parameters.

    :param mapping: Dictionary to validate
    :type mapping: dict[str, Any]
    :param ctx: Context string for error messages
    :type ctx: str
    :raises SystemExit: If an unknown parameter key is found
    """
    for key in mapping:
        if key in _PARAM_TYPES or key in RESOURCE_KEYS:
            continue
        sys.exit(f"Unknown parameter '{key}' in {ctx}. Must exist in config options.")


def _read_spec(path: Path) -> dict[str, Any]:
    """
    Read and parse a specification file (YAML or JSON).

    :param path: Path to the specification file
    :type path: Path
    :return: Parsed specification data
    :rtype: dict[str, Any]
    :raises SystemExit: If PyYAML is not installed for YAML files
    :raises json.JSONDecodeError: If JSON parsing fails
    :raises yaml.YAMLError: If YAML parsing fails
    """
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yml", ".yaml"}:
        if yaml is None:
            sys.exit("PyYAML not installed; install it or use a JSON spec file")
        result = yaml.safe_load(text)
        return dict(result) if isinstance(result, dict) else {}
    result = json.loads(text)
    return dict(result) if isinstance(result, dict) else {}


def _to_list(v: Any, *, ctx: str) -> list[Any]:
    """
    Convert a value to a list, with validation for common context.

    :param v: Value to convert to list
    :type v: Any
    :param ctx: Context for validation ('common' or 'grid')
    :type ctx: str
    :return: List representation of the value
    :rtype: list[Any]
    :raises SystemExit: If multiple values provided in 'common' context
    """
    if isinstance(v, list):
        if ctx == "common" and len(v) > 1:
            sys.exit(f"Only single values allowed in grid.common but got list {v}")
        return v
    return [v]


def _fetch(grid: dict[str, Any], common: dict[str, Any], key: str) -> list[Any]:
    """
    Fetch a parameter value from grid or common configuration.

    :param grid: Grid-specific configuration
    :type grid: dict[str, Any]
    :param common: Common configuration shared across grid
    :type common: dict[str, Any]
    :param key: Parameter key to fetch
    :type key: str
    :return: List of values for the parameter
    :rtype: list[Any]
    :raises SystemExit: If required key is not found
    """
    if key in grid:
        return _to_list(grid[key], ctx="grid")
    if key in common:
        return _to_list(common[key], ctx="common")
    sys.exit(f"Grid spec missing required key '{key}' (searched grid and grid.common)")


def _expand_grid(
    grid: dict[str, Any], starting_rid: int
) -> tuple[list[dict[str, Any]], int]:
    """
    Expand a grid specification into individual job configurations.

    :param grid: Grid specification with parameter combinations
    :type grid: dict[str, Any]
    :param starting_rid: Starting run ID for job numbering
    :type starting_rid: int
    :return: Tuple of (job rows, next available run ID)
    :rtype: tuple[list[dict[str, Any]], int]
    :raises SystemExit: If deprecated keys are found
    """
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
    rows: list[dict[str, Any]] = []
    for alg, t0, kp, curr_obs in itertools.product(algs, traf, kps, obs):
        rows.append(
            {
                "run_id": f"{rid:05}",
                "path_algorithm": alg,
                "erlang_start": t0,
                "erlang_stop": t0 + 50,
                "k_paths": kp,
                "obs_space": curr_obs,
                "is_rl": _is_rl(alg),
                **{
                    k: _cast(k, v)
                    for k, v in common.items()
                    if k not in {"path_algorithm", "erlang_start", "k_paths"}
                },
            }
        )
        rid += 1
    return rows, rid


def _explicit(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Process explicit job specifications into job configurations.

    :param jobs: List of explicit job specifications
    :type jobs: list[dict[str, Any]]
    :return: Processed job configurations
    :rtype: list[dict[str, Any]]
    """
    rows: list[dict[str, Any]] = []
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


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """
    Write job configurations to a CSV manifest file.

    :param path: Path where CSV file should be written
    :type path: Path
    :param rows: Job configuration data to write
    :type rows: list[dict[str, Any]]
    """
    cols: list[str] = []
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


def _resolve_spec_path(arg: str) -> Path:
    """
    Resolve a specification file path, checking current and specs/ directory.

    :param arg: Specification name or path argument
    :type arg: str
    :return: Resolved path to the specification file
    :rtype: Path
    :raises SystemExit: If specification file is not found
    """
    p = Path(arg)
    if p.exists():
        return p
    specs_dir = Path(__file__).resolve().parent / "specs"
    for ext in ("", ".yml", ".yaml", ".json"):
        trial = specs_dir / (arg + ext)
        if trial.exists():
            return trial
    sys.exit(f"Spec file '{arg}' not found (searched cwd and specs/ directory)")


def main() -> None:  # noqa: C901  (cyclomatic – fine here)
    """
    Main entry point for the manifest generation script.

    Parses command line arguments, reads specification files, and generates
    job manifests for cluster submission. Supports both grid-based parameter
    sweeps and explicit job definitions.

    :raises SystemExit: If invalid arguments or specification format
    """
    if len(sys.argv) != 2:
        sys.exit("Usage: make_manifest.py <spec_name_or_path>")

    spec_path = _resolve_spec_path(sys.argv[1])
    spec = _read_spec(spec_path)

    resources: dict[str, Any] = spec.get("resources", {})
    _validate_resource_keys(resources)

    if sum(k in spec for k in ("grid", "grids", "jobs")) > 1:
        sys.exit(
            "Spec must contain only one of 'grid', 'grids', or 'jobs', not multiple."
        )

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
    base_dir = Path(EXPERIMENTS_DIR) / now.strftime("%m%d") / now.strftime("%H%M%S")

    # Group rows by network
    network_groups: dict[str, list[dict[str, Any]]] = {}
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
        meta_file = net_dir / "manifest_meta.json"
        meta_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Wrote {len(network_groups)} manifests (one per network).")
    print(f"Base experiments dir → {base_dir}")


if __name__ == "__main__":
    main()
