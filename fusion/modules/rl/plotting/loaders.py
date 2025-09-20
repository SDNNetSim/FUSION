# Standard library imports
import ast
import json
import re
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

# Third-party imports
import numpy as np

# Local imports
from fusion.modules.rl.plotting.errors import DataLoadError

ROOT_OUTPUT_DIRECTORY = Path("../../data/output")
ROOT_INPUT_DIRECTORY = Path("../../data/input")
ROOT_LOGS_DIRECTORY = Path("../../logs")


def _compose_algo_label(path_algo: str, obs: str | None) -> str:
    """Compose a unique algorithm label from base name and observation space."""
    return f"{path_algo}_{obs}" if obs else path_algo


def _safe_load_json(file_path: Path) -> Any | None:
    """Read file_path safely, returning None on error."""
    try:
        with file_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[loaders] ❌ Could not load {file_path}: {exc}")
        return None


def discover_all_run_ids(
    network: str, dates: list[str], drl: bool, obs_filter: list[str] | None
) -> dict[str, list[str]]:
    """Discovers all run IDs for a given network, date list, and DRL flag."""
    algo_runs: dict[str, list[str]] = defaultdict(list)

    for date in dates:
        run_root = ROOT_OUTPUT_DIRECTORY / network / date

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
                obs = meta.get("obs_space")

                if (obs_filter and obs not in obs_filter) and (
                    algo_base in ("dqn", "ppo")
                ):
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
                    input_file_path = (
                        ROOT_INPUT_DIRECTORY
                        / network
                        / date
                        / timestamp
                        / f"sim_input_{s_num}.json"
                    )

                    algo = "unknown"
                    if input_file_path.exists():
                        input_data = _safe_load_json(input_file_path) or {}
                        method = input_data.get("route_method")
                        k_paths = input_data.get("k_paths", 0)
                        algo = (
                            f"{method}_{'inf' if k_paths > 4 else k_paths}"
                            if method == "k_shortest_path"
                            else method
                        )

                    unique = f"{run_id}@{timestamp}"
                    algo_runs[algo].append(unique)

    for algorithm in tuple(algo_runs):
        if algorithm == "unknown":
            raise DataLoadError("Algorithm not found in run metadata.")
        algo_runs[algorithm] = list(dict.fromkeys(algo_runs[algorithm]))

    print(f"[DEBUG] Discovered {'DRL' if drl else 'non-DRL'} runs: {dict(algo_runs)}")
    return dict(algo_runs)


def load_metric_for_runs(
    run_ids: Iterable[str],
    metric: str,
    drl: bool,
    network: str,
    dates: list[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, str], dict[str, str]]:
    """Load metric data for each run."""
    raw_runs: dict[str, dict[str, Any]] = {}
    runid_to_algo: dict[str, str] = {}
    start_stamps: dict[str, str] = {}

    for date in dates:
        for simulation_directory in (ROOT_OUTPUT_DIRECTORY / network / date).rglob(
            "s*"
        ):
            if not simulation_directory.is_dir():
                continue

            if drl:
                result = _handle_drl_run(
                    simulation_directory, run_ids, metric, network, date
                )
            else:
                result = _handle_non_drl_run(
                    simulation_directory, run_ids, metric, network, date
                )

            if result:
                unique_run_id, algo, metric_vals, stamp = result
                raw_runs[unique_run_id] = metric_vals
                runid_to_algo[unique_run_id] = algo
                if stamp:
                    start_stamps[unique_run_id] = stamp

    return raw_runs, runid_to_algo, start_stamps


def _handle_drl_run(
    simulation_directory: Path,
    run_ids: Iterable[str],
    metric: str,
    network: str,
    date: str,
):  # pylint: disable=unused-argument
    base_run_directory = simulation_directory.parent
    seed = simulation_directory.name

    if metric == "sim_times":
        max_seed = max(
            int(p.name.lstrip("s")) for p in simulation_directory.parent.glob("s*")
        )
        if int(seed.lstrip("s")) != max_seed:
            return None

    metadata_file_path = next(base_run_directory.glob("s*/meta.json"), None)
    metadata_run_id: str | None = None
    algo = "unknown"

    if metadata_file_path and metadata_file_path.exists():
        metadata = _safe_load_json(metadata_file_path) or {}
        algorithm_base = metadata.get("path_algorithm", "unknown")
        observation_space = metadata.get("obs_space")
        algo = _compose_algo_label(algorithm_base, observation_space)
        metadata_run_id = metadata.get("run_id")

    timestamp = base_run_directory.name
    metadata_run_id = metadata_run_id or timestamp  # pylint: disable=possibly-unused-variable
    composite_run_id = f"{metadata_run_id}@{timestamp}"
    unique_run_id = f"{composite_run_id}_{seed}"

    if run_ids and not any(
        rid in run_ids for rid in (unique_run_id, composite_run_id, metadata_run_id)
    ):
        return None

    metric_values = _load_metric_jsons(simulation_directory, metric)
    _augment_with_logs(
        metric,
        drl=True,
        metric_vals=metric_values,
        algo=algo,
        network=network,
        base_run_dir=base_run_directory,
    )

    stamp = (
        f"{simulation_directory.parent.parent.name}_{simulation_directory.parent.name}"
        if metric == "sim_times"
        else None
    )
    return unique_run_id, algo, metric_values, stamp


def _handle_non_drl_run(
    simulation_directory: Path,
    run_ids: Iterable[str],
    metric: str,
    network: str,
    date: str,
):  # pylint: disable=unused-argument
    parent_directory = simulation_directory.parent
    seed = simulation_directory.name
    timestamp = parent_directory.name
    date_string = parent_directory.parent.name
    unique_run_id = f"{seed}_{date_string}@{timestamp}"

    if run_ids and unique_run_id not in run_ids:
        return None

    input_file_path = (
        ROOT_INPUT_DIRECTORY
        / network
        / date_string
        / parent_directory.name
        / f"sim_input_{seed}.json"
    )
    algo = "unknown"
    if input_file_path.exists():
        input_data = _safe_load_json(input_file_path) or {}
        method = input_data.get("route_method")
        k_paths = input_data.get("k_paths", 0)
        algo = (
            f"{method}_{'inf' if k_paths > 4 else k_paths}"
            if method == "k_shortest_path"
            else method
        )

    metric_values = _load_metric_jsons(simulation_directory, metric)
    return unique_run_id, algo, metric_values, None


def _load_metric_jsons(simulation_directory: Path, _: str) -> dict[str, Any]:
    metric_values: dict[str, Any] = {}
    for file_path in simulation_directory.glob("*.json"):
        if file_path.name == "meta.json":
            continue
        match = re.match(r"(\d+\.?\d*)_erlang\.json", file_path.name)
        if match:
            metric_values[match.group(1)] = _safe_load_json(file_path)
    return metric_values


def _augment_with_logs(
    metric: str,
    drl: bool,
    metric_vals: dict[str, Any],
    algo: str,
    network: str,
    base_run_dir: Path,
):
    if metric == "memory" and drl:
        logs_file_path = (
            ROOT_LOGS_DIRECTORY
            / algo
            / network
            / base_run_dir.parent.name
            / base_run_dir.name
            / "memory_usage.npy"
        )
        if logs_file_path.exists():
            try:
                memory_array = np.load(logs_file_path)
                metric_vals["overall"] = float(memory_array.max() / (1024**2))
            except (OSError, ValueError) as exc:
                raise DataLoadError(
                    f"Failed to load memory usage data from {logs_file_path}: {exc}"
                ) from exc

    elif metric == "state_values" and drl:
        logs_directory = (
            ROOT_LOGS_DIRECTORY
            / algo
            / network
            / base_run_dir.parent.name
            / base_run_dir.name
        )
        if not logs_directory.is_dir():
            return

        state_values_regex = re.compile(
            r"state_vals_e(?P<erl>\d+\.?\d*)_.*?_t(?P<trial>\d+)\.json"
        )
        state_values: dict[str, dict[int, dict]] = defaultdict(dict)

        for file_path in logs_directory.glob("state_vals_*.json"):
            match = state_values_regex.match(file_path.name)
            if not match:
                continue
            erlang = match.group("erl")
            trial = int(match.group("trial"))
            try:
                with file_path.open("r", encoding="utf-8") as f:
                    state_values_data = json.load(f)
                    state_values_data = {
                        ast.literal_eval(k): v for k, v in state_values_data.items()
                    }
                    state_values[erlang][trial] = state_values_data
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                raise DataLoadError(
                    f"Failed to load state values from {file_path}: {exc}"
                ) from exc

        for erlang, trials in state_values.items():
            if erlang not in metric_vals:
                metric_vals[erlang] = {}
            metric_vals[erlang]["state_vals"] = trials
