import os
import json
from collections import defaultdict

import numpy as np


def _process_baseline(sim_time_path):
    """
    Process a baseline simulation time folder.

    Parameters:
        sim_time_path (str): Path to the simulation time folder.

    Returns:
        dict: Baseline data with keys like "s1" or "s2".
    """
    baseline_data = {}
    for sim_run in os.listdir(sim_time_path):
        sim_run_path = os.path.join(sim_time_path, sim_run)
        if not os.path.isdir(sim_run_path):
            continue
        run_data = defaultdict(list)
        for filename in os.listdir(sim_run_path):
            if not filename.endswith(".json"):
                continue
            parts = filename.split('.')
            if len(parts) < 3:
                continue
            try:
                tv = float(parts[0])
            except ValueError:
                continue

            file_path = os.path.join(sim_run_path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except (OSError, json.JSONDecodeError) as exc:
                print(f"Error reading {file_path}: {exc}")
                continue

            if not data or 'iter_stats' not in data:
                print(f"No data in file: {file_path}")
                continue

            last_iter = list(data['iter_stats'].keys())[-1]
            last_iter_data = data['iter_stats'][last_iter]
            blocking = last_iter_data.get("sim_block_list")
            if blocking is not None:
                run_data[tv].append(blocking)

        # Average the blocking values for each traffic volume within this run.
        avg_run_data = {}
        for tv, lists in run_data.items():
            if not lists:
                continue
            arr = np.array(lists)
            avg_run_data[str(tv)] = lists[0] if arr.shape[0] == 1 else np.mean(arr, axis=0).tolist()
        baseline_data[sim_run] = avg_run_data
    return baseline_data


def _process_sim_time(sim_time_path):
    """
    Process a non-baseline simulation time folder.

    Parameters:
        sim_time_path (str): Path to the simulation time folder.

    Returns:
        dict: A dictionary mapping traffic volume to a list of averaged blocking values.
    """
    sim_time_data = defaultdict(list)
    for sim_run in os.listdir(sim_time_path):
        sim_run_path = os.path.join(sim_time_path, sim_run)
        if not os.path.isdir(sim_run_path):
            continue
        for filename in os.listdir(sim_run_path):
            if not filename.endswith(".json"):
                continue
            parts = filename.split('.')
            if len(parts) < 3:
                continue
            try:
                traffic_volume = float(parts[0])
            except ValueError:
                continue

            file_path = os.path.join(sim_run_path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except (OSError, json.JSONDecodeError) as exc:
                print(f"Error reading {file_path}: {exc}")
                continue

            if not data or 'iter_stats' not in data:
                print(f"No data in file: {file_path}")
                continue

            last_iter = list(data['iter_stats'].keys())[-1]
            last_iter_data = data['iter_stats'][last_iter]
            blocking = last_iter_data.get("sim_block_list")
            if blocking is None:
                continue
            sim_time_data[traffic_volume].append(blocking)

    # Average blocking values column-wise for each traffic volume.
    averaged_data = {}
    for tv, lists in sim_time_data.items():
        if not lists:
            continue
        arr = np.array(lists)
        averaged_data[str(tv)] = lists[0] if arr.shape[0] == 1 else np.mean(arr, axis=0).tolist()
    return averaged_data


def load_blocking_data(simulation_times, base_dir):
    """
    Process simulation JSON files to compute blocking probabilities.

    Parameters:
        simulation_times (dict): Dictionary of simulation times keyed by algorithm.
        base_dir (str): Directory containing simulation folders.

    Returns:
        dict: Final blocking probability results.
    """
    final_result = {}
    for algorithm, sim_time_lists in simulation_times.items():
        alg_result = defaultdict(list)
        for sim_time_wrapper in sim_time_lists:
            sim_time = sim_time_wrapper[0]
            sim_time_path = os.path.join(base_dir, sim_time)

            if not os.path.isdir(sim_time_path):
                print(f"Warning: Path does not exist: {sim_time_path}")
                continue

            # Process baseline simulation separately.
            if sim_time == "14_43_00_326842":
                baseline_data = _process_baseline(sim_time_path)
                if "s1" in baseline_data:
                    final_result.setdefault("KSP Baseline", {}).update(baseline_data["s1"])
                if "s2" in baseline_data:
                    final_result.setdefault("SPF Baseline", {}).update(baseline_data["s2"])
            else:
                # Process non-baseline simulation times.
                avg_data = _process_sim_time(sim_time_path)
                for tv_str, avg_block in avg_data.items():
                    alg_result[tv_str].append(avg_block)
        # Update final_result for non-baseline algorithms.
        if alg_result:
            if algorithm in final_result:
                for tv, block_list in alg_result.items():
                    final_result[algorithm].setdefault(tv, []).extend(block_list)
            else:
                final_result[algorithm] = dict(alg_result)
    return final_result


def load_rewards(simulation_times, base_logs_dir, base_dir):
    """
    Load average rewards from .npy files for each non-baseline algorithm.

    Parameters:
        simulation_times (dict): Dictionary of simulation times keyed by algorithm.
        base_logs_dir (str): Base directory for the logs.
        base_dir (str): Base directory containing simulation-time folders.

    Returns:
        dict: Dictionary mapping algorithm -> traffic label -> rewards array.
    """
    rewards_data = {}
    for algorithm, sim_time_lists in simulation_times.items():
        if algorithm.lower() == "baselines":
            continue
        alg_snake = algorithm.lower().replace(" ", "_")
        rewards_data[algorithm] = {}
        for sim_time_wrapper in sim_time_lists:
            sim_time = sim_time_wrapper[0]
            reward_file = os.path.join(
                # TODO: (drl_path_agents) Hard coded
                base_logs_dir, alg_snake, "NSFNet", "0228", sim_time, "average_rewards.npy"
            )
            if not os.path.exists(reward_file):
                print(f"Warning: Reward file does not exist: {reward_file}")
                continue

            # Try to extract a traffic volume label from one JSON file in the simulation time folder.
            sim_time_path = os.path.join(base_dir, sim_time)
            traffic_label = None
            if os.path.isdir(sim_time_path):
                for sim_run in os.listdir(sim_time_path):
                    run_path = os.path.join(sim_time_path, sim_run)
                    if not os.path.isdir(run_path):
                        continue
                    for filename in os.listdir(run_path):
                        if not filename.endswith(".json"):
                            continue
                        parts = filename.split('.')
                        try:
                            traffic_label = str(float(parts[0]))
                        except ValueError:
                            traffic_label = filename.split('.')[0]
                        break
                    if traffic_label is not None:
                        break
            if traffic_label is None:
                traffic_label = sim_time  # Fallback if no label is found

            rewards = np.load(reward_file)
            rewards_data[algorithm].setdefault(traffic_label, rewards)
    return rewards_data
