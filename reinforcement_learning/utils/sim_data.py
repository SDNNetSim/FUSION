import os
import re
import json
from collections import defaultdict

import numpy as np


def load_and_average_state_values(simulation_times, base_logs_dir, base_dir):
    """
    Load state-value files for each traffic volume, then average them across seeds.
    """
    # We'll store data in a nested structure: traffic_data[traffic_label][(src, dst)][path_idx] = [...]
    traffic_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for algorithm, sim_time_lists in simulation_times.items():
        if algorithm.lower() == "baselines":
            continue

        alg_snake = algorithm.lower().replace(" ", "_")

        for sim_time_wrapper in sim_time_lists:
            sim_time = sim_time_wrapper[0]

            logs_dir = os.path.join(base_logs_dir, alg_snake, "NSFNet", "0228", sim_time)
            if not os.path.isdir(logs_dir):
                continue

            # Identify the traffic_label (the same way you do in load_rewards)
            # e.g., extracting from JSON or fallback to sim_time
            traffic_label = ...  # your existing logic here

            json_files = [f for f in os.listdir(logs_dir)
                          if f.startswith("state_vals") and f.endswith(".json")]

            for json_file in json_files:
                file_path = os.path.join(logs_dir, json_file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                except (OSError, json.JSONDecodeError):
                    continue

                for link_str, path_vals in data.items():
                    try:
                        link_tuple = eval(link_str)
                    except (SyntaxError, NameError):
                        continue
                    for p_idx, val in enumerate(path_vals):
                        traffic_data[traffic_label][link_tuple][p_idx].append(val)

    # Now average across seeds
    # final_data = { traffic_label: { (src,dst): [mean_path_0, mean_path_1, ...], ... } }
    final_data = {}
    for t_label, link_map in traffic_data.items():
        final_data[t_label] = {}
        for link_tuple, pidx_map in link_map.items():
            max_p_idx = max(pidx_map.keys())
            path_vals_list = []
            for p_idx in range(max_p_idx + 1):
                raw_list = pidx_map.get(p_idx, [])
                if raw_list:
                    path_vals_list.append(float(np.mean(raw_list)))
                else:
                    path_vals_list.append(0.0)
            final_data[t_label][link_tuple] = path_vals_list

    return final_data


def _extract_traffic_label(sim_time_path: str) -> str:
    """
    Extract a traffic label from the simulation time folder.
    Looks into the first JSON file encountered and attempts to parse its name.
    """
    if not os.path.isdir(sim_time_path):
        return ""
    for sim_run in os.listdir(sim_time_path):
        run_path = os.path.join(sim_time_path, sim_run)
        if not os.path.isdir(run_path):
            continue
        for filename in os.listdir(run_path):
            if filename.endswith(".json"):
                parts = filename.split('.')
                try:
                    return str(float(parts[0]))
                except ValueError:
                    return parts[0]
    return ""


def load_all_rewards_files(simulation_times, base_logs_dir, base_dir):
    """
    Load all per-episode, per-trial reward files for each algorithm.
    """
    pattern = re.compile(r"^rewards_e([0-9.]+)_routes_c\d+_t(\d+)_iter_(\d+)\.npy$")
    all_rewards_data = {}

    for algorithm, sim_time_lists in simulation_times.items():
        if algorithm.lower() == "baselines":
            continue

        alg_snake = algorithm.lower().replace(" ", "_")
        all_rewards_data[algorithm] = {}

        for sim_time_wrapper in sim_time_lists:
            sim_time = sim_time_wrapper[0]
            rewards_dir = os.path.join(base_logs_dir, alg_snake, "NSFNet", "0228", sim_time)
            if not os.path.exists(rewards_dir):
                print(f"Warning: Directory does not exist: {rewards_dir}")
                continue

            sim_time_path = os.path.join(base_dir, sim_time)
            traffic_label = _extract_traffic_label(sim_time_path)
            if not traffic_label:
                traffic_label = sim_time  # fallback

            if traffic_label not in all_rewards_data[algorithm]:
                all_rewards_data[algorithm][traffic_label] = {}

            for fname in os.listdir(rewards_dir):
                match = pattern.match(fname)
                if not match:
                    continue
                # Unpack, ignore erlang value since it's unused.
                _, trial_str, episode_str = match.groups()
                trial_index = int(trial_str)
                episode_index = int(episode_str)

                file_path = os.path.join(rewards_dir, fname)
                rewards_array = np.load(file_path)

                if trial_index not in all_rewards_data[algorithm][traffic_label]:
                    all_rewards_data[algorithm][traffic_label][trial_index] = {}
                all_rewards_data[algorithm][traffic_label][trial_index][episode_index] = rewards_array

    return all_rewards_data


def _process_baseline(sim_time_path: str):
    """
    Process a baseline simulation time folder.
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

        avg_run_data = {}
        for tv, lists in run_data.items():
            if not lists:
                continue
            arr = np.array(lists)
            avg_run_data[str(tv)] = lists[0] if arr.shape[0] == 1 else np.mean(arr, axis=0).tolist()
        baseline_data[sim_run] = avg_run_data
    return baseline_data


def _process_sim_time(sim_time_path: str):
    """
    Process a non-baseline simulation time folder.
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

    averaged_data = {}
    for tv, lists in sim_time_data.items():
        if not lists:
            continue
        arr = np.array(lists)
        averaged_data[str(tv)] = lists[0] if arr.shape[0] == 1 else np.mean(arr, axis=0).tolist()
    return averaged_data


def _process_single_sim_time(sim_time: str, base_dir: str):
    """
    Process a single simulation time folder based on its type.
    """
    sim_time_path = os.path.join(base_dir, sim_time)
    if not os.path.isdir(sim_time_path):
        print(f"Warning: Path does not exist: {sim_time_path}")
        return None
    if sim_time == "14_43_00_326842":
        return _process_baseline(sim_time_path)
    return _process_sim_time(sim_time_path)


def load_blocking_data(simulation_times, base_dir):
    """
    Process simulation JSON files to compute blocking probabilities.
    """
    final_result = {}
    for algorithm, sim_time_lists in simulation_times.items():
        alg_result = defaultdict(list)
        for sim_time_wrapper in sim_time_lists:
            sim_time = sim_time_wrapper[0]
            data = _process_single_sim_time(sim_time, base_dir)
            if data is None:
                continue

            if sim_time == "14_43_00_326842":
                if "s1" in data:
                    final_result.setdefault("KSP Baseline", {}).update(data["s1"])
                if "s2" in data:
                    final_result.setdefault("SPF Baseline", {}).update(data["s2"])
            else:
                for tv_str, avg_block in data.items():
                    alg_result[tv_str].append(avg_block)
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
                base_logs_dir, alg_snake, "NSFNet", "0228", sim_time, "average_rewards.npy"
            )
            if not os.path.exists(reward_file):
                print(f"Warning: Reward file does not exist: {reward_file}")
                continue

            sim_time_path = os.path.join(base_dir, sim_time)
            traffic_label = _extract_traffic_label(sim_time_path)
            if not traffic_label:
                traffic_label = sim_time  # Fallback if no label is found

            rewards = np.load(reward_file)
            rewards_data[algorithm].setdefault(traffic_label, rewards)
    return rewards_data
