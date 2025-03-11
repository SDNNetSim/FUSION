import os
import re
import json
from collections import defaultdict

import numpy as np


def _extract_traffic_label(sim_time_path):
    # Look for a file named like "600.0_erlang.json" or any JSON
    for item in os.listdir(sim_time_path):
        item_path = os.path.join(sim_time_path, item)
        if os.path.isdir(item_path):
            # seed folder "s1", "s2", etc.
            for f in os.listdir(item_path):
                if f.endswith("erlang.json"):
                    return f.split('_')[0]  # e.g. "600.0"
    return ""


def load_memory_usage(simulation_times, base_logs_dir, base_dir, network, date):
    memory_usage_data = {}
    for algorithm, sim_time_lists in simulation_times.items():
        alg_snake = algorithm.lower().replace(" ", "_")
        memory_usage_data[algorithm] = {}
        for sim_time_wrapper in sim_time_lists:
            sim_time = sim_time_wrapper[0]
            mem_file = os.path.join(base_logs_dir, alg_snake, network, date, sim_time, "memory_usage.npy")
            sim_time_path = os.path.join(base_dir, sim_time)

            # Use your existing extraction logic:
            traffic_label = _extract_traffic_label(sim_time_path)
            if not traffic_label:
                traffic_label = sim_time  # fallback

            if os.path.exists(mem_file):
                memory_usage = np.load(mem_file)
                memory_usage_data[algorithm].setdefault(traffic_label, memory_usage)
            else:
                print(f"Memory usage file not found: {mem_file}")
    return memory_usage_data


def _extract_traffic_label_from_filename(fname: str, fallback: str) -> str:
    """
    Try to parse something like 'e400.0' (the traffic volume) from the filename.
    If it fails, just use the fallback.
    """
    match = re.search(r'e(\d+(?:\.\d+)?)', fname)
    if match:
        return match.group(1)
    return fallback


# def load_and_average_state_values(simulation_times, base_logs_dir, date, network, base_dir):
#     """
#     Load state-value data from:
#       - JSON files ("state_vals_e###.json") for most algorithms.
#       - NumPy arrays (e.g. "e###.0_routes_cX_tY.npy") for Q-learning,
#         where each element is an array of 2 tuples: [(path_or_None, q_value), (path_or_None, q_value)].
#     Collect by (traffic_label, link_tuple, path_index), then average across seeds.
#     """
#     # traffic_data[traffic_label][(src, dst)][path_idx] = [values across seeds]
#     traffic_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
#
#     for algorithm, sim_time_lists in simulation_times.items():
#         # Skip "baselines"
#         if algorithm.lower() == "baselines":
#             continue
#
#         alg_snake = algorithm.lower().replace(" ", "_")
#
#         for sim_time_wrapper in sim_time_lists:
#             sim_time = sim_time_wrapper[0]
#             logs_dir = os.path.join(base_logs_dir, alg_snake, network, date, sim_time)
#             if not os.path.isdir(logs_dir):
#                 continue
#
#             # Fallback label if nothing else is parsed from filenames
#             dir_label = _extract_traffic_label(logs_dir) or sim_time
#
#             json_files = [
#                 fname for fname in os.listdir(logs_dir)
#                 if fname.startswith("state_vals") and fname.endswith(".json")
#             ]
#             for json_file in json_files:
#                 file_path = os.path.join(logs_dir, json_file)
#                 traffic_label = _extract_traffic_label_from_filename(json_file, dir_label)
#
#                 # Load JSON data
#                 try:
#                     with open(file_path, 'r', encoding='utf-8') as f:
#                         data = json.load(f)
#                 except (OSError, json.JSONDecodeError):
#                     continue
#
#                 # data: { "(src,dst)": [val_path0, val_path1, ...], ... }
#                 for link_str, path_vals in data.items():
#                     try:
#                         link_tuple = eval(link_str)  # e.g. "(1, 3)" -> (1,3)
#                     except (SyntaxError, NameError):
#                         continue
#                     for p_idx, val in enumerate(path_vals):
#                         traffic_data[traffic_label][link_tuple][p_idx].append(val)
#
#     final_data = {}
#     for t_label, link_map in traffic_data.items():
#         final_data[t_label] = {}
#         for link_tuple, pidx_map in link_map.items():
#             max_p_idx = max(pidx_map.keys()) if pidx_map else -1
#             path_means = []
#             for p_idx in range(max_p_idx + 1):
#                 values_for_path = pidx_map.get(p_idx, [])
#                 path_means.append(float(np.mean(values_for_path)) if values_for_path else 0.0)
#             final_data[t_label][link_tuple] = path_means
#
#     return final_data


def load_and_average_state_values(simulation_times, base_logs_dir, date, network, base_dir):
    """
    Load state-value data from:
      - JSON files ("state_vals_e###.json") for most algorithms.
      - NumPy arrays (e.g. "e###.0_routes_cX_tY.npy") for Q-learning,
        where each element is an array of 2 tuples: [(path_or_None, q_value), (path_or_None, q_value)].
    Collect by (traffic_label, link_tuple, path_index), then average across seeds.
    Returns a dict of the form:
      {
          algorithm_1: {
              traffic_label_1: { (src, dst): [path_val_0, path_val_1, ...], ... },
              traffic_label_2: ...
          },
          algorithm_2: { ... },
          ...
      }
    """
    all_algorithms_data = {}
    for algorithm, sim_time_lists in simulation_times.items():
        if algorithm.lower() == "baselines":
            continue
        alg_snake = algorithm.lower().replace(" ", "_")
        traffic_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for sim_time_wrapper in sim_time_lists:
            sim_time = sim_time_wrapper[0]
            logs_dir = os.path.join(base_logs_dir, alg_snake, network, date, sim_time)
            if not os.path.isdir(logs_dir):
                continue
            dir_label = _extract_traffic_label(logs_dir) or sim_time
            json_files = [
                fname for fname in os.listdir(logs_dir)
                if fname.startswith("state_vals") and fname.endswith(".json")
            ]
            for json_file in json_files:
                file_path = os.path.join(logs_dir, json_file)
                traffic_label = _extract_traffic_label_from_filename(json_file, dir_label)
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
        final_data = {}
        for t_label, link_map in traffic_data.items():
            final_data[t_label] = {}
            for link_tuple, pidx_map in link_map.items():
                max_p_idx = max(pidx_map.keys()) if pidx_map else -1
                path_means = []
                for p_idx in range(max_p_idx + 1):
                    values_for_path = pidx_map.get(p_idx, [])
                    if values_for_path:
                        path_means.append(float(np.mean(values_for_path)))
                    else:
                        path_means.append(0.0)
                final_data[t_label][link_tuple] = path_means
        all_algorithms_data[algorithm] = final_data
    return all_algorithms_data


def load_all_rewards_files(simulation_times, base_logs_dir, base_dir, network, date):
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
            rewards_dir = os.path.join(base_logs_dir, alg_snake, network, date, sim_time)
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

            if sim_time == "08_39_34_463723":
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


def load_rewards(simulation_times, base_logs_dir, base_dir, network, date):
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
                base_logs_dir, alg_snake, network, date, sim_time, "average_rewards.npy"
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
