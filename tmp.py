# TODO: Make sure this averages blocking probabilities

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Updated base directory (simulation-time folders are directly under this folder)
base_dir = "./data/output/NSFNet/0227"

# Your dictionary of simulation times, keyed by algorithm
simulation_times = {
    "Epsilon greedy bandit": [['15_23_01_094246'], ['15_26_56_957674'], ['15_27_00_995133'], ['15_27_02_126167'],
                              ['15_27_02_142582'], ['15_27_02_142773'], ['15_27_03_358067'], ['15_27_03_883949'],
                              ['15_27_03_884391'], ['15_27_03_884425'], ['15_27_03_884451'], ['15_27_05_600451'],
                              ['15_27_05_942815'], ['15_27_06_913650']],
    "Q learning": [['15_26_53_549216'], ['15_26_55_059995'], ['15_26_59_656966'], ['15_27_02_126206'],
                   ['15_27_02_142810'], ['15_27_03_883947'], ['15_27_03_884408'], ['15_27_03_884411'],
                   ['15_27_03_884415'], ['15_27_03_884422'], ['15_27_05_600238'], ['15_27_05_600442'],
                   ['15_27_05_607359'], ['15_27_05_943422'], ['15_27_07_958207']],
    "UCB bandit": [['15_26_55_060125'], ['15_26_56_957673'], ['15_27_00_767416'], ['15_27_01_460318'],
                   ['15_27_02_126255'], ['15_27_02_126261'], ['15_27_02_142768'], ['15_27_02_144078'],
                   ['15_27_03_883941'], ['15_27_03_883966'], ['15_27_03_884403'], ['15_27_03_884454'],
                   ['15_27_03_884461'], ['15_27_05_600487'], ['15_27_05_944750']],
    "PPO": [['14_43_00_326842'], ['21_25_56_483843'], ['21_26_01_920140'], ['21_26_01_920146'], ['21_26_01_920162'],
            ['21_26_01_920204'], ['21_26_01_920206'], ['21_26_01_920214'], ['21_26_01_920476'], ['21_26_01_920477'],
            ['21_26_01_921120'], ['21_26_01_921603'], ['21_26_01_921615'], ['21_26_01_921828']],
    # Baselines (they will be processed separately and not used for rewards)
    "Baselines": [['14_43_00_326842']]
}

final_result = {}

# --- Process Blocking Probability Data ---
for algorithm, sim_time_lists in simulation_times.items():
    # We'll combine non-baseline simulation times into alg_result.
    alg_result = defaultdict(list)

    for sim_time_wrapper in sim_time_lists:
        sim_time = sim_time_wrapper[0]
        sim_time_path = os.path.join(base_dir, sim_time)

        if not os.path.isdir(sim_time_path):
            print(f"Warning: Path does not exist: {sim_time_path}")
            continue

        # Special baseline case: process simulation time "11_53_53_905072" separately.
        if sim_time == "14_43_00_326842":
            baseline_data = {}  # keys: simulation run folder name
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
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")
                        continue
                    if data and 'iter_stats' in data:
                        last_iter = list(data['iter_stats'].keys())[-1]
                        last_iter_data = data['iter_stats'][last_iter]
                        blocking = last_iter_data.get("sim_block_list")
                        if blocking is not None:
                            run_data[tv].append(blocking)
                    else:
                        print(f"No data in file: {file_path}")
                # Average the blocking lists for each traffic volume (within this run)
                avg_run_data = {}
                for tv, lists in run_data.items():
                    arr = np.array(lists)
                    if arr.shape[0] == 0:
                        continue
                    if arr.shape[0] == 1:
                        avg_run_data[str(tv)] = lists[0]
                    else:
                        avg_run_data[str(tv)] = np.mean(arr, axis=0).tolist()
                baseline_data[sim_run] = avg_run_data

            # Insert baseline results into final_result with separate keys.
            if "s1" in baseline_data:
                final_result.setdefault("KSP Baseline", {}).update(baseline_data["s1"])
            if "s2" in baseline_data:
                final_result.setdefault("SPF Baseline", {}).update(baseline_data["s2"])
        else:
            # Normal processing for non-baseline simulation times.
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
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")
                        continue
                    if data and 'iter_stats' in data:
                        last_iter = list(data['iter_stats'].keys())[-1]
                        last_iter_data = data['iter_stats'][last_iter]
                        blocking = last_iter_data.get("sim_block_list")
                        if blocking is not None:
                            sim_time_data[traffic_volume].append(blocking)
                    else:
                        print(f"No data in file: {file_path}")

            # Average blocking values column-wise for each traffic volume.
            for tv, lists in sim_time_data.items():
                arr = np.array(lists)
                if arr.shape[0] == 0:
                    continue
                if arr.shape[0] == 1:
                    avg_block = lists[0]
                else:
                    avg_block = np.mean(arr, axis=0).tolist()
                alg_result[str(tv)].append(avg_block)

            # Update final_result under the current algorithm.
            if alg_result:
                if algorithm in final_result:
                    for tv, block_list in alg_result.items():
                        final_result[algorithm].setdefault(tv, []).extend(block_list)
                else:
                    final_result[algorithm] = dict(alg_result)

# --- Plot Blocking Probabilities ---
plt.figure(figsize=(8, 6))
for algo_key, traffic_data in final_result.items():
    erlang_values = []
    blocking_probs = []
    for tv_str, sim_block_vectors in traffic_data.items():
        try:
            tv = float(tv_str)
        except ValueError:
            continue
        final_blocks = []
        for block_vector in sim_block_vectors:
            if isinstance(block_vector, list) and len(block_vector) > 0:
                final_blocks.append(block_vector[-1])
            elif isinstance(block_vector, (int, float)):
                final_blocks.append(block_vector)
        if final_blocks:
            avg_block = np.mean(final_blocks)
            erlang_values.append(tv)
            blocking_probs.append(avg_block)
    if erlang_values:
        sorted_pairs = sorted(zip(erlang_values, blocking_probs), key=lambda x: x[0])
        sorted_erlangs, sorted_blocks = zip(*sorted_pairs)
        plt.plot(sorted_erlangs, sorted_blocks, marker='o', label=algo_key)
plt.xlabel('Erlang Values')
plt.ylabel('Blocking Probability')
plt.yscale('log')
plt.ylim(10 ** -4, 10 ** -0.5)
plt.title('Blocking Probability vs Erlang Values')
plt.legend()
plt.grid(True)
plt.show()

# --- Plot Average Rewards for Non-Baseline Algorithms ---
# We skip any algorithm labeled "Baselines" or baseline keys in final_result.
for algorithm, sim_time_lists in simulation_times.items():
    if algorithm.lower() == "baselines":
        continue

    # Convert algorithm name to snake case for logs path.
    alg_snake = algorithm.lower().replace(" ", "_")
    plt.figure(figsize=(8, 6))

    # For each simulation time, load the rewards file and attempt to get its traffic volume.
    for sim_time_wrapper in sim_time_lists:
        sim_time = sim_time_wrapper[0]

        reward_file = os.path.join("./logs", alg_snake, "NSFNet", "0227", sim_time, "average_rewards.npy")
        if not os.path.exists(reward_file):
            print(f"Warning: Reward file does not exist: {reward_file}")
            continue

        # Try to extract a traffic volume label from one JSON file in the simulation time folder.
        sim_time_path = os.path.join(base_dir, sim_time)
        traffic_label = None
        if os.path.isdir(sim_time_path):
            for sim_run in os.listdir(sim_time_path):
                run_path = os.path.join(sim_time_path, sim_run)
                if os.path.isdir(run_path):
                    for filename in os.listdir(run_path):
                        if filename.endswith(".json"):
                            parts = filename.split('.')
                            try:
                                traffic_label = str(float(parts[0]))
                            except ValueError:
                                traffic_label = filename.split('.')[0]
                            break
                    if traffic_label is not None:
                        break
        if traffic_label is None:
            traffic_label = sim_time  # fallback if no traffic volume found

        rewards = np.load(reward_file)
        episodes = np.arange(len(rewards))
        plt.plot(episodes, rewards, linewidth=1.5, label=traffic_label)

    plt.xlabel("Number of Episodes")
    plt.ylabel("Reward")
    plt.title(f"Average Rewards for {algorithm}")
    plt.legend(title="Traffic Volume")
    plt.grid(True)
    plt.show()
