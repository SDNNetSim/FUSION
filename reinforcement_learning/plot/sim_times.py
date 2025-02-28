import os
import json
from json import JSONDecodeError
import datetime
import matplotlib.pyplot as plt


def parse_time_string(time_str):
    """Parse a time string into a datetime object."""
    parts = time_str.split('_')
    if len(parts) == 4:
        time_str += '_000000'
    fmt = "%m%d_%H_%M_%S_%f"
    return datetime.datetime.strptime(time_str, fmt)


def collect_simulation_durations_for_algo(sim_list, base_dir):
    """Collect total simulation durations per traffic volume for one algorithm."""
    durations_by_traffic = {}
    parent_date = os.path.basename(base_dir)
    for sim_entry in sim_list:
        sim_time = sim_entry[0]
        sim_start_str = sim_time if sim_time.startswith(parent_date) else f"{parent_date}_{sim_time}"
        try:
            sim_start = parse_time_string(sim_start_str)
        except ValueError as exc:
            print(f"Error parsing simulation start {sim_start_str}: {exc}")
            continue

        sim_folder = os.path.join(base_dir, sim_time)
        if not os.path.isdir(sim_folder):
            sim_folder = os.path.join(base_dir, f"{parent_date}_{sim_time}")
            if not os.path.isdir(sim_folder):
                print(f"No folder found for {sim_time}")
                continue

        end_times_for_this_sim_time = {}
        for seed_dir in os.listdir(sim_folder):
            seed_path = os.path.join(sim_folder, seed_dir)
            if not os.path.isdir(seed_path):
                continue
            json_files = [f for f in os.listdir(seed_path) if f.endswith("erlang.json")]
            for jfile in json_files:
                traffic_volume = jfile.split('_')[0]
                file_path = os.path.join(seed_path, jfile)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f_obj:
                        data = json.load(f_obj)
                except (OSError, JSONDecodeError) as exc:
                    print(f"Error reading/parsing {file_path}: {exc}")
                    continue

                sim_end_time_str = data.get("sim_end_time")
                if not sim_end_time_str:
                    print(f"No sim_end_time in {file_path}")
                    continue
                try:
                    sim_end = parse_time_string(sim_end_time_str)
                except ValueError as exc:
                    print(f"Error parsing sim_end_time in {file_path}: {exc}")
                    continue

                end_times_for_this_sim_time.setdefault(traffic_volume, []).append(sim_end)

        for tvol, end_time_list in end_times_for_this_sim_time.items():
            if not end_time_list:
                continue
            final_end_time = max(end_time_list)
            total_duration = (final_end_time - sim_start).total_seconds()
            durations_by_traffic.setdefault(tvol, []).append(total_duration)
    return durations_by_traffic


def plot_simulation_times_for_algo(durations_by_traffic, algorithm):
    """Plot total simulation durations by traffic volume with the algorithm name in the title."""
    traffic_labels = sorted(durations_by_traffic.keys(), key=float)
    data = [durations_by_traffic[t] for t in traffic_labels]
    plt.figure(figsize=(6, 4), dpi=200)
    plt.boxplot(data, labels=traffic_labels, showmeans=True)
    plt.xlabel("Traffic Volume")
    plt.ylabel("Time Taken (seconds)")
    plt.title(f"Simulation Duration by Traffic Volume for {algorithm}")
    plt.grid(True)
    plt.show()


def plot_sim_times(simulation_times, base_dir):
    """Collect and plot total simulation durations per algorithm from simulation_times and base_dir."""
    for algorithm, sim_list in simulation_times.items():
        durations = collect_simulation_durations_for_algo(sim_list, base_dir)
        if not durations:
            print(f"No simulation duration data collected for {algorithm}.")
            continue
        plot_simulation_times_for_algo(durations, algorithm)
