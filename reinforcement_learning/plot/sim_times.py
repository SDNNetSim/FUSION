import os
import json
from json import JSONDecodeError
import datetime
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


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


def style_boxplot_elements(bp, box_algos, algo_colors):
    """
    Colorize the boxplot elements using the provided mappings.
    """
    whiskers_iter = iter(bp['whiskers'])
    caps_iter = iter(bp['caps'])
    for i, box in enumerate(bp['boxes']):
        algo = box_algos[i]
        c = algo_colors[algo]
        box.set_facecolor(c)
        box.set_edgecolor(c)
        bp['medians'][i].set_color(c)
        bp['means'][i].set(markerfacecolor=c, markeredgecolor='black')
        bp['fliers'][i].set(markerfacecolor=c, markeredgecolor=c)
        for _ in range(2):
            next(whiskers_iter).set_color(c)
        for _ in range(2):
            next(caps_iter).set_color(c)


def plot_sim_times(simulation_times, base_dir):
    """
    Shows a single figure with grouped boxplots:
    each traffic volume on the x-axis, one box per algorithm in each group,
    with correct color coding and a matching legend.
    """
    if not simulation_times:
        return

    sim_times_dict = {}
    all_algorithms = set()
    for algorithm, sim_list in simulation_times.items():
        durations_by_label = collect_simulation_durations_for_algo(sim_list, base_dir)
        for label, dlist in durations_by_label.items():
            sim_times_dict.setdefault(label, {})[algorithm] = dlist
        all_algorithms.add(algorithm)
    all_algorithms = sorted(all_algorithms)
    traffic_labels = sorted(sim_times_dict.keys(), key=float)

    data_for_boxplot = []
    box_algos = []
    positions = []
    group_width = 0.8
    box_width = group_width / len(all_algorithms)
    current_center = 1
    label_positions = []

    for label in traffic_labels:
        for i, algo in enumerate(all_algorithms):
            durations_list = sim_times_dict[label].get(algo, [])
            data_for_boxplot.append(durations_list)
            box_algos.append(algo)
            pos = current_center - group_width / 2 + (i + 0.5) * box_width
            positions.append(pos)
        label_positions.append(current_center)
        current_center += 2

    plt.figure(figsize=(10, 5), dpi=200)
    bp = plt.boxplot(
        data_for_boxplot,
        positions=positions,
        widths=box_width,
        showmeans=True,
        patch_artist=True
    )

    plt.xticks(label_positions, traffic_labels, rotation=45, ha='right')
    plt.xlabel("Traffic Volume")
    plt.ylabel("Simulation Time (seconds)")
    plt.title("Grouped Boxplot of Simulation Times (All Algorithms)")
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    algo_colors = {algo: colors[i % len(colors)] for i, algo in enumerate(all_algorithms)}
    style_boxplot_elements(bp, box_algos, algo_colors)

    legend_elements = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor=algo_colors[algo], label=algo)
        for algo in all_algorithms
    ]
    plt.legend(handles=legend_elements, title="Algorithm")
    plt.tight_layout()
    plt.show()
