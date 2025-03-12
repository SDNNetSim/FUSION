import numpy as np
import matplotlib.pyplot as plt


def plot_memory_usage(memory_usage_data):
    """
    Plot average memory usage for each traffic volume on the x-axis,
    with one line per algorithm in the legend.
    """
    available_styles = plt.style.available
    if 'seaborn-whitegrid' in available_styles:
        plt.style.use('seaborn-whitegrid')
    elif 'seaborn-white' in available_styles:
        plt.style.use('seaborn-white')
    else:
        plt.style.use('default')

    all_traffic_labels = set()
    for traffic_dict in memory_usage_data.values():
        all_traffic_labels.update(traffic_dict.keys())

    sorted_tvols = sorted(all_traffic_labels, key=float)
    x_vals = [float(tv) for tv in sorted_tvols]

    plt.figure(figsize=(10, 6), dpi=300)
    for algorithm, traffic_dict in memory_usage_data.items():
        y_vals = []
        for tvol in sorted_tvols:
            mem_array = traffic_dict.get(tvol, None)
            if mem_array is None or len(mem_array) == 0:
                y_vals.append(0.0)
            else:
                y_vals.append(np.mean(mem_array))
        plt.plot(x_vals, y_vals, marker='o', linewidth=2, markersize=6, label=algorithm)

    plt.xticks(x_vals, sorted_tvols, rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel("Traffic Volume", fontsize=14, fontweight='bold')
    plt.ylabel("Memory Usage (MB)", fontsize=14, fontweight='bold')
    plt.title("Memory Usage by Traffic Volume", fontsize=16, fontweight='bold')
    plt.legend(title="Algorithm", fontsize=12, title_fontsize=12,
               bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, which="both", linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()
