import numpy as np
import matplotlib.pyplot as plt


def plot_memory_usage(memory_usage_data):
    """
    Creates a grouped bar chart of average memory usage:
      - x-axis: traffic volumes (grouped)
      - bars: one per algorithm, showing the mean memory usage
    """
    all_traffic_labels = set()
    for traffic_dict in memory_usage_data.values():
        all_traffic_labels.update(traffic_dict.keys())
    traffic_labels = sorted(all_traffic_labels, key=float)
    all_algorithms = sorted(memory_usage_data.keys())

    means_matrix = []
    for tvol in traffic_labels:
        row = []
        for algo in all_algorithms:
            mem_array = memory_usage_data[algo].get(tvol, None)
            if mem_array is None or len(mem_array) == 0:
                row.append(0.0)
            else:
                row.append(np.mean(mem_array))
        means_matrix.append(row)
    means_matrix = np.array(means_matrix)

    x = np.arange(len(traffic_labels))
    bar_width = 0.8 / len(all_algorithms)
    plt.figure(figsize=(14, 6), dpi=300)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    algo_colors = {algo: colors[i % len(colors)] for i, algo in enumerate(all_algorithms)}

    for i, algo in enumerate(all_algorithms):
        offset = i * bar_width
        plt.bar(
            x + offset,
            means_matrix[:, i],
            bar_width,
            label=algo,
            color=algo_colors[algo],
            alpha=0.9
        )
    plt.xticks(
        x + bar_width * (len(all_algorithms) / 2 - 0.5),
        traffic_labels,
        rotation=45, ha='right', fontsize=12
    )
    plt.xlabel("Traffic Volume", fontsize=14, fontweight='bold')
    plt.ylabel("Mean Memory Usage (MB)", fontsize=14, fontweight='bold')
    plt.title("Memory Usage by Traffic Volume", fontsize=16, fontweight='bold')

    plt.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.legend(
        title="Algorithm",
        fontsize=12,
        title_fontsize=12,
        bbox_to_anchor=(1.05, 1),
        loc="upper left"
    )
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()
