import numpy as np
import matplotlib.pyplot as plt


def plot_memory_usage(memory_usage_data):
    """
    Plot average memory usage for each traffic volume on the x-axis,
    with one line per algorithm in the legend.
    """
    # Collect all traffic volumes across all algorithms
    all_traffic_labels = set()
    for traffic_dict in memory_usage_data.values():
        all_traffic_labels.update(traffic_dict.keys())

    # Sort traffic volumes numerically
    sorted_tvols = sorted(all_traffic_labels, key=float)
    x_vals = [float(tv) for tv in sorted_tvols]

    plt.figure(figsize=(6, 4), dpi=200)

    # Plot one line per algorithm
    for algorithm, traffic_dict in memory_usage_data.items():
        y_vals = []
        for tvol in sorted_tvols:
            mem_array = traffic_dict.get(tvol, None)
            if mem_array is None or len(mem_array) == 0:
                y_vals.append(0.0)
            else:
                y_vals.append(np.mean(mem_array))
        plt.plot(x_vals, y_vals, marker='o', label=algorithm)

    plt.xticks(x_vals, sorted_tvols)
    plt.xlabel("Traffic Volume")
    plt.ylabel("Memory Usage (MB)")
    plt.title("Memory Usage by Traffic Volume")
    plt.legend(title="Algorithm")
    plt.grid(True)
    plt.show()
