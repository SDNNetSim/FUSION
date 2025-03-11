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

    # Increase figure width, e.g. 8 wide and 4 tall
    plt.figure(figsize=(8, 4), dpi=200)

    # Plot one line per algorithm
    for algorithm, traffic_dict in memory_usage_data.items():
        y_vals = []
        for tvol in sorted_tvols:
            mem_array = traffic_dict.get(tvol, None)
            # Check for None or empty array
            if mem_array is None or len(mem_array) == 0:
                y_vals.append(0.0)
            else:
                y_vals.append(np.mean(mem_array))

        plt.plot(x_vals, y_vals, marker='o', label=algorithm)

    # Set the tick positions and labels
    plt.xticks(x_vals, sorted_tvols, rotation=45, ha='right')

    plt.xlabel("Traffic Volume")
    plt.ylabel("Memory Usage (MB)")
    plt.title("Memory Usage by Traffic Volume")
    plt.legend(title="Algorithm")
    plt.grid(True)

    # Use tight_layout to minimize label overlap
    plt.tight_layout()
    plt.show()
