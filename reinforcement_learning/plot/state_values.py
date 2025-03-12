import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def _plot_matrix(axes, color_matrix, n_nodes):
    """
    Plots a matrix of color-coded best-path values for a given algorithm and traffic label.
    """
    axes.imshow(color_matrix, origin='upper')
    axes.set_title("Best Path Matrix", fontsize=16, fontweight='bold')
    axes.set_xlabel("Destination Node", fontsize=14, fontweight='bold')
    axes.set_ylabel("Source Node", fontsize=14, fontweight='bold')

    ticks = np.arange(n_nodes)
    axes.set_xticks(ticks)
    axes.set_yticks(ticks)
    axes.set_xticklabels(ticks, fontsize=12)
    axes.set_yticklabels(ticks, fontsize=12)
    axes.tick_params(axis='both', which='major', pad=5)


def _plot_bar_chart(pair_dict, t_label, algo):
    """
    Plots a bar chart of the best-path usage count for a given algorithm and traffic label.
    The title is short, and the algorithm/traffic info is in an external legend.
    """
    max_paths = max((len(vals) for vals in pair_dict.values()), default=0)
    path_counts = np.zeros(max_paths, dtype=int)
    for vals in pair_dict.values():
        if vals:
            path_counts[np.argmax(vals)] += 1

    fig, ax = plt.subplots(figsize=(12, 5), dpi=300)
    ax.bar(np.arange(max_paths), path_counts, edgecolor='black')
    ax.set_xticks(np.arange(max_paths))
    ax.set_xticklabels([f"Path {i}" for i in range(max_paths)], fontsize=12)
    ax.set_xlabel("Path Index", fontsize=14, fontweight='bold')
    ax.set_ylabel("Count", fontsize=14, fontweight='bold')
    ax.set_title("Best Path Count", fontsize=16, fontweight='bold')
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    patches = [
        Patch(facecolor='none', edgecolor='none',
              label=rf"$\mathbf{{Algorithm:}}$ {algo}"),
        Patch(facecolor='none', edgecolor='none',
              label=rf"$\mathbf{{Traffic:}}$ {t_label}")
    ]
    ax.legend(
        handles=patches,
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
        fontsize=10,
        borderaxespad=0.5,
        labelspacing=0.5
    )
    fig.tight_layout(rect=[0, 0, 0.8, 1])
    plt.show()


def plot_best_path_matrix(averaged_state_values_by_volume, path_colors=None):
    """
    For each algorithm in averaged_state_values_by_volume, and each traffic label,
    plots:
      1) A matrix of best-path values (colored heatmap).
      2) A bar chart of best-path usage counts.
    """
    if path_colors is None:
        path_colors = [
            (1, 0, 0), (0, 1, 0), (0, 0, 1),
            (1, 1, 0), (1, 0, 1), (0, 1, 1),
            (0.5, 0.5, 0.5)
        ]
    available_styles = plt.style.available
    if 'seaborn-whitegrid' in available_styles:
        plt.style.use('seaborn-whitegrid')
    elif 'seaborn-white' in available_styles:
        plt.style.use('seaborn-white')
    else:
        plt.style.use('default')

    for algo, volumes_dict in averaged_state_values_by_volume.items():
        for t_label, pair_dict in volumes_dict.items():
            all_nodes = {node for (src, dst) in pair_dict.keys() for node in (src, dst)}
            if not all_nodes:
                continue

            n_nodes = max(all_nodes) + 1
            global_max_val = max(
                (max(vals) for vals in pair_dict.values() if vals),
                default=1e-12
            )

            color_matrix = np.ones((n_nodes, n_nodes, 3), dtype=float)
            for (src, dst), path_vals in pair_dict.items():
                if not path_vals:
                    continue
                best_path_idx = np.argmax(path_vals)
                scale = path_vals[best_path_idx] / global_max_val
                base_color = (
                    path_colors[best_path_idx]
                    if best_path_idx < len(path_colors)
                    else (1, 1, 1)
                )
                color_matrix[src, dst] = (
                        (1 - scale) * np.array([1, 1, 1])
                        + scale * np.array(base_color)
                )

            fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
            _plot_matrix(ax, color_matrix, n_nodes)
            max_paths = max((len(vals) for vals in pair_dict.values()), default=0)
            patches = [
                Patch(facecolor='none', edgecolor='none',
                      label=rf"$\mathbf{{Algorithm:}}$ {algo}"),
                Patch(facecolor='none', edgecolor='none',
                      label=rf"$\mathbf{{Traffic:}}$ {t_label}")
            ]
            for i in range(max_paths):
                if i < len(path_colors):
                    patches.append(Patch(facecolor=path_colors[i], label=f"Path {i}"))
                else:
                    patches.append(Patch(facecolor=(1, 1, 1), label=f"Path {i}"))

            ax.legend(
                handles=patches,
                title="Path Colors",
                bbox_to_anchor=(1.01, 1),
                loc="upper left",
                fontsize=10,
                title_fontsize=10,
                handlelength=1.0,
                handletextpad=0.5,
                labelspacing=0.5,
                borderaxespad=0.5
            )

            fig.tight_layout(rect=[0.1, 0, 0.8, 1])
            plt.show()
            _plot_bar_chart(pair_dict, t_label, algo)
