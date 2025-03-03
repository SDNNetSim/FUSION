#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def _plot_matrix(axes, color_matrix, n_nodes, t_label, path_colors, pair_dict, global_max_val):
    axes.imshow(color_matrix, origin='upper')
    axes.set_title(f"Best Path Matrix (Traffic: {t_label})")
    axes.set_xlabel("Destination Node")
    axes.set_ylabel("Source Node")
    ticks = np.arange(n_nodes)
    axes.set_xticks(ticks)
    axes.set_yticks(ticks)
    axes.set_xticklabels(ticks)
    axes.set_yticklabels(ticks)

    max_paths = max((len(vals) for vals in pair_dict.values()), default=0)
    patches = [Patch(facecolor=path_colors[i] if i < len(path_colors) else (1, 1, 1), label=f"Path {i}")
               for i in range(max_paths)]
    axes.legend(handles=patches, title="Path Colors", bbox_to_anchor=(1.05, 1), loc="upper left")


def _plot_bar_chart(pair_dict, t_label):
    max_paths = max((len(vals) for vals in pair_dict.values()), default=0)
    path_counts = np.zeros(max_paths, dtype=int)
    for vals in pair_dict.values():
        if vals:
            path_counts[np.argmax(vals)] += 1
    plt.figure(figsize=(6, 4), dpi=200)
    plt.bar(np.arange(max_paths), path_counts, color='gray')
    plt.xticks(np.arange(max_paths), [f"Path {i}" for i in range(max_paths)])
    plt.xlabel("Path Index")
    plt.ylabel("Count")
    plt.title(f"Best Path Count (Traffic: {t_label})")
    plt.tight_layout()
    plt.show()


def plot_best_path_matrix(averaged_state_values_by_volume, path_colors=None):
    """
    Plots a heat map of state-values for best paths.
    """
    if path_colors is None:
        path_colors = [
            (1, 0, 0), (0, 1, 0), (0, 0, 1),
            (1, 1, 0), (1, 0, 1), (0, 1, 1),
            (0.5, 0.5, 0.5)
        ]
    for t_label, pair_dict in averaged_state_values_by_volume.items():
        all_nodes = {node for (src, dst) in pair_dict.keys() for node in (src, dst)}
        if not all_nodes:
            continue
        n_nodes = max(all_nodes) + 1
        global_max_val = max((max(vals) for vals in pair_dict.values() if vals), default=1e-12)
        color_matrix = np.ones((n_nodes, n_nodes, 3), dtype=float)
        for (src, dst), path_vals in pair_dict.items():
            if not path_vals:
                continue
            best_path_idx = np.argmax(path_vals)
            scale = path_vals[best_path_idx] / global_max_val
            base_color = path_colors[best_path_idx] if best_path_idx < len(path_colors) else (1, 1, 1)
            color_matrix[src, dst] = (1 - scale) * np.array([1, 1, 1]) + scale * np.array(base_color)
        fig, ax = plt.subplots(figsize=(max(6, n_nodes * 0.5), max(6, n_nodes * 0.5)), dpi=150)
        _plot_matrix(ax, color_matrix, n_nodes, t_label, path_colors, pair_dict, global_max_val)
        fig.tight_layout()
        plt.show()
        _plot_bar_chart(pair_dict, t_label)
