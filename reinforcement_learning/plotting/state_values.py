# pylint: disable=unsupported-binary-operation

import math

from pathlib import Path

from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def _plot_matrix(ax, color_matrix, n_nodes):
    """Heat-map helper."""
    ax.imshow(color_matrix, origin="upper")
    ax.set_title("Best Path Matrix", fontsize=16, fontweight="bold")
    ax.set_xlabel("Destination Node", fontsize=14, fontweight="bold")
    ax.set_ylabel("Source Node", fontsize=14, fontweight="bold")
    ticks = np.arange(n_nodes)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    fontsize = 10 if n_nodes <= 24 else 8
    ax.set_xticklabels(ticks, fontsize=fontsize, rotation=90)
    ax.set_yticklabels(ticks, fontsize=fontsize)

    ax.tick_params(axis="x", which="major", pad=1)
    ax.tick_params(axis="y", which="major", pad=1)


def _plot_bar_chart(pair_dict, t_label, algo):
    """Bar-chart helper (path-usage histogram)."""
    max_paths = max((len(vals) for vals in pair_dict.values()), default=0)
    path_counts = np.zeros(max_paths, dtype=int)
    for vals in pair_dict.values():
        if vals:
            path_counts[np.argmax(vals)] += 1

    fig, ax = plt.subplots(figsize=(12, 5), dpi=300)
    ax.bar(np.arange(max_paths), path_counts, edgecolor="black")
    ax.set_xticks(np.arange(max_paths))
    ax.set_xticklabels([f"Path {i}" for i in range(max_paths)], fontsize=12)
    ax.set_xlabel("Path Index", fontsize=14, fontweight="bold")
    ax.set_ylabel("Count", fontsize=14, fontweight="bold")
    ax.set_title("Best Path Count", fontsize=16, fontweight="bold")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

    # mini-legend (algo / traffic)
    ax.legend(
        handles=[
            Patch(facecolor="none", edgecolor="none", label=f"Algorithm: {algo}"),
            Patch(facecolor="none", edgecolor="none", label=f"Traffic: {t_label}"),
        ],
        bbox_to_anchor=(0.5, 0.93),
        loc="upper left",
        fontsize=10,
        borderaxespad=0.5,
        labelspacing=0.4,
    )

    fig.tight_layout(rect=[0, 0, 0.8, 1])
    return fig


def generate_pairwise_legend_labels(path_colors, seen_pairs):
    """Generate a legend for path mismatches using blended colors, no duplicates."""
    legend_handles = [
        Patch(facecolor=(0.9, 0.9, 0.9), edgecolor="black", label="Agreement")
    ]
    unique_pairs = set()
    for a_idx, b_idx in seen_pairs:
        pair = tuple(sorted((a_idx, b_idx)))
        unique_pairs.add(pair)

    for (a_idx, b_idx) in sorted(unique_pairs):
        c_a = np.array(path_colors[a_idx % len(path_colors)])
        c_b = np.array(path_colors[b_idx % len(path_colors)])
        blend = np.clip(0.5 * (c_a + c_b), 0, 1)
        label = f"Path {a_idx + 1} vs Path {b_idx + 1}"
        legend_handles.append(Patch(facecolor=blend, edgecolor="black", label=label))
    return legend_handles


def _compute_colour_matrix(pair_dict, n_nodes, path_colors, negative_factor):
    colour = np.ones((n_nodes, n_nodes, 3))
    global_max = max((max(vals) for vals in pair_dict.values() if vals), default=1e-12)
    best_path_data = {}

    for (src, dst), path_vals in pair_dict.items():
        if not path_vals:
            continue
        best_idx = int(np.argmax(path_vals))
        val = path_vals[best_idx]
        base = path_colors[best_idx % len(path_colors)]
        best_path_data[(src, dst)] = best_idx

        if val == 0:
            colour[src, dst] = (1, 1, 1)
        else:
            scale = abs(val) / global_max
            if val < 0:
                scale *= negative_factor
            colour[src, dst] = (1 - scale) + scale * np.array(base)

    return colour, best_path_data


def _annotate_heatmap(ax, n_nodes):
    ticks = np.arange(n_nodes)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(ticks, fontsize=8, rotation=90)
    ax.set_yticklabels(ticks, fontsize=8)
    ax.set_xlabel("Destination Node", fontsize=10)
    ax.set_ylabel("Source Node", fontsize=10)


def _finalize_figure(fig, handles, title_pos=0.95):
    fig.legend(
        handles=handles,
        title="Best-Path Colour Key",
        loc="upper center",
        ncol=len(handles),
        fontsize=10,
        title_fontsize=11,
        bbox_to_anchor=(0.5, title_pos)
    )
    fig.tight_layout(rect=[0, 0, 1, title_pos - 0.04])


def _compute_colour_matrix(pair_dict, n_nodes, path_colors, negative_factor):
    colour = np.ones((n_nodes, n_nodes, 3))
    global_max = max((max(vals) for vals in pair_dict.values() if vals), default=1e-12)
    best_path_data = {}

    for (src, dst), path_vals in pair_dict.items():
        if not path_vals:
            continue
        best_idx = int(np.argmax(path_vals))
        val = path_vals[best_idx]
        base = path_colors[best_idx % len(path_colors)]
        best_path_data[(src, dst)] = best_idx

        if val == 0:
            colour[src, dst] = (1, 1, 1)
        else:
            scale = abs(val) / global_max
            if val < 0:
                scale *= negative_factor
            colour[src, dst] = (1 - scale) + scale * np.array(base)

    return colour, best_path_data


def _annotate_heatmap(ax, n_nodes):
    ticks = np.arange(n_nodes)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(ticks, fontsize=8, rotation=90)
    ax.set_yticklabels(ticks, fontsize=8)
    ax.set_xlabel("Destination Node", fontsize=10)
    ax.set_ylabel("Source Node", fontsize=10)


def _finalize_figure(fig, handles, title_pos=0.95):
    fig.legend(
        handles=handles,
        title="Best-Path Colour Key",
        loc="upper center",
        ncol=len(handles),
        fontsize=10,
        title_fontsize=11,
        bbox_to_anchor=(0.5, title_pos)
    )
    fig.tight_layout(rect=[0, 0, 1, title_pos - 0.04])


def _save_or_show(fig, algo, save_path, label):
    if save_path:
        p = Path(save_path)
        out_fp = p.parent / f"{p.stem}_{label}_{algo}{p.suffix or '.png'}"
        out_fp.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_fp, bbox_inches="tight")
        print(f"[{label}] ✅ Saved {out_fp}")
    else:
        plt.figure(fig.number)
        plt.show()


def plot_best_path_matrix(
        averaged_state_values_by_volume: dict,
        title: str = "State-Value Heat-maps",
        save_path: str | None = None,
        path_colors: list[tuple] | None = None,
):
    """
    Plots the best path as a histogram.
    """
    if path_colors is None:
        path_colors = [
            (1, 0, 0), (0, 1, 0), (0, 0, 1),
            (1, 1, 0), (1, 0, 1), (0, 1, 1),
            (0.5, 0.5, 0.5),
        ]

    plt.style.use("seaborn-whitegrid" if "seaborn-whitegrid" in plt.style.available else "default")
    created_figs = []
    negative_factor = 0.5
    best_path_data = defaultdict(lambda: defaultdict(dict))

    for algo, vol_dict in averaged_state_values_by_volume.items():
        _figs, best_data = _plot_algo_heatmaps(algo, vol_dict, title, path_colors, negative_factor, save_path)
        created_figs.extend(_figs)
        best_path_data[algo] = best_data

    created_figs.extend(_plot_diff_matrices(best_path_data, path_colors, save_path))
    return created_figs


def _plot_algo_heatmaps(algo, vol_dict, title, path_colors, negative_factor, save_path):
    t_labels = sorted(vol_dict.keys(), key=float)
    ncols = min(3, len(t_labels))
    nrows = math.ceil(len(t_labels) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.8 * nrows), dpi=150, squeeze=False)
    fig.suptitle(f"{title}: {algo}", fontsize=18, fontweight="bold")

    best_path_data = {}
    for idx, t_label in enumerate(t_labels):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        pair_dict = vol_dict[t_label]
        nodes = {n for (s, d) in pair_dict for n in (s, d)}
        n_nodes = max(nodes) + 1 if nodes else 1
        colour, best_map = _compute_colour_matrix(pair_dict, n_nodes, path_colors, negative_factor)
        best_path_data[t_label] = best_map
        ax.imshow(colour, origin="upper")
        ax.set_title(f"Traffic = {t_label}", fontsize=12, fontweight="bold")
        _annotate_heatmap(ax, n_nodes)

    for j in range(idx + 1, nrows * ncols):  # pylint: disable=undefined-loop-variable
        r, c = divmod(j, ncols)
        axes[r][c].axis("off")

    max_path_idx = max((int(np.argmax(vals)) for v in vol_dict.values() for vals in v.values() if vals), default=0)
    handles = [
        Patch(facecolor=path_colors[i % len(path_colors)], edgecolor="black", label=f"Path {i + 1}")
        for i in range(max_path_idx + 1)
    ]
    _finalize_figure(fig, handles)
    _save_or_show(fig, algo, save_path, "state_vals")
    plt.close(fig)
    return [fig], best_path_data


def _plot_diff_matrices(best_path_data, path_colors, save_path):
    created_figs = []
    desired_pairs = [
        ("epsilon_greedy_bandit", "ucb_bandit"),
        ("epsilon_greedy_bandit", "q_learning"),
        ("q_learning", "ucb_bandit"),
    ]
    for algo_a, algo_b in desired_pairs:
        if algo_a not in best_path_data or algo_b not in best_path_data:
            print(f"❌ Skipping missing pair: {algo_a} vs {algo_b}")
            continue

        traffic_levels = sorted(set(best_path_data[algo_a]) & set(best_path_data[algo_b]), key=float)
        if not traffic_levels:
            print(f"⚠️ No shared traffic levels between {algo_a} and {algo_b}")
            continue

        ncols = min(3, len(traffic_levels))
        nrows = math.ceil(len(traffic_levels) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.8 * nrows), dpi=150)
        axes = axes.reshape(nrows, ncols)
        fig.suptitle(f"Best Path Differences: {algo_a} vs. {algo_b}", fontsize=18, fontweight="bold")
        seen_pairs = set()

        for idx, traffic in enumerate(traffic_levels):
            r, c = divmod(idx, ncols)
            ax = axes[r][c]
            a_map = best_path_data[algo_a][traffic]
            b_map = best_path_data[algo_b][traffic]
            nodes = {n for (s, d) in a_map for n in (s, d)}
            n_nodes = max(nodes) + 1 if nodes else 1
            colour = np.ones((n_nodes, n_nodes, 3))

            for (src, dst), a_idx in a_map.items():
                b_idx = b_map.get((src, dst))
                if b_idx is None:
                    continue
                if a_idx == b_idx:
                    colour[src, dst] = (0.9, 0.9, 0.9)
                else:
                    seen_pairs.add((a_idx, b_idx))
                    c_a = np.array(path_colors[a_idx % len(path_colors)])
                    c_b = np.array(path_colors[b_idx % len(path_colors)])
                    colour[src, dst] = np.clip(0.5 * (c_a + c_b), 0, 1)

            ax.imshow(colour, origin="upper")
            ax.set_title(f"Traffic = {traffic}", fontsize=12, fontweight="bold")
            _annotate_heatmap(ax, n_nodes)

        for j in range(idx + 1, nrows * ncols):  # pylint: disable=undefined-loop-variable
            r, c = divmod(j, ncols)
            axes[r][c].axis("off")

        fig.tight_layout(rect=[0, 0, 1, 0.93])
        handles = generate_pairwise_legend_labels(path_colors, seen_pairs)
        fig.legend(
            handles=handles,
            title="Path Selection Difference",
            loc="upper center",
            ncol=3,
            fontsize=10,
            title_fontsize=11,
            bbox_to_anchor=(0.5, 0.97)
        )
        _save_or_show(fig, f"{algo_a}_vs_{algo_b}", save_path, "diff_plot")
        plt.close(fig)
        created_figs.append(fig)

    return created_figs
