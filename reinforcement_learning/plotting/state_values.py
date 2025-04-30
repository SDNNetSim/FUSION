from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# ─────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────────────────
def _plot_matrix(ax, color_matrix, n_nodes):
    """Heat-map helper."""
    ax.imshow(color_matrix, origin="upper")
    ax.set_title("Best Path Matrix", fontsize=16, fontweight="bold")
    ax.set_xlabel("Destination Node", fontsize=14, fontweight="bold")
    ax.set_ylabel("Source Node", fontsize=14, fontweight="bold")
    ticks = np.arange(n_nodes)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(ticks, fontsize=12)
    ax.set_yticklabels(ticks, fontsize=12)
    ax.tick_params(axis="both", which="major", pad=5)


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
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
        fontsize=10,
        borderaxespad=0.5,
        labelspacing=0.4,
    )

    fig.tight_layout(rect=[0, 0, 0.8, 1])
    return fig


# ─────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────
def plot_best_path_matrix(
        averaged_state_values_by_volume: dict,
        path_colors: list[tuple] | None = None,
        save_dir: str | None = None,
):
    """
    Produce, for every algorithm / traffic-volume pair:

    1. A heat-map showing the "best path" choice **per (src,dst)** pair
    2. A bar-chart summarising which path index is chosen most often.

    Parameters
    ----------
    averaged_state_values_by_volume : dict
        See input format description above.
    path_colors : list[tuple] | None
        RGB triples to colour the individual paths.  Defaults to 7 distinct colours.
    save_dir : str | None
        If supplied, each figure is saved into this directory
        (created if it doesn't exist).  File names:
        ``{algo}_{traffic}_matrix.png`` and ``{algo}_{traffic}_bars.png``

    Returns
    -------
    list[matplotlib.figure.Figure]
        All created figures, in order (matrix then bar for each pair).
    """
    # ─── style ────────────────────────────────────────────
    plt.style.use(
        "seaborn-whitegrid" if "seaborn-whitegrid" in plt.style.available else "default"
    )

    if path_colors is None:
        path_colors = [
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 1),
            (1, 1, 0),
            (1, 0, 1),
            (0, 1, 1),
            (0.5, 0.5, 0.5),
        ]

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    created_figs = []
    negative_factor = 0.5

    # ─── iterate over algo / traffic ─────────────────────
    for algo, vol_dict in averaged_state_values_by_volume.items():
        for t_label, pair_dict in vol_dict.items():
            if not pair_dict:
                continue

            # ---- data prep ---------------------------------------------------
            nodes = {n for (src, dst) in pair_dict for n in (src, dst)}
            n_nodes = max(nodes) + 1
            global_max = max((max(vals) for vals in pair_dict.values() if vals), 1e-12)
            color_matrix = np.ones((n_nodes, n_nodes, 3))

            for (src, dst), path_vals in pair_dict.items():
                if not path_vals:
                    continue
                best_idx = int(np.argmax(path_vals))
                val = path_vals[best_idx]
                base_color = path_colors[best_idx % len(path_colors)]

                if val == 0:
                    color_matrix[src, dst] = (1, 1, 1)
                else:
                    scale = abs(val) / global_max
                    if val < 0:
                        scale *= negative_factor
                    color_matrix[src, dst] = (1 - scale) + scale * np.array(base_color)

            # ---- heat-map ----------------------------------------------------
            fig_matrix, ax_matrix = plt.subplots(figsize=(10, 8), dpi=300)
            _plot_matrix(ax_matrix, color_matrix, n_nodes)

            # legend (path-colour swatches + algo/traffic)
            legend_handles = [
                Patch(facecolor="none", edgecolor="none", label=f"Algorithm: {algo}"),
                Patch(facecolor="none", edgecolor="none", label=f"Traffic: {t_label}"),
            ]
            for i in range(max(len(p) for p in pair_dict.values())):
                legend_handles.append(
                    Patch(
                        facecolor=path_colors[i % len(path_colors)],
                        label=f"Path {i}",
                    )
                )

            ax_matrix.legend(
                handles=legend_handles,
                title="Legend",
                bbox_to_anchor=(1.01, 1),
                loc="upper left",
                fontsize=10,
                title_fontsize=10,
                borderaxespad=0.5,
                labelspacing=0.5,
            )
            fig_matrix.tight_layout(rect=[0.05, 0, 0.8, 1])

            if save_dir:
                fig_matrix.savefig(
                    Path(save_dir) / f"{algo}_{t_label}_matrix.png",
                    bbox_inches="tight",
                )

            created_figs.append(fig_matrix)

            # ---- bar-chart ----------------------------------------------------
            fig_bar = _plot_bar_chart(pair_dict, t_label, algo)
            if save_dir:
                fig_bar.savefig(
                    Path(save_dir) / f"{algo}_{t_label}_bars.png",
                    bbox_inches="tight",
                )
            created_figs.append(fig_bar)

    return created_figs
