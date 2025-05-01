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
        bbox_to_anchor=(0.5, 0.93),
        loc="upper left",
        fontsize=10,
        borderaxespad=0.5,
        labelspacing=0.4,
    )

    fig.tight_layout(rect=[0, 0, 0.8, 1])
    return fig


def plot_best_path_matrix(
        averaged_state_values_by_volume: dict,
        title: str = "State-Value Heat-maps",
        save_path: str | None = None,
        path_colors: list[tuple] | None = None,
):
    """
    Create ONE figure per algorithm containing a grid of heat-maps
    (one traffic volume per subplot).

    averaged_state_values_by_volume
        {algo: {traffic: {(src,dst): [vals]}}}
    """
    plt.style.use(
        "seaborn-whitegrid" if "seaborn-whitegrid" in plt.style.available else "default"
    )

    if path_colors is None:
        path_colors = [
            (1, 0, 0), (0, 1, 0), (0, 0, 1),
            (1, 1, 0), (1, 0, 1), (0, 1, 1),
            (0.5, 0.5, 0.5),
        ]
    created_figs = []
    negative_factor = 0.5

    for algo, vol_dict in averaged_state_values_by_volume.items():
        # -------------- figure & grid layout -----------------
        t_labels = sorted(vol_dict.keys(), key=float)
        n_vol = len(t_labels)
        ncols = min(3, n_vol)  # up to 3 heat-maps per row
        nrows = int(np.ceil(n_vol / ncols))

        fig_width = 5.5 * ncols + 1  # wider  ➜ 5.5 in per column
        fig_height = 5.0 * nrows + 1  # taller ➜ 5.0 in per row
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(fig_width, fig_height),
            dpi=300,
            squeeze=False
        )
        fig.suptitle(f"{title}: {algo}", fontsize=18, fontweight="bold")

        # -------------- loop over traffic volumes ------------
        for idx, t_label in enumerate(t_labels):
            r, c = divmod(idx, ncols)
            ax = axes[r][c]
            pair_dict = vol_dict[t_label]

            # Build colour matrix --------------------------------
            nodes = {n for (s, d) in pair_dict for n in (s, d)}
            n_nodes = max(nodes) + 1 if nodes else 1
            colour = np.ones((n_nodes, n_nodes, 3))

            global_max = max(
                (max(vals) for vals in pair_dict.values() if vals),
                default=1e-12
            )

            for (src, dst), path_vals in pair_dict.items():
                if not path_vals:
                    continue
                best_idx = int(np.argmax(path_vals))
                val = path_vals[best_idx]
                base = path_colors[best_idx % len(path_colors)]

                if val == 0:
                    colour[src, dst] = (1, 1, 1)
                else:
                    scale = abs(val) / global_max
                    if val < 0:
                        scale *= negative_factor
                    colour[src, dst] = (1 - scale) + scale * np.array(base)

            # Draw heat-map --------------------------------------
            _plot_matrix(ax, colour, n_nodes)
            ax.set_title(f"Traffic = {t_label}", fontsize=12, fontweight="bold")

        # hide empty cells
        for j in range(idx + 1, nrows * ncols):
            r, c = divmod(j, ncols)
            axes[r][c].axis("off")

        fig.tight_layout(rect=[0, 0, 1, 0.94])
        # -------- legend that maps colour → path index ----------------
        # discover highest path index used across all volumes
        max_path_idx = 0
        for pair_dict in vol_dict.values():
            for vals in pair_dict.values():
                if vals:
                    max_path_idx = max(max_path_idx, int(np.argmax(vals)))

        handles = [
            plt.matplotlib.patches.Patch(
                facecolor=path_colors[i % len(path_colors)],
                edgecolor="black",
                label=f"Path {i + 1}"
            )
            for i in range(max_path_idx + 1)
        ]

        fig.legend(
            handles=handles,
            title="Best-Path Colour Key",
            loc="upper center",
            ncol=len(handles),
            fontsize=10,
            title_fontsize=11,
            bbox_to_anchor=(0.5, 0.95)  # ↓ push legend down
        )
        fig.tight_layout(rect=[0, 0, 1, 0.91])  # reserve room at top

        # -------- save with <stem>_<algo>.png ------------------
        if save_path:
            p = Path(save_path)
            stem = p.stem
            parent = p.parent
            suff = p.suffix or ".png"
            out_fp = parent / f"{stem}_{algo}{suff}"
            out_fp.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_fp, bbox_inches="tight")
            print(f"[state_vals] ✅ Saved {out_fp}")

        created_figs.append(fig)

    return created_figs
