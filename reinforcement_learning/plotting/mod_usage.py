from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BASE_FONT_SIZE = 11
TICK_FONT_SIZE = 8
TITLE_FONT_SIZE = 14
SUBPLOT_TITLE_SZ = 9
MOD_ORDER = ["QPSK", "16-QAM", "64-QAM"]
MOD_COLORS = {"QPSK": "tab:blue",
              "16-QAM": "tab:orange",
              "64-QAM": "tab:green"}


def _auto_bar_width(bws: list[float]) -> float:
    """60 % of the minimum gap between consecutive bandwidths (≥ 8 Gb/s)."""
    if len(bws) < 2:
        return 8.0
    gaps = np.diff(sorted(bws))
    return max(0.6 * gaps.min(), 8.0)


def _apply_axes_style(ax):
    """Match global professional style: light grid, muted spines, small ticks."""
    ax.grid(axis="y", color="0.5", alpha=0.2, linewidth=0.8)
    ax.grid(axis="x", visible=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("0.5")
        ax.spines[spine].set_linewidth(0.8)
    ax.tick_params(axis="both", labelsize=TICK_FONT_SIZE, width=0.8, length=3)


def plot_modulation_usage(data, save_path=None, title="Modulation Trends vs. Traffic Volume"):
    all_algos = sorted(data.keys())
    all_tvs = sorted({float(tv) for algo in data.values() for tv in algo})
    all_tvs_str = [str(tv) for tv in all_tvs]

    fig, axes = plt.subplots(
        1, len(MOD_ORDER),
        figsize=(17.5, 5),  # increased width
        dpi=300,
        sharex=True, sharey=True,
        gridspec_kw={'width_ratios': [1] * len(MOD_ORDER)}
    )

    if len(MOD_ORDER) == 1:
        axes = [axes]

    algo_lines = {}

    for mod_idx, mod in enumerate(MOD_ORDER):
        ax = axes[mod_idx]

        for algo in all_algos:
            mod_props = []
            for tv_str in all_tvs_str:
                bw_vals = data[algo].get(tv_str, {})
                all_counts = sum(
                    sum(bw_dict.values()) for bw_dict in bw_vals.values()
                )
                mod_counts = sum(
                    bw_dict.get(mod, 0) for bw_dict in bw_vals.values()
                )
                prop = (mod_counts / all_counts) if all_counts > 0 else 0
                mod_props.append(prop)

            line, = ax.plot(
                all_tvs,
                mod_props,
                linewidth=2,
                marker='o',
                label=algo if mod_idx == 0 else None
            )

            if mod_idx == 0:
                algo_lines[algo] = line

        ax.set_title(f"{mod}", fontsize=12, fontweight="bold")
        ax.set_ylim(0, 1.0)
        ax.set_yticks(np.linspace(0, 1.0, 6))
        _apply_axes_style(ax)

        if mod_idx == 0:
            ax.set_ylabel("Proportion of Requests", fontweight="bold")
        ax.set_xlabel("Traffic Volume (Er)", fontweight="bold")

    # Clean spacing between plots
    fig.subplots_adjust(left=0.06, right=0.87, top=0.93, bottom=0.12)

    # Add external legend (right side, fully outside plot)
    legend_ax = fig.add_axes([0.89, 0.2, 0.10, 0.6])  # more right
    legend_ax.axis('off')
    legend_ax.legend(
        handles=[algo_lines[a] for a in all_algos],
        labels=all_algos,
        loc='center',
        frameon=False,
        fontsize=9,
        title="Algorithm",
        title_fontsize=10
    )

    if save_path:
        save_path = Path(save_path)
        fig.savefig(save_path, dpi=300, bbox_inches=None)
        print(f"[plot_modulation_trends_all_algos] ✅ Saved {save_path}")
        plt.close(fig)
    else:
        plt.show()
