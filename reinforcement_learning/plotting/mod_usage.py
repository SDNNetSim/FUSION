import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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


def plot_modulation_usage(data, save_path=None,
                          title="Modulations per Bandwidth"):
    """
    data ::
        {algo: {tv (str): {bw (str): {mod: mean_cnt}}}}
    Generates one figure per traffic volume – grid of algorithm panels.
    """
    plt.rcParams.update({
        "font.size": BASE_FONT_SIZE,
        "axes.labelweight": "bold",
        "axes.titlesize": SUBPLOT_TITLE_SZ,
        "axes.titleweight": "regular",
        "legend.fontsize": BASE_FONT_SIZE - 1,
    })

    plt.style.use(
        "seaborn-whitegrid"
        if "seaborn-whitegrid" in plt.style.available
        else "default"
    )

    all_tvs = sorted({float(tv) for algo in data.values() for tv in algo})
    all_algos = sorted(data.keys())

    for tv in all_tvs:
        tv_str = str(tv)
        bws = sorted(
            {
                float(bw)
                for algo in all_algos
                for bw in data[algo].get(tv_str, {})
            }
        )
        if not bws:
            continue

        bar_w = _auto_bar_width(bws)

        n_algos = len(all_algos)
        n_cols = 3
        n_rows = math.ceil(n_algos / n_cols)
        fig_w = 3.2 * n_cols
        fig_h = 3.0 * n_rows

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(fig_w, fig_h),
            sharey=True, dpi=300,
        )
        axes = axes.flatten()

        for idx, algo in enumerate(all_algos):
            ax = axes[idx]
            bottoms = np.zeros(len(bws))
            bw_vals = data[algo].get(tv_str, {})

            for mod in MOD_ORDER:
                heights = [
                    bw_vals.get(str(int(bw)), {}).get(mod, 0) for bw in bws
                ]
                ax.bar(
                    bws,
                    heights,
                    width=bar_w,
                    bottom=bottoms,
                    color=MOD_COLORS[mod],
                    edgecolor="black",
                    linewidth=0.3,
                )
                bottoms += np.array(heights)

            ax.set_title(algo, fontsize=SUBPLOT_TITLE_SZ, pad=4)
            ax.set_xticks(
                bws, [int(b) for b in bws], rotation=45, fontsize=TICK_FONT_SIZE
            )
            if idx % n_cols == 0:
                ax.set_ylabel("# requests", fontweight="bold")
            _apply_axes_style(ax)

        # delete empty axes if algos % n_cols != 0
        for ax in axes[n_algos:]:
            fig.delaxes(ax)

        fig.suptitle(f"{title} – {tv} Er",
                     fontsize=TITLE_FONT_SIZE, fontweight="bold", y=0.98)

        mod_handles = [
            mpatches.Patch(color=MOD_COLORS[m], label=m) for m in MOD_ORDER
        ]
        leg = fig.legend(
            handles=mod_handles,
            title="Modulation",
            loc="upper right",
            bbox_to_anchor=(1.02, 1),
            frameon=True,
            edgecolor="0.5",
        )
        leg.get_frame().set_linewidth(0.8)

        fig.tight_layout(rect=[0, 0, 0.92, 0.94])

        if save_path:
            save_path = Path(save_path)
            fp = save_path.parent / f"{save_path.stem}_{tv}.png"
            fig.savefig(fp, dpi=300)
            print(f"[plot_mod_usage] ✅ Saved {fp}")
            plt.close(fig)
        else:
            plt.show()
            plt.clf()
