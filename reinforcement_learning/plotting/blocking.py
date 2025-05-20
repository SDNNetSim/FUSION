import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import numpy as np
from itertools import cycle

import pandas as pd


def plot_blocking_probabilities(
        blocking_data: dict,
        title: str = "Blocking Probability",
        save_path: str | None = None,
        selected_algos: list[str] | None = None
):
    """
    Plot blocking probability curves with shaded std and CI in legend.

    Parameters:
        blocking_data (dict): {
            algo: {
                erlang: {
                    "mean": float,
                    "std": float,
                    "ci": float
                }
            }
        }
        title (str): Plot title.
        save_path (str|Path): If given, saves the plot to this location.
        selected_algos (list): If given, filters to only these algorithms.
    """
    plt.style.use("seaborn-whitegrid" if "seaborn-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    if selected_algos:
        blocking_data = {k: v for k, v in blocking_data.items() if k in selected_algos}

    color_map = plt.cm.get_cmap("tab10")
    markers = cycle(['o', 's', 'D', '^', 'v', '*', 'X', 'P', '<', '>'])
    colors = cycle(color_map.colors)

    for algo in sorted(blocking_data.keys()):
        traffic_dict = blocking_data[algo]
        erlangs, means, stds, cis = [], [], [], []

        for tv_str, stats in traffic_dict.items():
            try:
                tv = float(tv_str)
                erlangs.append(tv)
                means.append(stats["mean"])
                stds.append(stats.get("std", 0.0))
                cis.append(stats.get("ci", 0.0))
            except (ValueError, KeyError, TypeError):
                continue

        if not erlangs:
            continue

        erlangs, means, stds, cis = map(np.array, zip(*sorted(zip(erlangs, means, stds, cis), key=lambda x: x[0])))

        color = next(colors)
        marker = next(markers)
        overall_ci = np.mean(cis) if len(cis) > 1 else cis[0]
        label = f"{algo} ±{overall_ci:.4f}"

        ax.plot(erlangs, means, label=label, color=color, linewidth=2, marker=marker, markersize=5)
        # ax.fill_between(erlangs, means - stds, means + stds, color=color, alpha=0.2)

    ax.set_xlabel("Erlang Values", fontsize=14, fontweight="bold")
    ax.set_ylabel("Blocking Probability", fontsize=14, fontweight="bold")
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_yscale("log")
    ax.set_ylim(10 ** -4, 10 ** -0.3)
    ax.set_xlim(50, 900)
    ax.tick_params(labelsize=12)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    ax.legend(title="Algorithm (95% ± CI)", loc="upper left", bbox_to_anchor=(1.05, 1), fontsize=11, title_fontsize=12)
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"[plot_blocking] ✅ Saved {save_path}")
        plt.close()
    else:
        plt.show()
        plt.clf()


def plot_blocking_stats_table(
        processed: dict,
        save_path: str | None = None,
        title: str = "Effect Size Comparison of Blocking Probabilities",
        group_col: str = "Erlang"
):
    # ────────────────────────────────────────────────────────
    # 1) Build a flattened DataFrame
    # ────────────────────────────────────────────────────────
    rows = []
    all_erlangs = sorted({float(tv) for a in processed.values() for tv in a})
    if not all_erlangs:
        print("[plot_blocking_stats] ⚠ No data to plot.")
        return

    keep_erl = {
        str(all_erlangs[0]),  # min
        str(all_erlangs[len(all_erlangs) // 2]),  # mid
        str(all_erlangs[-1])  # max
    }

    for algo, tv_dict in processed.items():
        for tv, stats in tv_dict.items():
            if tv not in keep_erl:
                continue
            for ck in ("vs_k_shortest_path_4", "vs_cong_aware"):
                if ck not in stats or stats[ck].get("d") is None:
                    continue
                d = stats[ck]["d"]
                mean_diff = stats[ck].get("mean_diff", 0.0)
                ci_width = 2 * stats.get("ci", 0.0)
                baseline = "k = 4" if "k_shortest" in ck else "cong_aware"

                # Cohen’s d interpretation
                if d <= -0.8:
                    inter = "Large Benefit"
                elif d <= -0.5:
                    inter = "Moderate Benefit"
                elif d <= -0.2:
                    inter = "Small Benefit"
                elif d < 0.2:
                    inter = "Negligible"
                elif d < 0.5:
                    inter = "Small Disadvantage"
                elif d < 0.8:
                    inter = "Moderate Disadvantage"
                else:
                    inter = "Large Disadvantage"

                rows.append([
                    algo.replace("_", " ").title(), baseline, float(tv),
                    f"{mean_diff:.4f}", f"{d:.3f}", f"±{ci_width:.4f}", inter
                ])

    df = (pd.DataFrame(rows, columns=[
        "Algorithm", "Baseline", "Erlang",
        "Δ Mean Blocking", "Cohen's d",
        "(95% ± CI)", "Interpretation"
    ])
          .sort_values([group_col, "Baseline", "Algorithm"])
          .reset_index(drop=True)
          )

    if save_path:
        df.to_csv(save_path.with_suffix(".csv"), index=False)

    # Mask that is True on the last row of each Erlang block
    boundary_mask = df[group_col].ne(df[group_col].shift(-1))

    # ────────────────────────────────────────────────────────
    # 2) Draw the table
    # ────────────────────────────────────────────────────────
    fig_h = 0.6 + 0.4 * len(df)  # dynamic height
    fig_w = 13
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300)
    ax.axis("off")
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.96)

    col_w = [0.16, 0.12, 0.09, 0.13, 0.11, 0.11, 0.28]
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        colWidths=col_w,
        loc="center",
        cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.4)

    # Colours for interpretation column
    palette = {
        "Large Benefit": to_rgba("mediumseagreen", 0.6),
        "Moderate Benefit": to_rgba("palegreen", 0.6),
        "Small Benefit": to_rgba("honeydew", 0.8),
        "Negligible": to_rgba("lightyellow", 0.8),
        "Small Disadvantage": to_rgba("moccasin", 0.8),
        "Moderate Disadvantage": to_rgba("lightsalmon", 0.7),
        "Large Disadvantage": to_rgba("lightcoral", 0.5),
    }

    n_cols = len(df.columns)

    for (row, col), cell in table.get_celld().items():
        if row == 0:  # header styling
            cell.set_text_props(weight="bold")
        if col == 0:  # left-align algorithm names
            cell._loc = "left"

        # Apply heat-map colour to Interpretation column
        if col == 6 and row > 0:
            cell.set_facecolor(
                palette[table[row, col].get_text().get_text()]
            )

        # Shade boundary rows (but NOT the Interpretation column)
        if row > 0 and boundary_mask.iloc[row - 1] and col != 6:
            cell.set_facecolor("#e6e6e6")  # light grey
            cell.set_text_props(weight="bold")

    # ────────────────────────────────────────────────────────
    # 3) Save or display
    # ────────────────────────────────────────────────────────
    if save_path:
        plt.savefig(save_path.with_suffix(".png"), bbox_inches="tight")
        plt.close()
    else:
        plt.tight_layout()
        plt.show()
