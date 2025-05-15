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
        ax.fill_between(erlangs, means - stds, means + stds, color=color, alpha=0.2)

    ax.set_xlabel("Erlang Values", fontsize=14, fontweight="bold")
    ax.set_ylabel("Blocking Probability", fontsize=14, fontweight="bold")
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_yscale("log")
    ax.set_ylim(10 ** -4, 10 ** -0.5)
    ax.set_xlim(50, 700)
    ax.tick_params(labelsize=12)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    ax.legend(title="Algorithm (±CI)", loc="upper left", bbox_to_anchor=(1.05, 1), fontsize=11, title_fontsize=12)
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
        title: str = "Effect Size Comparison of Blocking Probabilities"
):
    rows = []

    # Determine Erlang values
    all_erlangs = sorted({float(tv) for algo in processed.values() for tv in algo})
    if not all_erlangs:
        print("[plot_blocking_stats] ⚠ No data to plot.")
        return

    min_erl = all_erlangs[0]
    mid_erl = all_erlangs[len(all_erlangs) // 2]
    max_erl = all_erlangs[-1]
    selected_erlangs = {str(min_erl), str(mid_erl), str(max_erl)}

    for algo, tv_dict in processed.items():
        for tv, stats in tv_dict.items():
            if tv not in selected_erlangs:
                continue

            for compare_key in ["vs_k_shortest_path_4", "vs_cong_aware"]:
                if compare_key not in stats:
                    continue
                comp = stats[compare_key]
                d = comp.get("d")
                if d is None:
                    continue

                baseline_label = "k = 4" if "k_shortest" in compare_key else "cong_aware"
                mean_diff = comp.get("mean_diff", 0.0)
                ci_width = 2 * stats.get("ci", 0.0)

                # Interpret Cohen's d (direction-aware)
                if d <= -0.8:
                    interpretation = "Large Benefit"
                elif d <= -0.5:
                    interpretation = "Moderate Benefit"
                elif d <= -0.2:
                    interpretation = "Small Benefit"
                elif d < 0.2:
                    interpretation = "Negligible"
                elif d < 0.5:
                    interpretation = "Small Disadvantage"
                elif d < 0.8:
                    interpretation = "Moderate Disadvantage"
                else:
                    interpretation = "Large Disadvantage"

                rows.append([
                    algo.replace("_", " ").title(),
                    baseline_label,
                    float(tv),
                    f"{mean_diff:.4f}",
                    f"{d:.3f}",
                    f"±{ci_width:.4f}",
                    interpretation
                ])

    df = pd.DataFrame(rows, columns=["Algorithm", "Baseline", "Erlang", "Δ Mean Blocking", "Cohen's d", "95% CI Width",
                                     "Interpretation"])
    df.sort_values(by=["Erlang", "Baseline", "Algorithm"], inplace=True)

    if save_path:
        csv_fp = save_path.with_suffix(".csv")
        df.to_csv(csv_fp, index=False)
        print(f"[plot_blocking_stats] ✅ CSV saved to {csv_fp}")

    fig_height = 0.6 + 0.4 * len(df)
    fig, ax = plt.subplots(figsize=(12, fig_height), dpi=300)
    ax.axis("off")
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.96)

    table = ax.table(cellText=df.values, colLabels=df.columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.4)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold')
        if col == 0:
            cell._loc = 'left'

        if col == 6 and row > 0:
            value = table[row, col].get_text().get_text()
            if value == "Large Benefit":
                cell.set_facecolor(to_rgba("mediumseagreen", 0.6))
            elif value == "Moderate Benefit":
                cell.set_facecolor(to_rgba("palegreen", 0.6))
            elif value == "Small Benefit":
                cell.set_facecolor(to_rgba("honeydew", 0.8))
            elif value == "Negligible":
                cell.set_facecolor(to_rgba("lightyellow", 0.8))
            elif value == "Small Disadvantage":
                cell.set_facecolor(to_rgba("moccasin", 0.8))
            elif value == "Moderate Disadvantage":
                cell.set_facecolor(to_rgba("lightsalmon", 0.7))
            elif value == "Large Disadvantage":
                cell.set_facecolor(to_rgba("lightcoral", 0.5))

    if save_path:
        fig_fp = save_path.with_suffix(".png")
        plt.savefig(fig_fp, bbox_inches="tight")
        print(f"[plot_blocking_stats] ✅ Table saved to {fig_fp}")
        plt.close()
    else:
        plt.tight_layout()
        plt.show()
