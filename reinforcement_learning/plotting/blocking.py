from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.colors import TwoSlopeNorm
import numpy as np
from itertools import cycle

import pandas as pd
import seaborn as sns


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
    ax.set_ylim(10 ** -2.3, 10 ** -0.7)
    ax.set_xlim(200, 1300)
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


def plot_effect_heatmaps(
    processed: dict,
    title: str | None = None,
    save_path: str | Path | None = None
):
    save_path = Path(save_path) if save_path else None

    excluded = {"k_shortest_path_1", "k_shortest_path_4", "k_shortest_path_inf", "cong_aware"}
    rl_algorithms = sorted([algo for algo in processed if algo not in excluded])
    traffic_volumes = sorted({tv for algo in processed.values() for tv in algo}, key=lambda x: float(x))

    baselines = ["vs_cong_aware", "vs_k_shortest_path_4"]
    baseline_titles = {
        "vs_cong_aware": "Cohen's d vs Congestion-Aware Heuristic",
        "vs_k_shortest_path_4": "Cohen's d vs K-Shortest Path (k=4)"
    }

    d_maps = {b: pd.DataFrame(index=rl_algorithms, columns=traffic_volumes, dtype=float) for b in baselines}
    d_annots = {b: pd.DataFrame(index=rl_algorithms, columns=traffic_volumes, dtype=str) for b in baselines}
    ci_map = pd.DataFrame(index=rl_algorithms, columns=traffic_volumes, dtype=float)
    ci_annot = pd.DataFrame(index=rl_algorithms, columns=traffic_volumes, dtype=str)

    for algo in rl_algorithms:
        for tv in traffic_volumes:
            stats = processed.get(algo, {}).get(tv, {})
            for b in baselines:
                if b in stats and stats[b].get("d") is not None:
                    d = stats[b]["d"]
                    d_maps[b].loc[algo, tv] = d
                    d_annots[b].loc[algo, tv] = f"{d:.2f}"
            if "ci" in stats and stats["ci"] is not None:
                ci_width = 2 * stats["ci"]
                ci_map.loc[algo, tv] = ci_width
                ci_annot.loc[algo, tv] = f"±{ci_width:.2f}"

    plt.style.use("seaborn-whitegrid" if "seaborn-whitegrid" in plt.style.available else "default")

    for baseline in baselines:
        fig, ax = plt.subplots(figsize=(18, 6), dpi=300)
        sns.heatmap(
            d_maps[baseline],
            ax=ax,
            annot=d_annots[baseline],
            fmt="",
            cmap=sns.diverging_palette(145, 15, s=90, l=60, as_cmap=True),
            center=0,
            cbar=True,
            linewidths=0.5,
            linecolor="gray",
            annot_kws={"fontsize": 9, "weight": "bold"},
            square=False
        )
        full_title = f"{title + ': ' if title else ''}{baseline_titles[baseline]}"
        ax.set_title(full_title, fontsize=13, weight="bold", pad=10)
        ax.set_xlabel("Traffic Volume (Erlang)", fontsize=11, weight="bold")
        ax.set_ylabel("RL Algorithm", fontsize=11, weight="bold")
        ax.tick_params(axis='x', labelsize=8, rotation=40)
        ax.tick_params(axis='y', labelsize=9)
        plt.tight_layout()

        if save_path:
            full_path = save_path.with_stem(save_path.stem + f"_{baseline}")
            plt.savefig(full_path, bbox_inches="tight")
            print(f"[heatmap] ✅ Saved {full_path}")
            plt.close()
        else:
            plt.show()

    fig, ax = plt.subplots(figsize=(22, 6), dpi=300)
    sns.heatmap(
        ci_map,
        ax=ax,
        annot=ci_annot,
        fmt="",
        cmap=sns.color_palette("YlGnBu", as_cmap=True),
        cbar=True,
        linewidths=0.5,
        linecolor="gray",
        annot_kws={"fontsize": 9, "weight": "bold"},
        square=False
    )
    full_title = f"{title + ': ' if title else ''}Blocking Probability CI Width (±95%)"
    ax.set_title(full_title, fontsize=13, weight="bold", pad=10)
    ax.set_xlabel("Traffic Volume (Erlang)", fontsize=11, weight="bold")
    ax.set_ylabel("RL Algorithm", fontsize=11, weight="bold")
    ax.tick_params(axis='x', labelsize=8, rotation=40)
    ax.tick_params(axis='y', labelsize=9)
    plt.tight_layout()

    if save_path:
        full_path = save_path.with_stem(save_path.stem + "_blocking_ci")
        plt.savefig(full_path, bbox_inches="tight")
        print(f"[heatmap] ✅ Saved {full_path}")
        plt.close()
    else:
        plt.show()
