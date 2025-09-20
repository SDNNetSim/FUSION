from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_path_index(path_index_data, title="Path Index Usage", save_path=None):
    """
    Plots the path index histogram.
    """
    if not path_index_data:
        print("[plot_path_index_by_erlang] No data to plot.")
        return

    # Get all algorithms and Erlangs
    algos = sorted(path_index_data.keys())
    erlangs = sorted(
        {float(tv) for algo_dict in path_index_data.values() for tv in algo_dict}
    )

    for tv in erlangs:
        tv_str = str(tv)

        # Gather all used path indices for this Erlang
        all_path_indices = sorted(
            {
                idx
                for algo in algos
                for idx in path_index_data[algo].get(tv_str, {}).keys()
            }
        )
        if not all_path_indices:
            print(f"[plot_path_index_by_erlang] Skipping Erlang {tv} (no data)")
            continue

        x = np.arange(len(all_path_indices))  # Path index positions
        bar_w = 0.8 / len(algos)

        _, ax = plt.subplots(figsize=(10, 6), dpi=300)
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        for i, algo in enumerate(algos):
            algo_counts = [
                path_index_data[algo].get(tv_str, {}).get(idx, 0)
                for idx in all_path_indices
            ]
            ax.bar(
                x + i * bar_w,
                algo_counts,
                bar_w,
                label=algo,
                color=colors[i % len(colors)],
                edgecolor="black",
                linewidth=0.6,
            )
        # Final formatting
        ax.set_xticks(x + bar_w * (len(algos) / 2 - 0.5))
        ax.set_xticklabels([str(idx) for idx in all_path_indices], fontsize=11)
        ax.set_xlabel("Path Index", fontsize=13, fontweight="bold")
        ax.set_ylabel("Number of Requests Routed", fontsize=13, fontweight="bold")
        ax.set_title(f"{title} – Erlang {int(tv)}", fontsize=15, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        ax.legend(
            title="Algorithm",
            fontsize=10,
            title_fontsize=11,
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
        )
        plt.tight_layout(rect=[0, 0, 0.85, 1])

        if save_path:
            path = Path(save_path)
            filename = f"{path.stem}_erlang_{int(tv)}{path.suffix}"
            output_path = path.with_name(filename)
            plt.savefig(output_path, bbox_inches="tight")
            print(f"[plot_path_index_by_erlang] ✅ Saved: {output_path}")
            plt.close()
        else:
            plt.show()
