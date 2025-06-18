# reinforcement_learning/plotting/path_len.py
from pathlib import Path
from typing import Optional, Union

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_path_lengths(
        processed: dict,
        title: str = "Path-Length vs Erlang",
        save_path: Optional[Union[str, Path]] = None
):
    """
    Line plot: mean path-length per Erlang with ±1 std bands.
    """
    if not processed:
        print("[plot_path_lengths] No data to plot.")
        return

    sns.set_style("whitegrid" if "seaborn-whitegrid" in sns.axes_style() else "ticks")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    palette = plt.cm.get_cmap("tab10", len(processed))

    # --- gather all Erlang values so we can align x-axis order
    all_ers = sorted({float(er) for tvs in processed.values() for er in tvs})

    for i, (algo, tvs) in enumerate(sorted(processed.items())):
        means, stds = [], []
        for er in all_ers:
            arr = tvs.get(str(er), np.array([]))
            means.append(arr.mean() if arr.size else np.nan)
            stds.append(arr.std(ddof=0) if arr.size else np.nan)

        all_ers_arr = np.asarray(all_ers, float)
        means, stds = np.asarray(means), np.asarray(stds)

        ax.plot(
            all_ers_arr,
            means,
            label=algo,
            color=palette(i),
            lw=2,
            marker='o'
        )
        ax.fill_between(
            all_ers_arr,
            means - stds,
            means + stds,
            color=palette(i),
            alpha=0.2
        )

    ax.set_xlabel("Traffic Volume (Er)", fontweight="bold", fontsize=12)
    ax.set_ylabel("Average Path Length (KM)", fontweight="bold", fontsize=12)
    ax.set_title(title, fontweight="bold", fontsize=14, pad=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # legend outside
    ax.legend(
        title="Algorithm",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=False,
        fontsize=9,
        title_fontsize=10
    )

    fig.tight_layout(rect=[0, 0, 0.85, 1])

    if save_path:
        save_path = Path(save_path)
        fig.savefig(save_path, dpi=300)
        print(f"[plot_path_lengths] ✅ Saved {save_path}")
        plt.close(fig)
    else:
        plt.show()
