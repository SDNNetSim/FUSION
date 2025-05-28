from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ALG_COLORS = plt.cm.get_cmap("tab10")


def plot_bw_blocked(data, save_path=None, title="Blocked Bandwidth (Normalized)"):
    """
    Plots normalized blocked bandwidth by dividing blocked counts by request distribution.
    """
    request_distribution = {"25": 0.10, "50": 0.10, "100": 0.50, "200": 0.20, "400": 0.10}

    plt.style.use(
        'seaborn-whitegrid'
        if 'seaborn-whitegrid' in plt.style.available
        else 'default'
    )

    all_tvs = sorted({tv for algo in data.values() for tv in algo})
    all_algos = list(data.keys())

    for tv in all_tvs:
        bws = sorted({bw for algo in all_algos for bw in data[algo].get(tv, {})})
        if not bws:
            continue

        width = 0.8 / max(len(all_algos), 1)
        x_pos = np.arange(len(bws))

        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

        for i, algo in enumerate(all_algos):
            raw_vals = [data[algo].get(tv, {}).get(bw, 0) for bw in bws]
            scaled_vals = []
            for j, bw in enumerate(bws):
                freq = request_distribution.get(str(int(bw)), 1.0)
                scaled_vals.append(raw_vals[j] / freq if freq > 0 else 0)
            ax.bar(
                x_pos + i * width,
                scaled_vals,
                width=width,
                label=algo,
                color=ALG_COLORS(i),
            )

        for patch in ax.patches:
            patch.set_zorder(3)

        ax.set_axisbelow(True)
        ax.grid(
            axis="y",
            linestyle="--",
            linewidth=0.8,
            alpha=0.7,
            zorder=0
        )

        ax.set_xticks(x_pos + width * (len(all_algos) - 1) / 2)
        ax.set_xticklabels([int(bw) for bw in bws])

        ax.set_xlabel("Bandwidth blocked (Gb/s)", fontweight="bold")
        ax.set_ylabel("Blocked Requests (normalized)", fontweight="bold")
        ax.set_title(f"{title} – {tv} Er", fontsize=16, fontweight="bold")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")

        fig.tight_layout(rect=[0, 0, 0.83, 1])

        if save_path:
            save_path = Path(save_path)
            fp = save_path.parent / f"{save_path.stem}_{tv}_normalized.png"
            fig.savefig(fp)
            print(f"[plot_bw_blocked_scaled] ✅ Saved {fp}")
            plt.close(fig)
        else:
            plt.show()
