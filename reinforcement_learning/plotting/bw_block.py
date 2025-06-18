from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ALG_COLORS = plt.cm.get_cmap("tab10")


def plot_bw_blocked(data, save_path=None, title="Bw_block – NSFNet"):
    request_distribution = {25.0: 0.10, 50.0: 0.10, 100.0: 0.50, 200.0: 0.20, 400.0: 0.10}
    all_algos = sorted(data.keys())
    all_ers = sorted({float(er) for algo_dict in data.values() for er in algo_dict})
    all_bws = sorted({float(bw) for algo_dict in data.values() for er_dict in algo_dict.values() for bw in er_dict})

    ncols = 4
    nrows = int(np.ceil(len(all_ers) / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4.5 * ncols, 4.2 * nrows), dpi=300, sharey=False)
    axes = axes.flatten() if nrows > 1 else [axes]

    max_vals_per_er = {
        er: max(
            (data.get(algo, {}).get(er, {}).get(bw, 0.0) / request_distribution.get(bw, 1.0))
            for algo in all_algos for bw in all_bws
        ) or 1.0
        for er in all_ers
    }

    for ax, er in zip(axes, all_ers):
        x = np.arange(len(all_bws))
        bar_width = 0.8 / len(all_algos)

        for i, algo in enumerate(all_algos):
            blocked_vals = []
            for bw in all_bws:
                raw = data.get(algo, {}).get(er, {}).get(bw, 0.0)
                freq = request_distribution.get(bw, 1.0)
                norm = raw / freq if freq > 0 else 0.0
                norm /= max_vals_per_er[er]
                blocked_vals.append(norm)

            positions = x + i * bar_width
            ax.bar(positions, blocked_vals, width=bar_width, label=algo, color=ALG_COLORS(i))

        ax.set_title(f"Er = {er}", fontsize=11, fontweight="bold")
        ax.set_xticks(x + bar_width * (len(all_algos) - 1) / 2)
        ax.set_xticklabels([str(int(bw)) for bw in all_bws])
        ax.set_xlabel("Blocked Bandwidth (Gb/s)")
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        if ax is axes[0]:
            ax.set_ylabel("Normalized Blocked Requests")

    # Remove unused axes
    for j in range(len(all_ers), len(axes)):
        fig.delaxes(axes[j])

    # Adjust layout to make room for legend and title
    fig.subplots_adjust(left=0.06, right=0.86, top=0.95, bottom=0.12)

    # Add external legend in its own axis
    legend_ax = fig.add_axes([0.88, 0.25, 0.10, 0.5])  # [left, bottom, width, height]
    legend_ax.axis('off')
    legend_ax.legend(
        handles=[plt.Line2D([0], [0], color=ALG_COLORS(i), lw=4) for i in range(len(all_algos))],
        labels=all_algos,
        loc="center",
        fontsize=9,
        title="Algorithm",
        title_fontsize=10
    )

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.0)

    if save_path:
        save_path = Path(save_path)
        fig.savefig(save_path, dpi=300)
        print(f"[plot_bw_blocked_normalized_subplots] ✅ Saved {save_path}")
        plt.close(fig)
    else:
        plt.show()
