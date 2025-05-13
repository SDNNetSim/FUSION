import matplotlib.pyplot as plt
import numpy as np

# pick any colormap you like; tab10 gives up to 10 distinct hues
ALG_COLORS = plt.cm.get_cmap("tab10")


def plot_bw_blocked(data, save_path=None, title="Blocked Bandwidth"):
    """
    Parameters
    ----------
    data : dict
        Output of process_blocked_bandwidth():
        {algo: {traffic_volume (float): {bw (float): mean_blocked}}}
    save_path : pathlib.Path | str | None
        If provided, figures are written as
          <save_path.stem>_<traffic>.png
        alongside the existing plots.
    title : str
        Root title used in each figure.

    One **figure per traffic volume**.
    Each figure: grouped bars where colour = algorithm, x‑axis = bandwidth.
    """
    plt.style.use(
        'seaborn-whitegrid'
        if 'seaborn-whitegrid' in plt.style.available
        else 'default'
    )

    # ── gather global sets ─────────────────────────────────────────────
    all_tvs = sorted({tv for algo in data.values() for tv in algo})
    all_algos = list(data.keys())

    for tv in all_tvs:
        # union of bandwidths present for this traffic volume
        bws = sorted({bw for algo in all_algos for bw in data[algo].get(tv, {})})
        if not bws:
            # skip empty figures (no algo reported this traffic volume)
            continue

        width = 0.8 / max(len(all_algos), 1)  # group width
        x_pos = np.arange(len(bws))  # base positions

        plt.figure(figsize=(10, 6), dpi=300)

        for i, algo in enumerate(all_algos):
            heights = [data[algo].get(tv, {}).get(bw, 0) for bw in bws]
            plt.bar(
                x_pos + i * width,
                heights,
                width=width,
                label=algo,
                color=ALG_COLORS(i),
            )

        # ── axes formatting ────────────────────────────────────────────
        plt.xticks(
            x_pos + width * (len(all_algos) - 1) / 2,
            [int(bw) for bw in bws],
        )
        plt.xlabel("Bandwidth blocked (Gb/s)", fontweight="bold")
        plt.ylabel("# Blocked requests", fontweight="bold")
        plt.title(f"{title} – {tv} Er", fontsize=16, fontweight="bold")
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout(rect=[0, 0, 0.83, 1])

        # ── save or show ───────────────────────────────────────────────
        if save_path:
            from pathlib import Path

            save_path = Path(save_path)
            fp = save_path.parent / f"{save_path.stem}_{tv}.png"
            plt.savefig(fp)
            print(f"[plot_bw_block] ✅ Saved {fp}")
            plt.close()
        else:
            plt.show()
            plt.clf()
