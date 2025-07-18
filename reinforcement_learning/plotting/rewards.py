# pylint: disable=unsupported-binary-operation

import matplotlib.pyplot as plt
import numpy as np


def plot_rewards_mean_var(
        rewards_data: dict,
        title: str = "Averaged Rewards",
        save_path: str | None = None
):
    """
    Plot mean and variance of rewards.
    """
    plt.style.use('seaborn-whitegrid' if 'seaborn-whitegrid' in plt.style.available else 'default')

    for algo, tv_dict in rewards_data.items():
        plt.figure(figsize=(10, 6), dpi=300)

        vmin, vmax = float("inf"), float("-inf")
        sorted_tvs = sorted(tv_dict.items(), key=lambda x: float(x[0]))
        num_lines = len(sorted_tvs)
        color_map = plt.cm.get_cmap("tab20", num_lines)  # or "plasma", "viridis"

        for idx, (tv, rewards) in enumerate(sorted_tvs):
            rewards_arr = np.array(rewards["mean"], dtype=float)
            std_arr = np.array(rewards["std"], dtype=float)
            ci_arr = std_arr  # use raw std for shading

            if idx == 0:
                print(
                    f"[DBG PLOT] tv={tv:<6} std_min={np.nanmin(std_arr):.4f} "
                    f"std_max={np.nanmax(std_arr):.4f} "
                    f"overall_ci={rewards.get('overall_ci', 0.0):.4f}"
                )

            ep_labels = rewards.get("episodes")
            if ep_labels:
                episodes = list(range(len(ep_labels)))  # numeric ticks
                plt.xticks(episodes, ep_labels, rotation=45, ha='right')
            else:
                episodes = range(len(rewards_arr))

            episodes = np.arange(len(rewards_arr))
            color = color_map(idx)

            vmin = min(vmin, np.min(rewards_arr - ci_arr))
            vmax = max(vmax, np.max(rewards_arr + ci_arr))

            final_ci = rewards.get("overall_ci", 0.0)
            label = f"{tv} ±{final_ci:.4f}"

            plt.plot(episodes, rewards_arr, color=color, linewidth=2, marker='o', label=label)
            plt.fill_between(
                episodes,
                rewards_arr - ci_arr,
                rewards_arr + ci_arr,
                color=color,
                alpha=0.2,
            )

        y_range = vmax - vmin
        plt.ylim(vmin - 0.05 * y_range, vmax + 0.05 * y_range)

        plt.xlabel("Episode", fontsize=14, fontweight='bold')
        plt.ylabel("Mean Reward", fontsize=14, fontweight='bold')
        plt.title(f"{title}: {algo}", fontsize=16, fontweight='bold')

        plt.grid(True, linestyle='--', linewidth=0.5)
        plt.legend(
            title="Traffic (CI)",
            bbox_to_anchor=(1.05, 1),
            loc='upper left',
            fontsize=12,
            title_fontsize=12
        )
        plt.tight_layout(rect=[0, 0, 0.85, 1])

        if save_path:
            stem = save_path.stem
            parent = save_path.parent
            suffix = save_path.suffix or ".png"
            save_fp = parent / f"{stem}_{algo}{suffix}"
            plt.savefig(save_fp, bbox_inches="tight")
            print(f"[plot_rewards] ✅ Saved {save_fp}")
            plt.close()
        else:
            plt.show()
            plt.clf()
