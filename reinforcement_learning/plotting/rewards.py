import math

import matplotlib.pyplot as plt
import numpy as np


def plot_rewards_per_seed(all_rewards_data, title_prefix='Per-Seed Rewards', save_path=None):
    """
    Plots each seed as a separate line of mean rewards across episodes,
    with an overlay showing the aggregate average across seeds.

    Parameters:
        all_rewards_data (dict): {algorithm: {traffic_label: {trial_index: {episode: reward_list}}}}
        title_prefix (str): Title prefix for plots.
        save_path (str, optional): Path to save the plot.
    """
    plt.style.use('seaborn-whitegrid' if 'seaborn-whitegrid' in plt.style.available else 'default')

    for algorithm, traffic_dict in all_rewards_data.items():
        for traffic_label, trial_dict in traffic_dict.items():
            plt.figure(figsize=(10, 6), dpi=300)
            all_episodes = sorted({ep for trial in trial_dict.values() for ep in trial.keys()})

            # Plot each trial separately
            for trial_index, episodes_dict in trial_dict.items():
                episode_indices = sorted(episodes_dict.keys())
                means = [np.mean(episodes_dict[ep]) for ep in episode_indices]
                plt.plot(episode_indices, means, marker='o', linewidth=2, markersize=5, label=f"Trial {trial_index}")

            # Aggregate average
            agg_means = [np.mean([np.mean(trial_dict[trial][ep]) for trial in trial_dict if ep in trial_dict[trial]])
                         for ep in all_episodes]
            plt.plot(all_episodes, agg_means, 'k--', linewidth=2, label="Aggregate Average")

            plt.xlabel("Episode (iter)", fontsize=14, fontweight='bold')
            plt.ylabel("Reward (mean across steps)", fontsize=14, fontweight='bold')
            plt.title(f"{title_prefix}: {algorithm} (Traffic: {traffic_label})", fontsize=16, fontweight='bold')
            plt.grid(True, linestyle='--', linewidth=0.5)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout(rect=[0, 0, 0.85, 1])

            if save_path:
                plt.savefig(f"{save_path}_{algorithm}_{traffic_label}_per_seed.png", bbox_inches='tight')
            plt.show()


def plot_rewards_averaged_with_variance(all_rewards_data, title_prefix='Averaged Rewards with Variance',
                                        save_path=None):
    """
    Plots averaged rewards across all seeds per episode, including variance shading.

    Parameters:
        all_rewards_data (dict): Same format as above.
        title_prefix (str): Title prefix.
        save_path (str, optional): Save path.
    """
    plt.style.use('seaborn-whitegrid' if 'seaborn-whitegrid' in plt.style.available else 'default')

    for algorithm, traffic_dict in all_rewards_data.items():
        plt.figure(figsize=(10, 6), dpi=300)

        for _, (traffic_label, trial_dict) in enumerate(traffic_dict.items()):
            all_episodes = sorted({ep for trial in trial_dict.values() for ep in trial.keys()})
            means, stds = [], []

            for ep in all_episodes:
                ep_rewards = [np.mean(trial_dict[trial][ep]) for trial in trial_dict if ep in trial_dict[trial]]
                means.append(np.mean(ep_rewards))
                stds.append(np.std(ep_rewards))

            means, stds = np.array(means), np.array(stds)
            plt.plot(all_episodes, means, linewidth=2, label=f"Traffic: {traffic_label}")
            plt.fill_between(all_episodes, means - stds, means + stds, alpha=0.2)

        plt.xlabel("Episode (iter)", fontsize=14, fontweight='bold')
        plt.ylabel("Reward", fontsize=14, fontweight='bold')
        plt.title(f"{title_prefix}: {algorithm}", fontsize=16, fontweight='bold')
        plt.grid(True, linestyle='--', linewidth=0.5)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout(rect=[0, 0, 0.85, 1])

        if save_path:
            plt.savefig(f"{save_path}_{algorithm}_averaged_variance.png", bbox_inches='tight')
        plt.show()


def plot_average_rewards(rewards_data, title_prefix='Average Rewards', save_path=None):
    """
    Plots average rewards per algorithm and traffic volume clearly.

    Parameters:
        rewards_data (dict): {algorithm: {traffic_label: rewards_array}}
        title_prefix (str): Title prefix.
        save_path (str, optional): Save path.
    """
    plt.style.use('seaborn-whitegrid' if 'seaborn-whitegrid' in plt.style.available else 'default')

    for algorithm, traffic_rewards in rewards_data.items():
        plt.figure(figsize=(10, 6), dpi=300)
        cmap = plt.get_cmap("tab10")

        for idx, (traffic_label, rewards) in enumerate(traffic_rewards.items()):
            episodes = np.arange(len(rewards))
            plt.plot(episodes, rewards, linewidth=1.5, color=cmap(idx), label=f"{traffic_label}")

        plt.xlabel("Number of Episodes", fontsize=14, fontweight='bold')
        plt.ylabel("Reward", fontsize=14, fontweight='bold')
        plt.title(f"{title_prefix} for {algorithm}", fontsize=16, fontweight='bold')
        plt.grid(True, linestyle='--', linewidth=0.5)
        plt.legend(title="Traffic Volume", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout(rect=[0, 0, 0.85, 1])

        if save_path:
            plt.savefig(f"{save_path}_{algorithm}_average_rewards.png", bbox_inches='tight')
        plt.show()


def compute_confidence_interval_for_traffic(trial_dict, all_episodes):
    """Utility: Computes a 95% confidence interval."""
    se_list = []
    for ep in all_episodes:
        trial_means = [np.mean(trial_dict[trial][ep]) for trial in trial_dict if ep in trial_dict[trial]]
        if trial_means:
            se_list.append(np.std(trial_means) / math.sqrt(len(trial_means)))
    ci = 1.96 * np.mean(se_list) if se_list else 0
    return f"(CI: ±{ci:.2f})"
