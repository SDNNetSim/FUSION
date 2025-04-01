import math

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def plot_rewards_per_seed(all_rewards_data):
    """
    Plot each seed (trial) as a separate line of mean rewards across episodes,
    and overlay a dashed line representing the aggregate average across seeds.
    """
    available_styles = plt.style.available
    if 'seaborn-whitegrid' in available_styles:
        plt.style.use('seaborn-whitegrid')
    elif 'seaborn-white' in available_styles:
        plt.style.use('seaborn-white')
    else:
        plt.style.use('default')

    for algorithm, traffic_dict in all_rewards_data.items():
        for traffic_label, trial_dict in traffic_dict.items():
            plt.figure(figsize=(10, 6), dpi=300)
            all_episodes = set()
            for trial_index, episodes_dict in trial_dict.items():
                all_episodes.update(episodes_dict.keys())
            all_episodes = sorted(all_episodes)

            for trial_index, episodes_dict in trial_dict.items():
                episode_indices = sorted(episodes_dict.keys())
                means = [np.mean(episodes_dict[ep_idx]) for ep_idx in episode_indices]
                plt.plot(episode_indices, means,
                         marker='o', markersize=5, linewidth=2,
                         label=f"Trial {trial_index}")

            agg_means = []
            for ep_idx in all_episodes:
                trial_means = []
                for trial_index, episodes_dict in trial_dict.items():
                    if ep_idx in episodes_dict:
                        trial_means.append(np.mean(episodes_dict[ep_idx]))
                if trial_means:
                    agg_means.append(np.mean(trial_means))
                else:
                    agg_means.append(np.nan)

            plt.plot(all_episodes, agg_means,
                     'k--', linewidth=2, label="Aggregate Average")
            plt.xlabel("Episode (iter)", fontsize=14, fontweight='bold')
            plt.ylabel("Reward (mean across steps)", fontsize=14, fontweight='bold')
            plt.title(f"Per-Seed Rewards: {algorithm} (Traffic: {traffic_label})",
                      fontsize=16, fontweight='bold')
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.grid(True, which="both", linestyle='--', linewidth=0.5)
            plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)
            plt.tight_layout(rect=[0, 0, 0.85, 1])

            plt.show()


def compute_confidence_interval_for_traffic(trial_dict, all_episodes):
    """Compute a 95% confidence interval string for a traffic volume."""
    se_list = []
    for ep_idx in all_episodes:
        trial_means = []
        for trial_index in trial_dict:
            if ep_idx in trial_dict[trial_index]:
                rewards_array = trial_dict[trial_index][ep_idx]
                trial_means.append(np.mean(rewards_array))
        if trial_means:
            std = np.std(trial_means)
            n = len(trial_means)
            se = std / math.sqrt(n)
            se_list.append(se)
    if se_list:
        avg_se = np.mean(se_list)
        ci = 1.96 * avg_se
        return f"(CI: ±{ci:.2f})"
    return ""


def plot_rewards_averaged_with_variance(all_rewards_data):
    """
    Plot the average across all seeds, per episode, with variance shading
    (i.e., standard deviation across seeds) for each traffic volume (for a single algorithm)
    on the same plot.
    """
    available_styles = plt.style.available
    if 'seaborn-whitegrid' in available_styles:
        plt.style.use('seaborn-whitegrid')
    elif 'seaborn-white' in available_styles:
        plt.style.use('seaborn-white')
    else:
        plt.style.use('default')

    for algorithm, traffic_dict in all_rewards_data.items():
        plt.figure(figsize=(10, 6), dpi=300)
        for i, (traffic_label, trial_dict) in enumerate(traffic_dict.items()):
            # Collect all episodes across trials
            all_episodes = set()
            for trial_index in trial_dict:
                all_episodes.update(trial_dict[trial_index].keys())
            all_episodes = sorted(all_episodes)

            means = []
            lower_bounds = []
            upper_bounds = []

            for ep_idx in all_episodes:
                trial_means = []
                for trial_index in trial_dict:
                    if ep_idx in trial_dict[trial_index]:
                        rewards_array = trial_dict[trial_index][ep_idx]
                        trial_means.append(np.mean(rewards_array))
                if trial_means:
                    ep_mean = np.mean(trial_means)
                    ep_std = np.std(trial_means)
                    means.append(ep_mean)
                    lower_bounds.append(ep_mean - ep_std)
                    upper_bounds.append(ep_mean + ep_std)
                else:
                    means.append(np.nan)
                    lower_bounds.append(np.nan)
                    upper_bounds.append(np.nan)

            all_episodes_arr = np.array(all_episodes)
            means = np.array(means)
            lower_bounds = np.array(lower_bounds)
            upper_bounds = np.array(upper_bounds)

            ci_str = compute_confidence_interval_for_traffic(trial_dict, all_episodes)
            plt.plot(all_episodes_arr, means,
                     marker='^' if i < len(traffic_dict) / 2 else 'o', markersize=5, linewidth=2,
                     label=f"Traffic: {traffic_label} {ci_str}")
            plt.fill_between(all_episodes_arr, lower_bounds, upper_bounds, alpha=0.2)

        plt.title(f"Averaged Rewards: {algorithm}", fontsize=16, fontweight='bold')
        plt.xlabel("Episode (iter)", fontsize=14, fontweight='bold')
        plt.ylabel("Reward", fontsize=14, fontweight='bold')
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)
        plt.grid(True, which="both", linestyle='--', linewidth=0.5)
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.show()


def plot_average_rewards(rewards_data):
    """
    Plot average rewards per algorithm and traffic volume.

    Parameters:
        rewards_data (dict): Mapping from algorithm to traffic label to rewards array.
    """
    available_styles = plt.style.available
    if 'seaborn-whitegrid' in available_styles:
        plt.style.use('seaborn-whitegrid')
    elif 'seaborn-white' in available_styles:
        plt.style.use('seaborn-white')
    else:
        plt.style.use('default')

    for algorithm, traffic_rewards in rewards_data.items():
        plt.figure(figsize=(10, 6), dpi=300)

        num_lines = len(traffic_rewards)
        colors = sns.color_palette("bright", num_lines)
        items = list(traffic_rewards.items())

        half = num_lines // 2

        for idx, (traffic_label, rewards) in enumerate(items):
            episodes = np.arange(len(rewards))
            if idx < half:
                plt.plot(
                    episodes, rewards,
                    linewidth=1.5,
                    marker='^',
                    markevery=10,
                    color=colors[idx],
                    label=traffic_label
                )
            else:
                plt.plot(
                    episodes, rewards,
                    linewidth=1.5,
                    color=colors[idx],
                    label=traffic_label
                )

        plt.xlabel("Number of Episodes", fontsize=14, fontweight='bold')
        plt.ylabel("Reward", fontsize=14, fontweight='bold')
        plt.title(f"Average Rewards for {algorithm}", fontsize=16, fontweight='bold')
        plt.legend(
            title="Traffic Volume",
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            borderaxespad=0,
            fontsize=12,
            title_fontsize=12
        )
        plt.grid(True, which="both", linestyle='--', linewidth=0.5)
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.show()
