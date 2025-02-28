import math

import matplotlib.pyplot as plt
import numpy as np


def plot_rewards_per_seed(all_rewards_data):
    """
    Plot each seed (trial) as a separate line of mean rewards across episodes,
    and overlay a dashed line representing the aggregate average across seeds.
    """
    for algorithm, traffic_dict in all_rewards_data.items():
        for traffic_label, trial_dict in traffic_dict.items():
            plt.figure(figsize=(6, 4), dpi=200)
            all_episodes = set()
            for trial_index, episodes_dict in trial_dict.items():
                all_episodes.update(episodes_dict.keys())
            all_episodes = sorted(all_episodes)

            # Plot each seed's curve
            for trial_index, episodes_dict in trial_dict.items():
                episode_indices = sorted(episodes_dict.keys())
                means = [np.mean(episodes_dict[ep_idx]) for ep_idx in episode_indices]
                plt.plot(episode_indices, means, marker='o', label=f"Trial {trial_index}")

            # Compute aggregate average for each episode over all seeds
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
            plt.plot(all_episodes, agg_means, 'k--', linewidth=2, label="Aggregate Average")

            plt.title(f"Per-Seed Rewards: {algorithm} (Traffic: {traffic_label})")
            plt.xlabel("Episode (iter)")
            plt.ylabel("Reward (mean across steps)")
            plt.legend()
            plt.grid(True)
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
    across seeds (i.e., the standard deviation across seeds for that episode)
    for each traffic volume (for a single algorithm) on the same plot.
    """
    for algorithm, traffic_dict in all_rewards_data.items():
        plt.figure(figsize=(6, 4), dpi=200)
        for traffic_label, trial_dict in traffic_dict.items():
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
            plt.plot(all_episodes_arr, means, marker='o', label=f"Traffic: {traffic_label} {ci_str}")
            plt.fill_between(all_episodes_arr, lower_bounds, upper_bounds, alpha=0.2)
        plt.title(f"Averaged Rewards: {algorithm}")
        plt.xlabel("Episode (iter)")
        plt.ylabel("Reward (mean across steps, averaged across seeds)")
        plt.legend()
        plt.grid(True)
        plt.show()


def plot_average_rewards(rewards_data):
    """
    Plot average rewards per algorithm and traffic volume.

    Parameters:
        rewards_data (dict): Mapping from algorithm to traffic label to rewards array.
    """
    for algorithm, traffic_rewards in rewards_data.items():
        plt.figure(figsize=(6, 4), dpi=200)
        for traffic_label, rewards in traffic_rewards.items():
            episodes = np.arange(len(rewards))
            plt.plot(episodes, rewards, linewidth=1.5, label=traffic_label)
        plt.xlabel("Number of Episodes")
        plt.ylabel("Reward")
        plt.title(f"Average Rewards for {algorithm}")
        plt.legend(title="Traffic Volume")
        plt.grid(True)
        plt.show()
