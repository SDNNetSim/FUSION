import matplotlib.pyplot as plt
import numpy as np


def plot_rewards_per_seed_with_variance(all_rewards_data):
    """
    Plot each seed (trial) as a separate line with variance shading
    across the steps in each episode.
    """
    for algorithm, traffic_dict in all_rewards_data.items():
        for traffic_label, trial_dict in traffic_dict.items():
            plt.figure(figsize=(8, 5))
            for trial_index, episodes_dict in trial_dict.items():
                episode_indices = sorted(episodes_dict.keys())
                means = []
                lower_bounds = []
                upper_bounds = []

                for ep_idx in episode_indices:
                    rewards_array = episodes_dict[ep_idx]
                    mean_val = np.mean(rewards_array)
                    std_val = np.std(rewards_array)
                    means.append(mean_val)
                    lower_bounds.append(mean_val - std_val)
                    upper_bounds.append(mean_val + std_val)

                plt.plot(episode_indices, means, marker='o', label=f"Trial {trial_index}")
                plt.fill_between(episode_indices, lower_bounds, upper_bounds, alpha=0.2)

            plt.title(f"Per-Seed Rewards: {algorithm} (Traffic: {traffic_label})")
            plt.xlabel("Episode (iter)")
            plt.ylabel("Reward (mean across steps)")
            plt.legend()
            plt.grid(True)
            plt.show()


def plot_rewards_averaged_with_variance(all_rewards_data):
    """
    Plot the average across all seeds, per episode, with variance shading
    across seeds (i.e., the standard deviation across seeds for that episode).
    """
    for algorithm, traffic_dict in all_rewards_data.items():
        for traffic_label, trial_dict in traffic_dict.items():
            plt.figure(figsize=(8, 5))

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

            plt.plot(all_episodes_arr, means, marker='o', label="Average across seeds")
            plt.fill_between(all_episodes_arr, lower_bounds, upper_bounds, alpha=0.2)
            plt.title(f"Averaged Rewards: {algorithm} (Traffic: {traffic_label})")
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
        plt.figure(figsize=(8, 6))
        for traffic_label, rewards in traffic_rewards.items():
            episodes = np.arange(len(rewards))
            plt.plot(episodes, rewards, linewidth=1.5, label=traffic_label)
        plt.xlabel("Number of Episodes")
        plt.ylabel("Reward")
        plt.title(f"Average Rewards for {algorithm}")
        plt.legend(title="Traffic Volume")
        plt.grid(True)
        plt.show()
