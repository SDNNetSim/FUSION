import matplotlib.pyplot as plt
import numpy as np


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
