from reinforcement_learning.plotting.blocking import plot_blocking_probabilities
from reinforcement_learning.plotting.memory_usage import plot_memory_usage
from reinforcement_learning.plotting.rewards import plot_rewards_mean_var
from reinforcement_learning.plotting.sim_times import plot_sim_times
from reinforcement_learning.plotting.state_values import plot_best_path_matrix
from reinforcement_learning.plotting.mod_usage import plot_modulation_usage
from reinforcement_learning.plotting.bw_block import plot_bw_blocked
from reinforcement_learning.plotting.blocking import plot_effect_heatmaps
from reinforcement_learning.plotting.link_data import plot_link_usage, plot_link_throughput
from reinforcement_learning.plotting.path_index import plot_path_index
from reinforcement_learning.plotting.resource_stats import plot_resource_percent_delta_heatmaps

PLOTS = {
    "blocking": {"plot": plot_blocking_probabilities, "process": "process_blocking"},
    "memory": {"plot": plot_memory_usage, "process": "process_memory_usage"},
    "rewards": {"plot": plot_rewards_mean_var, "process": "process_rewards"},
    "sim_times": {"plot": plot_sim_times, "process": "process_sim_times"},
    "state_values": {"plot": plot_best_path_matrix, "process": "process_state_values"},
    "mod_usage": {
        "process": "process_modulation_usage",
        "plot": plot_modulation_usage,
    },
    "bw_block": {
        "process": "process_blocked_bandwidth",
        "plot": plot_bw_blocked,
    },
    "blocking_stats": {
        "plot": plot_effect_heatmaps,
        "process": "process_blocking",  # reuse existing processor
    },
    "resource_stats": {
        "process": "process_resource_metrics",
        "plot": plot_resource_percent_delta_heatmaps,
    },

    "link_usage": {
        "plot": plot_link_usage,
        "process": "process_link_data",
    },
    "link_throughput": {
        "plot": plot_link_throughput,
        "process": "process_link_data"
    },
    "path_index": {
        "plot": plot_path_index,
        "process": "process_path_index"
    }
}
