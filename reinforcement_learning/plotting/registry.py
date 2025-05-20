from reinforcement_learning.plotting.blocking import plot_blocking_probabilities
from reinforcement_learning.plotting.memory_usage import plot_memory_usage
from reinforcement_learning.plotting.rewards import plot_rewards_mean_var
from reinforcement_learning.plotting.sim_times import plot_sim_times
from reinforcement_learning.plotting.state_values import plot_best_path_matrix
from reinforcement_learning.plotting.mod_usage import plot_modulation_usage
from reinforcement_learning.plotting.bw_block import plot_bw_blocked
from reinforcement_learning.plotting.blocking import plot_blocking_stats_table
from reinforcement_learning.plotting.resource_stats import plot_resource_stats_table_entry

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
        "plot": plot_blocking_stats_table,
        "process": "process_blocking",  # reuse existing processor
    },
    "resource_stats": {
        "process": "process_resource_metrics",
        "plot": plot_resource_stats_table_entry,
    },

}
