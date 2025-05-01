"""
Maps a *plot nickname* to:

  • the plotting function         (from your updated plotter scripts)
  • the corresponding processor   (function name inside processors.py)
  • the default file-stem used when saving (optional convenience)
"""
from reinforcement_learning.plotting.blocking import plot_blocking_probabilities
from reinforcement_learning.plotting.memory_usage import plot_memory_usage
from reinforcement_learning.plotting.rewards import plot_rewards_mean_var
from reinforcement_learning.plotting.sim_times import plot_sim_times
from reinforcement_learning.plotting.state_values import plot_best_path_matrix

PLOTS = {
    "blocking": {"plot": plot_blocking_probabilities, "process": "process_blocking"},
    "memory": {"plot": plot_memory_usage, "process": "process_memory_usage"},
    "rewards": {"plot": plot_rewards_mean_var, "process": "process_rewards"},
    "sim_times": {"plot": plot_sim_times, "process": "process_sim_times"},
    "state_vals": {"plot": plot_best_path_matrix, "process": "process_rewards"},
    # state-values uses same raw format as rewards trials
}
