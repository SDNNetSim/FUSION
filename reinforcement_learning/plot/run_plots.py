import os

from reinforcement_learning.utils.sim_filters import find_times
from reinforcement_learning.utils.sim_data import (
    load_blocking_data,
    load_rewards,
    load_all_rewards_files,
    load_and_average_state_values,
    load_memory_usage,
)
from reinforcement_learning.plot.blocking import plot_blocking_probabilities
from reinforcement_learning.plot.state_values import plot_best_path_matrix
from reinforcement_learning.plot.sim_times import plot_sim_times
from reinforcement_learning.plot.memory_usage import plot_memory_usage
from reinforcement_learning.plot.rewards import (
    plot_rewards_per_seed,
    plot_rewards_averaged_with_variance,
    plot_average_rewards,
)

DATE = '0312'
NETWORK = 'NSFNet'


def main():
    """
    Controls the plotting module.
    """
    dates_dict = {
        DATE: NETWORK,
    }

    filter_dict = {
        'not_filter_list': [
            # Example: [['config', 'exclude'], 'value_to_exclude']
            # ['penalty', -1]
            ['path_algorithm', 'ppo']
        ],
        'or_filter_list': [
            # Example: [['config', 'include'], 'desired_value']
        ],
        'and_filter_list': [
            # Example: [['config', 'must_have'], 'expected_value']
            # ['path_algorithm', 'q_learning']
        ]
    }
    sims_info_dict = find_times(dates_dict=dates_dict, filter_dict=filter_dict)
    simulation_times = {}
    for idx, time_entry in enumerate(sims_info_dict['times_matrix']):
        time_str = time_entry[0]
        algorithms_matrix = sims_info_dict.get('algorithms_matrix', [])
        if idx < len(algorithms_matrix) and algorithms_matrix[idx]:
            algo_name = algorithms_matrix[idx][0]
        else:
            algo_name = "Unknown Algorithm"
        simulation_times.setdefault(algo_name, []).append([time_str])

    base_dir = os.path.join('..', '..', 'data', 'output', NETWORK, DATE)
    base_logs_dir = os.path.join('..', '..', 'logs')

    final_result = load_blocking_data(simulation_times, base_dir)
    rewards_data = load_rewards(simulation_times, base_logs_dir, base_dir, network=NETWORK, date=DATE)
    all_rewards_data = load_all_rewards_files(simulation_times, base_logs_dir, base_dir, date=DATE, network=NETWORK)
    averaged_state_values_by_volume = load_and_average_state_values(simulation_times, base_logs_dir, base_dir=base_dir, date=DATE, network=NETWORK)
    memory_usage_data = load_memory_usage(simulation_times, base_logs_dir, base_dir, network=NETWORK, date=DATE)

    # plot_blocking_probabilities(final_result)
    # plot_best_path_matrix(averaged_state_values_by_volume)  # plot_blocking_probabilities(final_result)
    # plot_rewards_per_seed(all_rewards_data)
    # plot_rewards_averaged_with_variance(all_rewards_data)
    # plot_average_rewards(rewards_data)
    plot_sim_times(simulation_times, base_dir)
    plot_memory_usage(memory_usage_data)
    # plot_sim_times(simulation_times, base_dir)
    plot_memory_usage(memory_usage_data)


if __name__ == '__main__':
    main()
