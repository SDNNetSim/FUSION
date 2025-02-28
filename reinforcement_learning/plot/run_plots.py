import os

from reinforcement_learning.utils.sim_filters import find_times
from reinforcement_learning.utils.sim_data import load_blocking_data, load_rewards
from reinforcement_learning.plot.blocking import plot_blocking_probabilities
from reinforcement_learning.plot.rewards import plot_average_rewards


def main():
    """
    Controls the plots module.
    """
    dates_dict = {
        '0228': 'NSFNet',
    }

    filter_dict = {
        'not_filter_list': [
            # Example: [['config', 'exclude'], 'value_to_exclude']
        ],
        'or_filter_list': [
            # Example: [['config', 'include'], 'desired_value']
        ],
        'and_filter_list': [
            # Example: [['config', 'must_have'], 'expected_value']
            ['path_algorithm', 'epsilon_greedy_bandit']
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

    # TODO: (drl_path_agents) Hard coded date
    base_dir = os.path.join('..', '..', 'data', 'output', 'NSFNet', '0228')
    base_logs_dir = os.path.join('..', '..', 'logs')

    final_result = load_blocking_data(simulation_times, base_dir)
    rewards_data = load_rewards(simulation_times, base_logs_dir, base_dir)

    plot_blocking_probabilities(final_result)
    plot_average_rewards(rewards_data)


if __name__ == '__main__':
    main()
