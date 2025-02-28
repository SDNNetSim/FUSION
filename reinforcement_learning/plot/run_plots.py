from reinforcement_learning.utils.sim_filters import find_times
from reinforcement_learning.utils.sim_data import load_blocking_data, load_rewards
from reinforcement_learning.plot.blocking import plot_blocking_probabilities
from reinforcement_learning.plot.rewards import plot_average_rewards


def main():
    """
    Controls the plots module.
    """
    dates_dict = {
        '0227': 'NSFNet',
        '0228': 'PanEuro'
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
        ]
    }
    sims_info_dict = find_times(dates_dict=dates_dict, filter_dict=filter_dict)
    simulation_times = {}
    for time_entry in sims_info_dict['times_matrix']:
        time_str = time_entry[0]
        simulation_times.setdefault("Example Algorithm", []).append([time_str])

    base_dir = "./data/output/NSFNet/0227"
    base_logs_dir = "../../logs"

    final_result = load_blocking_data(simulation_times, base_dir)
    rewards_data = load_rewards(simulation_times, base_logs_dir, base_dir)

    plot_blocking_probabilities(final_result)
    plot_average_rewards(rewards_data)


if __name__ == '__main__':
    main()
