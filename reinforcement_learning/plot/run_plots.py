from reinforcement_learning.utils.sim_data import load_blocking_data, load_rewards
from reinforcement_learning.plot.blocking import plot_blocking_probabilities
from reinforcement_learning.plot.rewards import plot_average_rewards


def main():
    """
    Controls the plotting module.
    """
    # Base directories (update these paths as needed)
    base_dir = "./data/output/NSFNet/0227"
    base_logs_dir = "../../logs"

    # Dictionary of simulation times, keyed by algorithm.
    simulation_times = {
        "Epsilon greedy bandit": [
            ['15_23_01_094246'], ['15_26_56_957674'], ['15_27_00_995133'], ['15_27_02_126167'],
            ['15_27_02_142582'], ['15_27_02_142773'], ['15_27_03_358067'], ['15_27_03_883949'],
            ['15_27_03_884391'], ['15_27_03_884425'], ['15_27_03_884451'], ['15_27_05_600451'],
            ['15_27_05_942815'], ['15_27_06_913650']
        ],
        "Q learning": [
            ['15_26_53_549216'], ['15_26_55_059995'], ['15_26_59_656966'], ['15_27_02_126206'],
            ['15_27_02_142810'], ['15_27_03_883947'], ['15_27_03_884408'], ['15_27_03_884411'],
            ['15_27_03_884415'], ['15_27_03_884422'], ['15_27_05_600238'], ['15_27_05_600442'],
            ['15_27_05_607359'], ['15_27_05_943422'], ['15_27_07_958207']
        ],
        "UCB bandit": [
            ['15_26_55_060125'], ['15_26_56_957673'], ['15_27_00_767416'], ['15_27_01_460318'],
            ['15_27_02_126255'], ['15_27_02_126261'], ['15_27_02_142768'], ['15_27_02_144078'],
            ['15_27_03_883941'], ['15_27_03_883966'], ['15_27_03_884403'], ['15_27_03_884454'],
            ['15_27_03_884461'], ['15_27_05_600487'], ['15_27_05_944750']
        ],
        "PPO": [
            ['14_43_00_326842'], ['21_25_56_483843'], ['21_26_01_920140'], ['21_26_01_920146'],
            ['21_26_01_920162'], ['21_26_01_920204'], ['21_26_01_920206'], ['21_26_01_920214'],
            ['21_26_01_920476'], ['21_26_01_920477'], ['21_26_01_921120'], ['21_26_01_921603'],
            ['21_26_01_921615'], ['21_26_01_921828']
        ],
        # Baselines will be processed separately.
        "Baselines": [
            ['14_43_00_326842']
        ]
    }

    # Load simulation data.
    final_result = load_blocking_data(simulation_times, base_dir)
    rewards_data = load_rewards(simulation_times, base_logs_dir, base_dir)

    # Generate plots.
    plot_blocking_probabilities(final_result)
    plot_average_rewards(rewards_data)


if __name__ == '__main__':
    main()
