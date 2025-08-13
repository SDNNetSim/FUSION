# fusion/cli/run_sim.py

import multiprocessing

from fusion.cli.main_parser import build_parser
from fusion.cli.config_setup import setup_config_from_cli
from fusion.sim.network_simulator import run as run_simulation


def main(stop_flag):
    """
    Controls the run_sim script.
    Entry point for running simulations from the command line.
    Parses arguments and delegates to the simulation runner.
    """
    parser = build_parser()
    args = parser.parse_args()

    config = setup_config_from_cli(args)
    print("✅ Parsed Config:\n", config)

    run_simulation(config, stop_flag=stop_flag)


if __name__ == "__main__":
    stop_flag = multiprocessing.Event()
    main(stop_flag)
