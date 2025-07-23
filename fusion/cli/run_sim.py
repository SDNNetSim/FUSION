# fusion/cli/run_sim.py

from fusion.cli.main_parser import build_parser
from fusion.cli.config_setup import setup_config_from_cli
from fusion.sim.network_simulator import run as run_simulation

def main():
    parser = build_parser()
    args = parser.parse_args()
    config = setup_config_from_cli(args)
    print("✅ Parsed Config:\n", config)

    # TODO: Not using stop_flag (multiprocessing) now, will fix later
    run_simulation(config, stop_flag=None)

if __name__ == "__main__":
    main()
