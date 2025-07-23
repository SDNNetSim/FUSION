# fusion/cli/run_sim.py

from fusion.cli.main_parser import build_parser
from fusion.cli.config_setup import setup_config_from_cli

def main():
    parser = build_parser()
    args = parser.parse_args()
    config = setup_config_from_cli(args)

    # TEMP: Stub to confirm CLI + config work
    print("✅ Parsed Config:\n", config)

if __name__ == "__main__":
    main()
