# fusion/cli/main_parser.py

import argparse
from fusion.cli.args.run_sim_args import register_run_sim_args

def build_parser():
    parser = argparse.ArgumentParser(description='FUSION Simulator CLI')
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Register subcommand parsers here
    register_run_sim_args(subparsers)

    return parser
