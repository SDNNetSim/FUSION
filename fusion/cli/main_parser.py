# fusion/cli/main_parser.py

import argparse
from fusion.cli.args.run_sim_args import register_run_sim_args
from fusion.cli.args.run_sim_args import add_run_sim_args
from fusion.cli.args.routing_args import add_routing_args
from fusion.cli.args.spectrum_args import add_spectrum_args
from fusion.cli.args.snr_args import add_snr_args
from fusion.cli.args.sdn_args import add_sdn_args
from fusion.cli.args.stats_args import add_stats_args


def build_parser():
    parser = argparse.ArgumentParser(description='FUSION Simulator CLI')
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Register subcommand parsers here
    register_run_sim_args(subparsers)

    return parser

def get_train_args():
    parser = argparse.ArgumentParser(description="Train an agent (RL or ML)")
    add_run_sim_args(parser)
    add_routing_args(parser)
    add_spectrum_args(parser)
    add_snr_args(parser)
    add_sdn_args(parser)
    add_stats_args(parser)
    parser.add_argument("--agent_type", choices=["rl", "ml"], required=True, help="Type of agent to train")
    return parser.parse_args()

