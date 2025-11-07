"""
CLI arguments for simulation configuration - compatibility module.
Imports and combines arguments from routing, network, and traffic modules.
"""

import argparse

from .network import add_network_args
from .routing import add_all_routing_args
from .shared import add_config_args, add_debug_args, add_output_args
from .survivability import add_survivability_args
from .traffic import add_traffic_args
from .training import add_machine_learning_args


def add_simulation_args(parser: argparse.ArgumentParser) -> None:
    """
    Add comprehensive simulation arguments to the parser.
    Includes routing, spectrum assignment, SNR, and SDN configuration.

    This is a compatibility function that maintains the original interface
    while delegating to the new focused modules.
    """
    add_all_routing_args(parser)


def add_run_sim_args(parser: argparse.ArgumentParser) -> None:
    """
    Add run_sim specific arguments. Legacy compatibility function.
    This consolidates arguments from routing, network, and traffic groups.
    """
    add_simulation_args(parser)
    add_network_args(parser)
    add_traffic_args(parser)


def register_run_sim_args(subparsers: argparse._SubParsersAction) -> None:
    """
    Register run_sim subcommand parser. Legacy compatibility function.
    """
    parser = subparsers.add_parser("run_sim", help="Run network simulation")
    add_config_args(parser)
    add_debug_args(parser)
    add_output_args(parser)
    add_run_sim_args(parser)
    add_machine_learning_args(parser)
    add_survivability_args(parser)
