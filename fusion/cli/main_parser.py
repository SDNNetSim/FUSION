# fusion/cli/main_parser.py

"""
Main CLI argument parser using the centralized registry system.
Implements the architecture plan's requirement for clean argument parsing.
"""

from .args.registry import args_registry


def build_parser():
    """
    Builds the main argument parser with subcommands.

    Returns:
        ArgumentParser: Configured main parser
    """
    return args_registry.create_main_parser()


def get_train_args():
    """
    Builds the argument parser for training simulations (RL or ML).

    Returns:
        Parsed arguments for training
    """
    train_groups = ["config", "debug", "simulation", "network", "traffic", "training", "statistics"]
    parser = args_registry.create_parser_with_groups(
        "Train an agent (RL or ML)",
        train_groups
    )
    parser.add_argument(
        "--agent_type",
        choices=["rl", "ml"],
        required=True,
        help="Type of agent to train"
    )
    return parser.parse_args()


def get_gui_args():
    """
    Builds the argument parser for GUI simulations.

    Returns:
        Parsed arguments for GUI
    """
    gui_groups = ["gui", "debug", "output"]
    parser = args_registry.create_parser_with_groups(
        "Launch GUI for FUSION",
        gui_groups
    )
    return parser.parse_args()
