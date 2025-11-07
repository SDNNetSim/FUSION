"""Main CLI argument parser using the centralized registry system."""

from argparse import ArgumentParser, Namespace

from .parameters.registry import args_registry

# Argument group configurations
TRAINING_GROUP_NAMES: list[str] = [
    "config",
    "debug",
    "simulation",
    "network",
    "traffic",
    "training",
    "statistics",
]
GUI_GROUP_NAMES: list[str] = ["gui", "debug", "output"]

# Agent type choices
AGENT_TYPE_CHOICES: list[str] = ["rl", "ml"]


def build_main_argument_parser() -> ArgumentParser:
    """
    Build the main CLI argument parser with all subcommands configured.

    Creates the primary argument parser that handles all CLI interactions
    for the FUSION simulator, including subcommands for different operations
    like training, GUI launch, and simulation execution.

    :return: Fully configured argument parser with subcommands
    :rtype: ArgumentParser
    """
    return args_registry.create_main_parser()


def create_training_argument_parser() -> Namespace:
    """
    Create and parse arguments for training simulations (RL or ML).

    Builds a specialized argument parser configured for machine learning
    and reinforcement learning training workflows. Includes all necessary
    argument groups for comprehensive training configuration.

    :return: Parsed command line arguments for training operations
    :rtype: Namespace
    :raises SystemExit: If required arguments are missing or invalid
    """
    training_parser = args_registry.create_parser_with_groups(
        "Train an agent (RL or ML)", TRAINING_GROUP_NAMES
    )
    training_parser.add_argument(
        "--agent_type",
        choices=AGENT_TYPE_CHOICES,
        required=True,
        help="Type of agent to train (rl=reinforcement learning, ml=machine learning)",
    )
    return training_parser.parse_args()


def create_gui_argument_parser() -> Namespace:
    """
    Create and parse arguments for GUI-based simulator interface.

    Builds a specialized argument parser configured for graphical user
    interface operations. Includes GUI-specific settings, debug options,
    and output configuration parameters.

    :return: Parsed command line arguments for GUI operations
    :rtype: Namespace
    :raises SystemExit: If argument parsing fails
    """
    gui_parser = args_registry.create_parser_with_groups(
        "Launch GUI for FUSION", GUI_GROUP_NAMES
    )
    return gui_parser.parse_args()


# Backward compatibility functions
def build_parser() -> ArgumentParser:
    """
    Legacy function name for building main argument parser.

    :return: Configured main parser
    :rtype: ArgumentParser
    :deprecated: Use build_main_argument_parser() instead
    """
    return build_main_argument_parser()


def get_train_args() -> Namespace:
    """
    Legacy function name for creating training argument parser.

    :return: Parsed arguments for training
    :rtype: Namespace
    :deprecated: Use create_training_argument_parser() instead
    """
    return create_training_argument_parser()


def get_gui_args() -> Namespace:
    """
    Legacy function name for creating GUI argument parser.

    :return: Parsed arguments for GUI
    :rtype: Namespace
    :deprecated: Use create_gui_argument_parser() instead
    """
    return create_gui_argument_parser()
