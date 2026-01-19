"""Main CLI argument parser using the centralized registry system."""

from argparse import ArgumentParser, Namespace

from .parameters.registry import args_registry


class GUINotSupportedError(Exception):
    """Raised when attempting to use the unsupported GUI module."""

    pass


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

# Agent type choices - RL (reinforcement learning) and SL (supervised learning)
AGENT_TYPE_CHOICES: list[str] = ["rl", "sl"]


def build_main_argument_parser() -> ArgumentParser:
    """
    Build the main CLI argument parser with all subcommands configured.

    Creates the primary argument parser that handles all CLI interactions
    for the FUSION simulator, including subcommands for training and
    simulation execution.

    :return: Fully configured argument parser with subcommands
    :rtype: ArgumentParser
    """
    return args_registry.create_main_parser()


def create_training_argument_parser() -> Namespace:
    """
    Create and parse arguments for training simulations.

    Builds a specialized argument parser configured for reinforcement
    learning (RL) and supervised learning (SL) training workflows.
    Includes all necessary argument groups for comprehensive training
    configuration.

    :return: Parsed command line arguments for training operations
    :rtype: Namespace
    :raises SystemExit: If required arguments are missing or invalid
    """
    training_parser = args_registry.create_parser_with_groups(
        "Train an agent using reinforcement learning (RL) or supervised learning (SL)",
        TRAINING_GROUP_NAMES,
    )
    training_parser.add_argument(
        "--agent_type",
        choices=AGENT_TYPE_CHOICES,
        required=True,
        help="Type of agent to train (rl=reinforcement learning, sl=supervised learning)",
    )
    return training_parser.parse_args()


def create_gui_argument_parser() -> Namespace:
    """
    Create and parse arguments for GUI-based simulator interface.

    :raises GUINotSupportedError: Always raised - GUI is not currently supported
    """
    raise GUINotSupportedError(
        "The FUSION GUI is currently under active development and not supported. "
        "This feature will be available in version 6.1.0. "
        "Please use the CLI interface instead: python -m fusion.cli.run_sim run_sim --config_path <path>"
    )


# TODO (v6.1.0): Remove backward compatibility functions below
def build_parser() -> ArgumentParser:
    """
    Legacy function name for building main argument parser.

    :return: Configured main parser
    :rtype: ArgumentParser
    :deprecated: Use build_main_argument_parser() instead
    """
    return build_main_argument_parser()


# TODO (v6.1.0): Remove backward compatibility function
def get_train_args() -> Namespace:
    """
    Legacy function name for creating training argument parser.

    :return: Parsed arguments for training
    :rtype: Namespace
    :deprecated: Use create_training_argument_parser() instead
    """
    return create_training_argument_parser()


# TODO (v6.1.0): Remove backward compatibility function
def get_gui_args() -> Namespace:
    """
    Legacy function name for creating GUI argument parser.

    :raises GUINotSupportedError: Always raised - GUI is not currently supported
    :deprecated: Use create_gui_argument_parser() instead
    """
    raise GUINotSupportedError(
        "The FUSION GUI is currently under active development and not supported. "
        "This feature will be available in version 6.1.0. "
        "Please use the CLI interface instead."
    )
