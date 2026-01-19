"""
Common CLI argument definitions used across multiple commands.
Centralizes shared arguments to reduce duplication and ensure consistency.
"""

import argparse


def add_config_args(parser: argparse.ArgumentParser) -> None:
    """
    Add configuration-related arguments to the parser.

    Provides common configuration arguments used across all CLI commands,
    including configuration file path and run identification.

    :param parser: ArgumentParser instance to add arguments to
    :type parser: argparse.ArgumentParser
    :return: None
    :rtype: None
    """
    parser.add_argument("--config_path", type=str, required=True, help="Path to INI configuration file")
    parser.add_argument(
        "--run_id",
        type=str,
        required=True,
        help="Unique identifier for this simulation run",
    )


def add_debug_args(parser: argparse.ArgumentParser) -> None:
    """
    Add debugging and logging arguments to the parser.

    Configures debug output control and verbose logging options
    for troubleshooting and development.

    :param parser: ArgumentParser instance to add arguments to
    :type parser: argparse.ArgumentParser
    :return: None
    :rtype: None
    """
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")


def add_output_args(parser: argparse.ArgumentParser) -> None:
    """
    Add output control arguments to the parser.

    Configures output directory, file saving options, snapshot settings,
    and result formatting parameters for simulation data management.

    :param parser: ArgumentParser instance to add arguments to
    :type parser: argparse.ArgumentParser
    :return: None
    :rtype: None
    """
    parser.add_argument("--output_dir", type=str, help="Directory to save output files")
    parser.add_argument("--save_results", action="store_true", help="Save simulation results to file")
    parser.add_argument("--save_snapshots", action="store_true", help="Save simulation snapshots")
    parser.add_argument("--snapshot_step", type=int, help="Step interval for saving snapshots")
    parser.add_argument("--print_step", type=int, help="Step interval for printing progress")
    parser.add_argument("--save_step", type=int, help="Step interval for saving results")
    parser.add_argument(
        "--save_start_end_slots",
        action="store_true",
        help="Save start and end slots information",
    )
    parser.add_argument("--file_type", type=str, help="Output file format type")
    parser.add_argument("--filter_mods", action="store_true", help="Enable modulation filtering")


# TODO (v6.1.0): Plotting functionality is not currently supported. These arguments
# are defined for future use when plotting is implemented.
def add_plot_format_args(
    parser: argparse.ArgumentParser | argparse._ArgumentGroup,
) -> None:
    """
    Add common plot format arguments to the parser.

    Provides standardized plot format options used by both analysis
    and plotting commands for consistent output formatting.

    NOTE: Plotting is not currently supported. These arguments are defined
    for future use and will be functional in v6.1.0.

    :param parser: ArgumentParser or ArgumentGroup instance to add arguments to
    :type parser: argparse.ArgumentParser | argparse._ArgumentGroup
    :return: None
    :rtype: None
    """
    parser.add_argument(
        "--plot_format",
        type=str,
        default=None,
        help="Output format for generated plots (not currently supported)",
    )
