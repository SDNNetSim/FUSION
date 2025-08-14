"""
Common CLI argument definitions used across multiple commands.
Centralizes shared arguments to reduce duplication and ensure consistency.
"""

import argparse


def add_config_args(parser: argparse.ArgumentParser) -> None:
    """
    Add configuration-related arguments to the parser.
    These are common across all CLI commands.
    """
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to INI configuration file"
    )
    parser.add_argument(
        "--run_id",
        type=str,
        required=True,
        help="Unique identifier for this simulation run"
    )


def add_debug_args(parser: argparse.ArgumentParser) -> None:
    """
    Add debugging and logging arguments to the parser.
    """
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )


def add_output_args(parser: argparse.ArgumentParser) -> None:
    """
    Add output control arguments to the parser.
    """
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory to save output files"
    )
    parser.add_argument(
        "--save_results",
        action="store_true",
        help="Save simulation results to file"
    )


def add_plot_format_args(parser: argparse.ArgumentParser) -> None:
    """
    Add common plot format arguments to the parser.
    Used by both analysis and plotting commands.
    """
    parser.add_argument(
        "--plot_format",
        type=str,
        choices=["png", "pdf", "svg", "eps"],
        default="png",
        help="Output format for generated plots"
    )
