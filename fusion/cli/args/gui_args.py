"""
GUI launcher CLI arguments.
Arguments specific to launching and configuring the graphical interface.
"""

import argparse

from .common_args import add_config_args


def add_gui_args(parser: argparse.ArgumentParser) -> None:
    """
    Add GUI launcher arguments to the parser.

    Args:
        parser: ArgumentParser instance to add arguments to
    """
    # Basic configuration (uses common_args now)
    add_config_args(parser)

    # GUI-specific arguments
    # Note: Currently no GUI-specific arguments are implemented
