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
    parser.add_argument(
        "--theme",
        type=str,
        choices=["dark", "light", "auto"],
        default="auto",
        help="GUI theme selection"
    )
    parser.add_argument(
        "--geometry",
        type=str,
        help="Window geometry (format: 'widthxheight+x+y')"
    )
    parser.add_argument(
        "--fullscreen",
        action="store_true",
        help="Launch GUI in fullscreen mode"
    )
    parser.add_argument(
        "--no_splash",
        action="store_true",
        help="Disable splash screen on startup"
    )
