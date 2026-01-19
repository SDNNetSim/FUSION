"""
GUI launcher CLI arguments.

NOTE: The GUI is currently under active development and not supported.
This module will be updated in version 6.1.0.
"""

import argparse


class GUINotSupportedError(Exception):
    """Raised when attempting to use the unsupported GUI module."""

    pass


def add_gui_args(parser: argparse.ArgumentParser) -> None:
    """
    Add GUI launcher arguments to the parser.

    :param parser: ArgumentParser instance to add arguments to
    :raises GUINotSupportedError: Always raised - GUI is not currently supported
    """
    raise GUINotSupportedError(
        "The FUSION GUI is currently under active development and not supported. "
        "This feature will be available in version 6.1.0. "
        "Please use the CLI interface instead."
    )


def add_all_gui_args(parser: argparse.ArgumentParser) -> None:
    """
    Add all GUI-related argument groups to the parser.

    :param parser: ArgumentParser instance to add arguments to
    :raises GUINotSupportedError: Always raised - GUI is not currently supported
    """
    raise GUINotSupportedError(
        "The FUSION GUI is currently under active development and not supported. "
        "This feature will be available in version 6.1.0. "
        "Please use the CLI interface instead."
    )
