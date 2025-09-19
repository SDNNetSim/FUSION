"""
GUI launcher CLI arguments.
Arguments specific to launching and configuring the graphical interface.
"""

import argparse

from .shared import add_config_args


def add_gui_args(parser: argparse.ArgumentParser) -> None:
    """
    Add GUI launcher arguments to the parser.
    
    Currently includes only basic configuration arguments.
    GUI-specific arguments will be added as needed.
    
    :param parser: ArgumentParser instance to add arguments to
    :type parser: argparse.ArgumentParser
    :return: None
    :rtype: None
    """
    # Basic configuration (shared with other modules)
    add_config_args(parser)

    # GUI-specific arguments pending - see cli/TODO.md for planned features
    # Future: window size, theme, display options, etc.


def add_all_gui_args(parser: argparse.ArgumentParser) -> None:
    """
    Add all GUI-related argument groups to the parser.
    
    Convenience function that adds all GUI arguments in a single call.
    Alias for add_gui_args to maintain consistency with other modules.
    
    :param parser: ArgumentParser instance to add arguments to
    :type parser: argparse.ArgumentParser
    :return: None
    :rtype: None
    """
    add_gui_args(parser)
