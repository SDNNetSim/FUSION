"""
CLI arguments for routing and spectrum allocation configuration.
Handles routing algorithms, spectrum assignment, SNR/modulation, and SDN settings.
"""

import argparse

from .snr import add_snr_args


def add_routing_args(parser: argparse.ArgumentParser) -> None:
    """
    Add routing algorithm arguments to the parser.

    Configures path selection algorithms and routing parameters
    for optical network simulation.

    :param parser: ArgumentParser instance to add arguments to
    :type parser: argparse.ArgumentParser
    :return: None
    :rtype: None
    """
    routing_group = parser.add_argument_group("Routing Configuration")
    routing_group.add_argument(
        "--route_method", type=str, help="Routing algorithm method"
    )
    routing_group.add_argument(
        "--k_paths",
        type=int,
        default=None,
        help="Number of candidate paths for k-shortest path routing",
    )


def add_spectrum_args(parser: argparse.ArgumentParser) -> None:
    """
    Add spectrum assignment arguments to the parser.

    Configures spectrum allocation methods, guard slots, and
    priority settings for optical spectrum assignment.

    :param parser: ArgumentParser instance to add arguments to
    :type parser: argparse.ArgumentParser
    :return: None
    :rtype: None
    """
    spectrum_group = parser.add_argument_group("Spectrum Assignment Configuration")
    spectrum_group.add_argument(
        "--allocation_method", type=str, help="Spectrum allocation method"
    )
    spectrum_group.add_argument(
        "--guard_slots",
        type=int,
        default=None,
        help="Number of guard slots between allocations",
    )
    spectrum_group.add_argument(
        "--spectrum_priority",
        type=str,
        choices=["BSC", "CSB"],
        help="Priority order for multi-band allocation (BSC or CSB)",
    )


def add_modulation_args(parser: argparse.ArgumentParser) -> None:
    """
    Add SNR and modulation arguments to the parser.

    Delegates to the SNR module to avoid code duplication.
    Provides compatibility interface for routing-related modulation settings.

    :param parser: ArgumentParser instance to add arguments to
    :type parser: argparse.ArgumentParser
    :return: None
    :rtype: None
    """
    add_snr_args(parser)


def add_sdn_args(parser: argparse.ArgumentParser) -> None:
    """
    Add SDN and dynamic switching arguments to the parser.

    Configures Software-Defined Networking features including
    dynamic lightpath switching and core allocation policies.

    :param parser: ArgumentParser instance to add arguments to
    :type parser: argparse.ArgumentParser
    :return: None
    :rtype: None
    """
    sdn_group = parser.add_argument_group("SDN Configuration")
    sdn_group.add_argument(
        "--dynamic_lps",
        action="store_true",
        help="Enable SDN dynamic lightpath switching",
    )
    sdn_group.add_argument(
        "--single_core",
        action="store_true",
        help="Force single-core allocation per request",
    )


def add_all_routing_args(parser: argparse.ArgumentParser) -> None:
    """
    Add all routing-related arguments to the parser.

    Convenience function that combines routing algorithms, spectrum assignment,
    modulation settings, and SDN configuration in a single call.

    :param parser: ArgumentParser instance to add arguments to
    :type parser: argparse.ArgumentParser
    :return: None
    :rtype: None
    """
    add_routing_args(parser)
    add_spectrum_args(parser)
    add_modulation_args(parser)
    add_sdn_args(parser)
