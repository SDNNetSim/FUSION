"""
CLI arguments for network topology and physical layer configuration.
Handles network topology, physical parameters, and spectrum band settings.
"""

import argparse


def add_topology_args(parser: argparse.ArgumentParser) -> None:
    """
    Add network topology configuration arguments to the parser.

    Configures network topology selection, core counts, bandwidth parameters,
    and link configuration options for optical network simulation.

    :param parser: ArgumentParser instance to add arguments to
    :type parser: argparse.ArgumentParser
    :return: None
    :rtype: None
    """
    network_group = parser.add_argument_group("Network Topology Configuration")
    network_group.add_argument(
        "--network",
        type=str,
        help="Network topology name (e.g., 'NSFNet', 'USbackbone60')",
    )
    network_group.add_argument(
        "--cores_per_link", type=int, default=1, help="Number of cores per fiber link"
    )
    network_group.add_argument(
        "--bw_per_slot",
        type=float,
        default=None,
        help="Bandwidth per spectral slot in GHz",
    )


def add_link_args(parser: argparse.ArgumentParser) -> None:
    """
    Add link configuration arguments to the parser.

    Configures link behavior including weights, directionality,
    and multi-fiber options for network links.

    :param parser: ArgumentParser instance to add arguments to
    :type parser: argparse.ArgumentParser
    :return: None
    :rtype: None
    """
    link_group = parser.add_argument_group("Link Configuration")
    link_group.add_argument(
        "--const_link_weight",
        action="store_true",
        help="Use constant link weights for routing",
    )
    link_group.add_argument(
        "--bi_directional", action="store_true", help="Enable bidirectional links"
    )
    link_group.add_argument(
        "--multi_fiber", action="store_true", help="Enable multi-fiber links"
    )


def add_node_args(parser: argparse.ArgumentParser) -> None:
    """
    Add network node configuration arguments to the parser.

    Configures node behavior and restrictions for network simulation.

    :param parser: ArgumentParser instance to add arguments to
    :type parser: argparse.ArgumentParser
    :return: None
    :rtype: None
    """
    node_group = parser.add_argument_group("Node Configuration")
    node_group.add_argument(
        "--is_only_core_node",
        action="store_true",
        help="Only allow core nodes to send requests",
    )


def add_spectrum_bands_args(parser: argparse.ArgumentParser) -> None:
    """
    Add spectrum band configuration arguments to the parser.

    Configures the number of spectral slots available in different
    optical transmission bands (C, L, S, E, O bands).

    :param parser: ArgumentParser instance to add arguments to
    :type parser: argparse.ArgumentParser
    :return: None
    :rtype: None
    """
    spectrum_group = parser.add_argument_group("Spectrum Band Configuration")
    spectrum_group.add_argument(
        "--c_band",
        type=int,
        default=96,
        help="Number of spectral slots in C-band (1530-1565nm)",
    )
    spectrum_group.add_argument(
        "--l_band",
        type=int,
        default=0,
        help="Number of spectral slots in L-band (1565-1625nm)",
    )
    spectrum_group.add_argument(
        "--s_band",
        type=int,
        default=0,
        help="Number of spectral slots in S-band (1460-1530nm)",
    )
    spectrum_group.add_argument(
        "--e_band",
        type=int,
        default=0,
        help="Number of spectral slots in E-band (1360-1460nm)",
    )
    spectrum_group.add_argument(
        "--o_band",
        type=int,
        default=0,
        help="Number of spectral slots in O-band (1260-1360nm)",
    )


def add_network_args(parser: argparse.ArgumentParser) -> None:
    """
    Add all network-related arguments to the parser.

    Combines topology, link, node, and spectrum band configuration
    arguments for comprehensive network setup.

    :param parser: ArgumentParser instance to add arguments to
    :type parser: argparse.ArgumentParser
    :return: None
    :rtype: None
    """
    add_topology_args(parser)
    add_link_args(parser)
    add_node_args(parser)
    add_spectrum_bands_args(parser)


def add_all_network_args(parser: argparse.ArgumentParser) -> None:
    """
    Add all network-related argument groups to the parser.

    Convenience function that adds all network arguments in a single call.
    Alias for add_network_args to maintain consistency with other modules.

    :param parser: ArgumentParser instance to add arguments to
    :type parser: argparse.ArgumentParser
    :return: None
    :rtype: None
    """
    add_network_args(parser)
