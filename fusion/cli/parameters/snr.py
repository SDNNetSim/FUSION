"""
CLI arguments for SNR and modulation configuration.
Handles signal-to-noise ratio calculations and modulation format selection.
"""

import argparse


def add_snr_args(parser: argparse.ArgumentParser) -> None:
    """
    Add SNR measurement and modulation arguments to the parser.

    Configures signal-to-noise ratio calculation methods, modulation format
    selection strategies, and physical layer parameters.

    :param parser: ArgumentParser instance to add arguments to
    :type parser: argparse.ArgumentParser
    :return: None
    :rtype: None
    """
    parser.add_argument("--mod_assumption", type=str, help="Modulation format selection strategy")
    parser.add_argument(
        "--mod_assumption_path",
        type=str,
        help="Path to modulation format configuration file",
    )
    parser.add_argument(
        "--snr_type",
        type=str,
        choices=["linear", "nonlinear", "egn"],
        help="SNR calculation method",
    )
    parser.add_argument("--input_power", type=float, default=None, help="Input power in Watts")
    parser.add_argument(
        "--egn_model",
        action="store_true",
        help="Enable Enhanced Gaussian Noise (EGN) model for SNR calculations",
    )
