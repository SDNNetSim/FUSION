"""
SNR measurement CLI arguments.
Deprecated: Use simulation_args.py instead.
Kept for backward compatibility.
"""

import argparse

def add_snr_args(parser: argparse.ArgumentParser) -> None:
    """
    Add SNR measurement arguments to the parser.

    Args:
        parser: ArgumentParser instance to add arguments to
    """
    parser.add_argument(
        "--mod_assumption",
        type=str,
        choices=["fixed", "adaptive", "precalculated"],
        help="Modulation format selection strategy"
    )
    parser.add_argument(
        "--mod_assumption_path",
        type=str,
        help="Path to modulation format configuration file"
    )
    parser.add_argument(
        "--snr_type",
        type=str,
        choices=["linear", "nonlinear", "egn"],
        help="SNR calculation method"
    )
    parser.add_argument(
        "--input_power",
        type=float,
        default=1e-3,
        help="Input power in Watts"
    )
