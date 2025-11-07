"""
CLI arguments for survivability experiments.
Handles failure injection, protection mechanisms, RL policies, and dataset logging.
"""

import argparse


def add_failure_args(parser: argparse.ArgumentParser) -> None:
    """
    Add failure injection configuration arguments to the parser.

    :param parser: Argument parser to add failure arguments to
    :type parser: argparse.ArgumentParser
    """
    failure_group = parser.add_argument_group("Failure Settings")
    failure_group.add_argument(
        "--failure_type",
        type=str,
        choices=["none", "link", "node", "srlg", "geo"],
        default=None,
        help="Type of failure to inject (none, link, node, srlg, geo)",
    )
    failure_group.add_argument(
        "--t_fail_arrival_index",
        type=int,
        default=None,
        help="Request arrival index when failure occurs (-1 = midpoint)",
    )
    failure_group.add_argument(
        "--t_repair_after_arrivals",
        type=int,
        default=None,
        help="Number of arrivals after failure until repair occurs",
    )
    failure_group.add_argument(
        "--failed_link_src",
        type=int,
        default=None,
        help="Source node of failed link (F1: link failure)",
    )
    failure_group.add_argument(
        "--failed_link_dst",
        type=int,
        default=None,
        help="Destination node of failed link (F1: link failure)",
    )
    failure_group.add_argument(
        "--failed_node_id",
        type=int,
        default=None,
        help="Node ID for node failure (F2: node failure)",
    )
    failure_group.add_argument(
        "--srlg_links",
        type=str,
        default=None,
        help='List of link tuples in SRLG (F3), e.g., "[(0,1), (2,3), (5,6)]"',
    )
    failure_group.add_argument(
        "--geo_center_node",
        type=int,
        default=None,
        help="Center node for geographic failure (F4: geographic failure)",
    )
    failure_group.add_argument(
        "--geo_hop_radius",
        type=int,
        default=None,
        help="Hop radius for geographic failure (F4: geographic failure)",
    )


def add_protection_args(parser: argparse.ArgumentParser) -> None:
    """
    Add protection mechanism configuration arguments to the parser.

    :param parser: Argument parser to add protection arguments to
    :type parser: argparse.ArgumentParser
    """
    protection_group = parser.add_argument_group("Protection Settings")
    protection_group.add_argument(
        "--protection_mode",
        type=str,
        choices=["none", "1plus1"],
        default=None,
        help="Protection mechanism to use (none, 1plus1)",
    )
    protection_group.add_argument(
        "--protection_switchover_ms",
        type=float,
        default=None,
        help="1+1 protection switchover latency in milliseconds",
    )
    protection_group.add_argument(
        "--restoration_latency_ms",
        type=float,
        default=None,
        help="Restoration compute + signaling latency in milliseconds",
    )
    protection_group.add_argument(
        "--revert_to_primary",
        action="store_true",
        help="Revert to primary path after repair",
    )


def add_offline_rl_args(parser: argparse.ArgumentParser) -> None:
    """
    Add RL policy configuration arguments to the parser.

    :param parser: Argument parser to add RL policy arguments to
    :type parser: argparse.ArgumentParser
    """
    rl_group = parser.add_argument_group("RL Policy Settings")
    rl_group.add_argument(
        "--policy_type",
        type=str,
        choices=["ksp_ff", "one_plus_one", "bc", "iql"],
        default=None,
        help="Policy to use for path selection (ksp_ff, one_plus_one, bc, iql)",
    )
    rl_group.add_argument(
        "--bc_model_path",
        type=str,
        default=None,
        help="Path to Behavior Cloning model (.pt file)",
    )
    rl_group.add_argument(
        "--iql_model_path",
        type=str,
        default=None,
        help="Path to IQL model (.pt file)",
    )
    rl_group.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "mps"],
        default=None,
        help="Compute device for model inference (cpu, cuda, mps)",
    )
    rl_group.add_argument(
        "--fallback_policy",
        type=str,
        choices=["ksp_ff", "one_plus_one"],
        default=None,
        help="Fallback policy when all actions are masked",
    )


def add_dataset_logging_args(parser: argparse.ArgumentParser) -> None:
    """
    Add offline dataset logging configuration arguments to the parser.

    :param parser: Argument parser to add dataset logging arguments to
    :type parser: argparse.ArgumentParser
    """
    dataset_group = parser.add_argument_group("Dataset Logging")
    dataset_group.add_argument(
        "--log_offline_dataset",
        action="store_true",
        help="Enable offline dataset logging for RL training",
    )
    dataset_group.add_argument(
        "--dataset_output_path",
        type=str,
        default=None,
        help="Output path for JSONL dataset file",
    )
    dataset_group.add_argument(
        "--epsilon_mix",
        type=float,
        default=None,
        help="Probability of selecting second-best path (0.0-1.0)",
    )


def add_recovery_timing_args(parser: argparse.ArgumentParser) -> None:
    """
    Add recovery timing configuration arguments to the parser.

    :param parser: Argument parser to add recovery timing arguments to
    :type parser: argparse.ArgumentParser
    """
    recovery_group = parser.add_argument_group("Recovery Timing")
    recovery_group.add_argument(
        "--failure_window_size",
        type=int,
        default=None,
        help="Number of arrivals for failure window blocking probability calculation",
    )


def add_reporting_args(parser: argparse.ArgumentParser) -> None:
    """
    Add results reporting configuration arguments to the parser.

    :param parser: Argument parser to add reporting arguments to
    :type parser: argparse.ArgumentParser
    """
    reporting_group = parser.add_argument_group("Survivability Reporting")
    reporting_group.add_argument(
        "--export_csv",
        action="store_true",
        help="Export results to CSV file",
    )
    reporting_group.add_argument(
        "--csv_output_path",
        type=str,
        default=None,
        help="CSV output path for results",
    )
    reporting_group.add_argument(
        "--aggregate_seeds",
        action="store_true",
        help="Aggregate results across multiple seeds",
    )
    reporting_group.add_argument(
        "--seed_list",
        type=int,
        nargs="+",
        default=None,
        help="List of seeds for multi-seed runs (space-separated integers)",
    )


def add_survivability_args(parser: argparse.ArgumentParser) -> None:
    """
    Add all survivability-related arguments to the parser.
    Convenience function that adds failure, protection, RL policy,
    dataset logging, recovery timing, and reporting arguments.

    :param parser: Argument parser to add all survivability arguments to
    :type parser: argparse.ArgumentParser
    """
    add_failure_args(parser)
    add_protection_args(parser)
    add_offline_rl_args(parser)
    add_dataset_logging_args(parser)
    add_recovery_timing_args(parser)
    add_reporting_args(parser)
