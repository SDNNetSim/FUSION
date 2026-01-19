"""
CLI arguments for policy, protection, and heuristic configuration (P5.6).

This module provides command-line argument definitions for:
- Policy selection and configuration
- Protection pipeline settings
- Heuristic policy parameters

Phase: P5.6 - Configuration + CLI Integration
"""

import argparse


def add_policy_args(parser: argparse.ArgumentParser) -> None:
    """
    Add policy configuration arguments.

    Args:
        parser: ArgumentParser to add arguments to
    """
    group = parser.add_argument_group("Policy Configuration (P5.6)")

    group.add_argument(
        "--policy-type",
        type=str,
        choices=["heuristic", "ml", "rl"],
        default=None,
        help="Type of policy: heuristic, ml, or rl (default: heuristic)",
    )

    group.add_argument(
        "--policy-name",
        type=str,
        choices=[
            "first_feasible",
            "shortest",
            "shortest_feasible",
            "least_congested",
            "random",
            "random_feasible",
            "load_balanced",
        ],
        default=None,
        help="Heuristic policy name (default: first_feasible)",
    )

    group.add_argument(
        "--policy-model-path",
        type=str,
        default=None,
        help="Path to ML/RL model file for ml/rl policy types",
    )

    group.add_argument(
        "--policy-fallback",
        type=str,
        default=None,
        help="Fallback policy name when ML/RL fails (default: first_feasible)",
    )

    group.add_argument(
        "--policy-k-paths",
        type=int,
        default=None,
        help="Number of candidate paths for policy (default: 3)",
    )

    group.add_argument(
        "--policy-seed",
        type=int,
        default=None,
        help="Random seed for policy (affects random policies)",
    )

    group.add_argument(
        "--policy-algorithm",
        type=str,
        default=None,
        help="RL algorithm name for rl policy type (e.g., PPO, MaskablePPO)",
    )

    group.add_argument(
        "--policy-device",
        type=str,
        choices=["cpu", "cuda", "auto"],
        default=None,
        help="Device for ML/RL inference (default: cpu)",
    )


def add_heuristic_args(parser: argparse.ArgumentParser) -> None:
    """
    Add heuristic policy configuration arguments.

    Args:
        parser: ArgumentParser to add arguments to
    """
    group = parser.add_argument_group("Heuristic Policy Configuration (P5.6)")

    group.add_argument(
        "--heuristic-alpha",
        type=float,
        default=None,
        help="Alpha parameter for LoadBalancedPolicy (0.0-1.0, default: 0.5)",
    )

    group.add_argument(
        "--heuristic-seed",
        type=int,
        default=None,
        help="Random seed for RandomFeasiblePolicy",
    )


def add_protection_args(parser: argparse.ArgumentParser) -> None:
    """
    Add protection pipeline configuration arguments.

    Args:
        parser: ArgumentParser to add arguments to
    """
    group = parser.add_argument_group("Protection Configuration (P5.6)")

    group.add_argument(
        "--protection-enabled",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=None,
        help="Enable 1+1 protection (default: False)",
    )

    group.add_argument(
        "--disjointness-type",
        type=str,
        choices=["link", "node"],
        default=None,
        help="Type of path disjointness: link or node (default: link)",
    )

    group.add_argument(
        "--protection-switchover-ms",
        type=float,
        default=None,
        help="Protection switchover latency in milliseconds (default: 50.0)",
    )

    group.add_argument(
        "--restoration-latency-ms",
        type=float,
        default=None,
        help="Restoration latency in milliseconds (default: 100.0)",
    )


def add_all_phase5_args(parser: argparse.ArgumentParser) -> None:
    """
    Add all Phase 5 configuration arguments.

    Convenience function to add all policy, heuristic, and protection
    arguments to a parser.

    Args:
        parser: ArgumentParser to add arguments to
    """
    add_policy_args(parser)
    add_heuristic_args(parser)
    add_protection_args(parser)
