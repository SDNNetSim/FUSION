"""
CLI arguments for traffic generation and request handling.
Handles traffic patterns, Erlang loads, and simulation parameters.
"""

import argparse


def add_erlang_args(parser: argparse.ArgumentParser) -> None:
    """
    Add Erlang load configuration arguments to the parser.
    """
    erlang_group = parser.add_argument_group("Erlang Load Configuration")
    erlang_group.add_argument(
        "--erlang_start", type=float, default=None, help="Starting Erlang load"
    )
    erlang_group.add_argument(
        "--erlang_stop", type=float, default=None, help="Ending Erlang load"
    )
    erlang_group.add_argument(
        "--erlang_step", type=float, default=None, help="Erlang load increment"
    )


def add_request_args(parser: argparse.ArgumentParser) -> None:
    """
    Add request handling arguments to the parser.
    """
    request_group = parser.add_argument_group("Request Configuration")
    request_group.add_argument(
        "--holding_time",
        type=float,
        default=None,
        help="Average holding time for requests",
    )
    request_group.add_argument(
        "--num_requests",
        type=int,
        default=None,
        help="Total number of requests to generate",
    )


def add_simulation_control_args(parser: argparse.ArgumentParser) -> None:
    """
    Add simulation control arguments to the parser.
    """
    control_group = parser.add_argument_group("Simulation Control")
    control_group.add_argument(
        "--max_iters",
        type=int,
        default=None,
        help="Maximum number of simulation iterations",
    )
    control_group.add_argument(
        "--thread_erlangs",
        action="store_true",
        help="Enable multi-threaded Erlang processing",
    )


def add_traffic_args(parser: argparse.ArgumentParser) -> None:
    """
    Add all traffic-related arguments to the parser.
    Convenience function that adds Erlang, request, and simulation control args.
    """
    add_erlang_args(parser)
    add_request_args(parser)
    add_simulation_control_args(parser)
