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
    erlang_group.add_argument("--erlang_start", type=float, default=None, help="Starting Erlang load")
    erlang_group.add_argument("--erlang_stop", type=float, default=None, help="Ending Erlang load")
    erlang_group.add_argument("--erlang_step", type=float, default=None, help="Erlang load increment")


def add_request_args(parser: argparse.ArgumentParser) -> None:
    """
    Add request handling arguments to the parser.
    """
    request_group = parser.add_argument_group("Request Configuration")
    request_group.add_argument("--traffic_matrix_sheet", default=None, help="Excel matrix worksheet name")
    request_group.add_argument(
        "--traffic_matrix_direction",
        choices=("directed", "undirected"),
        default=None,
        help="Undirected reads each symmetric matrix pair once",
    )
    request_group.add_argument("--traffic_type", choices=("dynamic", "static", "semi_static"), default=None)
    request_group.add_argument(
        "--request_sort", choices=("none", "bandwidth", "shortest_path", "hop_count", "bandwidth_distance"), default=None
    )
    request_group.add_argument("--request_sort_order", choices=("ascending", "descending"), default=None)
    request_group.add_argument("--traffic_matrix_file", default=None, help="Local traffic file; relative paths use the config directory")
    request_group.add_argument(
        "--traffic_matrix_format", choices=("auto", "csv", "json", "matrix_csv", "matrix_xlsx", "sndlib"), default=None
    )
    request_group.add_argument("--traffic_matrix_scale", type=float, default=None, help="Multiply imported bandwidth values")
    request_group.add_argument("--traffic_node_map_file", default=None, help="JSON mapping dataset nodes to topology nodes")
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
