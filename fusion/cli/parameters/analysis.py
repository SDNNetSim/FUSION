"""
CLI arguments for analysis, plotting, and statistics.
Consolidates visualization and data analysis arguments.
"""

import argparse

from .shared import add_plot_format_args


def add_statistics_args(parser: argparse.ArgumentParser) -> None:
    """
    Add statistics collection and analysis arguments to the parser.

    Configures arguments for simulation state snapshots, progress tracking,
    and detailed result saving during simulation execution.

    :param parser: ArgumentParser instance to add arguments to
    :type parser: argparse.ArgumentParser
    :return: None
    :rtype: None
    """
    stats_group = parser.add_argument_group("Statistics Configuration")
    stats_group.add_argument(
        "--save_snapshots",
        action="store_true",
        help="Save simulation state snapshots during execution",
    )
    stats_group.add_argument(
        "--snapshot_step",
        type=int,
        default=100,
        help="Number of requests between snapshots",
    )
    stats_group.add_argument(
        "--save_step",
        type=int,
        default=1000,
        help="Number of requests between saving results",
    )
    stats_group.add_argument(
        "--print_step",
        type=int,
        default=1000,
        help="Number of requests between progress updates",
    )
    stats_group.add_argument(
        "--save_start_end_slots",
        action="store_true",
        help="Save detailed slot allocation information",
    )


def add_plotting_args(parser: argparse.ArgumentParser) -> None:
    """
    Add plotting and visualization arguments to the parser.

    Configures arguments for plot generation, display options, and
    visualization formatting parameters.

    :param parser: ArgumentParser instance to add arguments to
    :type parser: argparse.ArgumentParser
    :return: None
    :rtype: None
    """
    plot_group = parser.add_argument_group("Plotting Configuration")
    plot_group.add_argument(
        "--plot_results",
        action="store_true",
        help="Generate plots from simulation results",
    )
    add_plot_format_args(plot_group)
    plot_group.add_argument(
        "--plot_dpi", type=int, default=300, help="Resolution (DPI) for generated plots"
    )
    plot_group.add_argument(
        "--show_plots",
        action="store_true",
        help="Display plots interactively (in addition to saving)",
    )


def add_export_args(parser: argparse.ArgumentParser) -> None:
    """
    Add data export and file handling arguments to the parser.

    Configures arguments for exporting simulation results to various
    formats including Excel, CSV, JSON, and TSV.

    :param parser: ArgumentParser instance to add arguments to
    :type parser: argparse.ArgumentParser
    :return: None
    :rtype: None
    """
    export_group = parser.add_argument_group("Export Configuration")
    export_group.add_argument(
        "--file_type",
        type=str,
        choices=["json", "csv", "excel", "tsv"],
        default="json",
        help="Default file format for data export",
    )


def add_filtering_args(parser: argparse.ArgumentParser) -> None:
    """
    Add arguments for filtering and post-processing simulation results.

    Configures filters for modulation formats, Erlang load ranges,
    core exclusions, and time window analysis.

    :param parser: ArgumentParser instance to add arguments to
    :type parser: argparse.ArgumentParser
    :return: None
    :rtype: None
    """
    filter_group = parser.add_argument_group("Filtering Configuration")
    filter_group.add_argument(
        "--filter_mods", action="store_true", help="Filter results by modulation format"
    )
    filter_group.add_argument(
        "--min_erlang", type=float, help="Minimum Erlang load to include in analysis"
    )
    filter_group.add_argument(
        "--max_erlang", type=float, help="Maximum Erlang load to include in analysis"
    )


def add_comparison_args(parser: argparse.ArgumentParser) -> None:
    """
    Add arguments for comparing multiple simulation runs.

    Configures arguments for multi-run comparisons, statistical tests,
    and baseline analysis across different simulation executions.

    :param parser: ArgumentParser instance to add arguments to
    :type parser: argparse.ArgumentParser
    :return: None
    :rtype: None
    """
    comparison_group = parser.add_argument_group("Comparison Configuration")
    comparison_group.add_argument(
        "--compare_runs", type=str, nargs="+", help="List of run IDs to compare"
    )
    comparison_group.add_argument(
        "--baseline_run", type=str, help="Run ID to use as baseline for comparison"
    )
    comparison_group.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        choices=[
            "blocking_probability",
            "path_length",
            "execution_time",
            "resource_utilization",
        ],
        help="Metrics to include in comparison",
    )
    comparison_group.add_argument(
        "--significance_test",
        type=str,
        choices=["t_test", "wilcoxon", "mann_whitney"],
        help="Statistical test for comparing results",
    )


def add_all_analysis_args(parser: argparse.ArgumentParser) -> None:
    """
    Add all analysis-related argument groups to the parser.

    Convenience function that adds statistics, plotting, export,
    and filtering arguments in a single call.

    :param parser: ArgumentParser instance to add arguments to
    :type parser: argparse.ArgumentParser
    :return: None
    :rtype: None
    """
    add_statistics_args(parser)
    add_plotting_args(parser)
    add_export_args(parser)
    add_comparison_args(parser)
    add_filtering_args(parser)
