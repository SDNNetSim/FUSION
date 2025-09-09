"""
CLI arguments for analysis, plotting, and statistics.
Consolidates visualization and data analysis arguments.
"""

import argparse

from .common_args import add_plot_format_args


def add_statistics_args(parser: argparse.ArgumentParser) -> None:
    """
    Add statistics collection and analysis arguments.
    """
    stats_group = parser.add_argument_group('Statistics Configuration')
    stats_group.add_argument(
        "--save_snapshots",
        action="store_true",
        help="Save simulation state snapshots during execution"
    )
    stats_group.add_argument(
        "--snapshot_step",
        type=int,
        default=100,
        help="Number of requests between snapshots"
    )
    stats_group.add_argument(
        "--save_step",
        type=int,
        default=1000,
        help="Number of requests between saving results"
    )
    stats_group.add_argument(
        "--print_step",
        type=int,
        default=1000,
        help="Number of requests between progress updates"
    )
    stats_group.add_argument(
        "--save_start_end_slots",
        action="store_true",
        help="Save detailed slot allocation information"
    )


def add_plotting_args(parser: argparse.ArgumentParser) -> None:
    """
    Add plotting and visualization arguments.
    """
    plot_group = parser.add_argument_group('Plotting Configuration')
    plot_group.add_argument(
        "--plot_results",
        action="store_true",
        help="Generate plots from simulation results"
    )
    add_plot_format_args(plot_group)
    plot_group.add_argument(
        "--plot_dpi",
        type=int,
        default=300,
        help="Resolution (DPI) for generated plots"
    )
    plot_group.add_argument(
        "--show_plots",
        action="store_true",
        help="Display plots interactively (in addition to saving)"
    )
    plot_group.add_argument(
        "--plot_style",
        type=str,
        choices=["seaborn", "matplotlib", "ggplot", "bmh"],
        default="seaborn",
        help="Plotting style to use"
    )


def add_export_args(parser: argparse.ArgumentParser) -> None:
    """
    Add data export and file handling arguments.
    """
    export_group = parser.add_argument_group('Export Configuration')
    export_group.add_argument(
        "--export_excel",
        action="store_true",
        help="Export results to Excel format"
    )
    export_group.add_argument(
        "--export_csv",
        action="store_true",
        help="Export results to CSV format"
    )
    export_group.add_argument(
        "--export_json",
        action="store_true",
        help="Export results to JSON format"
    )
    export_group.add_argument(
        "--file_type",
        type=str,
        choices=["json", "csv", "excel", "tsv"],
        default="json",
        help="Default file format for data export"
    )
    export_group.add_argument(
        "--compress_results",
        action="store_true",
        help="Compress output files to save space"
    )


def add_comparison_args(parser: argparse.ArgumentParser) -> None:
    """
    Add arguments for comparing multiple simulation runs.
    """
    comparison_group = parser.add_argument_group('Comparison Configuration')
    comparison_group.add_argument(
        "--compare_runs",
        type=str,
        nargs='+',
        help="List of run IDs to compare"
    )
    comparison_group.add_argument(
        "--baseline_run",
        type=str,
        help="Run ID to use as baseline for comparison"
    )
    comparison_group.add_argument(
        "--metrics",
        type=str,
        nargs='+',
        choices=["blocking_probability", "path_length", "execution_time", "resource_utilization"],
        help="Metrics to include in comparison"
    )
    comparison_group.add_argument(
        "--significance_test",
        type=str,
        choices=["t_test", "wilcoxon", "mann_whitney"],
        help="Statistical test for comparing results"
    )


def add_filtering_args(parser: argparse.ArgumentParser) -> None:
    """
    Add arguments for filtering and post-processing results.
    """
    filter_group = parser.add_argument_group('Filtering Configuration')
    filter_group.add_argument(
        "--filter_mods",
        action="store_true",
        help="Filter results by modulation format"
    )
    filter_group.add_argument(
        "--min_erlang",
        type=float,
        help="Minimum Erlang load to include in analysis"
    )
    filter_group.add_argument(
        "--max_erlang",
        type=float,
        help="Maximum Erlang load to include in analysis"
    )
    filter_group.add_argument(
        "--exclude_cores",
        type=int,
        nargs='+',
        help="Core numbers to exclude from analysis"
    )
    filter_group.add_argument(
        "--time_window",
        type=str,
        help="Time window for analysis (format: 'start,end')"
    )


def add_all_analysis_args(parser: argparse.ArgumentParser) -> None:
    """
    Add all analysis-related argument groups.
    Convenience function to add all analysis arguments at once.
    """
    add_statistics_args(parser)
    add_plotting_args(parser)
    add_export_args(parser)
    add_comparison_args(parser)
    add_filtering_args(parser)
