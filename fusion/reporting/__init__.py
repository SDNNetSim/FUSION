"""
Reporting module for FUSION simulation output.

This module provides utilities for formatting and reporting simulation results,
separating presentation concerns from data collection.
"""

from fusion.reporting.aggregation import (
    aggregate_seed_results,
    create_comparison_table,
    format_comparison_for_display,
)
from fusion.reporting.csv_export import (
    append_result_to_csv,
    export_aggregated_results,
    export_comparison_table,
    export_results_to_csv,
)
from fusion.reporting.dataset_logger import DatasetLogger
from fusion.reporting.simulation_reporter import SimulationReporter

__all__ = [
    "SimulationReporter",
    "DatasetLogger",
    "aggregate_seed_results",
    "create_comparison_table",
    "format_comparison_for_display",
    "export_results_to_csv",
    "export_aggregated_results",
    "export_comparison_table",
    "append_result_to_csv",
]
