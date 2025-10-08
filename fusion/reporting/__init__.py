"""
Reporting module for FUSION simulation output.

This module provides utilities for formatting and reporting simulation results,
separating presentation concerns from data collection.
"""

from fusion.reporting.simulation_reporter import SimulationReporter
from fusion.reporting.statistics import (
    GroomingStatistics,
    SimulationStatistics,
    export_grooming_stats_csv,
    generate_grooming_report,
)

__all__ = [
    "SimulationReporter",
    "GroomingStatistics",
    "SimulationStatistics",
    "generate_grooming_report",
    "export_grooming_stats_csv",
]
