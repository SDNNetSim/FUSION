"""Data I/O module for FUSION simulator.

This module provides functionality for:
- Data generation (generate.py)
- Data structuring (structure.py)
- Data export (exporter.py)
"""

from .exporter import ExporterRegistry, SimulationDataExporter
from .generate import create_bw_info, create_pt
from .structure import assign_core_nodes, assign_link_lengths, create_network

__version__ = "1.0.0"

__all__ = [
    "SimulationDataExporter",
    "ExporterRegistry",
    "create_pt",
    "create_bw_info",
    "create_network",
    "assign_link_lengths",
    "assign_core_nodes",
]
