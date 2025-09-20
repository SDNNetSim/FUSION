"""
fusion.visualization: Plotting, charting, and results visualization.

This package provides unified visualization tools for simulation results and data analysis. It includes:
- Plotting utilities for blocking probabilities, path metrics, spectrum usage
- Excel and CSV output tools
- Plot configuration and customization options
- Visualization properties and data structures

This package consolidates what was previously split across plot_stats.py and various
plot helper functions scattered throughout the codebase.
"""

from .properties import PlotArgs, PlotProps

__all__ = ["PlotProps", "PlotArgs"]
