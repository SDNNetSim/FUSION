"""
SNR module visualization extensions (BETA).

Status: BETA
    This module is currently in BETA and is actively being developed.
    The API may evolve in future releases.

This module provides SNR-specific visualization components that extend
the core FUSION visualization system (``fusion/visualization/``) via the
plugin architecture.

Components:
    - SNR vs distance plots
    - Q-factor analysis
    - OSNR margin visualization

Usage:
    The plugin must be loaded via the visualization system's plugin registry
    before use::

        from fusion.visualization.plugins import get_global_registry
        registry = get_global_registry()
        registry.discover_plugins()
        registry.load_plugin("snr")

See Also:
    - ``fusion/visualization/`` - Core visualization system
"""

from fusion.modules.snr.visualization.snr_plugin import SNRVisualizationPlugin

__all__ = ["SNRVisualizationPlugin"]
