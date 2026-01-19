"""Visualization plugins module.

This module provides the plugin system for extending the FUSION visualization
system with module-specific metrics and plot types.
"""

from fusion.visualization.plugins.base_plugin import BasePlugin, PlotTypeRegistration
from fusion.visualization.plugins.plugin_registry import (
    PluginRegistry,
    get_global_registry,
    reset_global_registry,
)

__all__ = [
    "BasePlugin",
    "PlotTypeRegistration",
    "PluginRegistry",
    "get_global_registry",
    "reset_global_registry",
]
