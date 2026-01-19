"""
Routing module visualization extensions (BETA).

Status: BETA
    This module is currently in BETA and is actively being developed.
    The API may evolve in future releases.

This module provides routing-specific visualization components that extend
the core FUSION visualization system (``fusion/visualization/``) via the
plugin architecture.

Components:
    - Path hop count analysis
    - Route length distribution
    - Computation time metrics

Usage:
    The plugin must be loaded via the visualization system's plugin registry
    before use::

        from fusion.visualization.plugins import get_global_registry
        registry = get_global_registry()
        registry.discover_plugins()
        registry.load_plugin("routing")

See Also:
    - ``fusion/visualization/`` - Core visualization system
    - ``fusion/modules/routing/visualization/routing_plugin.py`` - Plugin implementation
"""

from fusion.modules.routing.visualization.routing_plugin import (
    RoutingVisualizationPlugin,
)

__all__ = ["RoutingVisualizationPlugin"]
