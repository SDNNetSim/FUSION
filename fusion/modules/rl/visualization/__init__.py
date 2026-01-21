"""
RL module visualization extensions (BETA).

Status: BETA
    This module is currently in BETA and is actively being developed.
    The API may evolve in future releases.

This module provides RL-specific visualization components that extend
the core FUSION visualization system (``fusion/visualization/``) via the
plugin architecture.

Components:
    - Metrics for RL training (rewards, TD errors, Q-values)
    - Plot types for RL analysis (convergence, learning curves)
    - Processing strategies for RL data

Usage:
    The plugin must be loaded via the visualization system's plugin registry
    before use::

        from fusion.visualization.plugins import get_global_registry
        registry = get_global_registry()
        registry.discover_plugins()
        registry.load_plugin("rl")

See Also:
    - ``fusion/visualization/`` - Core visualization system
    - :ref:`rl-visualization` - Sphinx documentation
"""

from fusion.modules.rl.visualization.rl_plugin import RLVisualizationPlugin

__all__ = ["RLVisualizationPlugin"]
