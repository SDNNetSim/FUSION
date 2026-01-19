"""Base renderer interface for plot rendering.

This module provides the base class for plot renderers,
which are used by plugins to define custom rendering backends.
"""

from fusion.visualization.application.ports.plot_renderer_port import (
    PlotRendererPort,
    RenderResult,
)

# Aliases for backward compatibility and clearer plugin interface
BaseRenderer = PlotRendererPort
PlotResult = RenderResult  # Legacy alias

__all__ = ["BaseRenderer", "RenderResult", "PlotResult"]
