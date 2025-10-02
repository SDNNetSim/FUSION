"""Port interface for plot rendering."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fusion.visualization.domain.value_objects.plot_specification import (
    PlotSpecification,
)


@dataclass
class RenderResult:
    """Result of rendering a plot."""

    success: bool
    output_path: Path | None = None
    error: str | None = None
    metadata: dict[str, Any] | None = None


class PlotRendererPort(ABC):
    """
    Port for plot rendering.

    This interface abstracts the actual rendering of plots,
    allowing different rendering backends (matplotlib, plotly, etc.)
    """

    @abstractmethod
    def render(
        self,
        specification: PlotSpecification,
        output_path: Path,
        dpi: int = 300,
        format: str = "png",
    ) -> RenderResult:
        """
        Render a plot from specification.

        Args:
            specification: PlotSpecification to render
            output_path: Where to save the plot
            dpi: Resolution in dots per inch
            format: Output format (png, pdf, svg)

        Returns:
            RenderResult with success status and output path

        Raises:
            RenderError: If rendering fails
        """
        pass

    @abstractmethod
    def supports_format(self, format: str) -> bool:
        """
        Check if renderer supports a format.

        Args:
            format: Format to check (png, pdf, svg, etc.)

        Returns:
            True if format is supported
        """
        pass

    @abstractmethod
    def get_supported_formats(self) -> list[str]:
        """
        Get list of supported output formats.

        Returns:
            List of format strings
        """
        pass
