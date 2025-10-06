"""Matplotlib-based plot renderer."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

from fusion.visualization.application.ports import PlotRendererPort, RenderResult
from fusion.visualization.domain.exceptions import RenderError
from fusion.visualization.domain.value_objects.plot_specification import (
    PlotSpecification,
    PlotType,
)

logger = logging.getLogger(__name__)


class MatplotlibRenderer(PlotRendererPort):
    """
    Matplotlib-based plot renderer.

    This renders PlotSpecification objects using matplotlib,
    supporting various plot types and styling options.
    """

    SUPPORTED_FORMATS = ["png", "pdf", "svg", "jpg"]

    # Default styling
    DEFAULT_STYLE = "seaborn-v0_8-darkgrid"
    DEFAULT_COLORS = [
        "#1f77b4",  # Blue
        "#ff7f0e",  # Orange
        "#2ca02c",  # Green
        "#d62728",  # Red
        "#9467bd",  # Purple
        "#8c564b",  # Brown
        "#e377c2",  # Pink
        "#7f7f7f",  # Gray
        "#bcbd22",  # Olive
        "#17becf",  # Cyan
    ]

    def __init__(
        self,
        style: str | None = None,
        output_dir: Path | None = None,
    ):
        """
        Initialize renderer.

        Args:
            style: Matplotlib style to use (default: seaborn-v0_8-darkgrid)
            output_dir: Optional default output directory
                (currently unused, kept for compatibility)
        """
        self.style = style or self.DEFAULT_STYLE
        self.output_dir = Path(output_dir) if output_dir else None

        # Try to set style, fallback to default if not available
        try:
            plt.style.use(self.style)
        except Exception as e:
            logger.warning(f"Could not set style {self.style}: {e}")

    def supports_format(self, format: str) -> bool:
        """Check if format is supported."""
        return format.lower() in self.SUPPORTED_FORMATS

    def get_supported_formats(self) -> list[str]:
        """Get list of supported formats."""
        return self.SUPPORTED_FORMATS.copy()

    def render(
        self,
        specification: PlotSpecification,
        output_path: Path,
        dpi: int = 300,
        format: str = "png",
    ) -> RenderResult:
        """
        Render plot from specification.

        Args:
            specification: PlotSpecification to render
            output_path: Where to save the plot
            dpi: Resolution in dots per inch
            format: Output format

        Returns:
            RenderResult with success status
        """
        logger.info(f"Rendering {specification.plot_type.value} plot to {output_path}")

        try:
            # Validate format
            if not self.supports_format(format):
                raise RenderError(
                    f"Unsupported format: {format}. "
                    f"Supported: {', '.join(self.SUPPORTED_FORMATS)}"
                )

            # Create figure
            figsize = specification.metadata.get("figsize", (10, 6))
            fig, ax = plt.subplots(figsize=figsize)

            # Render based on plot type
            if specification.plot_type == PlotType.LINE:
                self._render_line_plot(ax, specification)
            elif specification.plot_type == PlotType.SCATTER:
                self._render_scatter_plot(ax, specification)
            elif specification.plot_type == PlotType.BAR:
                self._render_bar_plot(ax, specification)
            elif specification.plot_type == PlotType.HEATMAP:
                self._render_heatmap(ax, specification)
            else:
                # Default to line plot
                self._render_line_plot(ax, specification)

            # Set labels and title
            ax.set_xlabel(specification.x_label, fontsize=12)
            ax.set_ylabel(specification.y_label, fontsize=12)
            ax.set_title(specification.title, fontsize=14, fontweight="bold")

            # Grid
            ax.grid(True, alpha=0.3)

            # Legend
            if len(specification.y_data) > 1:
                ax.legend(loc="best", framealpha=0.9)

            # Tight layout
            plt.tight_layout()

            # Ensure output directory exists
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save figure
            plt.savefig(output_path, dpi=dpi, format=format, bbox_inches="tight")
            plt.close(fig)

            logger.info(f"Successfully rendered plot to {output_path}")

            return RenderResult(
                success=True,
                output_path=output_path,
                metadata={
                    "dpi": dpi,
                    "format": format,
                    "plot_type": specification.plot_type.value,
                },
            )

        except Exception as e:
            logger.exception(f"Error rendering plot: {e}")
            return RenderResult(
                success=False,
                error=str(e),
            )

    def _render_line_plot(self, ax: plt.Axes, spec: PlotSpecification) -> None:
        """Render line plot."""
        x_data = np.array(spec.x_data)

        for i, (algorithm, y_values) in enumerate(spec.y_data.items()):
            y_data = np.array(y_values)
            color = self.DEFAULT_COLORS[i % len(self.DEFAULT_COLORS)]

            # Plot line with markers
            ax.plot(
                x_data,
                y_data,
                marker="o",
                markersize=6,
                linewidth=2,
                label=algorithm,
                color=color,
            )[0]

            # Add error bars if available
            if spec.errors and algorithm in spec.errors:
                errors = np.array(spec.errors[algorithm])
                ax.fill_between(
                    x_data,
                    y_data - errors,
                    y_data + errors,
                    alpha=0.2,
                    color=color,
                )

    def _render_scatter_plot(self, ax: plt.Axes, spec: PlotSpecification) -> None:
        """Render scatter plot."""
        x_data = np.array(spec.x_data)

        for i, (algorithm, y_values) in enumerate(spec.y_data.items()):
            y_data = np.array(y_values)
            color = self.DEFAULT_COLORS[i % len(self.DEFAULT_COLORS)]

            # Plot scatter
            ax.scatter(
                x_data,
                y_data,
                s=100,
                alpha=0.6,
                label=algorithm,
                color=color,
            )

            # Add error bars if available
            if spec.errors and algorithm in spec.errors:
                errors = np.array(spec.errors[algorithm])
                ax.errorbar(
                    x_data,
                    y_data,
                    yerr=errors,
                    fmt="none",
                    color=color,
                    alpha=0.3,
                    capsize=5,
                )

    def _render_bar_plot(self, ax: plt.Axes, spec: PlotSpecification) -> None:
        """Render bar plot."""
        x_data = np.array(spec.x_data)
        n_algorithms = len(spec.y_data)

        # Calculate bar width and positions
        bar_width = 0.8 / n_algorithms
        x_positions = np.arange(len(x_data))

        for i, (algorithm, y_values) in enumerate(spec.y_data.items()):
            y_data = np.array(y_values)
            color = self.DEFAULT_COLORS[i % len(self.DEFAULT_COLORS)]

            # Calculate position offset
            offset = (i - n_algorithms / 2) * bar_width + bar_width / 2

            # Plot bars
            errors = None
            if spec.errors and algorithm in spec.errors:
                errors = np.array(spec.errors[algorithm])

            ax.bar(
                x_positions + offset,
                y_data,
                bar_width,
                label=algorithm,
                color=color,
                alpha=0.7,
                yerr=errors,
                capsize=5,
            )

        # Set x-tick labels
        ax.set_xticks(x_positions)
        ax.set_xticklabels([str(x) for x in x_data])

    def _render_heatmap(self, ax: plt.Axes, spec: PlotSpecification) -> None:
        """Render heatmap."""
        # For heatmap, y_data should be 2D array
        # This is a simplified implementation
        if len(spec.y_data) == 1:
            algorithm = list(spec.y_data.keys())[0]
            data = np.array(spec.y_data[algorithm])

            # Reshape if 1D
            if data.ndim == 1:
                data = data.reshape(1, -1)

            im = ax.imshow(data, cmap="viridis", aspect="auto")

            # Colorbar
            plt.colorbar(im, ax=ax, label=spec.y_label)

            # Set ticks
            ax.set_xticks(range(len(spec.x_data)))
            ax.set_xticklabels([str(x) for x in spec.x_data])

        else:
            # Multiple algorithms - create grid
            algorithms = list(spec.y_data.keys())
            data_matrix = np.array([spec.y_data[algo] for algo in algorithms])

            im = ax.imshow(data_matrix, cmap="viridis", aspect="auto")

            # Colorbar
            plt.colorbar(im, ax=ax, label=spec.y_label)

            # Set ticks
            ax.set_xticks(range(len(spec.x_data)))
            ax.set_xticklabels([str(x) for x in spec.x_data])
            ax.set_yticks(range(len(algorithms)))
            ax.set_yticklabels(algorithms)
