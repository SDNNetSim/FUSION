"""Spectrum visualization plugin.

This plugin provides visualization support for spectrum allocation and
utilization analysis in optical networks.
"""

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.figure import Figure

from fusion.visualization.domain.entities.metric import (
    AggregationStrategy,
    DataType,
    MetricDefinition,
)
from fusion.visualization.domain.value_objects.plot_specification import PlotSpecification
from fusion.visualization.infrastructure.renderers.base_renderer import (
    BaseRenderer,
    PlotResult,
)
from fusion.visualization.plugins.base_plugin import BasePlugin, PlotTypeRegistration


class SpectrumHeatmapRenderer(BaseRenderer):
    """Renderer for spectrum utilization heatmaps."""

    def supports_format(self, format: str) -> bool:
        """Check if format is supported."""
        return format in ["png", "pdf", "svg", "jpg"]

    def get_supported_formats(self) -> list[str]:
        """Get list of supported formats."""
        return ["png", "pdf", "svg", "jpg"]

    def render(
        self,
        specification: PlotSpecification,
        output_path: Path,
        dpi: int = 300,
        format: str = "png",
    ) -> PlotResult:
        """Render spectrum utilization heatmap.

        Args:
            specification: Plot specification
            output_path: Where to save the plot
            dpi: Resolution in dots per inch
            format: Output format (png, pdf, svg)

        Returns:
            PlotResult with rendered figure
        """
        data = specification.metadata.get("processed_data", {})

        # Create figure
        fig, ax = plt.subplots(figsize=specification.figsize)

        # Get spectrum data (assuming shape: [links, slots])
        spectrum_matrix = data.get("spectrum_utilization", np.array([]))

        if spectrum_matrix.size > 0:
            # Create heatmap
            im = ax.imshow(spectrum_matrix, aspect="auto", cmap="YlOrRd", interpolation="nearest")

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label("Utilization", rotation=270, labelpad=15)

            # Labels
            ax.set_xlabel("Spectrum Slot", fontsize=12)
            ax.set_ylabel("Link", fontsize=12)
            ax.set_title(specification.title or "Spectrum Utilization Heatmap", fontsize=14, fontweight="bold")

        plt.tight_layout()

        # Save
        fig.savefig(output_path, dpi=dpi, format=format, bbox_inches="tight")
        plt.close(fig)

        return PlotResult(
            success=True, output_path=output_path, metadata={"plot_type": "spectrum_heatmap"}
        )


class FragmentationPlotRenderer(BaseRenderer):
    """Renderer for spectrum fragmentation analysis."""

    def supports_format(self, format: str) -> bool:
        """Check if format is supported."""
        return format in ["png", "pdf", "svg", "jpg"]

    def get_supported_formats(self) -> list[str]:
        """Get list of supported formats."""
        return ["png", "pdf", "svg", "jpg"]

    def render(
        self,
        specification: PlotSpecification,
        output_path: Path,
        dpi: int = 300,
        format: str = "png",
    ) -> PlotResult:
        """Render fragmentation analysis plot.

        Args:
            specification: Plot specification
            output_path: Where to save the plot
            dpi: Resolution in dots per inch
            format: Output format (png, pdf, svg)

        Returns:
            PlotResult with rendered figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=specification.figsize)

        data = specification.metadata.get("processed_data", {})

        # Plot 1: Fragmentation over time/traffic
        for algo, algo_data in data.items():
            if "traffic_volumes" in algo_data and "fragmentation" in algo_data:
                ax1.plot(
                    algo_data["traffic_volumes"],
                    algo_data["fragmentation"],
                    marker="o",
                    label=algo,
                    linewidth=2,
                )

        ax1.set_xlabel("Traffic Volume (Erlang)", fontsize=12)
        ax1.set_ylabel("Fragmentation Index", fontsize=12)
        ax1.set_title("Spectrum Fragmentation vs Traffic", fontsize=12, fontweight="bold")
        ax1.legend(loc="best")
        ax1.grid(True, alpha=0.3)

        # Plot 2: Average fragment size distribution
        algos = list(data.keys())
        avg_frag_sizes = [data[a].get("avg_fragment_size", 0) for a in algos]

        x_pos = np.arange(len(algos))
        ax2.bar(x_pos, avg_frag_sizes, alpha=0.7)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(algos, rotation=45, ha="right")
        ax2.set_ylabel("Average Fragment Size (slots)", fontsize=12)
        ax2.set_title("Average Spectrum Fragment Size", fontsize=12, fontweight="bold")
        ax2.grid(True, alpha=0.3, axis="y")

        fig.suptitle(specification.title or "Spectrum Fragmentation Analysis", fontsize=14, fontweight="bold")
        plt.tight_layout()

        # Save
        fig.savefig(output_path, dpi=dpi, format=format, bbox_inches="tight")
        plt.close(fig)

        return PlotResult(
            success=True, output_path=output_path, metadata={"plot_type": "fragmentation_plot"}
        )


class SpectrumVisualizationPlugin(BasePlugin):
    """Plugin for spectrum-specific visualizations.

    This plugin extends the FUSION visualization system with:
    - Spectrum utilization metrics
    - Fragmentation analysis
    - Spectrum allocation visualizations
    """

    @property
    def name(self) -> str:
        """Return plugin name."""
        return "spectrum"

    @property
    def version(self) -> str:
        """Return plugin version."""
        return "1.0.0"

    @property
    def description(self) -> str:
        """Return plugin description."""
        return "Visualization plugin for spectrum allocation and utilization analysis"

    def register_metrics(self) -> List[MetricDefinition]:
        """Register spectrum-specific metrics.

        Returns:
            List of spectrum metric definitions
        """
        return [
            MetricDefinition(
                name="spectrum_utilization",
                display_name="Spectrum Utilization",
                data_type=DataType.ARRAY,
                source_path="$.spectrum.utilization",
                aggregation=AggregationStrategy.MEAN,
                unit="percentage",
                description="Spectrum slot utilization across links",
            ),
            MetricDefinition(
                name="fragmentation_index",
                display_name="Fragmentation Index",
                data_type=DataType.FLOAT,
                source_path="$.spectrum.fragmentation_index",
                aggregation=AggregationStrategy.MEAN,
                unit="index",
                description="Measure of spectrum fragmentation (0=no fragmentation, 1=max fragmentation)",
            ),
            MetricDefinition(
                name="avg_fragment_size",
                display_name="Average Fragment Size",
                data_type=DataType.FLOAT,
                source_path="$.spectrum.avg_fragment_size",
                aggregation=AggregationStrategy.MEAN,
                unit="slots",
                description="Average size of contiguous spectrum fragments",
            ),
            MetricDefinition(
                name="largest_fragment",
                display_name="Largest Fragment",
                data_type=DataType.INT,
                source_path="$.spectrum.largest_fragment",
                aggregation=AggregationStrategy.MAX,
                unit="slots",
                description="Size of largest contiguous spectrum fragment",
            ),
            MetricDefinition(
                name="num_fragments",
                display_name="Number of Fragments",
                data_type=DataType.INT,
                source_path="$.spectrum.num_fragments",
                aggregation=AggregationStrategy.MEAN,
                unit="count",
                description="Total number of spectrum fragments",
            ),
            MetricDefinition(
                name="spectrum_efficiency",
                display_name="Spectrum Efficiency",
                data_type=DataType.FLOAT,
                source_path="$.spectrum.efficiency",
                aggregation=AggregationStrategy.MEAN,
                unit="bps/Hz",
                description="Spectral efficiency (bits per second per Hertz)",
            ),
        ]

    def register_plot_types(self) -> Dict[str, PlotTypeRegistration]:
        """Register spectrum-specific plot types.

        Returns:
            Dictionary of plot type registrations
        """
        from fusion.visualization.domain.strategies.processing_strategies import (
            GenericMetricProcessingStrategy,
        )

        return {
            "spectrum_heatmap": PlotTypeRegistration(
                processor=GenericMetricProcessingStrategy(),
                renderer=SpectrumHeatmapRenderer(),
                description="Heatmap showing spectrum utilization across links and slots",
                required_metrics=["spectrum_utilization"],
                default_config={"colormap": "YlOrRd", "show_colorbar": True},
            ),
            "fragmentation_plot": PlotTypeRegistration(
                processor=GenericMetricProcessingStrategy(),
                renderer=FragmentationPlotRenderer(),
                description="Analysis of spectrum fragmentation vs traffic load",
                required_metrics=["fragmentation_index"],
                default_config={"show_distribution": True},
            ),
        }

    def get_config_schema(self) -> Dict:
        """Return JSON schema for spectrum plugin configuration.

        Returns:
            JSON schema dictionary
        """
        return {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "Spectrum Visualization Plugin Configuration",
            "type": "object",
            "properties": {
                "colormap": {"type": "string", "default": "YlOrRd", "description": "Colormap for heatmaps"},
                "show_fragmentation_stats": {
                    "type": "boolean",
                    "default": True,
                    "description": "Show fragmentation statistics in plots",
                },
            },
        }
