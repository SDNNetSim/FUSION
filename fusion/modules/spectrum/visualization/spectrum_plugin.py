"""
Spectrum visualization plugin (BETA).

Status: BETA
    This module is currently in BETA and is actively being developed.
    The API may evolve in future releases.

This plugin provides visualization support for spectrum allocation and
utilization analysis in optical networks. It extends the core FUSION
visualization system with spectrum-specific renderers and plot types.

Components
----------
SpectrumHeatmapRenderer
    Renders spectrum utilization as heatmaps across links and slots.
FragmentationPlotRenderer
    Renders fragmentation analysis with traffic correlation.
SpectrumVisualizationPlugin
    Main plugin class that registers metrics and plot types.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fusion.visualization.domain.entities.metric import (
    AggregationStrategy,
    DataType,
    MetricDefinition,
)
from fusion.visualization.domain.value_objects.plot_specification import (
    PlotSpecification,
)
from fusion.visualization.infrastructure.renderers.base_renderer import (
    BaseRenderer,
    PlotResult,
)
from fusion.visualization.plugins.base_plugin import BasePlugin, PlotTypeRegistration


class SpectrumHeatmapRenderer(BaseRenderer):
    """
    Renderer for spectrum utilization heatmaps.

    This renderer creates heatmap visualizations showing spectrum slot
    utilization across network links, with color intensity indicating
    occupancy levels.
    """

    def supports_format(self, format: str) -> bool:
        """
        Check if the given output format is supported.

        :param format: Output format string (e.g., 'png', 'pdf').
        :type format: str
        :return: True if format is supported, False otherwise.
        :rtype: bool
        """
        return format in ["png", "pdf", "svg", "jpg"]

    def get_supported_formats(self) -> list[str]:
        """
        Get list of supported output formats.

        :return: List of supported format strings.
        :rtype: list[str]
        """
        return ["png", "pdf", "svg", "jpg"]

    def render(
        self,
        specification: PlotSpecification,
        output_path: Path,
        dpi: int = 300,
        format: str = "png",
    ) -> PlotResult:
        """
        Render spectrum utilization heatmap.

        :param specification: Plot specification containing data and styling.
        :type specification: PlotSpecification
        :param output_path: File path where the plot will be saved.
        :type output_path: Path
        :param dpi: Resolution in dots per inch.
        :type dpi: int
        :param format: Output format (png, pdf, svg, jpg).
        :type format: str
        :return: Result object containing success status and output path.
        :rtype: PlotResult
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
            ax.set_title(
                specification.title or "Spectrum Utilization Heatmap",
                fontsize=14,
                fontweight="bold",
            )

        plt.tight_layout()

        # Save
        fig.savefig(output_path, dpi=dpi, format=format, bbox_inches="tight")
        plt.close(fig)

        return PlotResult(
            success=True,
            output_path=output_path,
            metadata={"plot_type": "spectrum_heatmap"},
        )


class FragmentationPlotRenderer(BaseRenderer):
    """
    Renderer for spectrum fragmentation analysis.

    This renderer creates dual-panel plots showing fragmentation trends
    vs traffic load and average fragment size distribution across algorithms.
    """

    def supports_format(self, format: str) -> bool:
        """
        Check if the given output format is supported.

        :param format: Output format string (e.g., 'png', 'pdf').
        :type format: str
        :return: True if format is supported, False otherwise.
        :rtype: bool
        """
        return format in ["png", "pdf", "svg", "jpg"]

    def get_supported_formats(self) -> list[str]:
        """
        Get list of supported output formats.

        :return: List of supported format strings.
        :rtype: list[str]
        """
        return ["png", "pdf", "svg", "jpg"]

    def render(
        self,
        specification: PlotSpecification,
        output_path: Path,
        dpi: int = 300,
        format: str = "png",
    ) -> PlotResult:
        """
        Render fragmentation analysis plot.

        :param specification: Plot specification containing data and styling.
        :type specification: PlotSpecification
        :param output_path: File path where the plot will be saved.
        :type output_path: Path
        :param dpi: Resolution in dots per inch.
        :type dpi: int
        :param format: Output format (png, pdf, svg, jpg).
        :type format: str
        :return: Result object containing success status and output path.
        :rtype: PlotResult
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

        fig.suptitle(
            specification.title or "Spectrum Fragmentation Analysis",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()

        # Save
        fig.savefig(output_path, dpi=dpi, format=format, bbox_inches="tight")
        plt.close(fig)

        return PlotResult(
            success=True,
            output_path=output_path,
            metadata={"plot_type": "fragmentation_plot"},
        )


class SpectrumVisualizationPlugin(BasePlugin):
    """
    Plugin for spectrum-specific visualizations.

    This plugin extends the FUSION visualization system with spectrum
    allocation analysis capabilities including utilization heatmaps,
    fragmentation tracking, and efficiency metrics.

    Registered Plot Types
    ---------------------
    spectrum_heatmap
        Shows spectrum slot utilization across network links.
    fragmentation_plot
        Analyzes fragmentation trends vs traffic load.
    """

    @property
    def name(self) -> str:
        """
        Return the plugin name.

        :return: Plugin identifier string.
        :rtype: str
        """
        return "spectrum"

    @property
    def version(self) -> str:
        """
        Return the plugin version.

        :return: Semantic version string.
        :rtype: str
        """
        return "1.0.0"

    @property
    def description(self) -> str:
        """
        Return the plugin description.

        :return: Human-readable description of the plugin.
        :rtype: str
        """
        return "Visualization plugin for spectrum allocation and utilization analysis"

    def register_metrics(self) -> list[MetricDefinition]:
        """
        Register spectrum-specific metrics with the visualization system.

        :return: List of metric definitions for spectrum-related measurements.
        :rtype: list[MetricDefinition]
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
                description=("Measure of spectrum fragmentation (0=no fragmentation, 1=max fragmentation)"),
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

    def register_plot_types(self) -> dict[str, PlotTypeRegistration]:
        """
        Register spectrum-specific plot types with the visualization system.

        :return: Dictionary mapping plot type names to their registrations.
        :rtype: dict[str, PlotTypeRegistration]
        """
        from fusion.visualization.domain.strategies.processing_strategies import (
            GenericMetricProcessingStrategy,
        )

        return {
            "spectrum_heatmap": PlotTypeRegistration(
                processor=GenericMetricProcessingStrategy(),
                renderer=SpectrumHeatmapRenderer(),
                description=("Heatmap showing spectrum utilization across links and slots"),
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

    def get_config_schema(self) -> dict:
        """
        Return JSON schema for spectrum plugin configuration.

        :return: JSON schema dictionary defining valid configuration options.
        :rtype: dict
        """
        return {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "Spectrum Visualization Plugin Configuration",
            "type": "object",
            "properties": {
                "colormap": {
                    "type": "string",
                    "default": "YlOrRd",
                    "description": "Colormap for heatmaps",
                },
                "show_fragmentation_stats": {
                    "type": "boolean",
                    "default": True,
                    "description": "Show fragmentation statistics in plots",
                },
            },
        }
