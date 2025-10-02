"""Routing visualization plugin.

This plugin provides visualization support for routing algorithm analysis
in optical networks.
"""

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

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


class HopCountPlotRenderer(BaseRenderer):
    """Renderer for hop count analysis plots."""

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
        """Render hop count plot.

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

        # Plot 1: Mean hop count vs traffic volume
        for algo, algo_data in data.items():
            if "traffic_volumes" in algo_data and "mean_hops" in algo_data:
                mean_hops = algo_data["mean_hops"]
                std_hops = algo_data.get("std_hops")

                line = ax1.plot(
                    algo_data["traffic_volumes"],
                    mean_hops,
                    marker="o",
                    label=algo,
                    linewidth=2,
                )

                # Add error bars if available
                if std_hops is not None:
                    color = line[0].get_color()
                    ax1.fill_between(
                        algo_data["traffic_volumes"],
                        np.array(mean_hops) - np.array(std_hops),
                        np.array(mean_hops) + np.array(std_hops),
                        alpha=0.2,
                        color=color,
                    )

        ax1.set_xlabel("Traffic Volume (Erlang)", fontsize=12)
        ax1.set_ylabel("Mean Hop Count", fontsize=12)
        ax1.set_title("Average Path Hops vs Traffic", fontsize=12, fontweight="bold")
        ax1.legend(loc="best")
        ax1.grid(True, alpha=0.3)

        # Plot 2: Hop count distribution
        algos = list(data.keys())
        hop_distributions = [data[a].get("hop_distribution", []) for a in algos]

        # Create box plot
        positions = np.arange(len(algos))
        bp = ax2.boxplot(hop_distributions, positions=positions, patch_artist=True, widths=0.6)

        # Color the boxes
        cmap = plt.cm.get_cmap('Set3')
        colors = cmap(np.linspace(0, 1, len(algos)))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax2.set_xticks(positions)
        ax2.set_xticklabels(algos, rotation=45, ha="right")
        ax2.set_ylabel("Hop Count", fontsize=12)
        ax2.set_title("Hop Count Distribution", fontsize=12, fontweight="bold")
        ax2.grid(True, alpha=0.3, axis="y")

        fig.suptitle(specification.title or "Routing Hop Count Analysis", fontsize=14, fontweight="bold")
        plt.tight_layout()

        # Save
        fig.savefig(output_path, dpi=dpi, format=format, bbox_inches="tight")
        plt.close(fig)

        return PlotResult(success=True, output_path=output_path, metadata={"plot_type": "hop_count_plot"})


class PathLengthPlotRenderer(BaseRenderer):
    """Renderer for path length analysis."""

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
        """Render path length plot.

        Args:
            specification: Plot specification
            output_path: Where to save the plot
            dpi: Resolution in dots per inch
            format: Output format (png, pdf, svg)

        Returns:
            PlotResult with rendered figure
        """
        fig, ax = plt.subplots(figsize=specification.figsize)

        data = specification.metadata.get("processed_data", {})

        # Plot path lengths vs traffic
        for algo, algo_data in data.items():
            if "traffic_volumes" in algo_data and "mean_length" in algo_data:
                ax.plot(
                    algo_data["traffic_volumes"],
                    algo_data["mean_length"],
                    marker="o",
                    label=algo,
                    linewidth=2,
                )

        ax.set_xlabel(specification.x_label or "Traffic Volume (Erlang)", fontsize=12)
        ax.set_ylabel(specification.y_label or "Mean Path Length (km)", fontsize=12)
        ax.set_title(specification.title or "Average Path Length vs Traffic", fontsize=14, fontweight="bold")
        ax.legend(loc="best", frameon=True, shadow=True)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save
        fig.savefig(output_path, dpi=dpi, format=format, bbox_inches="tight")
        plt.close(fig)

        return PlotResult(
            success=True, output_path=output_path, metadata={"plot_type": "path_length_plot"}
        )


class ComputationTimePlotRenderer(BaseRenderer):
    """Renderer for routing computation time analysis."""

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
        """Render computation time plot.

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

        # Plot 1: Mean computation time vs traffic
        for algo, algo_data in data.items():
            if "traffic_volumes" in algo_data and "mean_comp_time" in algo_data:
                ax1.plot(
                    algo_data["traffic_volumes"],
                    algo_data["mean_comp_time"],
                    marker="o",
                    label=algo,
                    linewidth=2,
                )

        ax1.set_xlabel("Traffic Volume (Erlang)", fontsize=12)
        ax1.set_ylabel("Mean Computation Time (ms)", fontsize=12)
        ax1.set_title("Routing Computation Time vs Traffic", fontsize=12, fontweight="bold")
        ax1.legend(loc="best")
        ax1.grid(True, alpha=0.3)

        # Plot 2: Cumulative distribution of computation times
        for algo, algo_data in data.items():
            if "comp_time_cdf" in algo_data:
                times = algo_data["comp_time_cdf"]["times"]
                cdf = algo_data["comp_time_cdf"]["cdf"]
                ax2.plot(times, cdf, label=algo, linewidth=2)

        ax2.set_xlabel("Computation Time (ms)", fontsize=12)
        ax2.set_ylabel("Cumulative Probability", fontsize=12)
        ax2.set_title("Computation Time CDF", fontsize=12, fontweight="bold")
        ax2.legend(loc="best")
        ax2.grid(True, alpha=0.3)

        # Add percentile lines
        for percentile in [50, 90, 99]:
            ax2.axhline(percentile / 100, linestyle="--", alpha=0.3, color="gray")
            ax2.text(
                ax2.get_xlim()[1] * 0.95,
                percentile / 100,
                f"P{percentile}",
                va="bottom",
                ha="right",
                fontsize=9,
                alpha=0.7,
            )

        fig.suptitle(specification.title or "Routing Computation Time Analysis", fontsize=14, fontweight="bold")
        plt.tight_layout()

        # Save
        fig.savefig(output_path, dpi=dpi, format=format, bbox_inches="tight")
        plt.close(fig)

        return PlotResult(
            success=True, output_path=output_path, metadata={"plot_type": "computation_time_plot"}
        )


class RoutingVisualizationPlugin(BasePlugin):
    """Plugin for routing-specific visualizations.

    This plugin extends the FUSION visualization system with:
    - Hop count analysis
    - Path length distribution
    - Routing computation time metrics
    - Algorithm comparison plots
    """

    @property
    def name(self) -> str:
        """Return plugin name."""
        return "routing"

    @property
    def version(self) -> str:
        """Return plugin version."""
        return "1.0.0"

    @property
    def description(self) -> str:
        """Return plugin description."""
        return "Visualization plugin for routing algorithm analysis"

    def register_metrics(self) -> List[MetricDefinition]:
        """Register routing-specific metrics.

        Returns:
            List of routing metric definitions
        """
        return [
            MetricDefinition(
                name="hop_count",
                display_name="Hop Count",
                data_type=DataType.INT,
                source_path="$.routing.hops_mean",
                aggregation=AggregationStrategy.MEAN,
                unit="hops",
                description="Number of hops in the routing path",
            ),
            MetricDefinition(
                name="path_length",
                display_name="Path Length",
                data_type=DataType.FLOAT,
                source_path="$.routing.lengths_mean",
                aggregation=AggregationStrategy.MEAN,
                unit="km",
                description="Physical length of the routing path in kilometers",
            ),
            MetricDefinition(
                name="computation_time",
                display_name="Computation Time",
                data_type=DataType.FLOAT,
                source_path="$.routing.computation_time",
                aggregation=AggregationStrategy.MEAN,
                unit="ms",
                description="Time taken to compute the route",
            ),
            MetricDefinition(
                name="path_diversity",
                display_name="Path Diversity",
                data_type=DataType.FLOAT,
                source_path="$.routing.path_diversity",
                aggregation=AggregationStrategy.MEAN,
                unit="index",
                description="Diversity of paths used (0=no diversity, 1=max diversity)",
            ),
            MetricDefinition(
                name="route_efficiency",
                display_name="Route Efficiency",
                data_type=DataType.FLOAT,
                source_path="$.routing.efficiency",
                aggregation=AggregationStrategy.MEAN,
                unit="ratio",
                description="Ratio of used path length to shortest possible path",
            ),
            MetricDefinition(
                name="k_paths_found",
                display_name="K-Paths Found",
                data_type=DataType.INT,
                source_path="$.routing.k_paths_found",
                aggregation=AggregationStrategy.MEAN,
                unit="count",
                description="Number of alternative paths found",
            ),
        ]

    def register_plot_types(self) -> Dict[str, PlotTypeRegistration]:
        """Register routing-specific plot types.

        Returns:
            Dictionary of plot type registrations
        """
        from fusion.visualization.domain.strategies.processing_strategies import (
            GenericMetricProcessingStrategy,
        )

        return {
            "hop_count_plot": PlotTypeRegistration(
                processor=GenericMetricProcessingStrategy(),
                renderer=HopCountPlotRenderer(),
                description="Analysis of routing path hop counts",
                required_metrics=["hop_count"],
                default_config={"show_distribution": True},
            ),
            "path_length_plot": PlotTypeRegistration(
                processor=GenericMetricProcessingStrategy(),
                renderer=PathLengthPlotRenderer(),
                description="Analysis of routing path physical lengths",
                required_metrics=["path_length"],
                default_config={},
            ),
            "computation_time_plot": PlotTypeRegistration(
                processor=GenericMetricProcessingStrategy(),
                renderer=ComputationTimePlotRenderer(),
                description="Analysis of routing algorithm computation time",
                required_metrics=["computation_time"],
                default_config={"show_percentiles": True, "show_cdf": True},
            ),
        }

    def get_config_schema(self) -> Dict:
        """Return JSON schema for routing plugin configuration.

        Returns:
            JSON schema dictionary
        """
        return {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "Routing Visualization Plugin Configuration",
            "type": "object",
            "properties": {
                "show_distribution": {
                    "type": "boolean",
                    "default": True,
                    "description": "Show hop count distribution",
                },
                "show_percentiles": {
                    "type": "boolean",
                    "default": True,
                    "description": "Show percentile markers in plots",
                },
            },
        }
