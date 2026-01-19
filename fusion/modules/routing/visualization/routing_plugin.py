"""
Routing visualization plugin for FUSION.

This module provides routing-specific visualization extensions that integrate
with FUSION's core visualization system via the plugin architecture.

Status: BETA
    This module is currently in BETA and is actively being developed.
    The API may evolve in future releases. It is designed to work with
    the core visualization system at ``fusion/visualization/``.

Overview:
    This plugin extends the FUSION visualization system with routing-specific
    plot types and metrics:

    - Hop count analysis (mean hops, distribution across algorithms)
    - Path length distribution (physical distance of routes)
    - Routing computation time metrics (latency analysis, CDF)

Architecture:
    This is a **plugin module**, not a standalone visualization system.
    It implements the ``BasePlugin`` interface from
    ``fusion.visualization.plugins.base_plugin`` and registers its components
    with the core visualization system.

    The plugin registers:
        - Custom metrics (hop_count, path_length, computation_time, etc.)
        - Custom plot types (hop_count_plot, path_length_plot, computation_time_plot)
        - Custom renderers (HopCountPlotRenderer, PathLengthPlotRenderer, etc.)

Usage:
    The plugin must be loaded before use via the plugin registry::

        from fusion.visualization.plugins import get_global_registry

        registry = get_global_registry()
        registry.discover_plugins()
        registry.load_plugin("routing")

        # Now routing plot types are available
        from fusion.visualization.application.use_cases.generate_plot import generate_plot
        result = generate_plot(
            config_path="my_experiment.yml",
            plot_type="hop_count_plot",
            output_path="plots/hop_count.png",
        )

Components:
    HopCountPlotRenderer
        Renders hop count analysis with mean vs. traffic and distribution boxplot.

    PathLengthPlotRenderer
        Renders path length (km) trends across different traffic volumes.

    ComputationTimePlotRenderer
        Renders routing computation time analysis with CDF and percentile markers.

    RoutingVisualizationPlugin
        Main plugin class that registers metrics and plot types with the core system.

See Also:
    - ``fusion/visualization/`` - Core visualization system
    - ``fusion/visualization/plugins/base_plugin.py`` - Plugin interface
    - ``fusion/modules/routing/`` - Routing algorithms that generate the data

Example:
    >>> from fusion.modules.routing.visualization import RoutingVisualizationPlugin
    >>> plugin = RoutingVisualizationPlugin()
    >>> print(plugin.name)
    'routing'
    >>> print(plugin.version)
    '1.0.0'
    >>> metrics = plugin.register_metrics()
    >>> print([m.name for m in metrics])
    ['hop_count', 'path_length', 'computation_time', ...]
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


class HopCountPlotRenderer(BaseRenderer):
    """
    Renderer for routing path hop count analysis plots.

    This renderer creates a two-panel visualization:
        1. Left panel: Mean hop count vs. traffic volume with error bands
        2. Right panel: Hop count distribution boxplot across algorithms

    The plot helps identify:
        - How path length (in hops) varies with network load
        - Comparative hop counts between different routing algorithms
        - Statistical distribution of hop counts for each algorithm

    Inherits From:
        BaseRenderer: Base class from fusion.visualization.infrastructure.renderers

    Supported Formats:
        PNG, PDF, SVG, JPG

    Example:
        >>> renderer = HopCountPlotRenderer()
        >>> result = renderer.render(
        ...     specification=plot_spec,
        ...     output_path=Path("hop_count.png"),
        ...     dpi=300,
        ...     format="png"
        ... )
        >>> print(result.success)
        True
    """

    def supports_format(self, format: str) -> bool:
        """
        Check if the specified output format is supported.

        :param format: File format extension (e.g., 'png', 'pdf')
        :type format: str
        :return: True if format is supported, False otherwise
        :rtype: bool
        """
        return format in ["png", "pdf", "svg", "jpg"]

    def get_supported_formats(self) -> list[str]:
        """
        Get list of all supported output formats.

        :return: List of format extensions supported by this renderer
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
        Render the hop count analysis plot.

        Creates a two-panel figure showing:
        - Mean hop count trend vs. traffic volume with confidence bands
        - Boxplot distribution of hop counts per algorithm

        :param specification: Plot specification containing processed data and styling
        :type specification: PlotSpecification
        :param output_path: Destination path for the rendered plot
        :type output_path: Path
        :param dpi: Resolution in dots per inch (default: 300)
        :type dpi: int
        :param format: Output format - one of 'png', 'pdf', 'svg', 'jpg' (default: 'png')
        :type format: str
        :return: PlotResult indicating success/failure and output path
        :rtype: PlotResult
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
        cmap = plt.cm.get_cmap("Set3")
        colors = cmap(np.linspace(0, 1, len(algos)))
        for patch, color in zip(bp["boxes"], colors, strict=False):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax2.set_xticks(positions)
        ax2.set_xticklabels(algos, rotation=45, ha="right")
        ax2.set_ylabel("Hop Count", fontsize=12)
        ax2.set_title("Hop Count Distribution", fontsize=12, fontweight="bold")
        ax2.grid(True, alpha=0.3, axis="y")

        fig.suptitle(
            specification.title or "Routing Hop Count Analysis",
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
            metadata={"plot_type": "hop_count_plot"},
        )


class PathLengthPlotRenderer(BaseRenderer):
    """
    Renderer for routing path physical length analysis.

    This renderer creates a single-panel line plot showing mean path length
    (in kilometers) versus traffic volume for each routing algorithm.

    Use Case:
        Analyze how different routing algorithms trade off between path length
        and other metrics. Longer paths may offer better load balancing but
        consume more spectrum and may have worse SNR.

    Inherits From:
        BaseRenderer: Base class from fusion.visualization.infrastructure.renderers

    Supported Formats:
        PNG, PDF, SVG, JPG

    Example:
        >>> renderer = PathLengthPlotRenderer()
        >>> result = renderer.render(
        ...     specification=plot_spec,
        ...     output_path=Path("path_length.png"),
        ... )
    """

    def supports_format(self, format: str) -> bool:
        """
        Check if the specified output format is supported.

        :param format: File format extension (e.g., 'png', 'pdf')
        :type format: str
        :return: True if format is supported, False otherwise
        :rtype: bool
        """
        return format in ["png", "pdf", "svg", "jpg"]

    def get_supported_formats(self) -> list[str]:
        """
        Get list of all supported output formats.

        :return: List of format extensions supported by this renderer
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
        Render the path length analysis plot.

        Creates a line plot showing mean path length (km) vs. traffic volume
        for each routing algorithm, with legend and grid.

        :param specification: Plot specification containing processed data and styling
        :type specification: PlotSpecification
        :param output_path: Destination path for the rendered plot
        :type output_path: Path
        :param dpi: Resolution in dots per inch (default: 300)
        :type dpi: int
        :param format: Output format - one of 'png', 'pdf', 'svg', 'jpg' (default: 'png')
        :type format: str
        :return: PlotResult indicating success/failure and output path
        :rtype: PlotResult
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
        ax.set_title(
            specification.title or "Average Path Length vs Traffic",
            fontsize=14,
            fontweight="bold",
        )
        ax.legend(loc="best", frameon=True, shadow=True)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save
        fig.savefig(output_path, dpi=dpi, format=format, bbox_inches="tight")
        plt.close(fig)

        return PlotResult(
            success=True,
            output_path=output_path,
            metadata={"plot_type": "path_length_plot"},
        )


class ComputationTimePlotRenderer(BaseRenderer):
    """
    Renderer for routing algorithm computation time analysis.

    This renderer creates a two-panel visualization for analyzing the
    computational performance of routing algorithms:
        1. Left panel: Mean computation time vs. traffic volume
        2. Right panel: Cumulative Distribution Function (CDF) with percentile markers

    Use Case:
        - Compare algorithmic complexity between routing strategies
        - Identify performance bottlenecks under high load
        - Ensure routing decisions meet real-time requirements

    Inherits From:
        BaseRenderer: Base class from fusion.visualization.infrastructure.renderers

    Supported Formats:
        PNG, PDF, SVG, JPG

    Example:
        >>> renderer = ComputationTimePlotRenderer()
        >>> result = renderer.render(
        ...     specification=plot_spec,
        ...     output_path=Path("comp_time.png"),
        ... )
    """

    def supports_format(self, format: str) -> bool:
        """
        Check if the specified output format is supported.

        :param format: File format extension (e.g., 'png', 'pdf')
        :type format: str
        :return: True if format is supported, False otherwise
        :rtype: bool
        """
        return format in ["png", "pdf", "svg", "jpg"]

    def get_supported_formats(self) -> list[str]:
        """
        Get list of all supported output formats.

        :return: List of format extensions supported by this renderer
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
        Render the computation time analysis plot.

        Creates a two-panel figure showing:
        - Mean routing computation time vs. traffic volume
        - CDF of computation times with P50, P90, P99 percentile markers

        :param specification: Plot specification containing processed data and styling
        :type specification: PlotSpecification
        :param output_path: Destination path for the rendered plot
        :type output_path: Path
        :param dpi: Resolution in dots per inch (default: 300)
        :type dpi: int
        :param format: Output format - one of 'png', 'pdf', 'svg', 'jpg' (default: 'png')
        :type format: str
        :return: PlotResult indicating success/failure and output path
        :rtype: PlotResult
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

        fig.suptitle(
            specification.title or "Routing Computation Time Analysis",
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
            metadata={"plot_type": "computation_time_plot"},
        )


class RoutingVisualizationPlugin(BasePlugin):
    """
    Plugin providing routing-specific visualization capabilities for FUSION.

    This plugin extends the FUSION core visualization system
    (``fusion/visualization/``) with routing algorithm analysis features.
    It implements the ``BasePlugin`` interface and registers custom metrics,
    plot types, and renderers.

    Status: BETA
        This plugin is currently in BETA. The API may change in future releases.

    Registered Components:
        Metrics:
            - ``hop_count``: Number of hops in the routing path
            - ``path_length``: Physical length of the route (km)
            - ``computation_time``: Time to compute the route (ms)
            - ``path_diversity``: Diversity index of paths used
            - ``route_efficiency``: Ratio of used path to shortest path
            - ``k_paths_found``: Number of alternative paths found

        Plot Types:
            - ``hop_count_plot``: Hop count analysis with distribution
            - ``path_length_plot``: Path length vs. traffic volume
            - ``computation_time_plot``: Computation time analysis with CDF

    How It Works:
        1. The core visualization system discovers this plugin via entry points
           or explicit registration
        2. When loaded, the plugin registers its metrics and plot types
        3. Users can then request routing-specific plots via the standard API

    Usage:
        >>> from fusion.visualization.plugins import get_global_registry
        >>> registry = get_global_registry()
        >>> registry.discover_plugins()
        >>> registry.load_plugin("routing")
        >>> # Now routing plot types are available in generate_plot()

    See Also:
        - ``fusion/visualization/plugins/base_plugin.py`` - Plugin interface
        - ``fusion/visualization/application/use_cases/generate_plot.py`` - Plot generation

    Example:
        >>> plugin = RoutingVisualizationPlugin()
        >>> print(plugin.name)
        'routing'
        >>> metrics = plugin.register_metrics()
        >>> plot_types = plugin.register_plot_types()
        >>> print(list(plot_types.keys()))
        ['hop_count_plot', 'path_length_plot', 'computation_time_plot']
    """

    @property
    def name(self) -> str:
        """
        Return the plugin identifier.

        :return: Plugin name used for registration and discovery
        :rtype: str
        """
        return "routing"

    @property
    def version(self) -> str:
        """
        Return the plugin version.

        :return: Semantic version string
        :rtype: str
        """
        return "1.0.0"

    @property
    def description(self) -> str:
        """
        Return a brief description of the plugin.

        :return: Human-readable description of plugin capabilities
        :rtype: str
        """
        return "Visualization plugin for routing algorithm analysis"

    def register_metrics(self) -> list[MetricDefinition]:
        """
        Register routing-specific metrics with the visualization system.

        Defines metrics that can be extracted from simulation output data
        and used in plots. Each metric specifies its data type, source path
        in the output JSON, and aggregation strategy.

        :return: List of MetricDefinition objects for routing metrics
        :rtype: list[MetricDefinition]
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

    def register_plot_types(self) -> dict[str, PlotTypeRegistration]:
        """
        Register routing-specific plot types with the visualization system.

        Each plot type registration includes:
            - processor: Strategy for processing raw data into plot-ready format
            - renderer: Class responsible for rendering the plot
            - description: Human-readable description
            - required_metrics: Metrics that must be present in data
            - default_config: Default configuration options

        :return: Dictionary mapping plot type names to PlotTypeRegistration objects
        :rtype: dict[str, PlotTypeRegistration]
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

    def get_config_schema(self) -> dict:
        """
        Return JSON schema for routing plugin configuration.

        The schema defines available configuration options that can be
        passed when generating plots. These options control aspects like
        whether to show distributions or percentile markers.

        :return: JSON schema dictionary following JSON Schema draft-07
        :rtype: dict
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
