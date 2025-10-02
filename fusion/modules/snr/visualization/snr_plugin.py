"""SNR visualization plugin.

This plugin provides visualization support for Signal-to-Noise Ratio (SNR)
analysis in optical networks.
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


class SNRvsDistancePlotRenderer(BaseRenderer):
    """Renderer for SNR vs distance plots."""

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
        """Render SNR vs distance plot.

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

        # Plot each modulation format or configuration
        for config, config_data in data.items():
            if "distances" in config_data and "snr" in config_data:
                distances = config_data["distances"]
                snr = config_data["snr"]
                std = config_data.get("std")

                # Plot mean SNR
                line = ax.plot(distances, snr, marker="o", label=config, linewidth=2)

                # Add confidence interval if available
                if std is not None:
                    color = line[0].get_color()
                    ax.fill_between(distances, snr - std, snr + std, alpha=0.2, color=color)

        # Add SNR threshold lines if specified
        if "snr_threshold" in specification.metadata:
            threshold = specification.metadata["snr_threshold"]
            ax.axhline(
                threshold, color="red", linestyle="--", linewidth=2, label=f"Threshold ({threshold} dB)"
            )

        # Styling
        ax.set_xlabel(specification.x_label or "Distance (km)", fontsize=12)
        ax.set_ylabel(specification.y_label or "SNR (dB)", fontsize=12)
        ax.set_title(specification.title or "SNR vs Distance", fontsize=14, fontweight="bold")
        ax.legend(loc="best", frameon=True, shadow=True)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save
        fig.savefig(output_path, dpi=dpi, format=format, bbox_inches="tight")
        plt.close(fig)

        return PlotResult(
            success=True, output_path=output_path, metadata={"plot_type": "snr_vs_distance"}
        )


class QFactorPlotRenderer(BaseRenderer):
    """Renderer for Q-factor analysis plots."""

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
        """Render Q-factor plot.

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

        # Plot 1: Q-factor vs distance
        for config, config_data in data.items():
            if "distances" in config_data and "q_factor" in config_data:
                ax1.plot(
                    config_data["distances"],
                    config_data["q_factor"],
                    marker="o",
                    label=config,
                    linewidth=2,
                )

        ax1.set_xlabel("Distance (km)", fontsize=12)
        ax1.set_ylabel("Q-Factor (dB)", fontsize=12)
        ax1.set_title("Q-Factor vs Distance", fontsize=12, fontweight="bold")
        ax1.legend(loc="best")
        ax1.grid(True, alpha=0.3)

        # Plot 2: BER vs SNR
        for config, config_data in data.items():
            if "snr" in config_data and "ber" in config_data:
                ax2.semilogy(config_data["snr"], config_data["ber"], marker="o", label=config, linewidth=2)

        ax2.set_xlabel("SNR (dB)", fontsize=12)
        ax2.set_ylabel("Bit Error Rate (BER)", fontsize=12)
        ax2.set_title("BER vs SNR", fontsize=12, fontweight="bold")
        ax2.legend(loc="best")
        ax2.grid(True, alpha=0.3, which="both")

        fig.suptitle(specification.title or "Q-Factor and BER Analysis", fontsize=14, fontweight="bold")
        plt.tight_layout()

        # Save
        fig.savefig(output_path, dpi=dpi, format=format, bbox_inches="tight")
        plt.close(fig)

        return PlotResult(success=True, output_path=output_path, metadata={"plot_type": "q_factor_plot"})


class OSNRMarginPlotRenderer(BaseRenderer):
    """Renderer for OSNR margin visualization."""

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
        """Render OSNR margin plot.

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

        # Plot OSNR margins
        configs = list(data.keys())
        margins = [data[c].get("osnr_margin", 0) for c in configs]
        colors = ["green" if m > 3 else "orange" if m > 1 else "red" for m in margins]

        x_pos = np.arange(len(configs))
        bars = ax.bar(x_pos, margins, color=colors, alpha=0.7, edgecolor="black")

        # Add threshold lines
        ax.axhline(3.0, color="green", linestyle="--", linewidth=2, alpha=0.5, label="Safe (>3 dB)")
        ax.axhline(1.0, color="orange", linestyle="--", linewidth=2, alpha=0.5, label="Warning (>1 dB)")
        ax.axhline(0.0, color="red", linestyle="--", linewidth=2, alpha=0.5, label="Critical (<0 dB)")

        # Add value labels on bars
        for bar, margin in zip(bars, margins):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{margin:.1f} dB",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        ax.set_xticks(x_pos)
        ax.set_xticklabels(configs, rotation=45, ha="right")
        ax.set_ylabel("OSNR Margin (dB)", fontsize=12)
        ax.set_title(specification.title or "OSNR Margin Analysis", fontsize=14, fontweight="bold")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()

        # Save
        fig.savefig(output_path, dpi=dpi, format=format, bbox_inches="tight")
        plt.close(fig)

        return PlotResult(
            success=True, output_path=output_path, metadata={"plot_type": "osnr_margin_plot"}
        )


class SNRVisualizationPlugin(BasePlugin):
    """Plugin for SNR-specific visualizations.

    This plugin extends the FUSION visualization system with:
    - SNR vs distance analysis
    - Q-factor visualization
    - BER analysis
    - OSNR margin plots
    """

    @property
    def name(self) -> str:
        """Return plugin name."""
        return "snr"

    @property
    def version(self) -> str:
        """Return plugin version."""
        return "1.0.0"

    @property
    def description(self) -> str:
        """Return plugin description."""
        return "Visualization plugin for SNR and signal quality analysis"

    def register_metrics(self) -> List[MetricDefinition]:
        """Register SNR-specific metrics.

        Returns:
            List of SNR metric definitions
        """
        return [
            MetricDefinition(
                name="snr",
                display_name="Signal-to-Noise Ratio",
                data_type=DataType.FLOAT,
                source_path="$.snr.value",
                aggregation=AggregationStrategy.MEAN,
                unit="dB",
                description="Signal-to-Noise Ratio in decibels",
            ),
            MetricDefinition(
                name="osnr",
                display_name="Optical SNR",
                data_type=DataType.FLOAT,
                source_path="$.snr.osnr",
                aggregation=AggregationStrategy.MEAN,
                unit="dB",
                description="Optical Signal-to-Noise Ratio",
            ),
            MetricDefinition(
                name="q_factor",
                display_name="Q-Factor",
                data_type=DataType.FLOAT,
                source_path="$.snr.q_factor",
                aggregation=AggregationStrategy.MEAN,
                unit="dB",
                description="Quality factor (Q-factor) of the signal",
            ),
            MetricDefinition(
                name="ber",
                display_name="Bit Error Rate",
                data_type=DataType.FLOAT,
                source_path="$.snr.ber",
                aggregation=AggregationStrategy.MEAN,
                unit="rate",
                description="Bit Error Rate (BER)",
            ),
            MetricDefinition(
                name="osnr_margin",
                display_name="OSNR Margin",
                data_type=DataType.FLOAT,
                source_path="$.snr.osnr_margin",
                aggregation=AggregationStrategy.MIN,
                unit="dB",
                description="OSNR margin (difference from required OSNR)",
            ),
            MetricDefinition(
                name="ase_noise",
                display_name="ASE Noise",
                data_type=DataType.FLOAT,
                source_path="$.snr.ase_noise",
                aggregation=AggregationStrategy.MEAN,
                unit="dBm",
                description="Amplified Spontaneous Emission noise power",
            ),
            MetricDefinition(
                name="signal_power",
                display_name="Signal Power",
                data_type=DataType.FLOAT,
                source_path="$.snr.signal_power",
                aggregation=AggregationStrategy.MEAN,
                unit="dBm",
                description="Optical signal power",
            ),
        ]

    def register_plot_types(self) -> Dict[str, PlotTypeRegistration]:
        """Register SNR-specific plot types.

        Returns:
            Dictionary of plot type registrations
        """
        from fusion.visualization.domain.strategies.processing_strategies import (
            GenericMetricProcessingStrategy,
        )

        return {
            "snr_vs_distance": PlotTypeRegistration(
                processor=GenericMetricProcessingStrategy(),
                renderer=SNRvsDistancePlotRenderer(),
                description="Plot showing SNR degradation with distance",
                required_metrics=["snr"],
                default_config={"show_threshold": True, "snr_threshold": 15.0},
            ),
            "q_factor_plot": PlotTypeRegistration(
                processor=GenericMetricProcessingStrategy(),
                renderer=QFactorPlotRenderer(),
                description="Q-factor and BER analysis plots",
                required_metrics=["q_factor", "ber"],
                default_config={"use_log_scale": True},
            ),
            "osnr_margin_plot": PlotTypeRegistration(
                processor=GenericMetricProcessingStrategy(),
                renderer=OSNRMarginPlotRenderer(),
                description="OSNR margin visualization with safety thresholds",
                required_metrics=["osnr_margin"],
                default_config={"show_thresholds": True, "safe_margin": 3.0, "warning_margin": 1.0},
            ),
        }

    def get_config_schema(self) -> Dict:
        """Return JSON schema for SNR plugin configuration.

        Returns:
            JSON schema dictionary
        """
        return {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "SNR Visualization Plugin Configuration",
            "type": "object",
            "properties": {
                "snr_threshold": {
                    "type": "number",
                    "default": 15.0,
                    "description": "SNR threshold for acceptable performance (dB)",
                },
                "safe_margin": {
                    "type": "number",
                    "default": 3.0,
                    "description": "Safe OSNR margin threshold (dB)",
                },
                "warning_margin": {
                    "type": "number",
                    "default": 1.0,
                    "description": "Warning OSNR margin threshold (dB)",
                },
            },
        }
