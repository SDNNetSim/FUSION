"""RL-specific plot renderers.

This module provides specialized plot types for RL visualization:
- Reward learning curves
- Q-value heatmaps
- Convergence plots
- Policy entropy plots
"""

from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.figure import Figure

from fusion.visualization.domain.strategies.processing_strategies import ProcessedMetric
from fusion.visualization.domain.value_objects.plot_specification import PlotSpecification
from fusion.visualization.infrastructure.renderers.base_renderer import (
    BaseRenderer,
    PlotResult,
)


class RewardLearningCurveRenderer(BaseRenderer):
    """Renderer for RL reward learning curves with confidence intervals."""

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
        """Render reward learning curve.

        Args:
            specification: Plot specification
            output_path: Where to save the plot
            dpi: Resolution in dots per inch
            format: Output format (png, pdf, svg)

        Returns:
            PlotResult with rendered figure
        """
        fig, ax = plt.subplots(figsize=specification.figsize)

        # Get processed data
        data = specification.metadata.get("processed_data", {})

        # Plot each algorithm
        for algo, algo_data in data.items():
            episodes = algo_data.get("episodes", [])
            mean = algo_data.get("mean", [])
            ci_lower = algo_data.get("ci_lower")
            ci_upper = algo_data.get("ci_upper")

            # Plot mean line
            line = ax.plot(episodes, mean, label=algo, linewidth=2)
            color = line[0].get_color()

            # Plot confidence interval
            if ci_lower is not None and ci_upper is not None:
                ax.fill_between(
                    episodes, ci_lower, ci_upper, alpha=0.2, color=color, label=f"{algo} (95% CI)"
                )

        # Styling
        ax.set_xlabel(specification.x_label or "Episode", fontsize=12)
        ax.set_ylabel(specification.y_label or "Episode Reward", fontsize=12)
        ax.set_title(specification.title or "RL Training: Reward Learning Curve", fontsize=14, fontweight="bold")
        ax.legend(loc="best", frameon=True, shadow=True)
        ax.grid(True, alpha=0.3)

        # Apply style
        if specification.plot_style:
            plt.style.use(specification.plot_style.value)

        plt.tight_layout()

        # Save
        fig.savefig(output_path, dpi=dpi, format=format, bbox_inches="tight")
        plt.close(fig)

        return PlotResult(
            success=True, output_path=output_path, metadata={"plot_type": "reward_learning_curve"}
        )


class QValueHeatmapRenderer(BaseRenderer):
    """Renderer for Q-value heatmaps."""

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
        """Render Q-value heatmap.

        Args:
            specification: Plot specification
            output_path: Where to save the plot
            dpi: Resolution in dots per inch
            format: Output format (png, pdf, svg)

        Returns:
            PlotResult with rendered figure
        """
        data = specification.metadata.get("processed_data", {})

        # Create subplots for each algorithm
        n_algos = len(data)
        fig, axes = plt.subplots(1, n_algos, figsize=(6 * n_algos, 5))
        if n_algos == 1:
            axes = [axes]

        for ax, (algo, algo_data) in zip(axes, data.items()):
            q_values = algo_data.get("mean")

            if q_values is not None:
                # Create heatmap
                sns.heatmap(
                    q_values.reshape(-1, 1) if q_values.ndim == 1 else q_values,
                    ax=ax,
                    cmap="viridis",
                    cbar=True,
                    annot=False,
                )
                ax.set_title(f"{algo} Q-Values", fontsize=12, fontweight="bold")
                ax.set_xlabel("Action")
                ax.set_ylabel("State/Episode")

        fig.suptitle(specification.title or "Q-Value Analysis", fontsize=14, fontweight="bold")
        plt.tight_layout()

        # Save
        fig.savefig(output_path, dpi=dpi, format=format, bbox_inches="tight")
        plt.close(fig)

        return PlotResult(
            success=True, output_path=output_path, metadata={"plot_type": "q_value_heatmap"}
        )


class ConvergencePlotRenderer(BaseRenderer):
    """Renderer for training convergence analysis."""

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
        """Render convergence plot.

        Args:
            specification: Plot specification
            output_path: Where to save the plot
            dpi: Resolution in dots per inch
            format: Output format (png, pdf, svg)

        Returns:
            PlotResult with rendered figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=specification.figsize)

        data = specification.metadata.get("processed_data", {})

        # Top plot: metric over time
        for algo, algo_data in data.items():
            if "episodes" in algo_data and "mean" in algo_data:
                episodes = algo_data["episodes"]
                mean = algo_data["mean"]
                ax1.plot(episodes, mean, label=algo, linewidth=2)

                # Mark convergence point if available
                if "convergence_episode" in algo_data:
                    conv_ep = algo_data["convergence_episode"]
                    if conv_ep is not None:
                        ax1.axvline(conv_ep, linestyle="--", alpha=0.7, label=f"{algo} convergence")

        ax1.set_xlabel("Episode", fontsize=12)
        ax1.set_ylabel("Metric Value", fontsize=12)
        ax1.set_title("Training Progress", fontsize=12, fontweight="bold")
        ax1.legend(loc="best")
        ax1.grid(True, alpha=0.3)

        # Bottom plot: convergence statistics
        algos = list(data.keys())
        convergence_episodes = [data[a].get("mean_convergence_episode", 0) for a in algos]
        convergence_std = [data[a].get("std_convergence_episode", 0) for a in algos]

        x_pos = np.arange(len(algos))
        ax2.bar(x_pos, convergence_episodes, yerr=convergence_std, capsize=5, alpha=0.7)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(algos, rotation=45, ha="right")
        ax2.set_ylabel("Convergence Episode", fontsize=12)
        ax2.set_title("Convergence Speed Comparison", fontsize=12, fontweight="bold")
        ax2.grid(True, alpha=0.3, axis="y")

        fig.suptitle(specification.title or "RL Training Convergence Analysis", fontsize=14, fontweight="bold")
        plt.tight_layout()

        # Save
        fig.savefig(output_path, dpi=dpi, format=format, bbox_inches="tight")
        plt.close(fig)

        return PlotResult(
            success=True, output_path=output_path, metadata={"plot_type": "convergence_plot"}
        )


class MultiMetricDashboardRenderer(BaseRenderer):
    """Renderer for multi-metric RL training dashboard."""

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
        """Render multi-metric dashboard.

        Args:
            specification: Plot specification
            output_path: Where to save the plot
            dpi: Resolution in dots per inch
            format: Output format (png, pdf, svg)

        Returns:
            PlotResult with rendered figure
        """
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

        # Get different metrics from spec data
        metrics_data = specification.metadata.get("processed_data", {})

        # Plot 1: Rewards
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_metric(ax1, metrics_data.get("rewards", {}), "Episode Reward", "Rewards")

        # Plot 2: Policy Loss
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_metric(ax2, metrics_data.get("policy_loss", {}), "Policy Loss", "Policy Loss")

        # Plot 3: Value Loss
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_metric(ax3, metrics_data.get("value_loss", {}), "Value Loss", "Value Loss")

        # Plot 4: Entropy
        ax4 = fig.add_subplot(gs[2, 0])
        self._plot_metric(ax4, metrics_data.get("entropy", {}), "Entropy", "Policy Entropy")

        # Plot 5: Q-Values
        ax5 = fig.add_subplot(gs[2, 1])
        self._plot_metric(ax5, metrics_data.get("q_values", {}), "Q-Value", "Mean Q-Value")

        fig.suptitle(specification.title or "RL Training Dashboard", fontsize=16, fontweight="bold")

        # Save
        fig.savefig(output_path, dpi=dpi, format=format, bbox_inches="tight")
        plt.close(fig)

        return PlotResult(
            success=True, output_path=output_path, metadata={"plot_type": "multi_metric_dashboard"}
        )

    def _plot_metric(self, ax: Any, data: Dict[str, Any], ylabel: str, title: str) -> None:
        """Helper to plot a single metric."""
        for algo, algo_data in data.items():
            if "episodes" in algo_data and "mean" in algo_data:
                episodes = algo_data["episodes"]
                mean = algo_data["mean"]
                ax.plot(episodes, mean, label=algo, linewidth=2)

                # Add confidence interval if available
                if "ci_lower" in algo_data and "ci_upper" in algo_data:
                    ax.fill_between(
                        episodes, algo_data["ci_lower"], algo_data["ci_upper"], alpha=0.2
                    )

        ax.set_xlabel("Episode", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
