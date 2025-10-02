"""Data Transfer Objects for plot requests."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fusion.visualization.domain.value_objects.plot_specification import PlotType


@dataclass
class PlotRequestDTO:
    """
    DTO for plot generation requests.

    This carries data from the interface layer to the application layer,
    decoupling external API from internal domain model.
    """

    # Required fields
    network: str
    dates: list[str]
    plot_type: PlotType

    # Optional filters and configuration
    algorithms: list[str] | None = None
    traffic_volumes: list[float] | None = None
    run_ids: list[str] | None = None
    metrics: list[str] | None = None  # Specific metrics to plot

    # Plot configuration
    title: str | None = None
    x_label: str | None = None
    y_label: str | None = None
    include_ci: bool = True
    include_baselines: bool = False

    # Output configuration
    save_path: Path | None = None
    dpi: int = 300
    figsize: tuple[float, float] = (10, 6)
    format: str = "png"

    # Advanced options
    cache_enabled: bool = True
    parallel_processing: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> list[str]:
        """
        Validate the request.

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        if not self.network:
            errors.append("network is required")

        if not self.dates:
            errors.append("at least one date is required")

        if self.traffic_volumes and any(tv <= 0 for tv in self.traffic_volumes):
            errors.append("traffic_volumes must be positive")

        if self.dpi <= 0:
            errors.append("dpi must be positive")

        if self.format not in ["png", "pdf", "svg", "jpg"]:
            errors.append(f"unsupported format: {self.format}")

        return errors


@dataclass
class BatchPlotRequestDTO:
    """DTO for batch plot generation requests."""

    network: str
    dates: list[str]
    plots: list[PlotRequestDTO]

    # Batch-specific configuration
    parallel: bool = True
    max_workers: int = 4
    stop_on_error: bool = False
    output_dir: Path | None = None

    def validate(self) -> list[str]:
        """Validate batch request."""
        errors = []

        if not self.network:
            errors.append("network is required")

        if not self.dates:
            errors.append("at least one date is required")

        if not self.plots:
            errors.append("at least one plot is required")

        # Validate individual plots
        for i, plot in enumerate(self.plots):
            plot_errors = plot.validate()
            for error in plot_errors:
                errors.append(f"plot {i}: {error}")

        return errors


@dataclass
class ComparisonRequestDTO:
    """DTO for algorithm comparison requests."""

    network: str
    dates: list[str]
    algorithms: list[str]
    metric: str  # e.g., "blocking_probability"

    # Comparison configuration
    traffic_volumes: list[float] | None = None
    include_statistical_tests: bool = True
    include_effect_sizes: bool = True
    confidence_level: float = 0.95

    # Output configuration
    save_path: Path | None = None
    dpi: int = 300
    format: str = "png"

    def validate(self) -> list[str]:
        """Validate comparison request."""
        errors = []

        if not self.network:
            errors.append("network is required")

        if not self.dates:
            errors.append("at least one date is required")

        if not self.algorithms or len(self.algorithms) < 2:
            errors.append("at least two algorithms required for comparison")

        if not self.metric:
            errors.append("metric is required")

        if not 0 < self.confidence_level < 1:
            errors.append("confidence_level must be between 0 and 1")

        return errors
