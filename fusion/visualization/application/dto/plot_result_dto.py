"""Data Transfer Objects for plot results."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


@dataclass
class PlotResultDTO:
    """
    DTO for plot generation results.

    This carries results from the application layer back to the interface layer.
    """

    # Success/failure status
    success: bool
    plot_id: str

    # Output information
    output_path: Path | None = None
    plot_type: str | None = None

    # Metadata about the generation
    algorithms: list[str] = field(default_factory=list)
    traffic_volumes: list[float] = field(default_factory=list)
    num_runs: int = 0

    # Timing information
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration: timedelta | None = None

    # Error information
    error: str | None = None
    error_details: str | None = None
    error_message: str | None = None  # Alias for error

    # Additional metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Handle error message aliases."""
        # If error_message is set but error is not, copy it
        if self.error_message and not self.error:
            object.__setattr__(self, "error", self.error_message)
        # If error is set but error_message is not, copy it
        elif self.error and not self.error_message:
            object.__setattr__(self, "error_message", self.error)

    @property
    def duration_seconds(self) -> float | None:
        """Get duration in seconds."""
        if self.duration:
            return self.duration.total_seconds()
        return None

    @property
    def algorithms_plotted(self) -> list[str]:
        """Alias for algorithms property (backward compatibility)."""
        return self.algorithms

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,
            "plot_id": self.plot_id,
            "output_path": str(self.output_path) if self.output_path else None,
            "plot_type": self.plot_type,
            "algorithms": self.algorithms,
            "traffic_volumes": self.traffic_volumes,
            "num_runs": self.num_runs,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
            "error": self.error,
            "error_details": self.error_details,
            "metadata": self.metadata,
        }


@dataclass
class BatchPlotResultDTO:
    """DTO for batch plot generation results."""

    results: list[PlotResultDTO]
    plots: list[PlotResultDTO] | None = None  # Alias for results

    # Overall batch status
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration: timedelta | None = None

    def __post_init__(self) -> None:
        """Handle results/plots aliases."""
        # If plots is set but results is not, copy it
        if self.plots and not self.results:
            object.__setattr__(self, "results", self.plots)
        # If results is set but plots is not, copy it
        elif self.results and not self.plots:
            object.__setattr__(self, "plots", self.results)

    @property
    def success_count(self) -> int:
        """Number of successful plots."""
        return sum(1 for r in self.results if r.success)

    @property
    def failure_count(self) -> int:
        """Number of failed plots."""
        return sum(1 for r in self.results if not r.success)

    @property
    def total_count(self) -> int:
        """Total number of plots."""
        return len(self.results)

    @property
    def success_rate(self) -> float:
        """Success rate as percentage."""
        if self.total_count == 0:
            return 0.0
        return (self.success_count / self.total_count) * 100

    @property
    def all_successful(self) -> bool:
        """Check if all plots succeeded."""
        return self.failure_count == 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "results": [r.to_dict() for r in self.results],
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "total_count": self.total_count,
            "success_rate": self.success_rate,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration.total_seconds() if self.duration else None,
        }


@dataclass
class StatisticalComparison:
    """Statistical comparison between two algorithms."""

    algorithm_a: str
    algorithm_b: str
    metric: str

    # Descriptive statistics
    mean_a: float
    mean_b: float
    std_a: float
    std_b: float

    # Statistical tests
    p_value: float | None = None
    test_statistic: float | None = None
    test_name: str = "t-test"

    # Effect sizes
    cohens_d: float | None = None
    effect_size_interpretation: str | None = None

    # Confidence intervals
    ci_lower_a: float | None = None
    ci_upper_a: float | None = None
    ci_lower_b: float | None = None
    ci_upper_b: float | None = None

    @property
    def is_significant(self) -> bool:
        """Check if difference is statistically significant (p < 0.05)."""
        return self.p_value is not None and self.p_value < 0.05

    @property
    def mean_difference(self) -> float:
        """Difference in means (B - A)."""
        return self.mean_b - self.mean_a

    @property
    def percent_improvement(self) -> float:
        """Percent improvement from A to B."""
        if self.mean_a == 0:
            return 0.0
        return ((self.mean_b - self.mean_a) / abs(self.mean_a)) * 100


@dataclass
class ComparisonResultDTO:
    """DTO for algorithm comparison results."""

    network: str
    dates: list[str]
    algorithms: list[str]
    metric: str

    # Comparison results
    comparisons: list[StatisticalComparison]

    # Output information
    output_path: Path | None = None
    success: bool = True
    error: str | None = None

    # Metadata
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration: timedelta | None = None

    def get_comparison(self, algo_a: str, algo_b: str) -> StatisticalComparison | None:
        """Get comparison between two algorithms."""
        for comp in self.comparisons:
            if comp.algorithm_a == algo_a and comp.algorithm_b == algo_b:
                return comp
            if comp.algorithm_a == algo_b and comp.algorithm_b == algo_a:
                # Return inverted comparison
                return StatisticalComparison(
                    algorithm_a=algo_a,
                    algorithm_b=algo_b,
                    metric=comp.metric,
                    mean_a=comp.mean_b,
                    mean_b=comp.mean_a,
                    std_a=comp.std_b,
                    std_b=comp.std_a,
                    p_value=comp.p_value,
                    test_statistic=-comp.test_statistic if comp.test_statistic else None,
                    test_name=comp.test_name,
                    cohens_d=-comp.cohens_d if comp.cohens_d else None,
                    ci_lower_a=comp.ci_lower_b,
                    ci_upper_a=comp.ci_upper_b,
                    ci_lower_b=comp.ci_lower_a,
                    ci_upper_b=comp.ci_upper_a,
                )
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "network": self.network,
            "dates": self.dates,
            "algorithms": self.algorithms,
            "metric": self.metric,
            "comparisons": [
                {
                    "algorithm_a": c.algorithm_a,
                    "algorithm_b": c.algorithm_b,
                    "mean_a": c.mean_a,
                    "mean_b": c.mean_b,
                    "mean_difference": c.mean_difference,
                    "percent_improvement": c.percent_improvement,
                    "p_value": c.p_value,
                    "is_significant": c.is_significant,
                    "cohens_d": c.cohens_d,
                    "effect_size_interpretation": c.effect_size_interpretation,
                }
                for c in self.comparisons
            ],
            "output_path": str(self.output_path) if self.output_path else None,
            "success": self.success,
            "error": self.error,
            "duration_seconds": self.duration.total_seconds() if self.duration else None,
        }
