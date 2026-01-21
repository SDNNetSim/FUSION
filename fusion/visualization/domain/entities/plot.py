"""Plot entity representing a visualization."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from fusion.visualization.domain.exceptions.domain_exceptions import (
    InvalidStateError,
    ValidationError,
)
from fusion.visualization.domain.value_objects.plot_id import PlotId
from fusion.visualization.domain.value_objects.plot_specification import (
    PlotSpecification,
    PlotType,
)


class PlotState(Enum):
    """States a plot can be in during its lifecycle."""

    PENDING = "pending"
    LOADING = "loading"
    LOADED = "loaded"
    PROCESSING = "processing"
    PROCESSED = "processed"
    RENDERING = "rendering"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class PlotConfiguration:
    """Configuration for a plot."""

    plot_type: PlotType
    metrics: list[str]  # Metric names to plot
    algorithms: list[str]
    traffic_volumes: list[float]
    title: str | None = None
    x_label: str | None = None
    y_label: str | None = None
    include_ci: bool = True  # Include confidence intervals
    include_baselines: bool = False
    dpi: int = 100
    figsize: tuple[float, float] = (10, 6)
    save_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Plot:
    """
    Domain entity representing a visualization.

    This is an aggregate root that manages the lifecycle of plot generation
    from configuration through rendering.
    """

    id: PlotId
    title: str
    configuration: PlotConfiguration
    created_at: datetime = field(default_factory=datetime.now)
    state: PlotState = PlotState.PENDING
    specification: PlotSpecification | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """
        Validate plot configuration.

        Raises:
            ValidationError: If validation fails
        """
        errors = []

        if not self.title:
            errors.append("Plot title cannot be empty")

        if not self.configuration.algorithms:
            errors.append("at least one algorithm must be specified")

        if not self.configuration.traffic_volumes:
            errors.append("At least one traffic volume must be specified")

        if not self.configuration.metrics:
            errors.append("At least one metric must be specified")

        if errors:
            raise ValidationError("; ".join(errors))

    def start_loading(self) -> None:
        """Mark plot as loading data."""
        if self.state != PlotState.PENDING:
            raise InvalidStateError(f"Cannot start loading from state {self.state.value}")
        self.state = PlotState.LOADING

    def mark_loaded(self) -> None:
        """Mark plot data as loaded."""
        if self.state != PlotState.LOADING:
            raise InvalidStateError(f"Cannot mark loaded from state {self.state.value}")
        self.state = PlotState.LOADED

    def start_processing(self) -> None:
        """Mark plot as processing."""
        if self.state != PlotState.LOADED:
            raise InvalidStateError(f"Cannot start processing from state {self.state.value}")
        self.state = PlotState.PROCESSING

    def mark_processed(self, specification: PlotSpecification) -> None:
        """
        Mark plot as processed with specification.

        Args:
            specification: The generated plot specification
        """
        if self.state != PlotState.PROCESSING:
            raise InvalidStateError(f"Cannot mark processed from state {self.state.value}")
        self.specification = specification
        self.state = PlotState.PROCESSED

    def start_rendering(self) -> None:
        """Mark plot as rendering."""
        if self.state != PlotState.PROCESSED:
            raise InvalidStateError(f"Cannot start rendering from state {self.state.value}")
        self.state = PlotState.RENDERING

    def mark_completed(self) -> None:
        """Mark plot as completed."""
        if self.state != PlotState.RENDERING:
            raise InvalidStateError(f"Cannot mark completed from state {self.state.value}")
        self.state = PlotState.COMPLETED

    def mark_failed(self, error: str) -> None:
        """
        Mark plot as failed.

        Args:
            error: Description of the failure
        """
        self.error = error
        self.state = PlotState.FAILED

    def is_completed(self) -> bool:
        """Check if plot is completed."""
        return self.state == PlotState.COMPLETED

    def is_failed(self) -> bool:
        """Check if plot failed."""
        return self.state == PlotState.FAILED

    def __repr__(self) -> str:
        """Return detailed representation."""
        return f"Plot(id={self.id}, title='{self.title}', state={self.state.value})"
