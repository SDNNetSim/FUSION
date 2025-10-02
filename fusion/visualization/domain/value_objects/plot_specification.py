"""Plot specification value object."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class PlotType(Enum):
    """Supported plot types."""

    LINE = "line"
    SCATTER = "scatter"
    HEATMAP = "heatmap"
    BAR = "bar"
    BOX = "box"
    VIOLIN = "violin"
    HISTOGRAM = "histogram"


class PlotStyle(Enum):
    """Supported plot styles."""

    DEFAULT = "default"
    SEABORN_WHITEGRID = "seaborn-whitegrid"
    SEABORN_DARKGRID = "seaborn-darkgrid"
    SEABORN_DARK = "seaborn-dark"
    SEABORN_WHITE = "seaborn-white"
    MATPLOTLIB_CLASSIC = "classic"


@dataclass(frozen=True)
class LegendConfiguration:
    """Configuration for plot legend."""

    show: bool = True
    location: str = "best"
    frameon: bool = True
    fontsize: int = 10
    ncol: int = 1


@dataclass(frozen=True)
class Annotation:
    """Annotation for plot."""

    text: str
    x: float
    y: float
    fontsize: int = 10
    color: str = "black"
    ha: str = "center"  # horizontal alignment
    va: str = "center"  # vertical alignment


@dataclass(frozen=True)
class PlotSpecification:
    """
    Immutable specification for rendering a plot.

    This is a renderer-agnostic specification that contains all the
    information needed to create a plot.
    """

    title: str
    x_data: np.ndarray
    y_data: dict[str, np.ndarray]  # algorithm/series name -> values
    x_label: str
    y_label: str
    plot_type: PlotType = PlotType.LINE
    plot_style: PlotStyle = PlotStyle.DEFAULT
    legend: LegendConfiguration = field(default_factory=LegendConfiguration)
    annotations: list[Annotation] = field(default_factory=list)
    error_bars: dict[str, np.ndarray] | None = None  # algorithm -> error values
    colors: dict[str, str] | None = None  # algorithm -> color
    line_styles: dict[str, str] | None = None  # algorithm -> line style
    markers: dict[str, str] | None = None  # algorithm -> marker
    grid: bool = True
    figsize: tuple[float, float] = (10, 6)
    dpi: int = 100
    metadata: dict[str, Any] = field(default_factory=dict)
    # Legacy parameters for backward compatibility
    errors: dict[str, np.ndarray] | None = None  # alias for error_bars
    y_ci_data: dict[str, np.ndarray] | None = None  # alias for error_bars
    include_ci: bool = True  # ignored, kept for compatibility
    include_legend: bool = True  # ignored, kept for compatibility

    def __post_init__(self) -> None:
        """Validate plot specification and handle legacy parameters."""
        # Handle legacy parameter aliases
        if self.y_ci_data is not None and self.error_bars is None:
            object.__setattr__(self, "error_bars", self.y_ci_data)
        elif self.errors is not None and self.error_bars is None:
            object.__setattr__(self, "error_bars", self.errors)

        # Validate x_data and y_data shapes
        for algorithm, y_values in self.y_data.items():
            if len(y_values) != len(self.x_data):
                raise ValueError(
                    f"Length mismatch: x_data has {len(self.x_data)} points "
                    f"but y_data[{algorithm}] has {len(y_values)} points"
                )

        # Validate error bars if present
        if self.error_bars:
            for algorithm, err_values in self.error_bars.items():
                if algorithm not in self.y_data:
                    raise ValueError(
                        f"Error bars specified for {algorithm} but no y_data found"
                    )
                if len(err_values) != len(self.x_data):
                    raise ValueError(f"Error bar length mismatch for {algorithm}")

    @property
    def algorithms(self) -> list[str]:
        """Return list of algorithms/series in this plot."""
        return list(self.y_data.keys())

    @property
    def num_series(self) -> int:
        """Return number of data series in this plot."""
        return len(self.y_data)

    def has_error_bars(self) -> bool:
        """Check if plot has error bars."""
        return self.error_bars is not None and len(self.error_bars) > 0

    def has_annotations(self) -> bool:
        """Check if plot has annotations."""
        return len(self.annotations) > 0

    def __repr__(self) -> str:
        """Return detailed representation."""
        return (
            f"PlotSpecification(title='{self.title}', "
            f"type={self.plot_type.value}, "
            f"series={self.num_series}, "
            f"points={len(self.x_data)})"
        )
