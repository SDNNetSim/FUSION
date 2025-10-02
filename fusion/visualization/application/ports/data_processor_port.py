"""Port interface for data processing."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from fusion.visualization.domain.entities.run import Run
from fusion.visualization.infrastructure.adapters.canonical_data import CanonicalData


@dataclass
class ProcessedData:
    """Container for processed plot data."""

    x_data: list[float]  # e.g., traffic volumes
    y_data: dict[str, list[float]]  # algorithm -> values
    errors: dict[str, list[float]] | None = (
        None  # algorithm -> error bars (std, CI, etc.)
    )
    metadata: dict[str, Any] | None = None


class DataProcessorPort(ABC):
    """
    Port for data processing strategies.

    This interface abstracts the processing of raw simulation data
    into plottable format, allowing different processing strategies
    for different plot types.
    """

    @abstractmethod
    def process(
        self,
        runs: list[Run],
        data: dict[str, dict[float, CanonicalData]],  # run_id -> traffic_volume -> data
        metric_name: str,
        traffic_volumes: list[float],
        include_ci: bool = True,
    ) -> ProcessedData:
        """
        Process raw data into plottable format.

        Args:
            runs: List of Run entities
            data: Nested dictionary of canonical data
            metric_name: Name of metric to extract and process
            traffic_volumes: List of traffic volumes to include
            include_ci: Whether to include confidence intervals

        Returns:
            ProcessedData ready for rendering

        Raises:
            ProcessingError: If processing fails
        """
        pass

    @abstractmethod
    def can_process(self, metric_name: str) -> bool:
        """
        Check if this processor can handle the given metric.

        Args:
            metric_name: Name of the metric

        Returns:
            True if this processor can handle the metric
        """
        pass

    @abstractmethod
    def get_supported_metrics(self) -> list[str]:
        """
        Get list of metrics this processor supports.

        Returns:
            List of metric names
        """
        pass
