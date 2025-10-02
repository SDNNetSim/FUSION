"""Multi-metric processor that delegates to specialized processors."""

from __future__ import annotations
import logging
from typing import Dict, List

from fusion.visualization.application.ports import DataProcessorPort, ProcessedData
from fusion.visualization.domain.entities.run import Run
from fusion.visualization.infrastructure.adapters.canonical_data import CanonicalData
from fusion.visualization.domain.exceptions import ProcessingError

logger = logging.getLogger(__name__)


class MultiMetricProcessor(DataProcessorPort):
    """
    Processor that delegates to specialized metric processors.

    This acts as a registry and router for different metric types,
    delegating to the appropriate specialized processor.
    """

    def __init__(self) -> None:
        """Initialize with empty processor registry."""
        self._processors: Dict[str, DataProcessorPort] = {}

    def register_processor(self, processor: DataProcessorPort) -> None:
        """
        Register a processor for its supported metrics.

        Args:
            processor: Processor to register
        """
        for metric in processor.get_supported_metrics():
            self._processors[metric] = processor
            logger.debug(f"Registered processor for metric: {metric}")

    def can_process(self, metric_name: str) -> bool:
        """Check if any registered processor can handle the metric."""
        return metric_name in self._processors

    def get_supported_metrics(self) -> List[str]:
        """Get list of all supported metrics."""
        return list(self._processors.keys())

    def process(
        self,
        runs: List[Run],
        data: Dict[str, Dict[float, CanonicalData]],
        metric_name: str,
        traffic_volumes: List[float],
        include_ci: bool = True,
    ) -> ProcessedData:
        """
        Process data by delegating to appropriate processor.

        Args:
            runs: List of Run entities
            data: Nested dictionary run_id -> traffic_volume -> CanonicalData
            metric_name: Name of metric to process
            traffic_volumes: List of traffic volumes
            include_ci: Whether to include confidence intervals

        Returns:
            ProcessedData from specialized processor

        Raises:
            ProcessingError: If no processor found or processing fails
        """
        if not self.can_process(metric_name):
            raise ProcessingError(
                f"No processor registered for metric: {metric_name}. "
                f"Supported metrics: {', '.join(self.get_supported_metrics())}"
            )

        processor = self._processors[metric_name]

        logger.info(
            f"Delegating processing of '{metric_name}' to {processor.__class__.__name__}"
        )

        return processor.process(
            runs=runs,
            data=data,
            metric_name=metric_name,
            traffic_volumes=traffic_volumes,
            include_ci=include_ci,
        )

    def get_processor_for_metric(self, metric_name: str) -> DataProcessorPort | None:
        """
        Get the processor for a specific metric.

        Args:
            metric_name: Name of the metric

        Returns:
            Processor instance or None if not found
        """
        return self._processors.get(metric_name)
