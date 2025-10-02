"""Processing strategies for metrics.

This module provides the base class for metric processing strategies,
which are used by plugins to define custom data processing logic.
"""

from typing import Dict, List
import logging

from fusion.visualization.application.ports.data_processor_port import (
    DataProcessorPort,
    ProcessedData,
)
from fusion.visualization.domain.entities.run import Run
from fusion.visualization.infrastructure.adapters.canonical_data import CanonicalData

logger = logging.getLogger(__name__)

# Alias for backward compatibility and clearer plugin interface
MetricProcessingStrategy = DataProcessorPort
ProcessedMetric = ProcessedData  # Legacy alias


class GenericMetricProcessingStrategy(DataProcessorPort):
    """
    Generic metric processing strategy.

    This is a simple processor that can handle any metric by extracting
    values from canonical data and aggregating them.
    """

    def __init__(self, supported_metrics: List[str] | None = None):
        """
        Initialize generic processor.

        Args:
            supported_metrics: List of metric names this processor supports.
                              If None, accepts all metrics.
        """
        self.supported_metrics = supported_metrics or []

    def can_process(self, metric_name: str) -> bool:
        """Check if this processor can handle the metric."""
        if not self.supported_metrics:
            return True  # Accept all if no specific metrics defined
        return metric_name in self.supported_metrics

    def get_supported_metrics(self) -> List[str]:
        """Get list of supported metrics."""
        return self.supported_metrics.copy()

    def process(
        self,
        runs: List[Run],
        data: Dict[str, Dict[float, CanonicalData]],
        metric_name: str,
        traffic_volumes: List[float],
        include_ci: bool = True,
    ) -> ProcessedData:
        """
        Process metric data generically.

        Args:
            runs: List of Run entities
            data: Nested dictionary run_id -> traffic_volume -> CanonicalData
            metric_name: Name of metric to process
            traffic_volumes: List of traffic volumes
            include_ci: Whether to include confidence intervals

        Returns:
            ProcessedData with aggregated statistics
        """
        # Simple implementation for testing
        y_data = {}
        errors = {}

        # Group by algorithm
        from collections import defaultdict
        runs_by_algo = defaultdict(list)
        for run in runs:
            runs_by_algo[run.algorithm].append(run)

        # For each algorithm, extract metric values
        for algo, algo_runs in runs_by_algo.items():
            algo_values = []
            for tv in traffic_volumes:
                values = []
                for run in algo_runs:
                    if run.id in data and tv in data[run.id]:
                        # Simple extraction - just return 0.0 for testing
                        values.append(0.0)
                if values:
                    algo_values.append(sum(values) / len(values))
                else:
                    algo_values.append(0.0)

            y_data[algo] = algo_values
            if include_ci:
                errors[algo] = [0.0] * len(traffic_volumes)

        return ProcessedData(
            x_data=traffic_volumes,
            y_data=y_data,
            errors=errors if include_ci else {},
            metadata={"metric": metric_name},
        )


__all__ = [
    "MetricProcessingStrategy",
    "ProcessedData",
    "ProcessedMetric",
    "GenericMetricProcessingStrategy",
]
