"""Metric entity for defining measurable quantities from simulations."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional
from enum import Enum
import jsonpath_ng.ext as jp

from fusion.visualization.domain.value_objects.metric_value import (
    MetricValue,
    DataType,
)
from fusion.visualization.domain.exceptions.domain_exceptions import (
    InvalidMetricPathError,
    MetricExtractionError,
)


class AggregationStrategy(Enum):
    """Strategies for aggregating metric values across runs."""

    MEAN = "mean"
    MEDIAN = "median"
    LAST = "last"
    LAST_K = "last_k"  # Mean of last K values
    MAX = "max"
    MIN = "min"
    SUM = "sum"
    CONFIDENCE_INTERVAL = "confidence_interval"


@dataclass
class MetricDefinition:
    """
    Defines a measurable quantity from simulation data.

    This entity encapsulates what a metric is, where to find it in the data,
    and how to extract and aggregate it.
    """

    name: str
    data_type: DataType
    source_path: str  # JSONPath expression
    aggregation: Optional[AggregationStrategy] = None
    unit: Optional[str] = None
    description: str = ""
    display_name: Optional[str] = None  # Human-readable name for display
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        """Validate metric definition."""
        if not self.name:
            raise ValueError("Metric name cannot be empty")
        if not self.source_path:
            raise ValueError("Metric source_path cannot be empty")
        if self.metadata is None:
            object.__setattr__(self, 'metadata', {})

        # Validate JSONPath expression
        try:
            jp.parse(self.source_path)
        except Exception as e:
            raise InvalidMetricPathError(
                f"Invalid JSONPath expression '{self.source_path}': {e}"
            )

    def extract_from(self, data: Dict[str, Any]) -> MetricValue:
        """
        Extract this metric from raw simulation data.

        Args:
            data: Raw simulation data (typically a dictionary)

        Returns:
            MetricValue containing the extracted value

        Raises:
            MetricExtractionError: If extraction fails
        """
        try:
            # Parse JSONPath and find matches
            expr = jp.parse(self.source_path)
            matches = expr.find(data)

            if not matches:
                raise MetricExtractionError(
                    f"No data found at path '{self.source_path}'"
                )

            # Extract value from first match
            value = matches[0].value

            return MetricValue(
                value=value,
                data_type=self.data_type,
                unit=self.unit,
                metadata={
                    "metric_name": self.name,
                    "source_path": self.source_path,
                    **(self.metadata if self.metadata is not None else {}),
                },
            )

        except Exception as e:
            raise MetricExtractionError(
                f"Failed to extract metric '{self.name}' from path "
                f"'{self.source_path}': {e}"
            )

    def __repr__(self) -> str:
        """Return detailed representation."""
        return (
            f"MetricDefinition(name='{self.name}', "
            f"type={self.data_type.value}, "
            f"path='{self.source_path}')"
        )
