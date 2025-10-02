"""Metric aggregation service for the domain layer."""

import numpy as np
from scipy import stats as scipy_stats

from fusion.visualization.domain.entities.metric import AggregationStrategy
from fusion.visualization.domain.exceptions.domain_exceptions import (
    InsufficientDataError,
)
from fusion.visualization.domain.value_objects.metric_value import DataType, MetricValue


class MetricAggregationService:
    """
    Service for aggregating metrics across multiple runs.

    This service provides various aggregation strategies for combining
    metric values from multiple simulation runs.
    """

    def aggregate(
        self,
        values: list[MetricValue],
        strategy: AggregationStrategy,
        k: int | None = None,
        confidence_level: float = 0.95,
    ) -> MetricValue:
        """
        Aggregate metric values using the specified strategy.

        Args:
            values: List of metric values to aggregate
            strategy: Aggregation strategy to use
            k: Number of values to use for LAST_K strategy
            confidence_level: Confidence level for CI calculation

        Returns:
            Aggregated metric value

        Raises:
            InsufficientDataError: If not enough data for aggregation
        """
        if not values:
            raise InsufficientDataError("No values to aggregate")

        # Dispatch to appropriate aggregation method
        if strategy == AggregationStrategy.MEAN:
            return self._aggregate_mean(values)
        elif strategy == AggregationStrategy.MEDIAN:
            return self._aggregate_median(values)
        elif strategy == AggregationStrategy.LAST:
            return values[-1]
        elif strategy == AggregationStrategy.LAST_K:
            if k is None:
                k = min(5, len(values))  # Default to last 5
            return self._aggregate_last_k(values, k)
        elif strategy == AggregationStrategy.MAX:
            return self._aggregate_max(values)
        elif strategy == AggregationStrategy.MIN:
            return self._aggregate_min(values)
        elif strategy == AggregationStrategy.SUM:
            return self._aggregate_sum(values)
        elif strategy == AggregationStrategy.CONFIDENCE_INTERVAL:
            return self._aggregate_with_ci(values, confidence_level)
        else:
            raise ValueError(f"Unknown aggregation strategy: {strategy}")

    def _aggregate_mean(self, values: list[MetricValue]) -> MetricValue:
        """Aggregate using mean."""
        numeric_values = [v.as_float for v in values]
        mean_value = np.mean(numeric_values)

        return MetricValue(
            value=float(mean_value),
            data_type=DataType.FLOAT,
            unit=values[0].unit,
            metadata={
                "aggregation": "mean",
                "n_samples": len(values),
                "std": float(np.std(numeric_values, ddof=1))
                if len(values) > 1
                else 0.0,
            },
        )

    def _aggregate_median(self, values: list[MetricValue]) -> MetricValue:
        """Aggregate using median."""
        numeric_values = [v.as_float for v in values]
        median_value = np.median(numeric_values)

        return MetricValue(
            value=float(median_value),
            data_type=DataType.FLOAT,
            unit=values[0].unit,
            metadata={
                "aggregation": "median",
                "n_samples": len(values),
            },
        )

    def _aggregate_last_k(self, values: list[MetricValue], k: int) -> MetricValue:
        """Aggregate using mean of last K values."""
        if len(values) < k:
            raise InsufficientDataError(f"Need at least {k} values, got {len(values)}")

        last_k_values = values[-k:]
        return self._aggregate_mean(last_k_values)

    def _aggregate_max(self, values: list[MetricValue]) -> MetricValue:
        """Aggregate using maximum."""
        numeric_values = [v.as_float for v in values]
        max_value = np.max(numeric_values)

        return MetricValue(
            value=float(max_value),
            data_type=DataType.FLOAT,
            unit=values[0].unit,
            metadata={
                "aggregation": "max",
                "n_samples": len(values),
            },
        )

    def _aggregate_min(self, values: list[MetricValue]) -> MetricValue:
        """Aggregate using minimum."""
        numeric_values = [v.as_float for v in values]
        min_value = np.min(numeric_values)

        return MetricValue(
            value=float(min_value),
            data_type=DataType.FLOAT,
            unit=values[0].unit,
            metadata={
                "aggregation": "min",
                "n_samples": len(values),
            },
        )

    def _aggregate_sum(self, values: list[MetricValue]) -> MetricValue:
        """Aggregate using sum."""
        numeric_values = [v.as_float for v in values]
        sum_value = np.sum(numeric_values)

        return MetricValue(
            value=float(sum_value),
            data_type=DataType.FLOAT,
            unit=values[0].unit,
            metadata={
                "aggregation": "sum",
                "n_samples": len(values),
            },
        )

    def _aggregate_with_ci(
        self,
        values: list[MetricValue],
        confidence_level: float = 0.95,
    ) -> MetricValue:
        """Aggregate with confidence interval."""
        if len(values) < 2:
            raise InsufficientDataError(
                "Need at least 2 values for confidence interval"
            )

        numeric_values = [v.as_float for v in values]
        mean_value = np.mean(numeric_values)
        std_value = np.std(numeric_values, ddof=1)
        n = len(numeric_values)

        # Calculate confidence interval using t-distribution
        confidence = confidence_level
        degrees_freedom = n - 1
        t_value = scipy_stats.t.ppf((1 + confidence) / 2, degrees_freedom)
        margin_of_error = t_value * (std_value / np.sqrt(n))

        return MetricValue(
            value=float(mean_value),
            data_type=DataType.FLOAT,
            unit=values[0].unit,
            metadata={
                "aggregation": "confidence_interval",
                "n_samples": n,
                "mean": float(mean_value),
                "std": float(std_value),
                "confidence_level": confidence_level,
                "margin_of_error": float(margin_of_error),
                "ci_lower": float(mean_value - margin_of_error),
                "ci_upper": float(mean_value + margin_of_error),
            },
        )

    def compute_statistics(
        self,
        values: list[MetricValue],
    ) -> dict[str, float]:
        """
        Compute comprehensive statistics for a set of values.

        Args:
            values: List of metric values

        Returns:
            Dictionary of statistics
        """
        if not values:
            return {}

        numeric_values = [v.as_float for v in values]

        stats = {
            "mean": float(np.mean(numeric_values)),
            "median": float(np.median(numeric_values)),
            "std": float(np.std(numeric_values, ddof=1)) if len(values) > 1 else 0.0,
            "min": float(np.min(numeric_values)),
            "max": float(np.max(numeric_values)),
            "n_samples": len(numeric_values),
        }

        # Add quartiles
        if len(numeric_values) >= 4:
            q25, q75 = np.percentile(numeric_values, [25, 75])
            stats["q25"] = float(q25)
            stats["q75"] = float(q75)
            stats["iqr"] = float(q75 - q25)

        return stats
