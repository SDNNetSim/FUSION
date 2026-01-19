"""Processor for blocking probability metrics."""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np

from fusion.visualization.application.ports import DataProcessorPort, ProcessedData
from fusion.visualization.domain.entities.run import Run
from fusion.visualization.domain.exceptions import ProcessingError
from fusion.visualization.infrastructure.adapters.canonical_data import CanonicalData

logger = logging.getLogger(__name__)


class BlockingProbabilityProcessor(DataProcessorPort):
    """
    Processor for blocking probability metrics.

    This aggregates blocking probability across multiple runs and
    computes statistics (mean, std, confidence intervals).
    """

    SUPPORTED_METRICS = [
        "blocking",
        "blocking_probability",
        "blocking_mean",
    ]

    def can_process(self, metric_name: str) -> bool:
        """Check if this processor can handle the metric."""
        return metric_name in self.SUPPORTED_METRICS

    def get_supported_metrics(self) -> list[str]:
        """Get list of supported metrics."""
        return self.SUPPORTED_METRICS.copy()

    def process(
        self,
        runs: list[Run],
        data: dict[str, dict[float, CanonicalData]] | None = None,
        metric_name: str = "blocking_probability",
        traffic_volumes: list[float] | None = None,
        include_ci: bool = True,
        run_data: dict[str, dict[float, CanonicalData]] | None = None,  # Legacy alias for data
    ) -> ProcessedData:
        """
        Process blocking probability data.

        Args:
            runs: List of Run entities
            data: Nested dictionary run_id -> traffic_volume -> CanonicalData
            metric_name: Name of metric to process
            traffic_volumes: List of traffic volumes
            include_ci: Whether to include confidence intervals
            run_data: Legacy alias for data parameter

        Returns:
            ProcessedData with aggregated statistics

        Raises:
            ProcessingError: If processing fails
        """
        # Handle legacy parameter alias
        if run_data is not None and data is None:
            data = run_data

        if data is None:
            raise ValueError("Either 'data' or 'run_data' parameter must be provided")
        if traffic_volumes is None:
            raise ValueError("traffic_volumes parameter must be provided")

        logger.info(f"Processing blocking probability for {len(runs)} runs across {len(traffic_volumes)} traffic volumes")

        try:
            # Group runs by algorithm
            runs_by_algorithm = defaultdict(list)
            for run in runs:
                runs_by_algorithm[run.algorithm].append(run)

            # Aggregate data by algorithm and traffic volume
            aggregated = self._aggregate_by_algorithm(
                runs_by_algorithm,
                data,
                traffic_volumes,
            )

            # Sort traffic volumes for consistent plotting
            sorted_tvs = sorted(traffic_volumes)

            # Prepare output data
            y_data: dict[str, list[float]] = {}
            errors: dict[str, list[float]] = {} if include_ci else {}

            for algorithm, tv_dict in aggregated.items():
                y_values = []
                error_values = []

                for tv in sorted_tvs:
                    if tv in tv_dict:
                        stats = tv_dict[tv]
                        y_values.append(stats["mean"])

                        if include_ci:
                            # Use 95% confidence interval
                            error_values.append(stats["ci"])
                    else:
                        # Missing data point
                        y_values.append(float(np.nan))
                        if include_ci:
                            error_values.append(0.0)

                y_data[algorithm] = [float(v) for v in y_values]

                if include_ci and errors is not None:
                    errors[algorithm] = error_values

            logger.info(f"Successfully processed data for {len(y_data)} algorithms")

            return ProcessedData(
                x_data=[float(tv) for tv in sorted_tvs],
                y_data=y_data,
                errors=errors,
                metadata={
                    "metric": metric_name,
                    "num_runs_by_algorithm": {algo: len(runs) for algo, runs in runs_by_algorithm.items()},
                },
            )

        except Exception as e:
            logger.exception(f"Error processing blocking probability: {e}")
            raise ProcessingError(f"Failed to process blocking probability: {e}") from e

    def _aggregate_by_algorithm(
        self,
        runs_by_algorithm: dict[str, list[Run]],
        data: dict[str, dict[float, CanonicalData]],
        traffic_volumes: list[float],
    ) -> dict[str, dict[float, dict[str, float]]]:
        """
        Aggregate data by algorithm and traffic volume.

        Returns:
            Nested dictionary: algorithm -> traffic_volume -> statistics
        """
        aggregated = {}

        for algorithm, runs in runs_by_algorithm.items():
            tv_stats = {}

            for tv in traffic_volumes:
                # Collect values for this traffic volume across all runs
                values = []

                for run in runs:
                    if run.id in data and tv in data[run.id]:
                        canonical_data = data[run.id][tv]
                        # Extract blocking probability
                        if hasattr(canonical_data, "blocking_probability"):
                            bp = canonical_data.blocking_probability
                            if bp is not None:
                                values.append(bp)

                # Compute statistics if we have data
                if values:
                    values_array = np.array(values)

                    mean = float(np.mean(values_array))
                    std = float(np.std(values_array, ddof=1)) if len(values) > 1 else 0.0

                    # 95% confidence interval (1.96 * SEM)
                    sem = std / np.sqrt(len(values)) if len(values) > 0 else 0.0
                    ci = 1.96 * sem

                    tv_stats[tv] = {
                        "mean": mean,
                        "std": std,
                        "sem": sem,
                        "ci": ci,
                        "n": len(values),
                        "min": float(np.min(values_array)),
                        "max": float(np.max(values_array)),
                    }
                else:
                    logger.warning(f"No data for algorithm {algorithm} at traffic volume {tv}")

            aggregated[algorithm] = tv_stats

        return aggregated
