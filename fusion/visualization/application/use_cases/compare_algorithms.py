"""Use case for comparing algorithms statistically."""

from __future__ import annotations
import logging
from datetime import datetime
from typing import List, Dict
from pathlib import Path
import numpy as np
from scipy import stats

from fusion.visualization.application.dto import (
    ComparisonRequestDTO,
    ComparisonResultDTO,
    StatisticalComparison,
)
from fusion.visualization.application.ports import CachePort
from fusion.visualization.domain.repositories import SimulationRepository
from fusion.visualization.domain.exceptions import RepositoryError, ValidationError
from fusion.visualization.infrastructure.adapters.canonical_data import CanonicalData

logger = logging.getLogger(__name__)


class CompareAlgorithmsUseCase:
    """
    Use case for statistical comparison of algorithms.

    This performs statistical tests (t-tests, effect sizes) to compare
    algorithm performance and generates comparison reports/plots.
    """

    def __init__(
        self,
        simulation_repository: SimulationRepository,
        cache: CachePort | None = None,
    ):
        """
        Initialize comparison use case.

        Args:
            simulation_repository: Repository for accessing simulation data
            cache: Optional cache for performance
        """
        self.simulation_repository = simulation_repository
        self.cache = cache

    def execute(self, request: ComparisonRequestDTO) -> ComparisonResultDTO:
        """
        Execute algorithm comparison.

        Args:
            request: Comparison request DTO

        Returns:
            ComparisonResultDTO with statistical comparison results
        """
        started_at = datetime.now()

        logger.info(
            f"Starting comparison of {len(request.algorithms)} algorithms "
            f"on metric '{request.metric}'"
        )

        try:
            # Validate request
            errors = request.validate()
            if errors:
                error_msg = f"Invalid request: {'; '.join(errors)}"
                logger.error(error_msg)
                return ComparisonResultDTO(
                    network=request.network,
                    dates=request.dates,
                    algorithms=request.algorithms,
                    metric=request.metric,
                    comparisons=[],
                    success=False,
                    error=error_msg,
                    started_at=started_at,
                    completed_at=datetime.now(),
                    duration=datetime.now() - started_at,
                )

            # Load data for all algorithms
            algorithm_data = self._load_algorithm_data(request)

            # Perform pairwise comparisons
            comparisons = self._perform_comparisons(
                algorithm_data,
                request.metric,
                request.include_statistical_tests,
                request.include_effect_sizes,
                request.confidence_level,
            )

            completed_at = datetime.now()

            logger.info(
                f"Completed {len(comparisons)} pairwise comparisons "
                f"in {(completed_at - started_at).total_seconds():.2f}s"
            )

            return ComparisonResultDTO(
                network=request.network,
                dates=request.dates,
                algorithms=request.algorithms,
                metric=request.metric,
                comparisons=comparisons,
                output_path=request.save_path,
                success=True,
                started_at=started_at,
                completed_at=completed_at,
                duration=completed_at - started_at,
            )

        except Exception as e:
            logger.exception(f"Error in comparison: {e}")
            return ComparisonResultDTO(
                network=request.network,
                dates=request.dates,
                algorithms=request.algorithms,
                metric=request.metric,
                comparisons=[],
                success=False,
                error=str(e),
                started_at=started_at,
                completed_at=datetime.now(),
                duration=datetime.now() - started_at,
            )

    def _load_algorithm_data(
        self, request: ComparisonRequestDTO
    ) -> Dict[str, List[float]]:
        """
        Load metric data for all algorithms.

        Returns:
            Dictionary mapping algorithm names to lists of metric values
        """
        algorithm_data: Dict[str, List[float]] = {algo: [] for algo in request.algorithms}

        for algorithm in request.algorithms:
            # Find runs for this algorithm
            runs = self.simulation_repository.find_runs(
                network=request.network,
                dates=request.dates,
                algorithm=algorithm,
            )

            if not runs:
                logger.warning(
                    f"No runs found for algorithm {algorithm} "
                    f"(network={request.network}, dates={request.dates})"
                )
                continue

            # Get traffic volumes
            traffic_volumes = request.traffic_volumes
            if not traffic_volumes:
                # Use traffic volumes from first run
                traffic_volumes = self.simulation_repository.get_available_traffic_volumes(
                    runs[0]
                )

            # Load metric values for each run and traffic volume
            for run in runs:
                for tv in traffic_volumes:
                    try:
                        data = self.simulation_repository.get_run_data(run, tv)
                        value = self._extract_metric(data, request.metric)
                        if value is not None:
                            algorithm_data[algorithm].append(value)

                    except Exception as e:
                        logger.warning(
                            f"Failed to load data for {algorithm} run {run.id} "
                            f"at {tv} Erlang: {e}"
                        )

        # Validate we have data
        empty_algorithms = [
            algo for algo, values in algorithm_data.items() if not values
        ]
        if empty_algorithms:
            raise RepositoryError(
                f"No data found for algorithms: {', '.join(empty_algorithms)}"
            )

        return algorithm_data

    def _extract_metric(self, data: CanonicalData, metric_name: str) -> float | None:
        """Extract metric value from canonical data."""
        # Map common metric names to canonical data fields
        metric_map = {
            "blocking_probability": lambda d: d.blocking_probability,
            "blocking": lambda d: d.blocking_probability,
            # Add more metric mappings as needed
        }

        extractor = metric_map.get(metric_name)
        if extractor:
            result = extractor(data)
            return float(result) if result is not None else None

        # Try to get from metadata
        if hasattr(data, "metadata") and metric_name in data.metadata:
            metadata_value = data.metadata[metric_name]
            return float(metadata_value) if metadata_value is not None else None

        logger.warning(f"Unknown metric: {metric_name}")
        return None

    def _perform_comparisons(
        self,
        algorithm_data: Dict[str, List[float]],
        metric: str,
        include_statistical_tests: bool,
        include_effect_sizes: bool,
        confidence_level: float,
    ) -> List[StatisticalComparison]:
        """Perform pairwise statistical comparisons."""
        comparisons = []
        algorithms = list(algorithm_data.keys())

        # Pairwise comparisons
        for i in range(len(algorithms)):
            for j in range(i + 1, len(algorithms)):
                algo_a = algorithms[i]
                algo_b = algorithms[j]

                data_a = np.array(algorithm_data[algo_a])
                data_b = np.array(algorithm_data[algo_b])

                comparison = self._compare_pair(
                    algo_a,
                    algo_b,
                    data_a,
                    data_b,
                    metric,
                    include_statistical_tests,
                    include_effect_sizes,
                    confidence_level,
                )

                comparisons.append(comparison)

        return comparisons

    def _compare_pair(
        self,
        algo_a: str,
        algo_b: str,
        data_a: np.ndarray,
        data_b: np.ndarray,
        metric: str,
        include_statistical_tests: bool,
        include_effect_sizes: bool,
        confidence_level: float,
    ) -> StatisticalComparison:
        """Compare two algorithms statistically."""
        # Descriptive statistics
        mean_a = float(np.mean(data_a))
        mean_b = float(np.mean(data_b))
        std_a = float(np.std(data_a, ddof=1))
        std_b = float(np.std(data_b, ddof=1))

        # Confidence intervals
        ci_a = stats.t.interval(
            confidence_level,
            len(data_a) - 1,
            loc=mean_a,
            scale=stats.sem(data_a),
        )
        ci_b = stats.t.interval(
            confidence_level,
            len(data_b) - 1,
            loc=mean_b,
            scale=stats.sem(data_b),
        )

        comparison = StatisticalComparison(
            algorithm_a=algo_a,
            algorithm_b=algo_b,
            metric=metric,
            mean_a=mean_a,
            mean_b=mean_b,
            std_a=std_a,
            std_b=std_b,
            ci_lower_a=float(ci_a[0]),
            ci_upper_a=float(ci_a[1]),
            ci_lower_b=float(ci_b[0]),
            ci_upper_b=float(ci_b[1]),
        )

        # Statistical test
        if include_statistical_tests:
            t_stat, p_value = stats.ttest_ind(data_a, data_b)
            comparison.test_statistic = float(t_stat)
            comparison.p_value = float(p_value)
            comparison.test_name = "Independent t-test"

        # Effect size (Cohen's d)
        if include_effect_sizes:
            pooled_std = np.sqrt(
                ((len(data_a) - 1) * std_a**2 + (len(data_b) - 1) * std_b**2)
                / (len(data_a) + len(data_b) - 2)
            )
            if pooled_std > 0:
                cohens_d = (mean_b - mean_a) / pooled_std
                comparison.cohens_d = float(cohens_d)

                # Interpret effect size
                abs_d = abs(cohens_d)
                if abs_d < 0.2:
                    interpretation = "negligible"
                elif abs_d < 0.5:
                    interpretation = "small"
                elif abs_d < 0.8:
                    interpretation = "medium"
                else:
                    interpretation = "large"

                comparison.effect_size_interpretation = interpretation

        return comparison
