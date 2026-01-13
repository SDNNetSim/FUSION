"""RL-specific data processing strategies.

This module provides processing strategies for RL-specific metrics:
- Reward smoothing and aggregation
- Q-value analysis
- Convergence detection
- Learning curve generation
"""

import numpy as np
from numpy import ndarray
from scipy.ndimage import uniform_filter1d

from fusion.visualization.application.ports.data_processor_port import (
    DataProcessorPort,
    ProcessedData,
)
from fusion.visualization.domain.entities.run import Run
from fusion.visualization.infrastructure.adapters.canonical_data import CanonicalData


class RewardProcessingStrategy(DataProcessorPort):
    """Process episode rewards with smoothing and statistics."""

    def __init__(self, window_size: int = 100, confidence_level: float = 0.95):
        """Initialize reward processor.

        Args:
            window_size: Size of smoothing window
            confidence_level: Confidence level for intervals
        """
        self.window_size = window_size
        self.confidence_level = confidence_level

    def can_process(self, metric_name: str) -> bool:
        """Check if this processor can handle the metric."""
        return metric_name in ["reward", "episode_reward", "training_reward"]

    def get_supported_metrics(self) -> list[str]:
        """Get list of supported metrics."""
        return ["reward", "episode_reward", "training_reward"]

    def process(
        self,
        runs: list[Run],
        data: dict[str, dict[float, CanonicalData]],
        metric_name: str,
        traffic_volumes: list[float],
        include_ci: bool = True,
    ) -> ProcessedData:
        """Process reward data with smoothing.

        Args:
            runs: List of simulation runs
            data: Nested dictionary of canonical data
            metric_name: Name of metric to extract and process
            traffic_volumes: List of traffic volumes to include
            include_ci: Whether to include confidence intervals

        Returns:
            ProcessedData with smoothed rewards and statistics
        """
        # Extract x_data (traffic volumes) and y_data (algorithm -> values)
        x_data = traffic_volumes
        y_data: dict[str, list[float]] = {}
        errors: dict[str, list[float]] = {} if include_ci else {}

        # Group by algorithm
        for run in runs:
            algo = run.algorithm
            if algo not in y_data:
                y_data[algo] = []
                if include_ci:
                    errors[algo] = []

            for volume in traffic_volumes:
                run_data = data.get(run.id, {}).get(volume)
                if run_data and hasattr(run_data, "training"):
                    rewards = self._extract_rewards(run_data)
                    if rewards is not None:
                        y_data[algo].append(float(np.mean(rewards)))
                        if include_ci:
                            errors[algo].append(float(np.std(rewards, ddof=1)))

        return ProcessedData(
            x_data=x_data,
            y_data=y_data,
            errors=errors if include_ci and errors is not None else None,
            metadata={
                "window_size": self.window_size,
                "confidence_level": self.confidence_level,
                "aggregation": "mean_with_smoothing",
            },
        )

    def _extract_rewards(self, data: CanonicalData) -> ndarray | None:
        """Extract reward sequence from data."""
        if hasattr(data, "training") and "episode_rewards" in data.training:
            rewards: ndarray = np.array(data.training["episode_rewards"])
            return rewards
        return None

    def _align_sequences(self, sequences: list[np.ndarray]) -> np.ndarray:
        """Align sequences to common length (truncate to shortest).

        Args:
            sequences: List of arrays to align

        Returns:
            2D array with aligned sequences
        """
        if not sequences:
            return np.array([])

        min_length = min(len(seq) for seq in sequences)
        return np.array([seq[:min_length] for seq in sequences])

    def _smooth(self, data: np.ndarray) -> np.ndarray:
        """Apply moving average smoothing.

        Args:
            data: Array to smooth

        Returns:
            Smoothed array
        """
        if len(data) < self.window_size:
            return data

        result: np.ndarray = uniform_filter1d(
            data, size=self.window_size, mode="nearest"
        )
        return result


class QValueProcessingStrategy(DataProcessorPort):
    """Process Q-value data for heatmap visualization."""

    def can_process(self, metric_name: str) -> bool:
        """Check if this processor can handle the metric."""
        return metric_name in ["q_value", "q_values", "action_values"]

    def get_supported_metrics(self) -> list[str]:
        """Get list of supported metrics."""
        return ["q_value", "q_values", "action_values"]

    def process(
        self,
        runs: list[Run],
        data: dict[str, dict[float, CanonicalData]],
        metric_name: str,
        traffic_volumes: list[float],
        include_ci: bool = True,
    ) -> ProcessedData:
        """Process Q-value data.

        Args:
            runs: List of simulation runs
            data: Nested dictionary of canonical data
            metric_name: Name of metric to extract and process
            traffic_volumes: List of traffic volumes to include
            include_ci: Whether to include confidence intervals

        Returns:
            ProcessedData with Q-value statistics
        """
        # Extract x_data (traffic volumes) and y_data (algorithm -> values)
        x_data = traffic_volumes
        y_data: dict[str, list[float]] = {}
        errors: dict[str, list[float]] = {} if include_ci else {}

        # Group by algorithm
        for run in runs:
            algo = run.algorithm
            if algo not in y_data:
                y_data[algo] = []
                if include_ci:
                    errors[algo] = []

            for volume in traffic_volumes:
                run_data = data.get(run.id, {}).get(volume)
                if run_data and hasattr(run_data, "training"):
                    q_values = self._extract_q_values(run_data)
                    if q_values is not None:
                        mean_q = float(np.mean(q_values))
                        y_data[algo].append(mean_q)
                        if include_ci:
                            errors[algo].append(float(np.std(q_values, ddof=1)))

        return ProcessedData(
            x_data=x_data,
            y_data=y_data,
            errors=errors if include_ci and errors is not None else None,
            metadata={"aggregation": "mean_over_actions_and_seeds"},
        )

    def _extract_q_values(self, data: CanonicalData) -> ndarray | None:
        """Extract Q-values from data."""
        if hasattr(data, "training") and "q_values" in data.training:
            return np.array(data.training["q_values"])
        return None


class ConvergenceDetectionStrategy(DataProcessorPort):
    """Detect convergence in RL training metrics."""

    def __init__(self, window_size: int = 100, threshold: float = 0.01):
        """Initialize convergence detector.

        Args:
            window_size: Window for convergence check
            threshold: Threshold for convergence (relative change)
        """
        self.window_size = window_size
        self.threshold = threshold

    def can_process(self, metric_name: str) -> bool:
        """Check if this processor can handle the metric."""
        return metric_name in ["convergence", "training_convergence"]

    def get_supported_metrics(self) -> list[str]:
        """Get list of supported metrics."""
        return ["convergence", "training_convergence"]

    def process(
        self,
        runs: list[Run],
        data: dict[str, dict[float, CanonicalData]],
        metric_name: str,
        traffic_volumes: list[float],
        include_ci: bool = True,
    ) -> ProcessedData:
        """Detect convergence in training.

        Args:
            runs: List of simulation runs
            data: Nested dictionary of canonical data
            metric_name: Name of metric to extract and process
            traffic_volumes: List of traffic volumes to include
            include_ci: Whether to include confidence intervals

        Returns:
            ProcessedData with convergence information
        """
        # Extract x_data (traffic volumes) and y_data
        # (algorithm -> convergence episodes)
        x_data = traffic_volumes
        y_data: dict[str, list[float]] = {}
        errors: dict[str, list[float]] = {} if include_ci else {}

        # Group by algorithm
        for run in runs:
            algo = run.algorithm
            if algo not in y_data:
                y_data[algo] = []
                if include_ci:
                    errors[algo] = []

            for volume in traffic_volumes:
                run_data = data.get(run.id, {}).get(volume)
                if run_data and hasattr(run_data, "training"):
                    metric_values = self._extract_metric(run_data, metric_name)
                    if metric_values is not None:
                        convergence_episode = self._detect_convergence(metric_values)
                        y_data[algo].append(
                            float(convergence_episode)
                            if convergence_episode is not None
                            else 0.0
                        )
                        if include_ci:
                            errors[algo].append(0.0)

        return ProcessedData(
            x_data=x_data,
            y_data=y_data,
            errors=errors if include_ci and errors is not None else None,
            metadata={
                "window_size": self.window_size,
                "threshold": self.threshold,
            },
        )

    def _extract_metric(self, data: CanonicalData, metric_name: str) -> ndarray | None:
        """Extract metric values from data."""
        # Simplified extraction - would use JSONPath in real implementation
        if hasattr(data, "training"):
            if metric_name in data.training:
                return np.array(data.training[metric_name])
        return None

    def _detect_convergence(self, values: ndarray) -> int | None:
        """Detect convergence episode.

        Args:
            values: Metric values over episodes

        Returns:
            Episode number where convergence occurred, or None
        """
        if len(values) < self.window_size * 2:
            return None

        for i in range(self.window_size, len(values) - self.window_size):
            window1 = values[i - self.window_size : i]
            window2 = values[i : i + self.window_size]

            mean1 = np.mean(window1)
            mean2 = np.mean(window2)

            if mean1 != 0:
                relative_change = abs(mean2 - mean1) / abs(mean1)
                if relative_change < self.threshold:
                    return i

        return None
