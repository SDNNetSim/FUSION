"""Canonical data model for internal representation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class IterationData:
    """Data for a single iteration."""

    iteration: int
    sim_block_list: list[float] | None = None
    hops_mean: float | None = None
    hops_list: list[float] | None = None
    lengths_mean: float | None = None
    lengths_list: list[float] | None = None
    snapshots_dict: dict[str, Any] | None = None
    mods_used_dict: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CanonicalData:
    """
    Canonical internal data format.

    This is the standardized format that all adapters convert to.
    It provides a stable interface for the rest of the system,
    insulating it from changes in external data formats.
    """

    version: str
    blocking_probability: float | None = None
    iterations: list[IterationData] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    # RL-specific fields
    rewards: list[float] | None = None
    td_errors: list[float] | None = None
    q_values: np.ndarray | None = None
    episode_lengths: list[int] | None = None

    # Spectrum-specific fields
    spectrum_usage: dict[str, Any] | None = None

    # SNR-specific fields
    snr_values: list[float] | None = None

    # Network metrics
    network_utilization: float | None = None
    link_utilization: dict[str, float] | None = None

    # Timing information
    sim_start_time: str | None = None
    sim_end_time: str | None = None
    duration_seconds: float | None = None

    def get_final_blocking_probability(self) -> float | None:
        """Get final blocking probability from iterations."""
        if self.blocking_probability is not None:
            return self.blocking_probability

        if self.iterations and self.iterations[-1].sim_block_list:
            return float(np.mean(self.iterations[-1].sim_block_list))

        return None

    def get_last_k_iterations(self, k: int) -> list[IterationData]:
        """Get last K iterations."""
        if len(self.iterations) <= k:
            return self.iterations
        return self.iterations[-k:]

    def get_metric(self, metric_path: str) -> Any:
        """
        Get a metric value using a simplified path notation.

        Args:
            metric_path: Path like "blocking", "iterations.0.hops_mean", etc.

        Returns:
            The metric value
        """
        parts = metric_path.split(".")

        if parts[0] == "blocking":
            return self.get_final_blocking_probability()
        elif parts[0] == "iterations":
            if len(parts) < 2:
                return self.iterations
            iter_idx = int(parts[1])
            if len(parts) == 2:
                return self.iterations[iter_idx]
            # Navigate deeper into iteration
            iter_data = self.iterations[iter_idx]
            attr_name = parts[2]
            return getattr(iter_data, attr_name, None)
        elif parts[0] == "rewards":
            return self.rewards
        elif parts[0] == "metadata":
            if len(parts) < 2:
                return self.metadata
            return self.metadata.get(parts[1])
        else:
            # Try to get as attribute
            return getattr(self, parts[0], None)

    def __repr__(self) -> str:
        """Return detailed representation."""
        return (
            f"CanonicalData(version={self.version}, "
            f"blocking={self.blocking_probability}, "
            f"iterations={len(self.iterations)})"
        )
