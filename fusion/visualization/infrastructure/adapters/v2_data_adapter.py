"""V2 data adapter for future format."""

from typing import Dict, Any
import logging

from fusion.visualization.infrastructure.adapters.data_adapter import DataAdapter
from fusion.visualization.infrastructure.adapters.canonical_data import (
    CanonicalData,
    IterationData,
)
from fusion.visualization.domain.value_objects.data_version import DataVersion
from fusion.visualization.domain.exceptions.domain_exceptions import (
    UnsupportedDataFormatError,
)

logger = logging.getLogger(__name__)


class V2DataAdapter(DataAdapter):
    """
    Adapter for future output format (V2).

    Format changes from V1:
    - Explicit 'version' field
    - `blocking_mean` → `metrics.blocking_probability`
    - `iter_stats` → `iterations` (list instead of dict)
    - ISO 8601 timestamps
    - More structured metadata
    """

    @property
    def version(self) -> DataVersion:
        """Return version identifier."""
        return DataVersion.from_string("v2")

    def can_handle(self, data: Dict[str, Any]) -> bool:
        """
        Check if this adapter can handle the data format.

        V2 format has 'metrics' at root level and 'iterations' as a list.
        """
        # Check for explicit version field
        if "version" in data:
            try:
                data_version = DataVersion.from_string(data["version"])
                return data_version.major == 2
            except (ValueError, KeyError):
                pass

        # Check for V2 structure indicators
        has_metrics = "metrics" in data
        has_iterations_list = "iterations" in data and isinstance(
            data["iterations"], list
        )

        return has_metrics or has_iterations_list

    def to_canonical(self, raw_data: Dict[str, Any]) -> CanonicalData:
        """
        Convert V2 format to canonical format.

        Args:
            raw_data: Raw V2 simulation data

        Returns:
            Data in canonical format

        Raises:
            UnsupportedDataFormatError: If data format is not V2
        """
        if not self.can_handle(raw_data):
            raise UnsupportedDataFormatError(
                "Data does not match V2 format"
            )

        # Extract metrics from structured metrics object
        metrics = raw_data.get("metrics", {})
        blocking_probability = metrics.get(
            "blocking_probability",
            raw_data.get("blocking_mean")  # Fallback to V1 field
        )

        # Parse iterations (list format in V2)
        iterations = []
        iter_data_list = raw_data.get("iterations", [])

        if isinstance(iter_data_list, list):
            for idx, iter_data in enumerate(iter_data_list):
                iteration = IterationData(
                    iteration=iter_data.get("iteration", idx),
                    sim_block_list=iter_data.get("sim_block_list"),
                    hops_mean=iter_data.get("hops_mean"),
                    hops_list=iter_data.get("hops_list"),
                    lengths_mean=iter_data.get("lengths_mean"),
                    lengths_list=iter_data.get("lengths_list"),
                    snapshots_dict=iter_data.get("snapshots_dict"),
                    mods_used_dict=iter_data.get("mods_used_dict"),
                    metadata=iter_data.get("metadata", {})
                )
                iterations.append(iteration)

        # Extract metadata (more structured in V2)
        metadata = raw_data.get("metadata", {})

        # Extract RL-specific fields
        rl_data = raw_data.get("reinforcement_learning", {})
        rewards = rl_data.get("rewards", raw_data.get("rewards"))
        td_errors = rl_data.get("td_errors", raw_data.get("td_errors"))
        episode_lengths = rl_data.get("episode_lengths", raw_data.get("episode_lengths"))
        q_values = rl_data.get("q_values")

        # Extract timing information
        timing = raw_data.get("timing", {})
        sim_start_time = timing.get(
            "start_time",
            metadata.get("sim_start_time", raw_data.get("sim_start_time"))
        )
        sim_end_time = timing.get(
            "end_time",
            metadata.get("sim_end_time", raw_data.get("sim_end_time"))
        )
        duration_seconds = timing.get("duration_seconds")

        # Extract network metrics
        network_metrics = raw_data.get("network", {})
        network_utilization = network_metrics.get("utilization")
        link_utilization = network_metrics.get("link_utilization")

        return CanonicalData(
            version=str(self.version),
            blocking_probability=blocking_probability,
            iterations=iterations,
            rewards=rewards,
            td_errors=td_errors,
            q_values=q_values,
            episode_lengths=episode_lengths,
            network_utilization=network_utilization,
            link_utilization=link_utilization,
            sim_start_time=sim_start_time,
            sim_end_time=sim_end_time,
            duration_seconds=duration_seconds,
            metadata=metadata,
        )

    def validate_data(self, data: Dict[str, Any]) -> bool:
        """
        Validate V2 data structure.

        Args:
            data: Data to validate

        Returns:
            True if valid, False otherwise
        """
        # Check for metrics or iterations
        if "metrics" not in data and "iterations" not in data:
            logger.warning("V2 data missing both 'metrics' and 'iterations'")
            return False

        # Validate iterations structure if present
        if "iterations" in data:
            iterations = data["iterations"]
            if not isinstance(iterations, list):
                logger.warning("V2 'iterations' should be a list")
                return False

        return True

    def __repr__(self) -> str:
        """Return representation."""
        return f"V2DataAdapter(version={self.version})"
