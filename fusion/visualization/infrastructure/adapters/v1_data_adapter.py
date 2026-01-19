"""V1 data adapter for legacy format."""

import logging
from typing import Any

from fusion.visualization.domain.exceptions.domain_exceptions import (
    UnsupportedDataFormatError,
)
from fusion.visualization.domain.value_objects.data_version import DataVersion
from fusion.visualization.infrastructure.adapters.canonical_data import (
    CanonicalData,
    IterationData,
)
from fusion.visualization.infrastructure.adapters.data_adapter import DataAdapter

logger = logging.getLogger(__name__)


class V1DataAdapter(DataAdapter):
    """
    Adapter for legacy output format (V1).

    Format characteristics:
    - `blocking_mean` at root level
    - `iter_stats` is dict of dicts (keyed by iteration number as string)
    - Timestamps use underscores
    - No explicit version field
    """

    @property
    def version(self) -> DataVersion:
        """Return version identifier."""
        return DataVersion.from_string("v1")

    def can_handle(self, data: dict[str, Any]) -> bool:
        """
        Check if this adapter can handle the data format.

        V1 format has 'blocking_mean' at root and 'iter_stats' as a dict.
        """
        # Check for V1 indicators
        has_blocking_mean = "blocking_mean" in data
        has_iter_stats = "iter_stats" in data

        if not (has_blocking_mean or has_iter_stats):
            return False

        # If iter_stats exists, check that it's a dict (not a list, which would be V2)
        if has_iter_stats:
            is_dict = isinstance(data["iter_stats"], dict)
            return is_dict

        return True

    def to_canonical(self, raw_data: dict[str, Any]) -> CanonicalData:
        """
        Convert V1 format to canonical format.

        Args:
            raw_data: Raw V1 simulation data

        Returns:
            Data in canonical format

        Raises:
            UnsupportedDataFormatError: If data format is not V1
        """
        if not self.can_handle(raw_data):
            raise UnsupportedDataFormatError("Data does not match V1 format")

        # Extract basic fields
        blocking_probability = raw_data.get("blocking_mean")

        # Parse iterations
        iterations = []
        iter_stats = raw_data.get("iter_stats", {})

        if isinstance(iter_stats, dict):
            # V1 format: iter_stats is a dict with string keys
            for iter_key, iter_data in sorted(iter_stats.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 0):
                iteration = IterationData(
                    iteration=int(iter_key) if iter_key.isdigit() else 0,
                    sim_block_list=iter_data.get("sim_block_list"),
                    hops_mean=iter_data.get("hops_mean"),
                    hops_list=iter_data.get("hops_list"),
                    lengths_mean=iter_data.get("lengths_mean"),
                    lengths_list=iter_data.get("lengths_list"),
                    snapshots_dict=iter_data.get("snapshots_dict"),
                    mods_used_dict=iter_data.get("mods_used_dict"),
                    metadata={},
                )
                iterations.append(iteration)

        # Extract metadata
        metadata = {}
        if "sim_end_time" in raw_data:
            metadata["sim_end_time"] = raw_data["sim_end_time"]
        if "sim_start_time" in raw_data:
            metadata["sim_start_time"] = raw_data["sim_start_time"]

        # Extract RL-specific fields if present
        rewards = raw_data.get("rewards")
        td_errors = raw_data.get("td_errors")
        episode_lengths = raw_data.get("episode_lengths")

        return CanonicalData(
            version=str(self.version),
            blocking_probability=blocking_probability,
            iterations=iterations,
            rewards=rewards,
            td_errors=td_errors,
            episode_lengths=episode_lengths,
            sim_start_time=raw_data.get("sim_start_time"),
            sim_end_time=raw_data.get("sim_end_time"),
            metadata=metadata,
        )

    def validate_data(self, data: dict[str, Any]) -> bool:
        """
        Validate V1 data structure.

        Args:
            data: Data to validate

        Returns:
            True if valid, False otherwise
        """
        # Check required fields
        if "blocking_mean" not in data and "iter_stats" not in data:
            logger.warning("V1 data missing both 'blocking_mean' and 'iter_stats'")
            return False

        # Validate iter_stats structure if present
        if "iter_stats" in data:
            iter_stats = data["iter_stats"]
            if not isinstance(iter_stats, dict):
                logger.warning("V1 'iter_stats' should be a dict")
                return False

        return True

    def __repr__(self) -> str:
        """Return representation."""
        return f"V1DataAdapter(version={self.version})"
