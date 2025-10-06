"""Run entity representing a single simulation execution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Run:
    """
    Represents a single execution of a simulation.

    A run is identified by a unique combination of network, date, and
    timestamp/run_id, and contains metadata about the algorithm and
    configuration used.
    """

    id: str  # Unique identifier (e.g., timestamp or run_id)
    network: str  # Network name (e.g., "NSFNet", "USNet")
    date: str  # Date string (e.g., "0606")
    algorithm: str  # Algorithm name (e.g., "ppo_obs_7")
    path: Path  # Path to run data directory
    metadata: dict[str, Any] | None = None  # Additional metadata

    def __post_init__(self) -> None:
        """Initialize default values."""
        if self.metadata is None:
            object.__setattr__(self, "metadata", {})
        if not isinstance(self.path, Path):
            object.__setattr__(self, "path", Path(self.path))

    @property
    def full_id(self) -> str:
        """Return fully qualified run identifier."""
        return f"{self.network}/{self.date}/{self.id}"

    def get_data_file(self, traffic_volume: float, extension: str = "json") -> Path:
        """
        Get path to data file for a specific traffic volume.

        Args:
            traffic_volume: Traffic volume (Erlang)
            extension: File extension (default: "json")

        Returns:
            Path to the data file
        """
        # Handle both integer and float traffic volumes
        if traffic_volume == int(traffic_volume):
            filename = f"{int(traffic_volume)}_erlang.{extension}"
        else:
            filename = f"{traffic_volume}_erlang.{extension}"

        return self.path / filename

    def exists(self) -> bool:
        """Check if run directory exists."""
        return self.path.exists() and self.path.is_dir()

    def __hash__(self) -> int:
        """Make Run hashable for use in sets and dicts."""
        return hash((self.network, self.date, self.id, self.algorithm))

    def __eq__(self, other: object) -> bool:
        """Compare runs for equality."""
        if not isinstance(other, Run):
            return NotImplemented
        return (
            self.network == other.network
            and self.date == other.date
            and self.id == other.id
            and self.algorithm == other.algorithm
        )

    def __repr__(self) -> str:
        """Return detailed representation."""
        return (
            f"Run(id='{self.id}', "
            f"network='{self.network}', "
            f"date='{self.date}', "
            f"algorithm='{self.algorithm}')"
        )
