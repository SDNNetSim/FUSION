"""DataSource entity for representing the origin of simulation data."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from fusion.visualization.domain.value_objects.data_version import DataVersion

if TYPE_CHECKING:
    pass


class SourceType(Enum):
    """Types of data sources."""

    FILE = "file"
    DATABASE = "database"
    STREAM = "stream"
    MEMORY = "memory"


class DataFormat(Enum):
    """Supported data formats."""

    JSON = "json"
    NPY = "npy"
    CSV = "csv"
    PARQUET = "parquet"
    HDF5 = "hdf5"


@dataclass
class DataSource:
    """
    Represents the origin of simulation data.

    This entity encapsulates where data comes from and what format
    it's in, enabling the system to select appropriate adapters.
    """

    source_type: SourceType
    location: Path
    format: DataFormat
    version: DataVersion

    def __post_init__(self) -> None:
        """Validate data source."""
        if self.source_type == SourceType.FILE:
            if not isinstance(self.location, Path):
                object.__setattr__(self, "location", Path(self.location))

    def exists(self) -> bool:
        """Check if the data source exists."""
        if self.source_type == SourceType.FILE:
            return self.location.exists()
        # For other source types, assume they exist
        # (would need connection check for database, etc.)
        return True

    def __repr__(self) -> str:
        """Return detailed representation."""
        return f"DataSource(type={self.source_type.value}, format={self.format.value}, version={self.version}, location={self.location})"
