"""Base data adapter interface."""

from abc import ABC, abstractmethod
from typing import Any

from fusion.visualization.domain.value_objects.data_version import DataVersion
from fusion.visualization.infrastructure.adapters.canonical_data import CanonicalData


class DataAdapter(ABC):
    """
    Abstract base class for data format adapters.

    Adapters convert between different external data formats and the
    canonical internal format, enabling the system to handle format
    changes gracefully.
    """

    @property
    @abstractmethod
    def version(self) -> DataVersion:
        """Return version identifier (e.g., v1, v2)."""

    @abstractmethod
    def can_handle(self, data: dict[str, Any]) -> bool:
        """
        Check if this adapter can handle the data format.

        Args:
            data: Raw data to check

        Returns:
            True if this adapter can process the data
        """

    @abstractmethod
    def to_canonical(self, raw_data: dict[str, Any]) -> CanonicalData:
        """
        Convert raw data to canonical internal format.

        Args:
            raw_data: Raw simulation data

        Returns:
            Data in canonical format

        Raises:
            UnsupportedDataFormatError: If data format is not supported
        """

    def validate_data(self, data: dict[str, Any]) -> bool:
        """
        Validate data structure.

        Args:
            data: Data to validate

        Returns:
            True if valid, False otherwise
        """
        # Default implementation - subclasses can override
        return True

    def __repr__(self) -> str:
        """Return representation."""
        return f"{self.__class__.__name__}(version={self.version})"
