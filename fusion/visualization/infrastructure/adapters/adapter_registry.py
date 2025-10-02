"""Registry for managing data adapters with auto-detection."""

import logging
from typing import Any

from fusion.visualization.domain.exceptions.domain_exceptions import (
    UnsupportedDataFormatError,
)
from fusion.visualization.domain.value_objects.data_version import DataVersion
from fusion.visualization.infrastructure.adapters.data_adapter import DataAdapter
from fusion.visualization.infrastructure.adapters.v1_data_adapter import V1DataAdapter
from fusion.visualization.infrastructure.adapters.v2_data_adapter import V2DataAdapter

logger = logging.getLogger(__name__)


class DataAdapterRegistry:
    """
    Factory for selecting appropriate data adapters.

    This registry auto-detects data format and returns the correct adapter.
    Adapters are tried in reverse order (newest first) to prefer newer formats.
    """

    def __init__(self, adapters: list[DataAdapter] | None = None):
        """
        Initialize the registry.

        Args:
            adapters: Optional list of adapters. If None, uses default adapters.
        """
        if adapters is None:
            # Default adapters in priority order (newest first)
            self._adapters: list[DataAdapter] = [
                V2DataAdapter(),
                V1DataAdapter(),
            ]
        else:
            self._adapters = adapters

    def register_adapter(self, adapter: DataAdapter) -> None:
        """
        Register a new adapter.

        Args:
            adapter: Adapter to register
        """
        # Insert at beginning to give priority to newer adapters
        self._adapters.insert(0, adapter)
        logger.info(f"Registered adapter: {adapter}")

    def get_adapter(self, data: dict[str, Any]) -> DataAdapter:
        """
        Auto-detect and return appropriate adapter for the data.

        Args:
            data: Raw data to analyze

        Returns:
            Adapter capable of handling the data

        Raises:
            UnsupportedDataFormatError: If no adapter can handle the data
        """
        for adapter in self._adapters:
            if adapter.can_handle(data):
                logger.debug(f"Selected adapter: {adapter} for data")
                return adapter

        # No adapter found
        supported_versions = ", ".join(str(a.version) for a in self._adapters)
        raise UnsupportedDataFormatError(
            f"No adapter found for data format. "
            f"Supported versions: {supported_versions}"
        )

    def get_adapter_by_version(self, version: DataVersion) -> DataAdapter | None:
        """
        Get adapter by specific version.

        Args:
            version: Version to look for

        Returns:
            Adapter for the version, or None if not found
        """
        for adapter in self._adapters:
            if adapter.version == version:
                return adapter
        return None

    def get_supported_versions(self) -> list[DataVersion]:
        """
        Get list of all supported versions.

        Returns:
            List of supported data versions
        """
        return [adapter.version for adapter in self._adapters]

    def clear(self) -> None:
        """Clear all registered adapters."""
        self._adapters.clear()

    def __repr__(self) -> str:
        """Return representation."""
        versions = ", ".join(str(a.version) for a in self._adapters)
        return f"DataAdapterRegistry(adapters=[{versions}])"


# Global registry instance
_default_registry = DataAdapterRegistry()


def get_default_registry() -> DataAdapterRegistry:
    """
    Get the default global adapter registry.

    Returns:
        Default adapter registry
    """
    return _default_registry


def register_adapter(adapter: DataAdapter) -> None:
    """
    Register an adapter with the default registry.

    Args:
        adapter: Adapter to register
    """
    _default_registry.register_adapter(adapter)


def get_adapter(data: dict[str, Any]) -> DataAdapter:
    """
    Get appropriate adapter from the default registry.

    Args:
        data: Raw data to analyze

    Returns:
        Adapter capable of handling the data
    """
    return _default_registry.get_adapter(data)
