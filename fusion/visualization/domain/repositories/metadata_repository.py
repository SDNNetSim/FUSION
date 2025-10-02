"""Abstract repository interface for metadata access and caching."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pathlib import Path


class MetadataRepository(ABC):
    """
    Abstract repository for run metadata access.

    This repository is optimized for fast run discovery and filtering,
    typically used before loading full simulation data.
    """

    @abstractmethod
    def discover_runs(
        self,
        base_path: Path,
        network: str,
        dates: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Discover all runs in the given network/dates.

        Args:
            base_path: Root directory for data
            network: Network name
            dates: List of date strings

        Returns:
            List of metadata dictionaries, each containing:
                - id: Run identifier
                - network: Network name
                - date: Date string
                - algorithm: Algorithm name
                - path: Path to run directory
                - Additional metadata fields

        Raises:
            RepositoryError: If discovery fails
        """
        pass

    @abstractmethod
    def get_run_metadata(self, run_path: Path) -> Dict[str, Any]:
        """
        Load metadata for a specific run.

        Args:
            run_path: Path to run directory

        Returns:
            Dictionary of metadata

        Raises:
            MetadataNotFoundError: If metadata file doesn't exist
            RepositoryError: If metadata cannot be loaded
        """
        pass

    @abstractmethod
    def cache_metadata(
        self,
        run_path: Path,
        metadata: Dict[str, Any],
    ) -> None:
        """
        Cache metadata for faster subsequent access.

        Args:
            run_path: Path to run directory
            metadata: Metadata to cache
        """
        pass

    @abstractmethod
    def clear_cache(self) -> None:
        """Clear all cached metadata."""
        pass

    @abstractmethod
    def get_cached_metadata(
        self,
        run_path: Path,
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached metadata if available.

        Args:
            run_path: Path to run directory

        Returns:
            Cached metadata or None if not cached
        """
        pass
