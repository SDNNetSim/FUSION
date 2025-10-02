"""Factory for creating repository instances."""

from __future__ import annotations

import logging
from pathlib import Path

from fusion.visualization.domain.repositories.metadata_repository import (
    MetadataRepository,
)
from fusion.visualization.domain.repositories.simulation_repository import (
    SimulationRepository,
)
from fusion.visualization.infrastructure.repositories.file_metadata_repository import (  # noqa: E501
    FileMetadataRepository,
)
from fusion.visualization.infrastructure.repositories.json_simulation_repository import (  # noqa: E501
    JsonSimulationRepository,
)

logger = logging.getLogger(__name__)


class RepositoryFactory:
    """
    Factory for creating repository instances.

    This factory encapsulates the creation logic for repositories,
    allowing for easy configuration and dependency injection.
    """

    def __init__(
        self,
        base_path: Path | None = None,
        cache_ttl_seconds: int = 3600,
        enable_cache: bool = True,
    ):
        """
        Initialize repository factory.

        Args:
            base_path: Root directory for data (defaults to ../../data/output)
            cache_ttl_seconds: Time-to-live for cache entries
            enable_cache: Whether to enable caching
        """
        if base_path is None:
            # Default to standard FUSION data directory
            base_path = (
                Path(__file__).parent.parent.parent.parent.parent / "data" / "output"
            )

        self.base_path = Path(base_path)
        self.cache_ttl_seconds = cache_ttl_seconds
        self.enable_cache = enable_cache

        # Singleton instances
        self._metadata_repository: MetadataRepository | None = None
        self._simulation_repository: SimulationRepository | None = None

        logger.info(
            f"RepositoryFactory initialized with base_path={self.base_path}, "
            f"cache_enabled={self.enable_cache}"
        )

    def create_metadata_repository(self) -> MetadataRepository:
        """
        Create or return cached metadata repository.

        Returns:
            MetadataRepository instance
        """
        if self._metadata_repository is None:
            self._metadata_repository = FileMetadataRepository(
                cache_ttl_seconds=self.cache_ttl_seconds,
                enable_cache=self.enable_cache,
            )
            logger.debug("Created FileMetadataRepository")

        return self._metadata_repository

    def create_simulation_repository(self) -> SimulationRepository:
        """
        Create or return cached simulation repository.

        Returns:
            SimulationRepository instance
        """
        if self._simulation_repository is None:
            # Create with metadata repository for optimized discovery
            metadata_repo = self.create_metadata_repository()

            self._simulation_repository = JsonSimulationRepository(
                base_path=self.base_path,
                metadata_repository=metadata_repo,
            )
            logger.debug(
                f"Created JsonSimulationRepository with base_path={self.base_path}"
            )

        return self._simulation_repository

    def configure(
        self,
        base_path: Path | None = None,
        cache_ttl_seconds: int | None = None,
        enable_cache: bool | None = None,
    ) -> None:
        """
        Reconfigure the factory (clears cached instances).

        Args:
            base_path: New root directory for data
            cache_ttl_seconds: New TTL for cache
            enable_cache: New cache enable setting
        """
        if base_path is not None:
            self.base_path = Path(base_path)

        if cache_ttl_seconds is not None:
            self.cache_ttl_seconds = cache_ttl_seconds

        if enable_cache is not None:
            self.enable_cache = enable_cache

        # Clear cached instances to pick up new config
        self._metadata_repository = None
        self._simulation_repository = None

        logger.info(
            f"RepositoryFactory reconfigured with base_path={self.base_path}, "
            f"cache_ttl={self.cache_ttl_seconds}s, cache_enabled={self.enable_cache}"
        )

    def clear_all_caches(self) -> None:
        """Clear all repository caches."""
        if self._metadata_repository is not None:
            self._metadata_repository.clear_cache()

        if self._simulation_repository is not None and hasattr(
            self._simulation_repository, "clear_cache"
        ):
            self._simulation_repository.clear_cache()

        logger.info("All repository caches cleared")

    def get_cache_stats(self) -> dict:
        """Get statistics from all repository caches."""
        stats = {}

        if self._metadata_repository is not None and hasattr(
            self._metadata_repository, "get_cache_stats"
        ):
            stats["metadata"] = self._metadata_repository.get_cache_stats()

        if self._simulation_repository is not None and hasattr(
            self._simulation_repository, "get_cache_stats"
        ):
            stats["simulation"] = self._simulation_repository.get_cache_stats()

        return stats


# Global default factory instance
_default_factory: RepositoryFactory | None = None


def get_default_factory() -> RepositoryFactory:
    """
    Get the default repository factory instance.

    Returns:
        Global RepositoryFactory instance
    """
    global _default_factory
    if _default_factory is None:
        _default_factory = RepositoryFactory()
    return _default_factory


def configure_default_factory(
    base_path: Path | None = None,
    cache_ttl_seconds: int | None = None,
    enable_cache: bool | None = None,
) -> None:
    """
    Configure the default repository factory.

    Args:
        base_path: Root directory for data
        cache_ttl_seconds: Time-to-live for cache entries
        enable_cache: Whether to enable caching
    """
    factory = get_default_factory()
    factory.configure(
        base_path=base_path,
        cache_ttl_seconds=cache_ttl_seconds,
        enable_cache=enable_cache,
    )


def get_simulation_repository() -> SimulationRepository:
    """
    Get a simulation repository from the default factory.

    Returns:
        SimulationRepository instance
    """
    return get_default_factory().create_simulation_repository()


def get_metadata_repository() -> MetadataRepository:
    """
    Get a metadata repository from the default factory.

    Returns:
        MetadataRepository instance
    """
    return get_default_factory().create_metadata_repository()
