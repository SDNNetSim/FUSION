"""File-based metadata repository with caching."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

from fusion.visualization.domain.exceptions.domain_exceptions import (
    RepositoryError,
)
from fusion.visualization.domain.repositories.metadata_repository import (
    MetadataRepository,
)

logger = logging.getLogger(__name__)


class FileMetadataRepository(MetadataRepository):
    """
    File-based metadata repository with in-memory caching.

    This repository caches metadata to speed up run discovery operations.
    """

    def __init__(
        self,
        base_path: Path | None = None,
        cache_ttl_seconds: int = 3600,
        enable_cache: bool = True,
    ):
        """
        Initialize metadata repository.

        Args:
            base_path: Optional base path for data discovery
                (currently unused, kept for compatibility)
            cache_ttl_seconds: Time-to-live for cache entries in seconds
            enable_cache: Whether to enable caching
        """
        self.base_path = Path(base_path) if base_path else None
        self.cache_ttl_seconds = cache_ttl_seconds
        self.enable_cache = enable_cache
        self._cache: dict[str, dict[str, Any]] = {}
        self._cache_timestamps: dict[str, float] = {}

    def find_metadata(
        self,
        network: str,
        dates: list[str],
        base_path: Path | None = None,
    ) -> list[dict[str, Any]]:
        """
        Find metadata for runs in the given network/dates.

        Args:
            network: Network name
            dates: List of dates to search
            base_path: Optional base path override

        Returns:
            List of metadata dictionaries
        """
        if base_path is None:
            if self.base_path is None:
                raise ValueError("base_path must be provided either in constructor or method call")
            base_path = self.base_path

        return self.discover_runs(base_path, network, dates)

    def discover_runs(
        self,
        base_path: Path,
        network: str,
        dates: list[str],
    ) -> list[dict[str, Any]]:
        """Discover all runs in the given network/dates."""
        discovered_runs: list[dict[str, Any]] = []

        for date in dates:
            date_path = base_path / network / date

            if not date_path.exists():
                logger.warning(f"Date directory not found: {date_path}")
                continue

            try:
                for run_dir in date_path.iterdir():
                    if not run_dir.is_dir():
                        continue

                    try:
                        # Get metadata (from cache or file)
                        metadata = self.get_run_metadata(run_dir)

                        # Add run info to metadata
                        run_info = {
                            "id": run_dir.name,
                            "network": network,
                            "date": date,
                            "algorithm": metadata.get("path_algorithm", "unknown"),
                            "path": str(run_dir),
                            **metadata,  # Include all metadata fields
                        }

                        discovered_runs.append(run_info)

                    except Exception as e:
                        logger.warning(f"Failed to load metadata for {run_dir}: {e}. Skipping.")
                        continue

            except PermissionError as e:
                raise RepositoryError(f"Permission denied accessing {date_path}: {e}") from e

        return discovered_runs

    def get_run_metadata(self, run_path: Path) -> dict[str, Any]:
        """Load metadata for a specific run."""
        # Check cache first
        cached = self.get_cached_metadata(run_path)
        if cached is not None:
            return cached

        # Load from file
        metadata = self._load_metadata_from_file(run_path)

        # Cache the result
        self.cache_metadata(run_path, metadata)

        return metadata

    def cache_metadata(
        self,
        run_path: Path,
        metadata: dict[str, Any],
    ) -> None:
        """Cache metadata for faster subsequent access."""
        if not self.enable_cache:
            return

        cache_key = self._get_cache_key(run_path)
        self._cache[cache_key] = metadata
        self._cache_timestamps[cache_key] = time.time()

    def clear_cache(self) -> None:
        """Clear all cached metadata."""
        self._cache.clear()
        self._cache_timestamps.clear()
        logger.info("Metadata cache cleared")

    def get_cached_metadata(
        self,
        run_path: Path,
    ) -> dict[str, Any] | None:
        """Get cached metadata if available and not expired."""
        if not self.enable_cache:
            return None

        cache_key = self._get_cache_key(run_path)

        # Check if in cache
        if cache_key not in self._cache:
            return None

        # Check if expired
        cached_time = self._cache_timestamps.get(cache_key, 0)
        age = time.time() - cached_time

        if age > self.cache_ttl_seconds:
            # Expired, remove from cache
            del self._cache[cache_key]
            del self._cache_timestamps[cache_key]
            return None

        return self._cache[cache_key]

    def _load_metadata_from_file(self, run_path: Path) -> dict[str, Any]:
        """
        Load metadata from file.

        Looks for metadata in multiple possible locations:
        1. metadata.json (DRL runs)
        2. input.json (standard runs)
        3. config.json (alternative naming)
        """
        metadata_files = [
            run_path / "metadata.json",
            run_path / "input.json",
            run_path / "config.json",
        ]

        for metadata_file in metadata_files:
            if metadata_file.exists():
                try:
                    with open(metadata_file) as f:
                        metadata: dict[str, Any] = json.load(f)
                        logger.debug(f"Loaded metadata from {metadata_file}")
                        return metadata
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON in {metadata_file}: {e}")
                    continue
                except Exception as e:
                    logger.warning(f"Failed to load metadata from {metadata_file}: {e}")
                    continue

        # No metadata found, return minimal info
        logger.debug(f"No metadata file found in {run_path}, using defaults")
        return {
            "path_algorithm": "unknown",
            "run_id": run_path.name,
        }

    def _get_cache_key(self, run_path: Path) -> str:
        """Generate cache key for a run path."""
        # Use hash of absolute path as cache key (not for security)
        path_str = str(run_path.absolute())
        return hashlib.md5(path_str.encode(), usedforsecurity=False).hexdigest()

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        now = time.time()
        valid_entries = sum(1 for key, timestamp in self._cache_timestamps.items() if (now - timestamp) <= self.cache_ttl_seconds)

        return {
            "total_entries": len(self._cache),
            "valid_entries": valid_entries,
            "expired_entries": len(self._cache) - valid_entries,
            "cache_ttl_seconds": self.cache_ttl_seconds,
            "cache_enabled": self.enable_cache,
        }

    def prune_expired_entries(self) -> int:
        """
        Remove expired entries from cache.

        Returns:
            Number of entries removed
        """
        now = time.time()
        expired_keys = [key for key, timestamp in self._cache_timestamps.items() if (now - timestamp) > self.cache_ttl_seconds]

        for key in expired_keys:
            del self._cache[key]
            del self._cache_timestamps[key]

        if expired_keys:
            logger.info(f"Pruned {len(expired_keys)} expired cache entries")

        return len(expired_keys)
