"""Application-level caching service."""

from __future__ import annotations

import logging
import pickle
import time
from collections.abc import Callable
from hashlib import sha256
from pathlib import Path
from typing import Any

from fusion.visualization.application.ports import CachePort

logger = logging.getLogger(__name__)


class InMemoryCacheService(CachePort):
    """
    Simple in-memory cache implementation.

    This stores cached values in memory with optional TTL.
    Suitable for single-process applications.
    """

    def __init__(self) -> None:
        """Initialize in-memory cache."""
        self._cache: dict[str, tuple[Any, float | None]] = {}
        # Cache structure: key -> (value, expiry_time)

    def get(self, key: str) -> Any | None:
        """Get value from cache."""
        if key not in self._cache:
            return None

        value, expiry_time = self._cache[key]

        # Check if expired
        if expiry_time is not None and time.time() > expiry_time:
            del self._cache[key]
            return None

        return value

    def set(self, key: str, value: Any, ttl_seconds: int | None = None) -> None:
        """Store value in cache."""
        expiry_time = None
        if ttl_seconds is not None:
            expiry_time = time.time() + ttl_seconds

        self._cache[key] = (value, expiry_time)

    def get_or_compute(
        self,
        key: str,
        compute_fn: Callable[[], Any],
        ttl_seconds: int | None = None,
    ) -> Any:
        """Get from cache or compute and cache result."""
        # Try to get from cache
        value = self.get(key)
        if value is not None:
            logger.debug(f"Cache hit: {key}")
            return value

        # Compute value
        logger.debug(f"Cache miss: {key}, computing...")
        value = compute_fn()

        # Store in cache
        self.set(key, value, ttl_seconds)

        return value

    def invalidate(self, key: str) -> None:
        """Invalidate cache entry."""
        if key in self._cache:
            del self._cache[key]
            logger.debug(f"Invalidated cache key: {key}")

    def clear(self) -> None:
        """Clear all cache entries."""
        count = len(self._cache)
        self._cache.clear()
        logger.info(f"Cleared {count} cache entries")

    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        if key not in self._cache:
            return False

        _, expiry_time = self._cache[key]

        # Check if expired
        if expiry_time is not None and time.time() > expiry_time:
            del self._cache[key]
            return False

        return True

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": len(self._cache),
            "type": "in_memory",
        }


class FileCacheService(CachePort):
    """
    File-based cache implementation.

    This stores cached values as pickle files on disk.
    Suitable for persistent caching across process restarts.
    """

    def __init__(self, cache_dir: Path):
        """
        Initialize file cache.

        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, key: str) -> Path:
        """Get file path for cache key."""
        # Hash the key to create filename
        key_hash = sha256(key.encode()).hexdigest()[:16]
        return self.cache_dir / f"{key_hash}.pkl"

    def get(self, key: str) -> Any | None:
        """Get value from cache."""
        cache_path = self._get_cache_path(key)

        if not cache_path.exists():
            return None

        try:
            with open(cache_path, "rb") as f:
                # Loading trusted cache file that we created
                data = pickle.load(f)  # nosec B301

            value = data["value"]
            expiry_time = data.get("expiry_time")

            # Check if expired
            if expiry_time is not None and time.time() > expiry_time:
                cache_path.unlink()
                return None

            return value

        except Exception as e:
            logger.warning(f"Error reading cache file {cache_path}: {e}")
            return None

    def set(self, key: str, value: Any, ttl_seconds: int | None = None) -> None:
        """Store value in cache."""
        cache_path = self._get_cache_path(key)

        expiry_time = None
        if ttl_seconds is not None:
            expiry_time = time.time() + ttl_seconds

        data = {
            "value": value,
            "expiry_time": expiry_time,
            "created_at": time.time(),
        }

        try:
            with open(cache_path, "wb") as f:
                pickle.dump(data, f)

        except Exception as e:
            logger.error(f"Error writing cache file {cache_path}: {e}")

    def get_or_compute(
        self,
        key: str,
        compute_fn: Callable[[], Any],
        ttl_seconds: int | None = None,
    ) -> Any:
        """Get from cache or compute and cache result."""
        # Try to get from cache
        value = self.get(key)
        if value is not None:
            logger.debug(f"Cache hit: {key}")
            return value

        # Compute value
        logger.debug(f"Cache miss: {key}, computing...")
        value = compute_fn()

        # Store in cache
        self.set(key, value, ttl_seconds)

        return value

    def invalidate(self, key: str) -> None:
        """Invalidate cache entry."""
        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            cache_path.unlink()
            logger.debug(f"Invalidated cache key: {key}")

    def clear(self) -> None:
        """Clear all cache entries."""
        count = 0
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()
            count += 1

        logger.info(f"Cleared {count} cache files")

    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        cache_path = self._get_cache_path(key)

        if not cache_path.exists():
            return False

        try:
            with open(cache_path, "rb") as f:
                # Loading trusted cache file that we created
                data = pickle.load(f)  # nosec B301

            expiry_time = data.get("expiry_time")

            # Check if expired
            if expiry_time is not None and time.time() > expiry_time:
                cache_path.unlink()
                return False

            return True

        except Exception:
            return False

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        cache_files = list(self.cache_dir.glob("*.pkl"))

        total_size = sum(f.stat().st_size for f in cache_files)

        return {
            "size": len(cache_files),
            "total_bytes": total_size,
            "cache_dir": str(self.cache_dir),
            "type": "file",
        }


class CacheService(FileCacheService):
    """
    Default cache service implementation.

    This is a convenience class that defaults to file-based caching
    for backward compatibility with existing code.
    """

    def __init__(self, cache_dir: Path | str | None = None):
        """
        Initialize cache service.

        Args:
            cache_dir: Optional directory for cache files.
                If None, uses in-memory cache.
        """
        if cache_dir is not None:
            super().__init__(Path(cache_dir))
        else:
            # For compatibility, if no cache_dir provided, still use a temp directory
            import tempfile

            super().__init__(Path(tempfile.gettempdir()) / "fusion_cache")


class CacheServiceFactory:
    """Factory for creating cache services."""

    @staticmethod
    def create_memory_cache() -> InMemoryCacheService:
        """Create in-memory cache service."""
        return InMemoryCacheService()

    @staticmethod
    def create_file_cache(cache_dir: Path | str) -> FileCacheService:
        """
        Create file-based cache service.

        Args:
            cache_dir: Directory for cache files

        Returns:
            FileCacheService instance
        """
        return FileCacheService(Path(cache_dir))

    @staticmethod
    def create_default_cache(cache_dir: Path | str | None = None) -> CachePort:
        """
        Create default cache service.

        If cache_dir is provided, uses file cache. Otherwise uses memory cache.

        Args:
            cache_dir: Optional cache directory

        Returns:
            Cache service instance
        """
        if cache_dir:
            return CacheServiceFactory.create_file_cache(cache_dir)
        else:
            return CacheServiceFactory.create_memory_cache()
