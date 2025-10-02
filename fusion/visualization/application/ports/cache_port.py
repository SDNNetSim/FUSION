"""Port interface for caching."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Optional, Callable


class CachePort(ABC):
    """
    Port for caching service.

    This interface abstracts the caching mechanism, allowing
    different implementations (in-memory, Redis, file-based, etc.)
    """

    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        pass

    @abstractmethod
    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        """
        Store value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: Time-to-live in seconds (None = no expiry)
        """
        pass

    @abstractmethod
    def get_or_compute(
        self,
        key: str,
        compute_fn: Callable[[], Any],
        ttl_seconds: Optional[int] = None,
    ) -> Any:
        """
        Get from cache or compute and cache result.

        Args:
            key: Cache key
            compute_fn: Function to compute value if not cached
            ttl_seconds: Time-to-live in seconds

        Returns:
            Cached or computed value
        """
        pass

    @abstractmethod
    def invalidate(self, key: str) -> None:
        """
        Invalidate cache entry.

        Args:
            key: Cache key to invalidate
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all cache entries."""
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """
        Check if key exists in cache.

        Args:
            key: Cache key

        Returns:
            True if key exists and is not expired
        """
        pass
