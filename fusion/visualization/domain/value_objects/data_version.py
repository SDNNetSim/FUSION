"""Data format version value object."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class DataVersion:
    """Immutable version identifier for data formats."""

    major: int
    minor: int = 0

    def __post_init__(self) -> None:
        """Validate version numbers."""
        if self.major < 0 or self.minor < 0:
            raise ValueError("Version numbers must be non-negative")

    @classmethod
    def from_string(cls, version_str: str) -> DataVersion:
        """
        Create DataVersion from string representation.

        Args:
            version_str: Version string like "v1", "v2.1", etc.

        Returns:
            DataVersion instance

        Raises:
            ValueError: If version string is invalid
        """
        # Remove 'v' prefix if present
        version_str = version_str.lower().lstrip('v')

        parts = version_str.split('.')
        if len(parts) == 1:
            return cls(major=int(parts[0]))
        elif len(parts) == 2:
            return cls(major=int(parts[0]), minor=int(parts[1]))
        else:
            raise ValueError(f"Invalid version string: {version_str}")

    def __str__(self) -> str:
        """Return string representation."""
        if self.minor == 0:
            return f"v{self.major}"
        return f"v{self.major}.{self.minor}"

    def __eq__(self, other: object) -> bool:
        """
        Compare with another DataVersion or string.

        Args:
            other: Another DataVersion object or a version string

        Returns:
            True if versions are equal
        """
        if isinstance(other, str):
            try:
                other = DataVersion.from_string(other)
            except (ValueError, AttributeError):
                return False
        if not isinstance(other, DataVersion):
            return False
        return (self.major, self.minor) == (other.major, other.minor)

    def __hash__(self) -> int:
        """Return hash for use in sets and dicts."""
        return hash((self.major, self.minor))

    def __lt__(self, other: DataVersion) -> bool:
        """Compare versions."""
        return (self.major, self.minor) < (other.major, other.minor)

    def __le__(self, other: DataVersion) -> bool:
        """Compare versions."""
        return (self.major, self.minor) <= (other.major, other.minor)

    def __gt__(self, other: DataVersion) -> bool:
        """Compare versions."""
        return (self.major, self.minor) > (other.major, other.minor)

    def __ge__(self, other: DataVersion) -> bool:
        """Compare versions."""
        return (self.major, self.minor) >= (other.major, other.minor)
