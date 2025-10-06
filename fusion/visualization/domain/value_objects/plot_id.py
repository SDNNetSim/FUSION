"""Type-safe identifier for plots."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID, uuid4


@dataclass(frozen=True)
class PlotId:
    """Immutable, type-safe identifier for plots."""

    value: UUID

    @classmethod
    def generate(cls) -> PlotId:
        """Generate a new unique PlotId."""
        return cls(value=uuid4())

    @classmethod
    def from_string(cls, value: str) -> PlotId:
        """Create PlotId from string representation."""
        return cls(value=UUID(value))

    def __str__(self) -> str:
        """Return string representation of the ID."""
        return str(self.value)

    def __repr__(self) -> str:
        """Return detailed representation."""
        return f"PlotId({self.value})"
