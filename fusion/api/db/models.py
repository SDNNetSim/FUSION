"""
SQLAlchemy ORM models for FUSION GUI.

Defines the database schema for simulation runs.
"""

from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from .database import Base


class Run(Base):
    """
    SQLAlchemy model for simulation runs.

    Tracks run metadata, process information, and progress state.
    """

    __tablename__ = "runs"

    # Primary key - 12 character hex string
    id: Mapped[str] = mapped_column(String(12), primary_key=True)

    # User-provided name (optional)
    name: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # Status: PENDING, RUNNING, COMPLETED, FAILED, CANCELLED
    status: Mapped[str] = mapped_column(String(20), nullable=False, index=True)

    # Configuration stored as JSON string
    config_json: Mapped[str] = mapped_column(Text, nullable=False)

    # Template used to create this run
    template: Mapped[str] = mapped_column(String(100), nullable=False, default="default")

    # Process tracking (for cancellation)
    pid: Mapped[int | None] = mapped_column(Integer, nullable=True)
    pgid: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    # Error information (if FAILED)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Progress cache (updated periodically from progress.jsonl)
    current_erlang: Mapped[float | None] = mapped_column(Float, nullable=True)
    total_erlangs: Mapped[int | None] = mapped_column(Integer, nullable=True)
    current_iteration: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_iterations: Mapped[int | None] = mapped_column(Integer, nullable=True)

    def __repr__(self) -> str:
        """Return string representation of the run."""
        return f"<Run(id={self.id!r}, name={self.name!r}, status={self.status!r})>"
