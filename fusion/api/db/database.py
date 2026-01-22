"""
SQLite database setup and session management.

Provides:
- SQLAlchemy engine and session factory
- Database initialization
- Session dependency for FastAPI
"""

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from ..config import settings


class Base(DeclarativeBase):
    """Base class for SQLAlchemy ORM models."""

    pass


engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False},  # SQLite specific
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db() -> None:
    """
    Initialize the database.

    Creates the runs directory and all tables if they don't exist.
    """
    settings.runs_dir.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    """
    Dependency for database sessions.

    Yields a database session and ensures it is closed after use.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
