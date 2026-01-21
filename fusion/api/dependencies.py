"""
FastAPI dependency injection providers.

Provides database sessions and other shared resources.
"""

from collections.abc import Generator
from typing import Annotated

from fastapi import Depends
from sqlalchemy.orm import Session

from .db.database import SessionLocal


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


# Type alias for dependency injection
DbSession = Annotated[Session, Depends(get_db)]
