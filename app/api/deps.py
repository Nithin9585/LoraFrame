"""
API Dependencies
Common dependencies for FastAPI routes (database sessions, etc.)
"""

from typing import Generator
from app.core.database import SessionLocal


def get_db() -> Generator:
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
