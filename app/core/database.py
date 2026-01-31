"""
Database Configuration
SQLAlchemy engine and session management.
Supports both SQLite (local dev) and PostgreSQL (production).
"""

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from app.core.config import settings

# Detect if using SQLite
is_sqlite = settings.DATABASE_URL.startswith("sqlite")

# Create database engine with appropriate settings
if is_sqlite:
    # SQLite settings
    engine = create_engine(
        settings.DATABASE_URL,
        connect_args={"check_same_thread": False},  # Needed for SQLite
    )
else:
    # PostgreSQL settings
    engine = create_engine(
        settings.DATABASE_URL,
        pool_pre_ping=True,
        pool_size=10,
        max_overflow=20,
    )

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()


def init_db():
    """Initialize database tables."""
    try:
        from app.models import Character, Job, EpisodicState  # noqa
        Base.metadata.create_all(bind=engine)
    except Exception as e:
        # In production, tables may already exist or filesystem may be read-only
        print(f"[WARNING] Could not create database tables: {e}")
        print("[INFO] Continuing with existing database...")
