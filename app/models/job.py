"""
Job Model
Database model for generation jobs.
"""

from datetime import datetime
from sqlalchemy import Column, String, Text, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship

from app.core.database import Base


class Job(Base):
    """Generation job model."""
    
    __tablename__ = "jobs"
    
    id = Column(String, primary_key=True)  # job_xxxx format
    character_id = Column(String, ForeignKey("characters.id"), nullable=False)
    
    # Request
    prompt = Column(Text, nullable=False)
    pose_image_url = Column(String, nullable=True)
    options = Column(JSON, default={})  # Changed from JSONB for SQLite
    
    # Status: queued, running, success, failed
    status = Column(String, default="queued", index=True)
    error_message = Column(Text, nullable=True)
    
    # Result
    result_url = Column(String, nullable=True)
    
    # Metrics
    metrics = Column(JSON, default={})  # Changed from JSONB for SQLite
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    
    # Relationships
    character = relationship("Character", back_populates="jobs")
