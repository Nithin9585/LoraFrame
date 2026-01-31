"""
Episodic State Model
Database model for character episodic memory.
"""

from datetime import datetime
from sqlalchemy import Column, String, Text, DateTime, Integer, ForeignKey, JSON
from sqlalchemy.orm import relationship

from app.core.database import Base


class EpisodicState(Base):
    """Episodic memory state model."""
    
    __tablename__ = "episodic_states"
    
    id = Column(String, primary_key=True)
    character_id = Column(String, ForeignKey("characters.id"), nullable=False)
    
    # Scene info
    scene_index = Column(Integer, nullable=False)
    
    # Memory data (use JSON for SQLite compatibility)
    tags = Column(JSON, default=[])  # ["injured", "wet", "wearing_coat"]
    state_data = Column(JSON, default={})  # detailed state
    
    # Image
    image_url = Column(String, nullable=True)
    notes = Column(Text, nullable=True)
    
    # Vector DB reference
    vector_id = Column(String, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    character = relationship("Character", back_populates="episodic_states")
