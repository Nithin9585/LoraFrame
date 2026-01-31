"""
Character Model
Database model for character profiles and identity metadata.
"""

from datetime import datetime
from sqlalchemy import Column, String, Text, DateTime, Integer, JSON
from sqlalchemy.orm import relationship

from app.core.database import Base


class Character(Base):
    """Character profile model."""
    
    __tablename__ = "characters"
    
    id = Column(String, primary_key=True)  # char_xxxx format
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    
    # Vector DB reference
    semantic_vector_id = Column(String, nullable=True)
    semantic_dim = Column(Integer, default=512)
    
    # Images
    base_image_url = Column(String, nullable=True)
    
    # Metadata & consent (use JSON for SQLite compatibility)
    char_metadata = Column(JSON, default={})  # canonical sheet (hair, eye color, tags)
    consent_given_at = Column(DateTime, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    episodic_states = relationship("EpisodicState", back_populates="character", cascade="all, delete-orphan")
    jobs = relationship("Job", back_populates="character", cascade="all, delete-orphan")
    lora_models = relationship("LoraModel", back_populates="character", cascade="all, delete-orphan")
    
    @property
    def active_lora(self):
        """Get the currently active LoRA model for this character."""
        for lora in self.lora_models:
            if lora.status == "active":
                return lora
        return None
