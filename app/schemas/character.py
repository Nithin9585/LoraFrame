"""
Character Schemas
Pydantic models for character API requests and responses.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel


class CharacterCreate(BaseModel):
    """Schema for character creation (used with form data)."""
    name: str
    description: Optional[str] = None
    consent: bool


class CharacterUpdate(BaseModel):
    """Schema for character update."""
    name: Optional[str] = None
    description: Optional[str] = None
    char_metadata: Optional[Dict[str, Any]] = None
    
    class Config:
        # Allow partial updates
        extra = "ignore"


class CharacterResponse(BaseModel):
    """Schema for character response."""
    id: str
    name: str
    description: Optional[str]
    semantic_vector_id: Optional[str]
    base_image_url: Optional[str]
    char_metadata: Dict[str, Any] = {}
    consent_given_at: Optional[datetime]
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class EpisodicStateResponse(BaseModel):
    """Schema for episodic state response."""
    id: str
    scene_index: int
    tags: List[str] = []
    image_url: Optional[str]
    notes: Optional[str]
    created_at: datetime
    
    class Config:
        from_attributes = True


class CharacterHistory(BaseModel):
    """Schema for character history with episodic states."""
    character: CharacterResponse
    episodic_states: List[EpisodicStateResponse] = []
    total_scenes: int
