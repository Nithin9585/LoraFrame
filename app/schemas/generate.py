"""
Generate Schemas
Pydantic models for generation API requests and responses.
"""

from typing import Optional, List
from pydantic import BaseModel


class GenerateOptions(BaseModel):
    """Options for image generation."""
    video: bool = False
    refine_face: bool = True
    aspect_ratio: str = "16:9"  # Gemini supports: 1:1, 2:3, 3:2, 3:4, 4:3, 4:5, 5:4, 9:16, 16:9, 21:9
    style_overrides: List[str] = []


class GenerateRequest(BaseModel):
    """Schema for generation request."""
    character_id: str
    prompt: str
    pose_image_url: Optional[str] = None
    options: Optional[GenerateOptions] = None


class GenerateResponse(BaseModel):
    """Schema for generation response."""
    job_id: str
    status: str
    message: str
