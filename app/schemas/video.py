"""
Video Generation Schemas
Pydantic models for video generation API requests and responses.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class VideoDialogue(BaseModel):
    """A single dialogue line in a video."""
    speaker: str = Field(..., description="Name or description of the speaker")
    line: str = Field(..., description="The dialogue text to speak")
    emotion: Optional[str] = Field(None, description="Emotion/delivery style (e.g., 'excitedly', 'whispering')")


class VideoGenerateOptions(BaseModel):
    """Options for video generation."""
    aspect_ratio: str = Field("16:9", description="16:9 (landscape) or 9:16 (portrait)")
    resolution: str = Field("720p", description="720p, 1080p, or 4k")
    duration_seconds: int = Field(8, description="Video duration: 4, 5, 6, or 8 seconds")
    negative_prompt: Optional[str] = Field(None, description="Things to avoid in the video")
    person_generation: str = Field("allow_adult", description="allow_all, allow_adult, or dont_allow")
    seed: Optional[int] = Field(None, description="Optional seed for more deterministic results")
    
    # Dialogue-specific options
    dialogue: Optional[List[VideoDialogue]] = Field(None, description="List of dialogue lines")
    sound_effects: Optional[List[str]] = Field(None, description="Sound effects to include")
    camera_movement: Optional[str] = Field(None, description="Camera direction (e.g., 'slow zoom in')")
    
    # Style options
    style: Optional[str] = Field(None, description="Visual style: cinematic, documentary, animation")
    lighting: Optional[str] = Field(None, description="Lighting description")
    color_grade: Optional[str] = Field(None, description="Color grading style")


class VideoGenerateRequest(BaseModel):
    """Schema for video generation request."""
    character_id: str = Field(..., description="Character ID for identity consistency")
    prompt: str = Field(..., description="Scene description including dialogue")
    options: Optional[VideoGenerateOptions] = None
    
    # Optional: Use specific images for video generation
    use_first_frame: bool = Field(False, description="Generate character image first as starting frame")
    use_reference_images: bool = Field(True, description="Use character's reference images for consistency")
    first_frame_prompt: Optional[str] = Field(None, description="Override prompt for the first frame image")
    

class VideoExtendRequest(BaseModel):
    """Schema for video extension request."""
    job_id: str = Field(..., description="Original video job ID")
    continuation_prompt: Optional[str] = Field(None, description="Prompt to guide the continuation")
    duration_seconds: int = Field(8, description="Duration of the extension")


class VideoTransitionRequest(BaseModel):
    """Schema for frame-to-frame video transition."""
    character_id: str = Field(..., description="Character ID")
    transition_prompt: str = Field(..., description="Description of the transition/action")
    first_frame_prompt: str = Field(..., description="Prompt for the starting pose/image")
    last_frame_prompt: str = Field(..., description="Prompt for the ending pose/image")
    options: Optional[VideoGenerateOptions] = None


class VideoGenerateResponse(BaseModel):
    """Schema for video generation response."""
    job_id: str
    status: str
    message: str
    result_url: Optional[str] = None
    video_duration_seconds: Optional[int] = None
    generation_time_seconds: Optional[float] = None
    scene_index: Optional[int] = None
    
    # Additional metadata
    model_used: Optional[str] = None
    has_audio: bool = True  # Veo 3.1 generates native audio
    
    class Config:
        extra = "allow"  # Allow additional fields


class VideoJobStatus(BaseModel):
    """Schema for checking video job status."""
    job_id: str
    status: str  # queued, running, success, failed
    progress_percent: Optional[int] = None
    estimated_time_remaining: Optional[int] = None
    result_url: Optional[str] = None
    error_message: Optional[str] = None
