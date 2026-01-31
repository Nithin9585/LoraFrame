"""
Job Schemas
Pydantic models for job API requests and responses.
"""

from datetime import datetime
from typing import Optional, Dict, Any
from enum import Enum
from pydantic import BaseModel


class JobStatus(str, Enum):
    """Job status enum."""
    QUEUED = "queued"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


class JobResponse(BaseModel):
    """Schema for job response."""
    id: str
    character_id: str
    prompt: str
    pose_image_url: Optional[str]
    options: Dict[str, Any] = {}
    status: str
    error_message: Optional[str]
    result_url: Optional[str]
    metrics: Dict[str, Any] = {}
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime]
    
    class Config:
        from_attributes = True
