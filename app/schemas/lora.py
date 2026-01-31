"""
LoRA Pydantic Schemas
Request/Response models for LoRA API endpoints.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


# --- Request Schemas ---

class LoraCreateRequest(BaseModel):
    """Request to create a new LoRA model for a character."""
    character_id: str = Field(..., description="Character ID to create LoRA for")
    
    class Config:
        json_schema_extra = {
            "example": {
                "character_id": "char_abc123"
            }
        }


class LoraTrainingConfigUpdate(BaseModel):
    """Update training configuration for a LoRA model."""
    rank: Optional[int] = Field(None, ge=4, le=128, description="LoRA rank (default: 32)")
    alpha: Optional[int] = Field(None, ge=4, le=128, description="LoRA alpha (default: 32)")
    learning_rate: Optional[float] = Field(None, ge=1e-6, le=1e-2, description="Learning rate")
    batch_size: Optional[int] = Field(None, ge=1, le=8, description="Training batch size")
    max_steps: Optional[int] = Field(None, ge=100, le=5000, description="Max training steps")


class LoraActivateRequest(BaseModel):
    """Request to activate a specific LoRA version."""
    lora_id: str = Field(..., description="LoRA model ID to activate")


class LoraAddImageRequest(BaseModel):
    """Request to manually add an image to training dataset."""
    image_url: str = Field(..., description="URL of image to add")
    idr_score: float = Field(..., ge=0.85, le=1.0, description="IDR score (must be >= 0.85)")
    caption: Optional[str] = Field(None, description="Optional caption for training")
    prompt_used: Optional[str] = Field(None, description="Original generation prompt")


# --- Response Schemas ---

class LoraTrainingImageResponse(BaseModel):
    """Training image in dataset."""
    id: str
    image_url: str
    idr_score: float
    caption: Optional[str] = None
    preprocessed: bool = False
    collected_at: datetime
    
    class Config:
        from_attributes = True


class LoraModelResponse(BaseModel):
    """Full LoRA model details."""
    id: str
    character_id: str
    version: int
    status: str
    
    # File info
    file_path: Optional[str] = None
    file_size_mb: Optional[float] = None
    
    # Training info
    training_images: int = 0
    training_steps: Optional[int] = None
    training_loss: Optional[float] = None
    training_time_seconds: Optional[float] = None
    training_config: Dict[str, Any] = {}
    
    # IDR metrics
    baseline_idr: Optional[float] = None
    final_idr: Optional[float] = None
    idr_improvement: Optional[float] = None
    
    # Validation
    validation_passed: Optional[bool] = None
    validation_samples: Optional[int] = None
    
    # Timestamps
    created_at: datetime
    training_started_at: Optional[datetime] = None
    training_completed_at: Optional[datetime] = None
    activated_at: Optional[datetime] = None
    
    # Error info
    error_message: Optional[str] = None
    
    class Config:
        from_attributes = True


class LoraModelSummary(BaseModel):
    """Brief LoRA model summary for lists."""
    id: str
    version: int
    status: str
    training_images: int
    final_idr: Optional[float] = None
    is_active: bool = False
    created_at: datetime
    
    class Config:
        from_attributes = True


class LoraListResponse(BaseModel):
    """List of LoRA models for a character."""
    character_id: str
    total: int
    active_lora_id: Optional[str] = None
    models: List[LoraModelSummary]


class LoraDatasetResponse(BaseModel):
    """Training dataset details."""
    lora_id: str
    total_images: int
    min_images_required: int = 30
    can_start_training: bool = False
    images: List[LoraTrainingImageResponse]
    
    # Statistics
    avg_idr: Optional[float] = None
    min_idr: Optional[float] = None
    max_idr: Optional[float] = None


class LoraTrainingStatusResponse(BaseModel):
    """Training job status."""
    lora_id: str
    status: str
    progress: float = 0.0  # 0.0 to 1.0
    current_step: Optional[int] = None
    total_steps: Optional[int] = None
    current_loss: Optional[float] = None
    eta_seconds: Optional[int] = None
    message: str = ""


class LoraValidationResponse(BaseModel):
    """Validation results."""
    lora_id: str
    passed: bool
    baseline_idr: float
    final_idr: float
    improvement_percent: float
    test_samples: int
    details: Dict[str, Any] = {}
    recommendation: str  # "activate", "retrain", "archive"


# --- Status Update Schemas ---

class LoraStatusUpdate(BaseModel):
    """Generic status update message."""
    lora_id: str
    status: str
    message: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# Export all
__all__ = [
    # Requests
    "LoraCreateRequest",
    "LoraTrainingConfigUpdate",
    "LoraActivateRequest",
    "LoraAddImageRequest",
    # Responses
    "LoraTrainingImageResponse",
    "LoraModelResponse",
    "LoraModelSummary",
    "LoraListResponse",
    "LoraDatasetResponse",
    "LoraTrainingStatusResponse",
    "LoraValidationResponse",
    "LoraStatusUpdate"
]
