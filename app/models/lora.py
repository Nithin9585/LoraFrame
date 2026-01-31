"""
LoRA Model
Database model for LoRA (Low-Rank Adaptation) models trained per character.

LoRA models are lightweight fine-tuned adapters that improve character consistency
by learning from high-quality generated images (Golden Images with IDR > 0.85).
"""

from datetime import datetime
from sqlalchemy import Column, String, Text, DateTime, Integer, Float, ForeignKey, JSON, Boolean
from sqlalchemy.orm import relationship

from app.core.database import Base


class LoraModelStatus:
    """LoRA model status constants."""
    COLLECTING = "collecting"      # Gathering training images
    TRAINING = "training"          # Currently training
    VALIDATING = "validating"      # Running validation tests
    ACTIVE = "active"              # Currently in use for generation
    ARCHIVED = "archived"          # Superseded by newer version
    FAILED = "failed"              # Training or validation failed


class LoraModel(Base):
    """
    LoRA model record for a character.
    
    Each character can have multiple LoRA versions. Only one can be active at a time.
    The system automatically trains new versions when enough golden images are collected.
    """
    
    __tablename__ = "lora_models"
    
    # Primary identifier: lora_char_xxx_v1
    id = Column(String, primary_key=True)
    
    # Character reference
    character_id = Column(String, ForeignKey("characters.id"), nullable=False, index=True)
    
    # Version tracking
    version = Column(Integer, default=1)
    
    # File storage
    file_path = Column(String, nullable=True)  # Path to .safetensors file
    file_size_mb = Column(Float, nullable=True)  # File size in MB
    
    # Training metrics
    training_images = Column(Integer, default=0)  # Number of images used
    training_steps = Column(Integer, nullable=True)  # Training steps completed
    training_loss = Column(Float, nullable=True)  # Final training loss
    training_time_seconds = Column(Float, nullable=True)  # Total training time
    
    # Training configuration (stored for reproducibility)
    training_config = Column(JSON, default=lambda: {
        "rank": 32,
        "alpha": 32,
        "learning_rate": 1e-4,
        "batch_size": 1,
        "max_steps": 1000,
        "target_modules": ["q_proj", "v_proj", "k_proj", "out_proj"]
    })
    
    # IDR metrics
    baseline_idr = Column(Float, nullable=True)  # Average IDR before LoRA
    final_idr = Column(Float, nullable=True)      # Average IDR after LoRA
    idr_improvement = Column(Float, nullable=True)  # Percentage improvement
    
    # Validation results
    validation_samples = Column(Integer, nullable=True)  # Number of test images
    validation_passed = Column(Boolean, nullable=True)   # Whether validation succeeded
    validation_details = Column(JSON, nullable=True)     # Detailed validation results
    
    # Status
    status = Column(String, default=LoraModelStatus.COLLECTING)
    error_message = Column(Text, nullable=True)  # Error details if failed
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    training_started_at = Column(DateTime, nullable=True)
    training_completed_at = Column(DateTime, nullable=True)
    activated_at = Column(DateTime, nullable=True)
    archived_at = Column(DateTime, nullable=True)
    
    # Relationship back to character
    character = relationship("Character", back_populates="lora_models")
    
    # Relationship to training images
    training_dataset = relationship("LoraTrainingImage", back_populates="lora_model", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<LoraModel {self.id} v{self.version} ({self.status})>"
    
    @property
    def is_active(self) -> bool:
        """Check if this LoRA is currently active."""
        return self.status == LoraModelStatus.ACTIVE
    
    @property
    def is_ready(self) -> bool:
        """Check if this LoRA is ready for use (active or validated)."""
        return self.status in [LoraModelStatus.ACTIVE, LoraModelStatus.VALIDATING]
    
    @property
    def can_train(self) -> bool:
        """Check if training can be started."""
        return self.status == LoraModelStatus.COLLECTING and self.training_images >= 30


class LoraTrainingImage(Base):
    """
    Training image record for LoRA dataset.
    
    Stores metadata about images collected for LoRA training.
    Only images with IDR > 0.85 (golden images) are collected.
    """
    
    __tablename__ = "lora_training_images"
    
    id = Column(String, primary_key=True)  # img_xxx format
    
    # LoRA model reference
    lora_model_id = Column(String, ForeignKey("lora_models.id"), nullable=False, index=True)
    
    # Image reference
    image_url = Column(String, nullable=False)
    job_id = Column(String, nullable=True)  # Original generation job
    
    # Quality metrics
    idr_score = Column(Float, nullable=False)  # IDR score (must be > 0.85)
    
    # Training metadata
    caption = Column(Text, nullable=True)  # Auto-generated caption for training
    prompt_used = Column(Text, nullable=True)  # Original generation prompt
    scene_index = Column(Integer, nullable=True)  # Scene number
    
    # Preprocessing status
    preprocessed = Column(Boolean, default=False)  # Whether image has been prepared
    preprocessed_path = Column(String, nullable=True)  # Path to processed image
    
    # Timestamps
    collected_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship back to LoRA model
    lora_model = relationship("LoraModel", back_populates="training_dataset")
    
    def __repr__(self):
        return f"<LoraTrainingImage {self.id} IDR={self.idr_score:.3f}>"


# Export all
__all__ = [
    "LoraModelStatus",
    "LoraModel",
    "LoraTrainingImage"
]
