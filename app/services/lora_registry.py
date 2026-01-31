"""
LoRA Registry Service
Manages LoRA models: creation, activation, archiving, and retrieval.
"""

import logging
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple

from sqlalchemy.orm import Session

from app.core.database import SessionLocal
from app.core.config import settings
from app.models.lora import LoraModel, LoraTrainingImage, LoraModelStatus
from app.services.storage import StorageService

logger = logging.getLogger(__name__)


class LoraRegistryService:
    """
    Service for managing LoRA models.
    
    Handles:
    - Creating new LoRA records for characters
    - Tracking training images (golden images with IDR > 0.85)
    - Activating/deactivating LoRA versions
    - Retrieving active LoRA for generation
    """
    
    # Minimum images required to start training
    MIN_TRAINING_IMAGES = 30
    
    # IDR threshold for golden images
    GOLDEN_IDR_THRESHOLD = 0.85
    
    def __init__(self, db: Optional[Session] = None):
        self._db = db
        self._storage = StorageService()
    
    @property
    def db(self) -> Session:
        """Get database session."""
        if self._db is None:
            self._db = SessionLocal()
        return self._db
    
    def close(self):
        """Close database session if we created it."""
        if self._db:
            self._db.close()
    
    # --- LoRA Model Management ---
    
    def create_lora(self, character_id: str, training_config: Optional[Dict] = None) -> LoraModel:
        """
        Create a new LoRA model record for a character.
        
        Args:
            character_id: Character to create LoRA for
            training_config: Optional custom training configuration
            
        Returns:
            New LoraModel instance
        """
        # Get next version number
        existing = self.list_loras(character_id)
        version = max([l.version for l in existing], default=0) + 1
        
        # Generate ID
        lora_id = f"lora_{character_id}_{uuid.uuid4().hex[:8]}_v{version}"
        
        # Create model
        lora = LoraModel(
            id=lora_id,
            character_id=character_id,
            version=version,
            status=LoraModelStatus.COLLECTING,
            training_images=0
        )
        
        # Apply custom config if provided
        if training_config:
            lora.training_config = {**lora.training_config, **training_config}
        
        self.db.add(lora)
        self.db.commit()
        self.db.refresh(lora)
        
        logger.info(f"Created LoRA model: {lora_id} for character {character_id}")
        return lora
    
    def get_lora(self, lora_id: str) -> Optional[LoraModel]:
        """Get a LoRA model by ID."""
        return self.db.query(LoraModel).filter(LoraModel.id == lora_id).first()
    
    def get_active_lora(self, character_id: str) -> Optional[LoraModel]:
        """
        Get the currently active LoRA model for a character.
        
        Args:
            character_id: Character ID
            
        Returns:
            Active LoraModel or None
        """
        return self.db.query(LoraModel).filter(
            LoraModel.character_id == character_id,
            LoraModel.status == LoraModelStatus.ACTIVE
        ).first()
    
    def get_active_lora_path(self, character_id: str) -> Optional[str]:
        """
        Get the file path of the active LoRA for a character.
        
        Args:
            character_id: Character ID
            
        Returns:
            Path to .safetensors file or None
        """
        lora = self.get_active_lora(character_id)
        return lora.file_path if lora else None
    
    def list_loras(self, character_id: str) -> List[LoraModel]:
        """
        List all LoRA models for a character.
        
        Args:
            character_id: Character ID
            
        Returns:
            List of LoraModel instances ordered by version desc
        """
        return self.db.query(LoraModel).filter(
            LoraModel.character_id == character_id
        ).order_by(LoraModel.version.desc()).all()
    
    def get_or_create_collecting_lora(self, character_id: str) -> LoraModel:
        """
        Get the current collecting LoRA or create a new one.
        
        This is used when adding golden images - we need a LoRA
        in 'collecting' status to add images to.
        
        Args:
            character_id: Character ID
            
        Returns:
            LoraModel in collecting status
        """
        # Find existing collecting LoRA
        lora = self.db.query(LoraModel).filter(
            LoraModel.character_id == character_id,
            LoraModel.status == LoraModelStatus.COLLECTING
        ).first()
        
        if lora:
            return lora
        
        # Create new one
        return self.create_lora(character_id)
    
    # --- Training Image Management ---
    
    def add_training_image(
        self,
        character_id: str,
        image_url: str,
        idr_score: float,
        job_id: Optional[str] = None,
        caption: Optional[str] = None,
        prompt_used: Optional[str] = None,
        scene_index: Optional[int] = None
    ) -> Optional[LoraTrainingImage]:
        """
        Add a golden image to the training dataset.
        
        Only images with IDR >= 0.85 should be added.
        
        Args:
            character_id: Character ID
            image_url: URL of the image
            idr_score: IDR score (must be >= 0.85)
            job_id: Original generation job ID
            caption: Training caption
            prompt_used: Original generation prompt
            scene_index: Scene number
            
        Returns:
            LoraTrainingImage or None if IDR too low
        """
        if idr_score < self.GOLDEN_IDR_THRESHOLD:
            logger.debug(f"Image IDR {idr_score:.3f} below threshold {self.GOLDEN_IDR_THRESHOLD}")
            return None
        
        # Get or create collecting LoRA
        lora = self.get_or_create_collecting_lora(character_id)
        
        # Check for duplicate
        existing = self.db.query(LoraTrainingImage).filter(
            LoraTrainingImage.lora_model_id == lora.id,
            LoraTrainingImage.image_url == image_url
        ).first()
        
        if existing:
            logger.debug(f"Image already in dataset: {image_url}")
            return existing
        
        # Create training image record
        img_id = f"img_{uuid.uuid4().hex[:12]}"
        
        training_img = LoraTrainingImage(
            id=img_id,
            lora_model_id=lora.id,
            image_url=image_url,
            idr_score=idr_score,
            job_id=job_id,
            caption=caption or self._generate_caption(prompt_used),
            prompt_used=prompt_used,
            scene_index=scene_index
        )
        
        self.db.add(training_img)
        
        # Update count
        lora.training_images += 1
        
        self.db.commit()
        self.db.refresh(training_img)
        
        logger.info(f"Added golden image to {lora.id}: IDR={idr_score:.3f} (total: {lora.training_images})")
        
        # Check if ready for training
        if lora.training_images >= self.MIN_TRAINING_IMAGES:
            logger.info(f"LoRA {lora.id} has {lora.training_images} images - ready for training!")
        
        return training_img
    
    def get_training_images(self, lora_id: str) -> List[LoraTrainingImage]:
        """Get all training images for a LoRA model."""
        return self.db.query(LoraTrainingImage).filter(
            LoraTrainingImage.lora_model_id == lora_id
        ).order_by(LoraTrainingImage.collected_at.desc()).all()
    
    def get_dataset_stats(self, lora_id: str) -> Dict[str, Any]:
        """
        Get statistics about a LoRA's training dataset.
        
        Returns:
            Dict with count, avg/min/max IDR, etc.
        """
        images = self.get_training_images(lora_id)
        
        if not images:
            return {
                "total_images": 0,
                "min_images_required": self.MIN_TRAINING_IMAGES,
                "can_train": False,
                "avg_idr": None,
                "min_idr": None,
                "max_idr": None
            }
        
        idr_scores = [img.idr_score for img in images]
        
        return {
            "total_images": len(images),
            "min_images_required": self.MIN_TRAINING_IMAGES,
            "can_train": len(images) >= self.MIN_TRAINING_IMAGES,
            "avg_idr": sum(idr_scores) / len(idr_scores),
            "min_idr": min(idr_scores),
            "max_idr": max(idr_scores),
            "preprocessed_count": sum(1 for img in images if img.preprocessed)
        }
    
    def _generate_caption(self, prompt: Optional[str]) -> str:
        """Generate a training caption from the generation prompt."""
        if not prompt:
            return "a person"
        
        # Simplify prompt for training caption
        # Remove technical terms, keep descriptive content
        caption = prompt.lower()
        
        # Remove common technical prefixes
        remove_phrases = [
            "generate an image of",
            "create a photo of",
            "a realistic image of",
            "high quality photo of",
        ]
        for phrase in remove_phrases:
            caption = caption.replace(phrase, "")
        
        return caption.strip()[:500]  # Limit length
    
    # --- Status Management ---
    
    def activate_lora(self, lora_id: str) -> bool:
        """
        Activate a LoRA model (deactivates any other active versions).
        
        Args:
            lora_id: LoRA model ID to activate
            
        Returns:
            True if activated, False if not found or invalid status
        """
        lora = self.get_lora(lora_id)
        
        if not lora:
            logger.error(f"LoRA not found: {lora_id}")
            return False
        
        if lora.status not in [LoraModelStatus.VALIDATING, LoraModelStatus.ARCHIVED]:
            logger.error(f"Cannot activate LoRA in status: {lora.status}")
            return False
        
        # Deactivate current active LoRA for this character
        current_active = self.get_active_lora(lora.character_id)
        if current_active:
            current_active.status = LoraModelStatus.ARCHIVED
            current_active.archived_at = datetime.utcnow()
            logger.info(f"Archived previous active LoRA: {current_active.id}")
        
        # Activate new one
        lora.status = LoraModelStatus.ACTIVE
        lora.activated_at = datetime.utcnow()
        
        self.db.commit()
        
        logger.info(f"Activated LoRA: {lora_id}")
        return True
    
    def archive_lora(self, lora_id: str) -> bool:
        """
        Archive a LoRA model.
        
        Args:
            lora_id: LoRA model ID to archive
            
        Returns:
            True if archived, False if not found
        """
        lora = self.get_lora(lora_id)
        
        if not lora:
            return False
        
        lora.status = LoraModelStatus.ARCHIVED
        lora.archived_at = datetime.utcnow()
        
        self.db.commit()
        
        logger.info(f"Archived LoRA: {lora_id}")
        return True
    
    def update_training_status(
        self,
        lora_id: str,
        status: str,
        error_message: Optional[str] = None,
        **metrics
    ) -> bool:
        """
        Update LoRA training status and metrics.
        
        Args:
            lora_id: LoRA model ID
            status: New status
            error_message: Error message if failed
            **metrics: Additional metrics to update
            
        Returns:
            True if updated
        """
        lora = self.get_lora(lora_id)
        
        if not lora:
            return False
        
        lora.status = status
        lora.error_message = error_message
        
        # Update timestamps based on status
        if status == LoraModelStatus.TRAINING and not lora.training_started_at:
            lora.training_started_at = datetime.utcnow()
        elif status in [LoraModelStatus.VALIDATING, LoraModelStatus.ACTIVE, LoraModelStatus.FAILED]:
            lora.training_completed_at = datetime.utcnow()
        
        # Update any provided metrics
        for key, value in metrics.items():
            if hasattr(lora, key):
                setattr(lora, key, value)
        
        self.db.commit()
        
        logger.info(f"Updated LoRA {lora_id} status: {status}")
        return True
    
    def register_trained_lora(
        self,
        lora_id: str,
        file_path: str,
        training_steps: int,
        training_loss: float,
        training_time_seconds: float,
        baseline_idr: float,
        final_idr: float
    ) -> bool:
        """
        Register a completed LoRA training.
        
        Args:
            lora_id: LoRA model ID
            file_path: Path to .safetensors file
            training_steps: Number of training steps
            training_loss: Final training loss
            training_time_seconds: Total training time
            baseline_idr: Average IDR before LoRA
            final_idr: Average IDR after LoRA
            
        Returns:
            True if registered
        """
        lora = self.get_lora(lora_id)
        
        if not lora:
            return False
        
        lora.file_path = file_path
        lora.training_steps = training_steps
        lora.training_loss = training_loss
        lora.training_time_seconds = training_time_seconds
        lora.baseline_idr = baseline_idr
        lora.final_idr = final_idr
        lora.idr_improvement = ((final_idr - baseline_idr) / baseline_idr * 100) if baseline_idr > 0 else 0
        lora.status = LoraModelStatus.VALIDATING
        lora.training_completed_at = datetime.utcnow()
        
        # Get file size
        try:
            import os
            if os.path.exists(file_path):
                lora.file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        except Exception:
            pass
        
        self.db.commit()
        
        logger.info(
            f"Registered trained LoRA: {lora_id} | "
            f"IDR: {baseline_idr:.3f} -> {final_idr:.3f} ({lora.idr_improvement:+.1f}%)"
        )
        return True
    
    def record_validation(
        self,
        lora_id: str,
        passed: bool,
        validation_samples: int,
        validation_details: Dict[str, Any]
    ) -> bool:
        """
        Record validation results for a LoRA.
        
        Args:
            lora_id: LoRA model ID
            passed: Whether validation passed
            validation_samples: Number of test samples
            validation_details: Detailed validation results
            
        Returns:
            True if recorded
        """
        lora = self.get_lora(lora_id)
        
        if not lora:
            return False
        
        lora.validation_passed = passed
        lora.validation_samples = validation_samples
        lora.validation_details = validation_details
        
        if passed:
            lora.status = LoraModelStatus.VALIDATING  # Ready to activate
        else:
            lora.status = LoraModelStatus.FAILED
            lora.error_message = "Validation failed - IDR did not improve"
        
        self.db.commit()
        
        logger.info(f"Validation {'PASSED' if passed else 'FAILED'} for LoRA: {lora_id}")
        return True


# Singleton instance
_registry: Optional[LoraRegistryService] = None


def get_lora_registry(db: Optional[Session] = None) -> LoraRegistryService:
    """Get LoRA registry service instance."""
    global _registry
    if _registry is None or db is not None:
        _registry = LoraRegistryService(db)
    return _registry


# Convenience functions
def get_active_lora_path(character_id: str) -> Optional[str]:
    """Get active LoRA path for a character."""
    return get_lora_registry().get_active_lora_path(character_id)


def add_golden_image(
    character_id: str,
    image_url: str,
    idr_score: float,
    **kwargs
) -> Optional[LoraTrainingImage]:
    """Add a golden image to training dataset."""
    return get_lora_registry().add_training_image(
        character_id=character_id,
        image_url=image_url,
        idr_score=idr_score,
        **kwargs
    )


# Export all
__all__ = [
    "LoraRegistryService",
    "get_lora_registry",
    "get_active_lora_path",
    "add_golden_image"
]
