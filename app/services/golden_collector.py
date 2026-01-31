"""
Golden Image Collector Service
Automatically collects high-quality images (IDR > 0.85) for LoRA training.

This service hooks into the image generation pipeline to:
- Monitor generated images for high IDR scores
- Automatically add qualifying images to LoRA training datasets
- Track collection progress and notify when training threshold is reached
"""

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime

from app.core.database import SessionLocal
from app.models.lora import LoraModel, LoraModelStatus
from app.models.character import Character
from app.services.lora_dataset import (
    add_golden_image_to_dataset,
    GOLDEN_IDR_THRESHOLD,
    MIN_TRAINING_IMAGES
)

logger = logging.getLogger(__name__)


class GoldenImageCollector:
    """
    Collects golden images (IDR > 0.85) for LoRA training.
    
    Integrates with the generation pipeline to automatically
    capture high-quality images as they are generated.
    """
    
    def __init__(self):
        """Initialize the collector."""
        self.threshold = GOLDEN_IDR_THRESHOLD
        self.min_images = MIN_TRAINING_IMAGES
        
        # Stats tracking (in-memory, resets on restart)
        self._session_stats: Dict[str, Dict[str, int]] = {}
    
    def evaluate_and_collect(
        self,
        character_id: str,
        image_path: str,
        idr_score: float,
        job_id: Optional[str] = None,
        prompt: Optional[str] = None,
        scene_index: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate an image and collect if it qualifies as golden.
        
        This is the main entry point called from the generation pipeline
        after IDR verification is complete.
        
        Args:
            character_id: The character ID
            image_path: Path to the generated image
            idr_score: IDR score from identity verification
            job_id: Generation job ID
            prompt: Generation prompt used
            scene_index: Scene index in the job
            metadata: Additional metadata to store
            
        Returns:
            Dict with collection result and status
        """
        result = {
            "collected": False,
            "character_id": character_id,
            "idr_score": idr_score,
            "threshold": self.threshold,
            "is_golden": idr_score >= self.threshold
        }
        
        # Check if qualifies as golden
        if idr_score < self.threshold:
            result["reason"] = f"IDR {idr_score:.3f} below threshold {self.threshold}"
            return result
        
        # Attempt to add to dataset
        try:
            add_result = add_golden_image_to_dataset(
                character_id=character_id,
                image_path=image_path,
                idr_score=idr_score,
                job_id=job_id,
                prompt=prompt,
                scene_index=scene_index
            )
            
            if add_result:
                result["collected"] = True
                result["image_id"] = add_result.get("image_id")
                result["ready_for_training"] = add_result.get("ready_for_training", False)
                result["total_images"] = add_result.get("total_images", 0)
                
                # Update session stats
                self._update_stats(character_id, collected=True)
                
                logger.info(
                    f"Golden image collected for {character_id}: "
                    f"IDR={idr_score:.3f}, total={result.get('total_images', '?')}"
                )
                
                # Check if ready for training
                if result["ready_for_training"]:
                    self._on_training_ready(character_id)
            else:
                result["reason"] = "Image rejected by dataset builder (duplicate or quality)"
                self._update_stats(character_id, rejected=True)
                
        except Exception as e:
            logger.error(f"Failed to collect golden image: {e}")
            result["error"] = str(e)
            result["reason"] = "Collection failed due to error"
        
        return result
    
    def _update_stats(self, character_id: str, collected: bool = False, rejected: bool = False):
        """Update session statistics."""
        if character_id not in self._session_stats:
            self._session_stats[character_id] = {
                "collected": 0,
                "rejected": 0,
                "evaluated": 0
            }
        
        self._session_stats[character_id]["evaluated"] += 1
        if collected:
            self._session_stats[character_id]["collected"] += 1
        if rejected:
            self._session_stats[character_id]["rejected"] += 1
    
    def _on_training_ready(self, character_id: str):
        """
        Called when a character has enough images for training.
        
        This can trigger automatic training or notify the user.
        """
        logger.info(f"🎯 Character {character_id} ready for LoRA training!")
        
        # Update LoRA model status to indicate readiness
        db = SessionLocal()
        try:
            lora = db.query(LoraModel).filter(
                LoraModel.character_id == character_id,
                LoraModel.status == LoraModelStatus.COLLECTING
            ).order_by(LoraModel.created_at.desc()).first()
            
            if lora:
                # Add a flag or metadata indicating training is available
                # The actual training trigger is handled separately
                logger.info(
                    f"LoRA {lora.id} has {lora.training_images} images, "
                    f"ready for training"
                )
        finally:
            db.close()
    
    def get_collection_status(self, character_id: str) -> Dict[str, Any]:
        """
        Get collection status for a character.
        
        Returns:
            Dict with collection progress and status
        """
        db = SessionLocal()
        try:
            # Get character
            character = db.query(Character).filter(
                Character.id == character_id
            ).first()
            
            if not character:
                return {
                    "error": "Character not found",
                    "character_id": character_id
                }
            
            # Get active/collecting LoRA
            lora = db.query(LoraModel).filter(
                LoraModel.character_id == character_id,
                LoraModel.status.in_([LoraModelStatus.COLLECTING, LoraModelStatus.ACTIVE])
            ).order_by(LoraModel.created_at.desc()).first()
            
            status = {
                "character_id": character_id,
                "character_name": character.name,
                "threshold": self.threshold,
                "min_images_required": self.min_images
            }
            
            if lora:
                images_collected = lora.training_images or 0
                status.update({
                    "lora_id": lora.id,
                    "lora_status": lora.status,
                    "images_collected": images_collected,
                    "images_needed": max(0, self.min_images - images_collected),
                    "progress_percent": min(100, (images_collected / self.min_images) * 100),
                    "ready_for_training": images_collected >= self.min_images
                })
            else:
                status.update({
                    "lora_id": None,
                    "lora_status": "none",
                    "images_collected": 0,
                    "images_needed": self.min_images,
                    "progress_percent": 0,
                    "ready_for_training": False
                })
            
            # Add session stats if available
            if character_id in self._session_stats:
                status["session_stats"] = self._session_stats[character_id]
            
            return status
            
        finally:
            db.close()
    
    def get_all_collection_status(self) -> List[Dict[str, Any]]:
        """Get collection status for all characters with LoRA models."""
        db = SessionLocal()
        try:
            # Get all characters with collecting or active LoRAs
            loras = db.query(LoraModel).filter(
                LoraModel.status.in_([LoraModelStatus.COLLECTING, LoraModelStatus.ACTIVE])
            ).all()
            
            statuses = []
            seen_characters = set()
            
            for lora in loras:
                if lora.character_id in seen_characters:
                    continue
                seen_characters.add(lora.character_id)
                
                status = self.get_collection_status(lora.character_id)
                statuses.append(status)
            
            return statuses
            
        finally:
            db.close()
    
    def reset_session_stats(self):
        """Reset in-memory session statistics."""
        self._session_stats.clear()
        logger.info("Session statistics reset")


# Global collector instance
_collector: Optional[GoldenImageCollector] = None


def get_collector() -> GoldenImageCollector:
    """Get or create the global collector instance."""
    global _collector
    if _collector is None:
        _collector = GoldenImageCollector()
    return _collector


def collect_golden_image(
    character_id: str,
    image_path: str,
    idr_score: float,
    job_id: Optional[str] = None,
    prompt: Optional[str] = None,
    scene_index: Optional[int] = None
) -> Dict[str, Any]:
    """
    Convenience function to collect a golden image.
    
    This is the main function to call from the generation pipeline.
    
    Args:
        character_id: Character ID
        image_path: Path to image file
        idr_score: IDR score (only collected if > 0.85)
        job_id: Generation job ID
        prompt: Generation prompt
        scene_index: Scene index
        
    Returns:
        Collection result dict
    """
    collector = get_collector()
    return collector.evaluate_and_collect(
        character_id=character_id,
        image_path=image_path,
        idr_score=idr_score,
        job_id=job_id,
        prompt=prompt,
        scene_index=scene_index
    )


def get_collection_progress(character_id: str) -> Dict[str, Any]:
    """Get collection progress for a character."""
    collector = get_collector()
    return collector.get_collection_status(character_id)


# Export all
__all__ = [
    "GoldenImageCollector",
    "get_collector",
    "collect_golden_image",
    "get_collection_progress",
    "GOLDEN_IDR_THRESHOLD",
    "MIN_TRAINING_IMAGES"
]
