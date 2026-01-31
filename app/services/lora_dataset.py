"""
LoRA Dataset Builder Service
Prepares and preprocesses images for LoRA training.

This service handles:
- Image quality filtering (IDR > 0.85 threshold)
- Face detection and alignment
- Image resizing and normalization
- Caption generation for training
- Dataset organization into training structure
- Deduplication to avoid redundant training data
"""

import os
import uuid
import hashlib
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime

import numpy as np
from PIL import Image
import cv2

from app.core.config import settings
from app.core.database import SessionLocal
from app.models.lora import LoraModel, LoraTrainingImage, LoraModelStatus
from app.services.identity import IdentityService

logger = logging.getLogger(__name__)


# Dataset configuration
GOLDEN_IDR_THRESHOLD = 0.85  # Minimum IDR score for golden images
TARGET_IMAGE_SIZE = (512, 512)  # Standard LoRA training size
MIN_FACE_SIZE = 64  # Minimum face size in pixels
MAX_FACE_ROTATION = 45  # Maximum face rotation in degrees
QUALITY_THRESHOLD = 0.7  # Minimum image quality score
MIN_TRAINING_IMAGES = 30  # Minimum images to start training


class LoraDatasetBuilder:
    """
    Builds and manages LoRA training datasets.
    
    Handles preprocessing, quality filtering, and organization
    of images for LoRA fine-tuning.
    """
    
    def __init__(self, lora_model_id: str, character_id: str):
        """
        Initialize dataset builder for a specific LoRA model.
        
        Args:
            lora_model_id: The LoRA model ID (e.g., "lora_char_xxx_v1")
            character_id: The character ID for identity verification
        """
        self.lora_model_id = lora_model_id
        self.character_id = character_id
        self.identity_service = IdentityService()
        
        # Setup directories
        self.base_dir = Path(settings.UPLOAD_DIR) / "lora" / lora_model_id
        self.raw_dir = self.base_dir / "raw"
        self.processed_dir = self.base_dir / "processed"
        self.captions_dir = self.base_dir / "captions"
        
        # Create directories
        for dir_path in [self.raw_dir, self.processed_dir, self.captions_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Track image hashes for deduplication
        self._image_hashes: set = set()
        self._load_existing_hashes()
    
    def _load_existing_hashes(self):
        """Load hashes of existing images to prevent duplicates."""
        for img_path in self.processed_dir.glob("*.png"):
            try:
                img = Image.open(img_path)
                img_hash = self._compute_image_hash(img)
                self._image_hashes.add(img_hash)
            except Exception as e:
                logger.warning(f"Failed to hash existing image {img_path}: {e}")
    
    def _compute_image_hash(self, image: Image.Image) -> str:
        """
        Compute perceptual hash for deduplication.
        
        Uses average hash (aHash) for speed while maintaining
        reasonable duplicate detection.
        """
        # Resize to 8x8 and convert to grayscale
        small = image.resize((8, 8), Image.Resampling.LANCZOS).convert('L')
        pixels = list(small.getdata())
        avg = sum(pixels) / len(pixels)
        
        # Create hash from pixel comparisons
        bits = ''.join('1' if p > avg else '0' for p in pixels)
        return hex(int(bits, 2))[2:].zfill(16)
    
    def _is_duplicate(self, image: Image.Image) -> bool:
        """Check if image is a duplicate of existing dataset images."""
        img_hash = self._compute_image_hash(image)
        
        if img_hash in self._image_hashes:
            return True
        
        self._image_hashes.add(img_hash)
        return False
    
    def _load_image(self, image_path: str) -> Optional[Image.Image]:
        """
        Load image from local path or URL.
        
        Handles:
        - Local file paths
        - /files/ API URLs
        - HTTP/HTTPS URLs
        """
        import io
        
        try:
            # Handle /files/ API URLs (convert to local path)
            if image_path.startswith("/files/"):
                local_path = Path(settings.UPLOAD_DIR) / image_path.replace("/files/", "")
                if local_path.exists():
                    return Image.open(local_path).convert('RGB')
                else:
                    logger.warning(f"Local file not found: {local_path}")
                    return None
            
            # Handle local file paths
            elif image_path.startswith("/") or (len(image_path) > 1 and image_path[1] == ":"):
                return Image.open(image_path).convert('RGB')
            
            # Handle HTTP URLs
            elif image_path.startswith("http"):
                import httpx
                response = httpx.get(image_path, timeout=30.0)
                if response.status_code == 200:
                    return Image.open(io.BytesIO(response.content)).convert('RGB')
                else:
                    logger.warning(f"Failed to download image: HTTP {response.status_code}")
                    return None
            
            # Try as local path
            else:
                # Could be a relative path
                local_path = Path(settings.UPLOAD_DIR) / image_path
                if local_path.exists():
                    return Image.open(local_path).convert('RGB')
                # Try as absolute path
                elif Path(image_path).exists():
                    return Image.open(image_path).convert('RGB')
                else:
                    logger.warning(f"Image path not found: {image_path}")
                    return None
                    
        except Exception as e:
            logger.error(f"Failed to load image from {image_path}: {e}")
            return None

    def add_image(
        self,
        image_path: str,
        idr_score: float,
        job_id: Optional[str] = None,
        prompt: Optional[str] = None,
        scene_index: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Add an image to the training dataset.
        
        Args:
            image_path: Path to the source image (local path or URL)
            idr_score: Identity consistency score (must be > 0.85)
            job_id: Original generation job ID
            prompt: Generation prompt used
            scene_index: Scene number in the job
            
        Returns:
            Dict with processing results or None if rejected
        """
        # Validate IDR threshold
        if idr_score < GOLDEN_IDR_THRESHOLD:
            logger.info(f"Image rejected: IDR {idr_score:.3f} below threshold {GOLDEN_IDR_THRESHOLD}")
            return None
        
        try:
            # Load image - handle both local paths and URLs
            image = self._load_image(image_path)
            if image is None:
                logger.warning(f"Failed to load image from: {image_path}")
                return None
            
            # Check for duplicates
            if self._is_duplicate(image):
                logger.info(f"Image rejected: Duplicate detected")
                return None
            
            # Preprocess image
            processed_image, face_info = self._preprocess_image(image)
            
            if processed_image is None:
                logger.info(f"Image rejected: Preprocessing failed")
                return None
            
            # Generate unique ID
            image_id = f"img_{uuid.uuid4().hex[:12]}"
            
            # Save processed image
            processed_path = self.processed_dir / f"{image_id}.png"
            processed_image.save(processed_path, "PNG", quality=95)
            
            # Save raw image for reference
            raw_path = self.raw_dir / f"{image_id}_raw.png"
            image.save(raw_path, "PNG")
            
            # Generate caption
            caption = self._generate_caption(prompt, face_info)
            caption_path = self.captions_dir / f"{image_id}.txt"
            caption_path.write_text(caption, encoding='utf-8')
            
            # Record in database
            db = SessionLocal()
            try:
                training_image = LoraTrainingImage(
                    id=image_id,
                    lora_model_id=self.lora_model_id,
                    image_url=str(processed_path),
                    job_id=job_id,
                    idr_score=idr_score,
                    caption=caption,
                    prompt_used=prompt,
                    scene_index=scene_index,
                    preprocessed=True,
                    preprocessed_path=str(processed_path)
                )
                db.add(training_image)
                
                # Update LoRA model image count
                lora_model = db.query(LoraModel).filter(
                    LoraModel.id == self.lora_model_id
                ).first()
                
                if lora_model:
                    lora_model.training_images = (lora_model.training_images or 0) + 1
                
                db.commit()
                
                logger.info(f"Added training image {image_id} (IDR: {idr_score:.3f})")
                
                return {
                    "image_id": image_id,
                    "processed_path": str(processed_path),
                    "caption": caption,
                    "idr_score": idr_score,
                    "face_info": face_info
                }
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Failed to add image: {e}")
            return None
    
    def _preprocess_image(
        self,
        image: Image.Image
    ) -> Tuple[Optional[Image.Image], Optional[Dict[str, Any]]]:
        """
        Preprocess image for LoRA training.
        
        Steps:
        1. Detect face
        2. Check face quality (size, rotation)
        3. Align face
        4. Crop and resize to target size
        5. Normalize
        
        Returns:
            Tuple of (processed_image, face_info) or (None, None) if failed
        """
        try:
            # Convert to numpy for face detection
            img_array = np.array(image)
            
            # Detect faces using identity service
            faces = self.identity_service.detect_faces(img_array)
            
            if not faces or len(faces) == 0:
                logger.debug("No face detected in image")
                return None, None
            
            # Get the largest/most prominent face
            face = self._select_best_face(faces, img_array.shape)
            
            if face is None:
                logger.debug("No suitable face found")
                return None, None
            
            # Extract face info
            face_info = self._extract_face_info(face)
            
            # Validate face quality
            if not self._validate_face_quality(face_info, img_array.shape):
                logger.debug(f"Face quality check failed: {face_info}")
                return None, None
            
            # Crop and align face region with context
            aligned_image = self._crop_and_align(image, face_info)
            
            if aligned_image is None:
                return None, None
            
            # Resize to target size
            final_image = aligned_image.resize(TARGET_IMAGE_SIZE, Image.Resampling.LANCZOS)
            
            return final_image, face_info
            
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            return None, None
    
    def _select_best_face(self, faces: List, image_shape: tuple) -> Optional[Any]:
        """Select the best face from detected faces."""
        if not faces:
            return None
        
        best_face = None
        best_score = 0
        
        img_height, img_width = image_shape[:2]
        img_center = (img_width / 2, img_height / 2)
        
        for face in faces:
            # Get bounding box
            bbox = getattr(face, 'bbox', None)
            if bbox is None:
                continue
            
            x1, y1, x2, y2 = bbox[:4]
            face_width = x2 - x1
            face_height = y2 - y1
            face_area = face_width * face_height
            
            # Calculate center distance (prefer centered faces)
            face_center = ((x1 + x2) / 2, (y1 + y2) / 2)
            center_dist = np.sqrt(
                (face_center[0] - img_center[0])**2 + 
                (face_center[1] - img_center[1])**2
            )
            max_dist = np.sqrt(img_width**2 + img_height**2) / 2
            center_score = 1 - (center_dist / max_dist)
            
            # Combined score: area * center_score
            score = (face_area / (img_width * img_height)) * center_score
            
            if score > best_score:
                best_score = score
                best_face = face
        
        return best_face
    
    def _extract_face_info(self, face: Any) -> Dict[str, Any]:
        """Extract face information from detection result."""
        bbox = getattr(face, 'bbox', [0, 0, 0, 0])
        landmarks = getattr(face, 'kps', None)
        
        # Calculate face dimensions
        x1, y1, x2, y2 = bbox[:4]
        width = x2 - x1
        height = y2 - y1
        center = ((x1 + x2) / 2, (y1 + y2) / 2)
        
        # Estimate rotation from landmarks if available
        rotation = 0
        if landmarks is not None and len(landmarks) >= 2:
            # Use eye positions to estimate rotation
            left_eye = landmarks[0]
            right_eye = landmarks[1]
            dy = right_eye[1] - left_eye[1]
            dx = right_eye[0] - left_eye[0]
            rotation = np.degrees(np.arctan2(dy, dx))
        
        return {
            "bbox": list(map(float, bbox[:4])),
            "width": float(width),
            "height": float(height),
            "center": list(map(float, center)),
            "rotation": float(rotation),
            "landmarks": landmarks.tolist() if landmarks is not None else None
        }
    
    def _validate_face_quality(self, face_info: Dict[str, Any], image_shape: tuple) -> bool:
        """Validate face meets quality requirements."""
        # Check minimum face size
        if face_info["width"] < MIN_FACE_SIZE or face_info["height"] < MIN_FACE_SIZE:
            logger.debug(f"Face too small: {face_info['width']:.0f}x{face_info['height']:.0f}")
            return False
        
        # Check rotation
        if abs(face_info["rotation"]) > MAX_FACE_ROTATION:
            logger.debug(f"Face rotation too extreme: {face_info['rotation']:.1f}°")
            return False
        
        # Check face is reasonably sized relative to image
        img_height, img_width = image_shape[:2]
        face_area_ratio = (face_info["width"] * face_info["height"]) / (img_width * img_height)
        
        if face_area_ratio < 0.01:  # Face less than 1% of image
            logger.debug(f"Face too small relative to image: {face_area_ratio:.2%}")
            return False
        
        return True
    
    def _crop_and_align(self, image: Image.Image, face_info: Dict[str, Any]) -> Optional[Image.Image]:
        """Crop image around face with context and optionally align."""
        try:
            x1, y1, x2, y2 = face_info["bbox"]
            face_width = x2 - x1
            face_height = y2 - y1
            
            # Add context padding (50% on each side)
            padding_x = face_width * 0.5
            padding_y = face_height * 0.7  # More vertical for hair
            
            # Calculate crop region
            crop_x1 = max(0, x1 - padding_x)
            crop_y1 = max(0, y1 - padding_y)
            crop_x2 = min(image.width, x2 + padding_x)
            crop_y2 = min(image.height, y2 + padding_y * 0.5)  # Less bottom padding
            
            # Make crop square (important for LoRA training)
            crop_width = crop_x2 - crop_x1
            crop_height = crop_y2 - crop_y1
            
            if crop_width > crop_height:
                # Expand height
                diff = crop_width - crop_height
                crop_y1 = max(0, crop_y1 - diff / 2)
                crop_y2 = min(image.height, crop_y2 + diff / 2)
            else:
                # Expand width
                diff = crop_height - crop_width
                crop_x1 = max(0, crop_x1 - diff / 2)
                crop_x2 = min(image.width, crop_x2 + diff / 2)
            
            # Crop
            cropped = image.crop((int(crop_x1), int(crop_y1), int(crop_x2), int(crop_y2)))
            
            # Align if rotated
            rotation = face_info.get("rotation", 0)
            if abs(rotation) > 2:  # Only align if rotation > 2 degrees
                cropped = cropped.rotate(-rotation, resample=Image.Resampling.BICUBIC, expand=False)
            
            return cropped
            
        except Exception as e:
            logger.error(f"Crop and align failed: {e}")
            return None
    
    def _generate_caption(self, prompt: Optional[str], face_info: Optional[Dict[str, Any]]) -> str:
        """
        Generate training caption for the image.
        
        For LoRA training, captions help guide the model to learn
        the specific features we want it to capture.
        """
        # Base caption with trigger word
        caption_parts = [f"a photo of {self.character_id}"]
        
        if prompt:
            # Extract key descriptors from original prompt
            # Remove common generation words
            skip_words = {'generate', 'create', 'make', 'image', 'of', 'a', 'an', 'the'}
            words = prompt.lower().split()
            descriptors = [w for w in words if w not in skip_words and len(w) > 2]
            
            # Add relevant descriptors
            if descriptors:
                caption_parts.append(', '.join(descriptors[:5]))
        
        # Add face-related descriptors if available
        if face_info:
            if face_info.get("rotation", 0) > 15:
                caption_parts.append("side view")
            elif face_info.get("rotation", 0) < -15:
                caption_parts.append("side view")
            else:
                caption_parts.append("front view")
        
        return ", ".join(caption_parts)
    
    def get_dataset_stats(self) -> Dict[str, Any]:
        """Get statistics about the current dataset."""
        db = SessionLocal()
        try:
            images = db.query(LoraTrainingImage).filter(
                LoraTrainingImage.lora_model_id == self.lora_model_id
            ).all()
            
            if not images:
                return {
                    "total_images": 0,
                    "preprocessed_images": 0,
                    "avg_idr_score": 0,
                    "min_idr_score": 0,
                    "max_idr_score": 0,
                    "ready_for_training": False
                }
            
            idr_scores = [img.idr_score for img in images]
            preprocessed = [img for img in images if img.preprocessed]
            
            return {
                "total_images": len(images),
                "preprocessed_images": len(preprocessed),
                "avg_idr_score": sum(idr_scores) / len(idr_scores),
                "min_idr_score": min(idr_scores),
                "max_idr_score": max(idr_scores),
                "ready_for_training": len(preprocessed) >= MIN_TRAINING_IMAGES,
                "images_needed": max(0, MIN_TRAINING_IMAGES - len(preprocessed))
            }
            
        finally:
            db.close()
    
    def prepare_training_directory(self) -> Optional[str]:
        """
        Prepare final training directory structure.
        
        Creates the standard LoRA training format:
        /training_data/
            /images/
                00001.png
                00001.txt
                00002.png
                00002.txt
                ...
        
        Returns:
            Path to training directory or None if not ready
        """
        stats = self.get_dataset_stats()
        
        if not stats["ready_for_training"]:
            logger.warning(f"Not enough images for training. Have {stats['preprocessed_images']}, need {MIN_TRAINING_IMAGES}")
            return None
        
        # Create training directory
        training_dir = self.base_dir / "training_data" / "images"
        training_dir.mkdir(parents=True, exist_ok=True)
        
        db = SessionLocal()
        try:
            images = db.query(LoraTrainingImage).filter(
                LoraTrainingImage.lora_model_id == self.lora_model_id,
                LoraTrainingImage.preprocessed == True
            ).order_by(LoraTrainingImage.idr_score.desc()).all()
            
            for idx, img in enumerate(images):
                # Copy image with sequential naming
                src_path = Path(img.preprocessed_path)
                dst_path = training_dir / f"{idx:05d}.png"
                
                if src_path.exists():
                    # Copy file
                    import shutil
                    shutil.copy2(src_path, dst_path)
                    
                    # Write caption
                    caption_path = training_dir / f"{idx:05d}.txt"
                    caption_path.write_text(img.caption or f"a photo of {self.character_id}", encoding='utf-8')
            
            logger.info(f"Prepared training directory with {len(images)} images at {training_dir}")
            return str(training_dir.parent)
            
        finally:
            db.close()
    
    def cleanup(self, keep_processed: bool = True):
        """
        Clean up dataset files.
        
        Args:
            keep_processed: Whether to keep processed images
        """
        import shutil
        
        # Always remove raw images
        if self.raw_dir.exists():
            shutil.rmtree(self.raw_dir)
        
        if not keep_processed:
            if self.processed_dir.exists():
                shutil.rmtree(self.processed_dir)
            if self.captions_dir.exists():
                shutil.rmtree(self.captions_dir)
        
        logger.info(f"Cleaned up dataset for {self.lora_model_id}")


def build_dataset_for_character(character_id: str) -> Optional[Dict[str, Any]]:
    """
    Convenience function to build or update dataset for a character.
    
    Args:
        character_id: The character ID
        
    Returns:
        Dataset statistics or None if no LoRA model found
    """
    from app.services.lora_registry import LoraRegistryService
    
    db = SessionLocal()
    try:
        registry = LoraRegistryService(db)
        lora = registry.get_active_lora(character_id)
        
        if not lora:
            # Get the latest collecting LoRA
            lora = db.query(LoraModel).filter(
                LoraModel.character_id == character_id,
                LoraModel.status == LoraModelStatus.COLLECTING
            ).order_by(LoraModel.created_at.desc()).first()
        
        if not lora:
            logger.info(f"No LoRA model found for character {character_id}")
            return None
        
        builder = LoraDatasetBuilder(lora.id, character_id)
        return builder.get_dataset_stats()
        
    finally:
        db.close()


def add_golden_image_to_dataset(
    character_id: str,
    image_path: str,
    idr_score: float,
    job_id: Optional[str] = None,
    prompt: Optional[str] = None,
    scene_index: Optional[int] = None
) -> Optional[Dict[str, Any]]:
    """
    Add a golden image to character's LoRA dataset.
    
    This is the main entry point for automatically collecting
    training images during generation.
    
    Args:
        character_id: Character ID
        image_path: Path to the image file
        idr_score: IDR score (must be > 0.85)
        job_id: Generation job ID
        prompt: Generation prompt
        scene_index: Scene index in job
        
    Returns:
        Processing result or None if rejected/failed
    """
    if idr_score < GOLDEN_IDR_THRESHOLD:
        return None
    
    from app.services.lora_registry import LoraRegistryService
    
    db = SessionLocal()
    try:
        registry = LoraRegistryService(db)
        
        # Get or create collecting LoRA model
        lora = db.query(LoraModel).filter(
            LoraModel.character_id == character_id,
            LoraModel.status == LoraModelStatus.COLLECTING
        ).order_by(LoraModel.created_at.desc()).first()
        
        if not lora:
            # Create new LoRA model for collection
            lora = registry.create_lora(character_id)
        
        if not lora:
            logger.error(f"Failed to get/create LoRA for character {character_id}")
            return None
        
        # Build dataset
        builder = LoraDatasetBuilder(lora.id, character_id)
        result = builder.add_image(
            image_path=image_path,
            idr_score=idr_score,
            job_id=job_id,
            prompt=prompt,
            scene_index=scene_index
        )
        
        # Check if ready for training
        if result:
            stats = builder.get_dataset_stats()
            if stats.get("ready_for_training"):
                logger.info(f"Character {character_id} has enough images for LoRA training!")
                result["ready_for_training"] = True
                result["total_images"] = stats["total_images"]
        
        return result
        
    finally:
        db.close()


# Export all
__all__ = [
    "LoraDatasetBuilder",
    "build_dataset_for_character",
    "add_golden_image_to_dataset",
    "GOLDEN_IDR_THRESHOLD",
    "MIN_TRAINING_IMAGES",
    "TARGET_IMAGE_SIZE"
]
