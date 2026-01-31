"""
Face Refiner Worker
Handles face refinement when IDR (Identity Retention) score is below threshold.

This is the "fix-up" loop that ensures facial consistency across generated images.

Refinement Pipeline:
┌─────────────────────────────────────────────────────────────────────┐
│                      FACE REFINER LOOP                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Generated Image (IDR < 0.75)                                        │
│         ↓                                                            │
│  ┌─────────────────┐                                                 │
│  │  Face Detection │                                                 │
│  │  & Cropping     │                                                 │
│  └────────┬────────┘                                                 │
│           ↓                                                          │
│  ┌─────────────────┐    ┌─────────────────┐                         │
│  │  Face Region    │    │ Character Sheet │                         │
│  │  (Expanded)     │    │ (Identity Data) │                         │
│  └────────┬────────┘    └────────┬────────┘                         │
│           ↓                      ↓                                   │
│      ┌───────────────────────────────┐                              │
│      │    Regenerate Face Region     │                              │
│      │    (Identity-focused prompt)  │                              │
│      └────────────────┬──────────────┘                              │
│                       ↓                                              │
│      ┌───────────────────────────────┐                              │
│      │    Blend Back into Original   │                              │
│      │    (Feathered edge blending)  │                              │
│      └────────────────┬──────────────┘                              │
│                       ↓                                              │
│      ┌───────────────────────────────┐                              │
│      │    Recompute IDR Score        │                              │
│      │    (Should be higher now)     │                              │
│      └───────────────────────────────┘                              │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
"""

import io
import numpy as np
from typing import Tuple, Optional
from app.core.config import settings

# Try to import image processing libraries
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class FaceRefiner:
    """
    Handles face refinement for improved identity consistency.
    
    When a generated image has low IDR (Identity Retention) score,
    this refiner:
    1. Extracts the face region
    2. Regenerates it with identity-focused prompts
    3. Blends it back into the original image
    """
    
    def __init__(self):
        from app.services.identity import IdentityService
        from app.services.storage import StorageService
        from app.services.gemini_image import GeminiImageService
        
        self.storage_service = StorageService()
        self.identity_service = IdentityService(storage_service=self.storage_service)
        self.gemini_service = GeminiImageService(storage_service=self.storage_service)
        self.max_refinement_attempts = 2
        self.idr_threshold = settings.IDR_THRESHOLD
    
    async def refine_if_needed(
        self,
        image_bytes: bytes,
        character_id: str,
        character_data: dict,
        current_idr: float
    ) -> Tuple[bytes, float, bool]:
        """
        Check if image needs refinement and refine if necessary.
        
        Args:
            image_bytes: The generated image
            character_id: Character ID for identity lookup
            character_data: Character metadata (face, hair, etc.)
            current_idr: Current IDR score
            
        Returns:
            Tuple of (refined_image_bytes, new_idr_score, was_refined)
        """
        if current_idr >= self.idr_threshold:
            print(f"[OK] IDR {current_idr:.3f} >= threshold {self.idr_threshold}, no refinement needed")
            return image_bytes, current_idr, False
        
        print(f"[WARNING] IDR {current_idr:.3f} < threshold {self.idr_threshold}, attempting refinement...")
        
        for attempt in range(self.max_refinement_attempts):
            print(f"  Refinement attempt {attempt + 1}/{self.max_refinement_attempts}")
            
            refined_bytes, new_idr = await self._refine_face(
                image_bytes=image_bytes,
                character_id=character_id,
                character_data=character_data
            )
            
            if new_idr >= self.idr_threshold:
                print(f"[OK] Refinement successful! IDR: {current_idr:.3f} -> {new_idr:.3f}")
                return refined_bytes, new_idr, True
            
            # Use refined image for next attempt
            image_bytes = refined_bytes
            current_idr = new_idr
        
        print(f"[WARNING] Refinement attempts exhausted. Final IDR: {current_idr:.3f}")
        return image_bytes, current_idr, True
    
    async def _refine_face(
        self,
        image_bytes: bytes,
        character_id: str,
        character_data: dict
    ) -> Tuple[bytes, float]:
        """
        Perform face refinement on an image.
        
        Returns:
            Tuple of (refined_image_bytes, new_idr_score)
        """
        try:
            # Convert bytes to numpy array
            image = self._bytes_to_image(image_bytes)
            if image is None:
                return image_bytes, 0.0
            
            # Detect and crop face region
            face_region, bbox = self._extract_face_region(image, expand_ratio=1.8)
            if face_region is None:
                print("  Could not detect face for refinement")
                return image_bytes, 0.0
            
            # Build identity-focused prompt
            identity_prompt = self._build_identity_prompt(character_data)
            
            # Convert face region to bytes
            face_bytes = self._image_to_bytes(face_region)
            
            # Regenerate face using Gemini
            refined_face_bytes = await self.gemini_service.refine_face(
                original_image_bytes=image_bytes,
                face_region=face_bytes,
                character_prompt=identity_prompt
            )
            
            # Convert refined face to numpy
            refined_face = self._bytes_to_image(refined_face_bytes)
            if refined_face is None:
                return image_bytes, 0.0
            
            # Resize refined face to match original face region size
            target_size = (bbox[2] - bbox[0], bbox[3] - bbox[1])
            refined_face = self._resize_image(refined_face, target_size)
            
            # Blend refined face back into original
            blended = self._blend_face(image, refined_face, bbox)
            
            # Convert back to bytes
            blended_bytes = self._image_to_bytes(blended)
            
            # Compute new IDR
            from app.services.vectordb import VectorDBService
            vectordb = VectorDBService()
            semantic = await vectordb.query_semantic(character_id)
            
            if semantic is not None:
                # Save temp file for IDR computation
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
                    f.write(blended_bytes)
                    temp_path = f.name
                
                new_idr = await self.identity_service.compute_idr(semantic, temp_path)
                
                import os
                os.unlink(temp_path)
            else:
                new_idr = 0.5  # Default if no semantic embedding
            
            return blended_bytes, new_idr
            
        except Exception as e:
            print(f"  Refinement error: {e}")
            return image_bytes, 0.0
    
    def _build_identity_prompt(self, character_data: dict) -> str:
        """Build an identity-focused prompt from character data."""
        parts = []
        
        if character_data.get("name"):
            parts.append(f"Character: {character_data['name']}")
        
        if character_data.get("face"):
            parts.append(f"Face: {character_data['face']}")
        
        if character_data.get("eyes"):
            parts.append(f"Eyes: {character_data['eyes']}")
        
        if character_data.get("hair"):
            parts.append(f"Hair: {character_data['hair']}")
        
        if character_data.get("distinctives"):
            parts.append(f"Distinctive marks: {character_data['distinctives']}")
        
        if character_data.get("age_range"):
            parts.append(f"Age: {character_data['age_range']}")
        
        return "\n".join(parts) if parts else "Portrait of a person"
    
    def _bytes_to_image(self, image_bytes: bytes) -> Optional[np.ndarray]:
        """Convert image bytes to numpy array."""
        if CV2_AVAILABLE:
            nparr = np.frombuffer(image_bytes, np.uint8)
            return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        elif PIL_AVAILABLE:
            img = Image.open(io.BytesIO(image_bytes))
            return np.array(img.convert('RGB'))[:, :, ::-1]  # RGB to BGR
        return None
    
    def _image_to_bytes(self, image: np.ndarray, format: str = 'jpg') -> bytes:
        """Convert numpy array to image bytes."""
        if CV2_AVAILABLE:
            if format == 'jpg':
                _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            else:
                _, buffer = cv2.imencode('.png', image)
            return buffer.tobytes()
        elif PIL_AVAILABLE:
            # Convert BGR to RGB
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = image[:, :, ::-1]
            img = Image.fromarray(image)
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG' if format == 'jpg' else 'PNG', quality=95)
            return buffer.getvalue()
        return b''
    
    def _extract_face_region(
        self, 
        image: np.ndarray, 
        expand_ratio: float = 1.5
    ) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
        """Extract face region with optional expansion."""
        face_crop = self.identity_service.crop_face(image, expand_ratio)
        bbox = self.identity_service.get_face_bbox(image)
        
        if face_crop is None or bbox is None:
            return None, None
        
        # Expand bbox
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        
        new_w, new_h = int(w * expand_ratio), int(h * expand_ratio)
        new_x1 = max(0, cx - new_w // 2)
        new_y1 = max(0, cy - new_h // 2)
        new_x2 = min(image.shape[1], cx + new_w // 2)
        new_y2 = min(image.shape[0], cy + new_h // 2)
        
        return face_crop, (new_x1, new_y1, new_x2, new_y2)
    
    def _resize_image(self, image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        """Resize image to target size."""
        if CV2_AVAILABLE:
            return cv2.resize(image, size, interpolation=cv2.INTER_LANCZOS4)
        elif PIL_AVAILABLE:
            img = Image.fromarray(image[:, :, ::-1])  # BGR to RGB
            img = img.resize(size, Image.LANCZOS)
            return np.array(img)[:, :, ::-1]  # RGB to BGR
        return image
    
    def _blend_face(
        self, 
        original: np.ndarray, 
        refined_face: np.ndarray, 
        bbox: Tuple[int, int, int, int],
        feather_size: int = 20
    ) -> np.ndarray:
        """
        Blend refined face back into original image with feathered edges.
        
        Uses alpha blending with a gradient mask for smooth transitions.
        """
        x1, y1, x2, y2 = bbox
        target_h, target_w = y2 - y1, x2 - x1
        
        # Resize refined face if needed
        if refined_face.shape[:2] != (target_h, target_w):
            refined_face = self._resize_image(refined_face, (target_w, target_h))
        
        # Create feathered mask
        mask = self._create_feathered_mask(target_w, target_h, feather_size)
        
        # Ensure 3-channel mask for color blending
        if len(mask.shape) == 2:
            mask = np.stack([mask] * 3, axis=-1)
        
        # Extract original face region
        original_region = original[y1:y2, x1:x2].astype(np.float32)
        refined_float = refined_face.astype(np.float32)
        
        # Blend
        blended_region = (mask * refined_float + (1 - mask) * original_region).astype(np.uint8)
        
        # Copy blended region back
        result = original.copy()
        result[y1:y2, x1:x2] = blended_region
        
        return result
    
    def _create_feathered_mask(
        self, 
        width: int, 
        height: int, 
        feather_size: int
    ) -> np.ndarray:
        """Create a feathered edge mask for smooth blending."""
        mask = np.ones((height, width), dtype=np.float32)
        
        # Create gradient at edges
        for i in range(feather_size):
            alpha = i / feather_size
            # Top
            if i < height:
                mask[i, :] = min(mask[i, 0], alpha)
            # Bottom
            if height - 1 - i >= 0:
                mask[height - 1 - i, :] = min(mask[height - 1 - i, 0], alpha)
            # Left
            if i < width:
                mask[:, i] = np.minimum(mask[:, i], alpha)
            # Right
            if width - 1 - i >= 0:
                mask[:, width - 1 - i] = np.minimum(mask[:, width - 1 - i], alpha)
        
        return mask


async def refine_face_task(
    job_id: str,
    image_url: str,
    character_id: str,
    character_data: dict,
    current_idr: float
) -> dict:
    """
    Worker task for face refinement.
    
    Called when a generated image has low IDR score.
    
    Returns:
        Dict with refined_url, new_idr, was_refined
    """
    from app.services.storage import StorageService
    
    try:
        refiner = FaceRefiner()
        storage = StorageService()
        
        # Load image
        image_bytes = await storage.download_file(image_url)
        if image_bytes is None:
            return {"error": "Could not load image"}
        
        # Attempt refinement
        refined_bytes, new_idr, was_refined = await refiner.refine_if_needed(
            image_bytes=image_bytes,
            character_id=character_id,
            character_data=character_data,
            current_idr=current_idr
        )
        
        if was_refined:
            # Save refined image
            refined_path = image_url.replace('/result.', '/result_refined.')
            refined_url = await storage.upload_bytes(refined_bytes, refined_path)
            
            return {
                "success": True,
                "refined_url": refined_url,
                "original_idr": current_idr,
                "new_idr": new_idr,
                "was_refined": True
            }
        else:
            return {
                "success": True,
                "refined_url": image_url,
                "original_idr": current_idr,
                "new_idr": current_idr,
                "was_refined": False
            }
            
    except Exception as e:
        print(f"[ERROR] Face refinement task failed: {e}")
        return {"error": str(e)}
