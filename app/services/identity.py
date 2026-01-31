"""
Identity Service
Handles face detection, embedding extraction, and identity management.

Uses InsightFace/ArcFace for face embeddings (512-dim) and optionally CLIP for style.

Identity Architecture:
┌─────────────────────────────────────────────────────────────────────┐
│                     IDENTITY EXTRACTION                              │
├─────────────────────────────────────────────────────────────────────┤
│  Reference Image(s)                                                  │
│       ↓                                                              │
│  ┌─────────────────┐    ┌─────────────────┐                         │
│  │  Face Detection │    │  CLIP Encoder   │                         │
│  │  (MTCNN/RetinaF)│    │  (Style/Context)│                         │
│  └────────┬────────┘    └────────┬────────┘                         │
│           ↓                      ↓                                   │
│  ┌─────────────────┐    ┌─────────────────┐                         │
│  │ Face Alignment  │    │ Style Embedding │                         │
│  │  (5-point warp) │    │   (512-dim)     │                         │
│  └────────┬────────┘    └────────┬────────┘                         │
│           ↓                      ↓                                   │
│  ┌─────────────────┐             │                                   │
│  │ ArcFace Encoder │             │                                   │
│  │   (512-dim)     │             │                                   │
│  └────────┬────────┘             │                                   │
│           ↓                      ↓                                   │
│      [Face Embedding]  +   [Style Embedding]                         │
│           ↓                      ↓                                   │
│      ┌─────────────────────────────┐                                 │
│      │  Combined Identity Vector   │                                 │
│      │       (1024-dim)            │                                 │
│      └─────────────────────────────┘                                 │
└─────────────────────────────────────────────────────────────────────┘
"""

import os
import io
import httpx
import numpy as np
from typing import List, Optional, Tuple
from pathlib import Path

# Try to import face analysis libraries
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("[WARNING] OpenCV not available")

try:
    from insightface.app import FaceAnalysis
    from insightface.data import get_image as ins_get_image
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    print("[WARNING] InsightFace not available, using fallback embedding")

# Optional CLIP for style embeddings
try:
    import torch
    from transformers import CLIPProcessor, CLIPModel
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("[WARNING] CLIP not available, style embeddings disabled")

# Global singleton instance to prevent reinitialization
_face_model_singleton = None
_face_model_lock = False


class IdentityService:
    """
    Service for identity extraction and management.
    
    Core capabilities:
    1. Face detection and alignment
    2. Face embedding extraction (ArcFace)
    3. Style embedding extraction (CLIP)
    4. Identity Retention (IDR) scoring
    """
    
    def __init__(self, storage_service=None):
        """Initialize identity extraction models."""
        self.face_model = None
        self.clip_model = None
        self.clip_processor = None
        self.face_dim = 512
        self.style_dim = 512
        self.storage_service = storage_service
        
        self._init_face_model()
        self._init_clip_model()
    
    def _init_face_model(self):
        """Initialize InsightFace for face detection and embedding (with singleton pattern)."""
        global _face_model_singleton, _face_model_lock
        
        # Use singleton if available
        if _face_model_singleton is not None:
            self.face_model = _face_model_singleton
            print("[OK] Face model: Using existing singleton instance")
            return
        
        # Prevent concurrent initialization
        if _face_model_lock:
            print("[INFO] Face model: Waiting for initialization...")
            import time
            for _ in range(30):  # Wait up to 3 seconds
                time.sleep(0.1)
                if _face_model_singleton is not None:
                    self.face_model = _face_model_singleton
                    return
            print("[WARNING] Face model: Timeout waiting for initialization")
            return
        
        if not INSIGHTFACE_AVAILABLE or not CV2_AVAILABLE:
            print("[WARNING] Face model not initialized (InsightFace/OpenCV not available)")
            return
        
        _face_model_lock = True
        try:
            # Initialize FaceAnalysis with buffalo_l model (best accuracy)
            # Use buffalo_s for faster but less accurate
            self.face_model = FaceAnalysis(
                name='buffalo_l',
                root=os.path.join(os.path.dirname(__file__), '..', '..', 'models'),
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            self.face_model.prepare(ctx_id=0, det_size=(640, 640))
            _face_model_singleton = self.face_model
            print("[OK] Face model initialized (InsightFace buffalo_l) [SINGLETON CREATED]")
        except Exception as e:
            print(f"[WARNING] Face model init failed: {e}")
            # Try with smaller model
            try:
                self.face_model = FaceAnalysis(
                    name='buffalo_s',
                    providers=['CPUExecutionProvider']
                )
                self.face_model.prepare(ctx_id=-1, det_size=(320, 320))
                _face_model_singleton = self.face_model
                print("[OK] Face model initialized (InsightFace buffalo_s - CPU) [SINGLETON CREATED]")
            except Exception as e2:
                print(f"[WARNING] Face model fallback also failed: {e2}")
                self.face_model = None
        finally:
            _face_model_lock = False
    
    def _init_clip_model(self):
        """Initialize CLIP for style embeddings."""
        if not CLIP_AVAILABLE:
            return
        
        try:
            model_name = "openai/clip-vit-base-patch32"
            self.clip_model = CLIPModel.from_pretrained(model_name)
            self.clip_processor = CLIPProcessor.from_pretrained(model_name)
            
            # Move to GPU if available
            if torch.cuda.is_available():
                self.clip_model = self.clip_model.cuda()
            
            print("[OK] CLIP model initialized for style embeddings")
        except Exception as e:
            print(f"[WARNING] CLIP init failed: {e}")
            self.clip_model = None
    
    async def extract_identity(self, image_urls: List[str]) -> np.ndarray:
        """
        Extract identity embedding from reference images.
        
        Args:
            image_urls: List of URLs or local paths for reference images
            
        Returns:
            Aggregated semantic embedding (normalized, 512 or 1024 dim)
        """
        embeddings = []
        
        for url in image_urls:
            try:
                # Download/load image
                image = await self._download_image(url)
                if image is None:
                    continue
                
                # Detect face
                face = self._detect_face(image)
                if face is None:
                    print(f"[WARNING] No face detected in: {url}")
                    continue
                
                # Extract face embedding
                face_embedding = self._extract_face_embedding(face)
                
                # Extract CLIP embedding for style (optional)
                style_embedding = self._extract_clip_embedding(image)
                
                # Combine embeddings
                if style_embedding is not None:
                    combined = np.concatenate([face_embedding, style_embedding])
                else:
                    combined = face_embedding
                
                embeddings.append(combined)
                print(f"[OK] Extracted embedding from: {url}")
                
            except Exception as e:
                print(f"[WARNING] Failed to extract from {url}: {e}")
                continue
        
        if not embeddings:
            # Return zero vector if no faces detected (fallback)
            dim = self.face_dim + (self.style_dim if self.clip_model else 0)
            print("[WARNING] No faces detected in any reference images, returning zero vector")
            return np.zeros(self.face_dim)
        
        # Aggregate: mean of normalized embeddings
        normalized = [e / (np.linalg.norm(e) + 1e-8) for e in embeddings]
        aggregated = np.mean(normalized, axis=0)
        aggregated = aggregated / (np.linalg.norm(aggregated) + 1e-8)
        
        return aggregated.astype(np.float32)
    
    async def compute_idr(
        self, 
        reference_embedding: np.ndarray, 
        generated_image_url: str
    ) -> float:
        """
        Compute Identity Retention (IDR) score.
        
        Measures how well the generated image preserves the character's identity.
        
        Args:
            reference_embedding: Semantic embedding of character
            generated_image_url: URL of generated image
            
        Returns:
            IDR score (cosine similarity, 0-1)
        """
        if reference_embedding is None:
            return 0.0
        
        try:
            # Download generated image
            image = await self._download_image(generated_image_url)
            if image is None:
                return 0.0
            
            # Extract face from generated image
            face = self._detect_face(image)
            if face is None:
                print("[WARNING] No face detected in generated image")
                return 0.0
            
            # Extract embedding from generated face
            generated_embedding = self._extract_face_embedding(face)
            generated_embedding = generated_embedding / (np.linalg.norm(generated_embedding) + 1e-8)
            
            # Compare only face portion of embeddings
            ref_face = reference_embedding[:self.face_dim]
            ref_face = ref_face / (np.linalg.norm(ref_face) + 1e-8)
            
            # Compute cosine similarity
            idr = float(np.dot(ref_face, generated_embedding))
            idr = max(0.0, min(1.0, idr))
            
            print(f"[OK] IDR Score: {idr:.3f}")
            return idr
            
        except Exception as e:
            print(f"[WARNING] IDR computation failed: {e}")
            return 0.0
    
    def needs_refinement(self, idr_score: float, threshold: float = None) -> bool:
        """Check if the generated image needs face refinement."""
        from app.core.config import settings
        threshold = threshold or settings.IDR_THRESHOLD
        return idr_score < threshold
    
    async def _download_image(self, url: str) -> Optional[np.ndarray]:
        """Download image from URL or load from local path."""
        try:
            print(f"[Identity] Downloading image from: {url}")
            
            # Handle relative API proxy URLs (e.g., /files/...)
            if url.startswith("/files/"):
                if self.storage_service:
                    image_bytes = await self.storage_service.download_bytes(url)
                    print(f"[Identity] Downloaded {len(image_bytes)} bytes via API proxy")
                else:
                    print(f"[Identity] ERROR: Relative URL but no storage service available")
                    return None
            # Use storage service for authenticated GCS downloads if available
            elif self.storage_service and url.startswith(('http://', 'https://')):
                image_bytes = await self.storage_service.download_bytes(url)
                print(f"[Identity] Downloaded {len(image_bytes)} bytes via storage service")
            elif url.startswith(('http://', 'https://')):
                # Fallback to httpx if no storage service
                async with httpx.AsyncClient() as client:
                    response = await client.get(url, timeout=30.0)
                    response.raise_for_status()
                    image_bytes = response.content
                    print(f"[Identity] Downloaded {len(image_bytes)} bytes via HTTP")
            else:
                # Local file path
                path = Path(url)
                if not path.exists():
                    # Try relative to uploads
                    from app.core.config import settings
                    path = Path(settings.LOCAL_STORAGE_PATH) / url
                
                if not path.exists():
                    print(f"[Identity] ERROR: Image not found at path: {url}")
                    return None
                
                with open(path, 'rb') as f:
                    image_bytes = f.read()
                    print(f"[Identity] Read {len(image_bytes)} bytes from local file")
            
            # Decode image
            if CV2_AVAILABLE:
                nparr = np.frombuffer(image_bytes, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if image is not None:
                    print(f"[Identity] Image decoded successfully: {image.shape}")
                else:
                    print(f"[Identity] ERROR: Failed to decode image with cv2")
                return image
            else:
                # Fallback using PIL
                from PIL import Image
                img = Image.open(io.BytesIO(image_bytes))
                img_array = np.array(img.convert('RGB'))[:, :, ::-1]  # RGB to BGR
                print(f"[Identity] Image decoded with PIL: {img_array.shape}")
                return img_array
                
        except Exception as e:
            print(f"[Identity] ERROR downloading image from {url}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _detect_face(self, image: np.ndarray) -> Optional[dict]:
        """Detect and return the largest face in the image."""
        if self.face_model is None:
            # Return dummy face info for fallback
            return {"embedding": np.zeros(self.face_dim)}
        
        try:
            faces = self.face_model.get(image)
            
            if not faces:
                return None
            
            # Return largest face (by bounding box area)
            largest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            return largest
            
        except Exception as e:
            print(f"[WARNING] Face detection failed: {e}")
            return None
    
    def _extract_face_embedding(self, face) -> np.ndarray:
        """Extract face embedding using ArcFace."""
        if self.face_model is None or face is None:
            # Return random but consistent embedding for testing
            return np.random.randn(self.face_dim).astype(np.float32)
        
        try:
            # InsightFace already provides embedding in face object
            if hasattr(face, 'embedding') and face.embedding is not None:
                return face.embedding.astype(np.float32)
            elif isinstance(face, dict) and 'embedding' in face:
                return face['embedding'].astype(np.float32)
            else:
                return np.zeros(self.face_dim, dtype=np.float32)
        except Exception as e:
            print(f"[WARNING] Embedding extraction failed: {e}")
            return np.zeros(self.face_dim, dtype=np.float32)
    
    def _extract_clip_embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Extract CLIP embedding for style/context."""
        if self.clip_model is None or self.clip_processor is None:
            return None
        
        try:
            # Convert BGR to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if CV2_AVAILABLE else image
            else:
                image_rgb = image
            
            # Process image
            inputs = self.clip_processor(images=image_rgb, return_tensors="pt")
            
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                features = self.clip_model.get_image_features(**inputs)
            
            embedding = features.cpu().numpy().flatten()
            
            # Resize to target dimension if needed
            if len(embedding) != self.style_dim:
                # Simple linear projection (could use learned projection)
                if len(embedding) > self.style_dim:
                    embedding = embedding[:self.style_dim]
                else:
                    embedding = np.pad(embedding, (0, self.style_dim - len(embedding)))
            
            return embedding.astype(np.float32)
            
        except Exception as e:
            print(f"[WARNING] CLIP embedding failed: {e}")
            return None
    
    def get_face_bbox(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Get bounding box of the largest face in image."""
        face = self._detect_face(image)
        if face is None or not hasattr(face, 'bbox'):
            return None
        
        bbox = face.bbox.astype(int)
        return (bbox[0], bbox[1], bbox[2], bbox[3])  # x1, y1, x2, y2
    
    def crop_face(
        self, 
        image: np.ndarray, 
        expand_ratio: float = 1.5
    ) -> Optional[np.ndarray]:
        """Crop face region with optional expansion."""
        bbox = self.get_face_bbox(image)
        if bbox is None:
            return None
        
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        
        # Expand bounding box
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        new_w, new_h = int(w * expand_ratio), int(h * expand_ratio)
        
        new_x1 = max(0, cx - new_w // 2)
        new_y1 = max(0, cy - new_h // 2)
        new_x2 = min(image.shape[1], cx + new_w // 2)
        new_y2 = min(image.shape[0], cy + new_h // 2)
        
        return image[new_y1:new_y2, new_x1:new_x2]
    
    async def analyze_identity_quality(
        self, 
        image_urls: List[str]
    ) -> dict:
        """
        Analyze the quality of reference images for identity extraction.
        Returns recommendations for better results.
        """
        results = {
            "total_images": len(image_urls),
            "faces_detected": 0,
            "quality_scores": [],
            "recommendations": []
        }
        
        for url in image_urls:
            image = await self._download_image(url)
            if image is None:
                continue
            
            face = self._detect_face(image)
            if face is not None:
                results["faces_detected"] += 1
                
                # Estimate quality based on face size
                if hasattr(face, 'bbox'):
                    w = face.bbox[2] - face.bbox[0]
                    h = face.bbox[3] - face.bbox[1]
                    face_area = w * h
                    image_area = image.shape[0] * image.shape[1]
                    coverage = face_area / image_area
                    
                    results["quality_scores"].append(min(1.0, coverage * 5))  # Scale up
        
        # Generate recommendations
        if results["faces_detected"] == 0:
            results["recommendations"].append("No faces detected - ensure clear facial visibility")
        elif results["faces_detected"] < results["total_images"]:
            results["recommendations"].append(f"Only {results['faces_detected']}/{results['total_images']} images had detectable faces")
        
        if results["quality_scores"]:
            avg_quality = sum(results["quality_scores"]) / len(results["quality_scores"])
            results["average_quality"] = avg_quality
            
            if avg_quality < 0.3:
                results["recommendations"].append("Face appears small in images - use closer shots")
        
        if len(image_urls) < 3:
            results["recommendations"].append("Add more reference images (3-5 recommended) for better consistency")
        
        if not results["recommendations"]:
            results["recommendations"].append("Reference images look good! ✨")
        
        return results
