"""
Gemini "Nano Banana" Image Generation Service
Uses native Gemini image generation models (gemini-2.5-flash-image).
Also provides vision analysis for state extraction.
Documentation: https://ai.google.dev/gemini-api/docs/image-generation
"""

import base64
import io
import httpx
import numpy as np
import time
from typing import Optional, Dict, Any, List
from google import genai
from google.genai import types
from app.core.config import settings


class GeminiImageService:
    """Service for image generation and vision analysis using Gemini models."""
    
    def __init__(self, storage_service=None):
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)
        self.storage_service = storage_service
        # Use model from config, fallback to gemini-2.5-flash-image
        self.model_name = settings.GEMINI_MODEL if settings.GEMINI_MODEL else "gemini-2.5-flash-image"
        # Model for character consistency (when using reference images)
        self.character_model = getattr(settings, 'GEMINI_MODEL_CHARACTER', self.model_name)
        # Vision model for state extraction
        self.vision_model = "gemini-2.5-flash"
        print(f"[GeminiImageService] Initialized with model: {self.model_name}") 
        print(f"[GeminiImageService] Character model: {self.character_model}") 
    
    async def _load_image_bytes(self, image_url: str) -> bytes:
        """Load image bytes from URL or local file path."""
        if not image_url:
            return None
            
        try:
            print(f"[Gemini] Loading image from: {image_url}")
            
            # Handle API proxy URLs (e.g., /files/...)
            if image_url.startswith("/files/"):
                if self.storage_service:
                    image_bytes = await self.storage_service.download_bytes(image_url)
                    print(f"[Gemini] Loaded {len(image_bytes)} bytes via API proxy")
                    return image_bytes
                else:
                    print(f"[Gemini] ERROR: Relative URL but no storage service available")
                    return None
            # Handle local file paths
            elif image_url.startswith("file://"):
                file_path = image_url.replace("file://", "")
                with open(file_path, "rb") as f:
                    image_bytes = f.read()
                    print(f"[Gemini] Read {len(image_bytes)} bytes from local file")
                    return image_bytes
            elif image_url.startswith("/") or (len(image_url) > 1 and image_url[1] == ":"):
                # Absolute path (Unix or Windows)
                with open(image_url, "rb") as f:
                    image_bytes = f.read()
                    print(f"[Gemini] Read {len(image_bytes)} bytes from absolute path")
                    return image_bytes
            else:
                # HTTP URL - use storage service for authenticated GCS downloads
                if self.storage_service:
                    image_bytes = await self.storage_service.download_bytes(image_url)
                    print(f"[Gemini] Downloaded {len(image_bytes)} bytes via storage service")
                    return image_bytes
                else:
                    # Fallback to httpx if no storage service
                    async with httpx.AsyncClient() as client:
                        response = await client.get(image_url, timeout=30.0)
                        if response.status_code == 200:
                            print(f"[Gemini] Downloaded {len(response.content)} bytes via HTTP")
                            return response.content
                        else:
                            print(f"[Gemini] ERROR: Failed to download image: HTTP {response.status_code}")
                            return None
        except Exception as e:
            print(f"[Gemini] ERROR: Failed to load image from {image_url}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    async def generate(
        self, 
        prompt: str, 
        aspect_ratio: str = "1:1",
        reference_image_url: str = None,
        reference_images: list = None,
        character_data: dict = None,
        semantic_vector: Optional[np.ndarray] = None  # Face embedding for identity verification
    ) -> bytes:
        """
        Generate image using Gemini native image generation.
        
        Args:
            prompt: The text prompt for image generation
            aspect_ratio: Aspect ratio
            reference_image_url: Primary reference image URL
            reference_images: Additional reference images
            character_data: Dict containing explicit character traits (face, distinctives, etc.)
            semantic_vector: Face embedding vector (512-dim) for identity preservation
        """
        try:
            print(f"[Gemini] Generating image with model: {self.model_name}")
            print(f"[Gemini] Prompt: {prompt[:100]}...")
            print(f"[Gemini] Aspect Ratio: {aspect_ratio}")
            print(f"[Gemini] Reference Image: {reference_image_url is not None}")
            
            # Build content list - can include both text and images
            contents = []
            
            # Load and add reference image(s) if provided
            ref_image_bytes = None
            if reference_image_url:
                ref_image_bytes = await self._load_image_bytes(reference_image_url)
                if ref_image_bytes:
                    print(f"[Gemini] [OK] Reference image loaded ({len(ref_image_bytes)} bytes)")
                    contents.append(
                        types.Part.from_bytes(
                            data=ref_image_bytes,
                            mime_type="image/jpeg"
                        )
                    )
            
            # Add additional reference images if provided
            if reference_images:
                for idx, ref_url in enumerate(reference_images[:2]):  # Max 2 additional
                    extra_bytes = await self._load_image_bytes(ref_url)
                    if extra_bytes:
                        print(f"[Gemini] [OK] Additional reference {idx+1} loaded")
                        contents.append(
                            types.Part.from_bytes(
                                data=extra_bytes,
                                mime_type="image/jpeg"
                            )
                        )
            
            # Create the prompt with identity instruction if we have reference images
            if ref_image_bytes:
                # Extract explicit traits to FORCE into the generation
                traits_prompt = ""
                identity_confidence = ""
                
                if character_data:
                    face = character_data.get("face", "")
                    distinctives = character_data.get("distinctives", "")
                    hair = character_data.get("hair", "")
                    eyes = character_data.get("eyes", "")
                    eyebrows = character_data.get("eyebrows", "")
                    build = character_data.get("build", "")
                    skin_tone = character_data.get("skin_tone", "")
                    initial_outfit = character_data.get("initial_outfit", "")
                    accessories = character_data.get("accessories", "")
                    tags = character_data.get("tags", [])
                    
                    traits_list = []
                    # CRITICAL IDENTITY FEATURES - Must be preserved
                    if face and len(face) > 3: traits_list.append(f"FACE: {face}")
                    if eyes and len(eyes) > 3: traits_list.append(f"EYES: {eyes}")
                    if eyebrows and len(eyebrows) > 3: traits_list.append(f"EYEBROWS: {eyebrows}")
                    if hair and len(hair) > 3: traits_list.append(f"HAIR: {hair}")
                    if skin_tone and len(skin_tone) > 3: traits_list.append(f"SKIN TONE: {skin_tone}")
                    if build and len(build) > 3: traits_list.append(f"BUILD: {build}")
                    if distinctives and len(distinctives) > 3: traits_list.append(f"DISTINCTIVE MARKS: {distinctives}")
                    # OUTFIT - Use initial if no scene change
                    if initial_outfit and len(initial_outfit) > 3: traits_list.append(f"OUTFIT: {initial_outfit}")
                    if accessories and len(accessories) > 3: traits_list.append(f"ACCESSORIES: {accessories}")
                    if tags: traits_list.append(f"Style: {', '.join(tags[:5])}")
                    
                    if traits_list:
                        traits_prompt = "\n".join(traits_list)
                    
                    # If we have semantic vector, emphasize identity preservation
                    if semantic_vector is not None:
                        identity_confidence = "\n\n⚠️ CRITICAL: Maintain EXACT facial features, skin tone, and body proportions from reference!"
                        print(f"[Gemini] Using semantic vector for identity preservation (embedding shape: {semantic_vector.shape})")

                # Enhanced prompt that tells Gemini to use the reference for identity + explicit traits
                identity_prompt = f"""REFERENCE-BASED CHARACTER GENERATION

You MUST generate an image of the EXACT SAME PERSON shown in the reference image.

[MANDATORY CHARACTER IDENTITY - DO NOT CHANGE]
{traits_prompt}{identity_confidence}

[SCENE REQUEST]
{prompt}

[GENERATION RULES]
1. The person in the generated image MUST have IDENTICAL features to the reference
2. Same face shape, eye color, hair color, skin tone, body build
3. Keep the same outfit unless the scene explicitly changes it
4. Copy ALL distinctive marks (scars, moles, tattoos) to their exact locations
5. Generate the scene while maintaining 100% character identity"""
                contents.append(identity_prompt)
            else:
                # No reference image, just use the prompt
                contents.append(prompt)
            
            # Configure the generation request - use TEXT and IMAGE for image editing
            # Reference: https://ai.google.dev/gemini-api/docs/image-generation
            response_modalities = ['TEXT', 'IMAGE'] if ref_image_bytes else ['IMAGE']
            
            config = types.GenerateContentConfig(
                response_modalities=response_modalities,
                # ImageConfig removed in newer versions - aspect_ratio now in config params
                # image_config=types.ImageConfig(aspect_ratio=aspect_ratio),
                safety_settings=[
                    types.SafetySetting(
                        category="HARM_CATEGORY_DANGEROUS_CONTENT",
                        threshold="BLOCK_NONE"  # Internal tool - we control prompts
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_HATE_SPEECH",
                        threshold="BLOCK_NONE"  # Internal tool - we control prompts
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_HARASSMENT",
                        threshold="BLOCK_NONE"  # Internal tool - we control prompts
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        threshold="BLOCK_NONE"  # Internal tool - we control prompts
                    ),
                ]
            )
            
            # Use character model when we have reference images for better identity preservation
            # Otherwise use the default model
            model_to_use = self.character_model if ref_image_bytes else self.model_name
            print(f"[Gemini] Using model: {model_to_use}")
            
            # Generate content with reference image(s) + prompt
            # Attempt generation with retry logic for safety blocking
            response = None
            attempt = 1
            max_attempts = 3
            
            while attempt <= max_attempts and response is None:
                try:
                    print(f"[Gemini] Generation attempt {attempt}/{max_attempts}")
                    response = self.client.models.generate_content(
                        model=model_to_use,
                        contents=contents,
                        config=config
                    )
                    print(f"[Gemini] Response received on attempt {attempt}")
                    break  # Success
                    
                except Exception as gen_error:
                    print(f"[Gemini] [WARNING] Attempt {attempt} failed: {gen_error}")
                    
                    if attempt < max_attempts:
                        # Progressive fallback strategy
                        if attempt == 1 and ref_image_bytes:
                            # Attempt 2: Simplify to just reference + basic prompt
                            print(f"[Gemini] Retrying with simplified prompt...")
                            contents = []
                            if ref_image_bytes:
                                contents.append(types.Part.from_bytes(data=ref_image_bytes, mime_type="image/jpeg"))
                            contents.append(f"Generate: {prompt}")
                            
                        elif attempt == 2:
                            # Attempt 3: Remove reference, prompt only
                            print(f"[Gemini] Retrying without reference image...")
                            contents = [prompt]
                            model_to_use = self.model_name  # Use default model
                    else:
                        # All attempts failed
                        raise gen_error
                    
                    attempt += 1
            
            if response is None:
                raise Exception("Failed to generate image after all retry attempts")
            
            print(f"[Gemini] Response received, parsing...")
            
            # Parse response according to official documentation
            # Method 1: Direct parts access (recommended by docs)
            if hasattr(response, 'parts') and response.parts:
                for part in response.parts:
                    if part.text is not None:
                        print(f"[Gemini] Text part: {part.text}")
                    elif part.inline_data is not None:
                        print(f"[Gemini] [OK] Image generated successfully!")
                        # The inline_data.data is already base64 decoded bytes
                        return part.inline_data.data
            
            # Method 2: Check candidates (fallback)
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                
                # Log finish reason and safety ratings
                print(f"[Gemini] Finish Reason: {candidate.finish_reason}")
                if hasattr(candidate, 'safety_ratings') and candidate.safety_ratings:
                    for rating in candidate.safety_ratings:
                        # Highlight safety blocks
                        if rating.probability in ["HIGH", "MEDIUM"]:
                            print(f"[Gemini] [SAFETY BLOCK] {rating.category}: {rating.probability}")
                        else:
                            print(f"[Gemini] Safety - {rating.category}: {rating.probability}")
                
                # Try to extract image from candidate content
                if hasattr(candidate, 'content') and candidate.content:
                    for part in candidate.content.parts:
                        if part.inline_data is not None:
                            print(f"[Gemini] [OK] Image found in candidate!")
                            return part.inline_data.data
            
            # If we reach here, no image was generated
            block_reason = "Unknown"
            error_details = []
            
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                block_reason = candidate.finish_reason
                
                # Collect safety ratings for debugging
                if hasattr(candidate, 'safety_ratings') and candidate.safety_ratings:
                    for rating in candidate.safety_ratings:
                        error_details.append(f"{rating.category}={rating.probability}")
            
            error_msg = f"No image generated. Finish Reason: {block_reason}"
            if error_details:
                error_msg += f" | Safety: {', '.join(error_details)}"
                
            raise Exception(error_msg)
            
        except Exception as e:
            print(f"[Gemini] [ERROR] Error: {str(e)}")
            raise Exception(f"Gemini Image Generation Failed: {str(e)}")
    
    async def analyze_image(
        self, 
        image_url: str = None,
        image_bytes: bytes = None,
        analysis_prompt: str = None
    ) -> Dict[str, Any]:
        """
        Analyze an image using Gemini Vision to extract state information.
        
        This is used for episodic state extraction after image generation.
        
        Args:
            image_url: URL or local path to the image
            image_bytes: Raw image bytes (alternative to URL)
            analysis_prompt: Custom analysis prompt
            
        Returns:
            Dict with extracted state: clothing, physical_state, props, pose, environment, tags
        """
        try:
            print(f"[Gemini Vision] Analyzing image...")
            
            # Load image
            if image_bytes is None and image_url:
                image_bytes = await self._load_image(image_url)
            
            if image_bytes is None:
                print(f"[Gemini Vision] [ERROR] Could not load image bytes")
                return {"error": "Could not load image"}
            
            # Detect mime type
            mime_type = "image/jpeg"
            if image_bytes.startswith(b'\x89PNG\r\n\x1a\n'):
                mime_type = "image/png"
            elif image_bytes.startswith(b'\xff\xd8'):
                mime_type = "image/jpeg"
            elif image_bytes.startswith(b'RIFF') and image_bytes[8:12] == b'WEBP':
                mime_type = "image/webp"
            
            # Default analysis prompt
            if analysis_prompt is None:
                analysis_prompt = """Analyze this image of a character and extract the following details in JSON format.
This is for a fictional story database. Be purely descriptive.

{
    "clothing": ["list", "of", "clothing", "items", "colors", "materials"],
    "physical_state": ["visual_conditions", "e.g.", "wet", "dirty", "injured", "glowing"],
    "props": ["objects", "held", "or", "nearby"],
    "pose": "concise description of pose and action",
    "environment": "concise setting description",
    "tags": ["5-10", "descriptive", "tags", "visual_style", "mood"]
}

Return ONLY the valid JSON object."""
            
            # Encode image to base64
            image_b64 = base64.b64encode(image_bytes).decode('utf-8')
            
            # Configure safety settings to avoid blocking harmless analysis
            # We are analyzing our own generated images, so we can be permissive
            safety_settings = [
                types.SafetySetting(
                    category="HARM_CATEGORY_HARASSMENT",
                    threshold="BLOCK_NONE",
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_HATE_SPEECH",
                    threshold="BLOCK_NONE",
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    threshold="BLOCK_NONE",
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT",
                    threshold="BLOCK_NONE",
                ),
            ]
            
            # Call Gemini Vision with retry logic for rate limiting
            # Note: inline_data is deprecated in some versions, sticking to it for now
            # but wrapping in correct Part structure
            max_retries = 3
            retry_delay = 2  # Start with 2 seconds
            
            for attempt in range(max_retries):
                try:
                    response = self.client.models.generate_content(
                        model=self.vision_model,
                        contents=[
                            types.Content(
                                parts=[
                                    types.Part(text=analysis_prompt),
                                    types.Part(
                                        inline_data=types.Blob(
                                            mime_type=mime_type,
                                            data=image_b64
                                        )
                                    )
                                ]
                            )
                        ],
                        config=types.GenerateContentConfig(
                            safety_settings=safety_settings,
                            temperature=0.4, # Lower temperature for more deterministic JSON
                            response_mime_type="application/json" # Force JSON mode
                        )
                    )
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    error_str = str(e)
                    # Check if it's a rate limit error
                    if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                        if attempt < max_retries - 1:
                            print(f"[Gemini Vision] [WARNING] Rate limited, retrying in {retry_delay}s (attempt {attempt + 1}/{max_retries})...")
                            time.sleep(retry_delay)
                            retry_delay *= 2  # Exponential backoff
                            continue
                        else:
                            print(f"[Gemini Vision] [ERROR] Rate limit exhausted after {max_retries} attempts")
                            raise  # Re-raise after all retries
                    else:
                        # Non-rate-limit error, raise immediately
                        raise
            
            # Check finish reason
            # if response.candidates and response.candidates[0].finish_reason != "STOP":
            #    print(f"[Gemini Vision] [WARNING] Finish reason: {response.candidates[0].finish_reason}")
            
            # Parse response
            text = response.text if hasattr(response, 'text') else ""
            
            if not text:
               print(f"[Gemini Vision] [ERROR] Empty response from model")
               return {"error": "Empty response from vision model"}

            # Parse JSON
            import json
            try:
                # Clean up markdown code blocks if present (though response_mime_type should prevent this)
                cleaned_text = text.replace("```json", "").replace("```", "").strip()
                result = json.loads(cleaned_text)
                
                # If analysis_prompt was provided (character traits), return raw result
                if analysis_prompt and "face" in analysis_prompt.lower():
                    print(f"[Gemini Vision] [OK] Extracted character traits")
                    return result
                
                # Otherwise normalize for state extraction
                normalized = {
                    "clothing": result.get("clothing", []),
                    "physical_state": result.get("physical_state", []),
                    "props": result.get("props", []),
                    "pose": result.get("pose", "Unknown"),
                    "environment": result.get("environment", "Unknown"),
                    "tags": result.get("tags", [])
                }
                
                print(f"[Gemini Vision] [OK] Extracted state: {len(normalized['tags'])} tags")
                return normalized
                
            except json.JSONDecodeError as e:
                print(f"[Gemini Vision] [ERROR] JSON Parse Error: {e}")
                print(f"Raw text: {text}")
                return {"error": "Failed to parse JSON response"}
            
        except Exception as e:
            print(f"[Gemini Vision] [ERROR] Exception: {str(e)}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}
    
    async def _load_image(self, url: str) -> Optional[bytes]:
        """Load image from URL or local path."""
        try:
            if url.startswith(('http://', 'https://')):
                async with httpx.AsyncClient() as client:
                    response = await client.get(url, timeout=30.0)
                    response.raise_for_status()
                    return response.content
            else:
                # Local file
                from pathlib import Path
                path = Path(url)
                if not path.exists():
                    path = Path(settings.LOCAL_STORAGE_PATH) / url
                
                if path.exists():
                    with open(path, 'rb') as f:
                        return f.read()
                
                return None
        except Exception as e:
            print(f"[Gemini] Failed to load image: {e}")
            return None
    
    async def refine_face(
        self,
        original_image_bytes: bytes,
        face_region: bytes,
        character_prompt: str
    ) -> bytes:
        """
        Refine the face region for better identity consistency.
        
        This is called when IDR score is below threshold.
        
        Args:
            original_image_bytes: The full generated image
            face_region: Cropped face region
            character_prompt: Identity-focused prompt for the character
            
        Returns:
            Refined face region bytes
        """
        try:
            print("[Gemini] Refining face region...")
            
            # Create a face-focused prompt
            face_prompt = f"""Generate a detailed, consistent portrait that EXACTLY matches this face:
{character_prompt}

Focus on:
- Precise facial features
- Exact eye color and shape
- Hair style and color
- Any distinctive marks

Style: Photorealistic, high detail, 8K, sharp focus on face"""
            
            # Save face region temporarily to pass as reference
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
                f.write(face_region)
                face_temp_path = f.name
            
            try:
                # Generate refined face WITH the original face as reference for identity!
                refined_bytes = await self.generate(
                    prompt=face_prompt,
                    aspect_ratio="1:1",  # Square for face
                    reference_image_url=face_temp_path  # CRITICAL: Pass face as reference!
                )
            finally:
                os.unlink(face_temp_path)
            
            print("[Gemini] [OK] Face refined successfully")
            return refined_bytes
            
        except Exception as e:
            print(f"[Gemini] Face refinement failed: {e}")
            # Return original on failure
            return face_region
