"""
Character Video Generation API
Complete pipeline for generating videos using character identity and embeddings.

WORKFLOW:
1. Select character by character_id
2. Fetch character's canonical image (base_image_url)
3. Fetch character metadata and embeddings
4. Generate video using Veo with character image as first frame
5. Upload video to GCS/S3/Local storage
6. Return video URL
"""

import uuid
import time
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from google import genai
from google.genai import types

from app.api.deps import get_db
from app.models.character import Character
from app.schemas.video import VideoGenerateRequest, VideoGenerateResponse
from app.services.storage import StorageService
from app.core.config import settings

router = APIRouter()


def _wait_for_operation(client, operation, message: str):
    """Poll operation until complete."""
    while not operation.done:
        print(message)
        time.sleep(10)
        operation = client.operations.get(operation)
    return operation


@router.post("/generate-character-video", response_model=VideoGenerateResponse, status_code=status.HTTP_202_ACCEPTED)
async def generate_character_video(
    request: VideoGenerateRequest,
    db: Session = Depends(get_db),
):
    """
    Generate a video using a character's canonical image and identity embeddings.
    
    PIPELINE:
    1. Validate character exists and has consent
    2. Fetch character's canonical image (base_image_url)
    3. Fetch character metadata (face, hair, eyes, distinctives)
    4. Generate enhanced prompt using character traits
    5. Generate video using Veo 3.0 with character image as first frame
    6. Download video from Veo
    7. Upload to storage (GCS/S3/Local)
    8. Return video URL
    
    Request Body:
    {
        "character_id": "char_abc123",
        "prompt": "walking through a park, smiling",
        "options": {
            "aspect_ratio": "16:9",
            "duration_seconds": 8,
            "negative_prompt": "cartoon, anime"
        }
    }
    """
    print(f"\n{'='*60}")
    print(f"[Character Video] Generate request for: {request.character_id}")
    print(f"[Character Video] Prompt: {request.prompt}")
    print(f"{'='*60}\n")
    
    # STEP 1: Validate character exists
    character = db.query(Character).filter(Character.id == request.character_id).first()
    
    if not character:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Character '{request.character_id}' not found"
        )
    
    # STEP 2: Check consent
    if not character.consent_given_at:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Character '{character.name}' consent not given. Cannot generate video."
        )
    
    # STEP 3: Get character's canonical image
    if not character.base_image_url:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Character '{character.name}' has no canonical image. Please create character with an image first."
        )
    
    canonical_image_url = character.base_image_url
    print(f"[Character Video] Character: {character.name}")
    print(f"[Character Video] Canonical Image: {canonical_image_url}")
    
    try:
        start_time = datetime.utcnow()
        
        # STEP 4: Download character's canonical image
        storage = StorageService()
        image_bytes = await storage.download_bytes(canonical_image_url)
        print(f"[Character Video] ✓ Downloaded character image: {len(image_bytes)} bytes")
        
        # Detect mime type
        if image_bytes.startswith(b'\x89PNG\r\n\x1a\n'):
            mime_type = "image/png"
        elif image_bytes.startswith(b'\xff\xd8'):
            mime_type = "image/jpeg"
        elif image_bytes.startswith(b'RIFF') and image_bytes[8:12] == b'WEBP':
            mime_type = "image/webp"
        else:
            mime_type = "image/jpeg"
        print(f"[Character Video] Image format: {mime_type}")
        
        # STEP 5: Get character metadata for enhanced prompt
        char_metadata = character.char_metadata or {}
        
        # Extract character traits
        face = char_metadata.get("face", "")
        hair = char_metadata.get("hair", "")
        eyes = char_metadata.get("eyes", "")
        distinctives = char_metadata.get("distinctives", "")
        build = char_metadata.get("build", "")
        skin_tone = char_metadata.get("skin_tone", "")
        
        # Build enhanced prompt with character identity
        identity_description = f"{character.name}"
        
        # Add physical traits if available
        traits = []
        if hair and hair != "Not specified": traits.append(f"{hair} hair")
        if eyes and eyes != "Not specified": traits.append(f"{eyes} eyes")
        if skin_tone and skin_tone != "Not specified": traits.append(f"{skin_tone} skin")
        if distinctives and distinctives.lower() not in ["none", "not specified"]:
            traits.append(distinctives)
        
        if traits:
            identity_description += f" ({', '.join(traits)})"
        
        # Build final prompt
        enhanced_prompt = f"{identity_description} {request.prompt}"
        
        print(f"[Character Video] Identity: {identity_description}")
        print(f"[Character Video] Enhanced Prompt: {enhanced_prompt}")
        
        # STEP 6: Initialize Veo API client
        api_key = settings.GEMINI_API_KEY
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set in environment")
        
        client = genai.Client(api_key=api_key)
        
        # Create image object for Veo
        image = types.Image(
            image_bytes=image_bytes,
            mime_type=mime_type,
        )
        
        # Get options from request (with defaults)
        # CRITICAL: request.options can be None, so we need safe attribute access
        if request.options:
            aspect_ratio = getattr(request.options, 'aspect_ratio', None) or "16:9"
            duration_seconds = getattr(request.options, 'duration_seconds', None) or 8
            negative_prompt = getattr(request.options, 'negative_prompt', None) or "cartoon, drawing, low quality, blurry"
        else:
            # Default values when options is None
            aspect_ratio = "16:9"
            duration_seconds = 8
            negative_prompt = "cartoon, drawing, low quality, blurry"
        
        print(f"\n[Character Video] 🎬 Generating video with Veo 3.0...")
        print(f"[Character Video] Aspect Ratio: {aspect_ratio}")
        print(f"[Character Video] Duration: {duration_seconds}s")
        print(f"[Character Video] Negative Prompt: {negative_prompt}")
        
        # STEP 7: Generate video from character image
        operation = client.models.generate_videos(
            model="veo-3.0-generate-001",
            prompt=enhanced_prompt,
            image=image,  # Use character's canonical image as first frame
            config={
                "aspect_ratio": aspect_ratio,
                "negative_prompt": negative_prompt,
            },
        )
        
        # STEP 8: Wait for video generation to complete (async operation)
        print(f"[Character Video] ⏳ Waiting for Veo to generate video...")
        operation = _wait_for_operation(
            client,
            operation,
            f"[Character Video] Still generating video for {character.name}...",
        )
        
        generated_video = operation.response.generated_videos[0]
        print(f"[Character Video] ✓ Video generation complete!")
        
        # STEP 9: Download video from Veo
        print(f"[Character Video] 📥 Downloading video from Veo...")
        
        video_file = generated_video.video
        
        # Download the file to get the video bytes
        # The Veo API returns a file object with uri or data
        if hasattr(video_file, 'uri'):
            # If it's a URI, download it via HTTP
            import httpx
            print(f"[Character Video] Downloading from URI: {video_file.uri[:100]}...")
            async with httpx.AsyncClient() as http_client:
                response = await http_client.get(video_file.uri, timeout=300.0)
                response.raise_for_status()
                video_bytes = response.content
        elif hasattr(video_file, 'data'):
            # If bytes are directly available
            video_bytes = video_file.data
        else:
            # Fallback: try to read as bytes
            video_bytes = bytes(video_file)
        
        print(f"[Character Video] ✓ Downloaded video: {len(video_bytes)} bytes ({len(video_bytes) / 1024 / 1024:.2f} MB)")
        
        # STEP 10: Upload to storage (GCS, S3, or Local)
        job_id = f"video_{uuid.uuid4().hex[:12]}"
        video_path = f"videos/characters/{request.character_id}/{job_id}.mp4"
        
        print(f"[Character Video] 📤 Uploading to storage: {video_path}")
        video_url = await storage.upload_bytes(video_bytes, video_path, content_type="video/mp4")
        print(f"[Character Video] ✓ Video uploaded: {video_url}")
        
        # Calculate metrics
        end_time = datetime.utcnow()
        generation_time = (end_time - start_time).total_seconds()
        
        # Get video metadata
        video_duration = getattr(generated_video, 'duration_seconds', duration_seconds)
        
        print(f"\n{'='*60}")
        print(f"[Character Video] ✅ SUCCESS!")
        print(f"[Character Video] Character: {character.name}")
        print(f"[Character Video] Video URL: {video_url}")
        print(f"[Character Video] Duration: {video_duration}s")
        print(f"[Character Video] Generation Time: {generation_time:.1f}s")
        print(f"{'='*60}\n")
        
        return {
            "job_id": job_id,
            "status": "success",
            "message": f"Video generated successfully for {character.name} in {generation_time:.1f}s",
            "result_url": video_url,
            "video_duration_seconds": video_duration,
            "generation_time_seconds": generation_time,
            "model_used": "veo-3.0-generate-001",
            "has_audio": True,
        }
        
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"[Character Video] ❌ ERROR: {str(e)}")
        print(f"{'='*60}\n")
        import traceback
        traceback.print_exc()
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Video generation failed: {str(e)}"
        )
