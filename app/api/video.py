"""
Video Generation API Routes
Handles video generation requests using Veo 3.1 with dialogue and audio.
"""

import uuid
import traceback
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.models.character import Character
from app.models.job import Job
from app.schemas.video import (
    VideoGenerateRequest,
    VideoGenerateResponse,
    VideoExtendRequest,
    VideoTransitionRequest,
    VideoJobStatus,
)

router = APIRouter()


@router.post("/video/generate", response_model=VideoGenerateResponse, status_code=status.HTTP_202_ACCEPTED)
async def generate_video(
    request: VideoGenerateRequest,
    db: Session = Depends(get_db),
):
    """
    Generate a video for a character using Veo 3.1.
    
    Features:
    - Text-to-video with dialogue and native audio
    - Image-to-video (use character image as first frame)
    - Reference images for character consistency
    
    The video generation runs synchronously and returns the result URL.
    """
    print(f"[Video API] Generate request for char: {request.character_id}")
    
    # Validate character exists
    character = db.query(Character).filter(Character.id == request.character_id).first()
    
    if not character:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Character not found"
        )
    
    # Check consent
    if not character.consent_given_at:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Character consent not given"
        )
    
    # Create job record
    job_id = f"video_{uuid.uuid4().hex[:12]}"
    
    options_dict = request.options.model_dump() if request.options else {}
    
    db_job = Job(
        id=job_id,
        character_id=request.character_id,
        prompt=request.prompt,
        options={**options_dict, "type": "video"},
        status="running",
    )
    db.add(db_job)
    db.commit()
    
    try:
        print("[Video API] Starting video generation...")
        
        # Import services
        from app.services.veo_video import VeoVideoService, VeoPromptOptimizer
        from app.services.gemini_image import GeminiImageService
        from app.services.groq_llm import GroqLLMService
        from app.services.storage import StorageService
        from app.core.config import settings
        
        start_time = datetime.utcnow()
        
        # Initialize services
        veo = VeoVideoService()
        storage = StorageService()
        gemini = GeminiImageService(storage_service=storage)
        groq = GroqLLMService()
        
        # Build character data for prompt optimization
        character_data = {
            "name": character.name,
            "face": character.char_metadata.get("face", ""),
            "hair": character.char_metadata.get("hair", ""),
            "eyes": character.char_metadata.get("eyes", ""),
            "distinctives": character.char_metadata.get("distinctives", ""),
            "tags": character.char_metadata.get("tags", []),
            **(character.char_metadata or {})
        }
        
        # Get options
        opts = request.options or {}
        aspect_ratio = getattr(opts, 'aspect_ratio', "16:9") if opts else "16:9"
        resolution = getattr(opts, 'resolution', "720p") if opts else "720p"
        duration = getattr(opts, 'duration_seconds', 8) if opts else 8
        negative_prompt = getattr(opts, 'negative_prompt', None) if opts else None
        person_generation = getattr(opts, 'person_generation', "allow_adult") if opts else "allow_adult"
        
        # Build optimized prompt
        if opts and hasattr(opts, 'dialogue') and opts.dialogue:
            # Use dialogue optimizer
            dialogue_list = [{"speaker": d.speaker, "line": d.line, "emotion": d.emotion} for d in opts.dialogue]
            optimized_prompt = VeoPromptOptimizer.create_dialogue_prompt(
                scene_description=request.prompt,
                dialogue=dialogue_list,
                sound_effects=opts.sound_effects if hasattr(opts, 'sound_effects') else None,
                camera_movement=opts.camera_movement if hasattr(opts, 'camera_movement') else None,
            )
        elif opts and hasattr(opts, 'style') and opts.style:
            # Use cinematic optimizer
            optimized_prompt = VeoPromptOptimizer.create_cinematic_prompt(
                scene_description=request.prompt,
                style=opts.style,
                lighting=opts.lighting if hasattr(opts, 'lighting') else None,
                color_grade=opts.color_grade if hasattr(opts, 'color_grade') else None,
            )
        else:
            # Use LLM to enhance prompt with character details
            optimized_prompt = await groq.generate_prompt(
                character_data,
                f"[VIDEO SCENE] {request.prompt}",
                []  # No episodic memory for first iteration
            )
        
        print(f"[Video API] Optimized prompt: {optimized_prompt[:150]}...")
        
        # Generate video based on mode
        video_result = None
        
        if request.use_first_frame:
            # Mode 1: Generate character image first, then animate it
            print("[Video API] Generating first frame image...")
            
            first_frame_prompt = request.first_frame_prompt or f"Portrait of {character.name}: {request.prompt}"
            
            # Generate the starting frame with Gemini
            image_bytes = await gemini.generate(
                prompt=first_frame_prompt,
                aspect_ratio=aspect_ratio,
                reference_image_url=character.base_image_url,
                character_data=character_data
            )
            
            print(f"[Video API] First frame generated ({len(image_bytes)} bytes)")
            
            # Now generate video from that image
            video_result = await veo.generate_from_image(
                prompt=optimized_prompt,
                image_bytes=image_bytes,
                aspect_ratio=aspect_ratio,
                resolution=resolution,
                duration_seconds=duration,
                negative_prompt=negative_prompt,
                person_generation=person_generation,
            )
            
        elif request.use_reference_images and character.base_image_url:
            # Mode 2: Use reference images for consistency
            print("[Video API] Loading reference images...")
            
            ref_bytes = await gemini._load_image_bytes(character.base_image_url)
            
            if ref_bytes:
                video_result = await veo.generate_with_reference_images(
                    prompt=optimized_prompt,
                    reference_images=[ref_bytes],
                    aspect_ratio=aspect_ratio,
                    resolution=resolution,
                    duration_seconds=duration,
                    negative_prompt=negative_prompt,
                    person_generation=person_generation,
                )
            else:
                # Fallback to text-only
                video_result = await veo.generate_from_text(
                    prompt=optimized_prompt,
                    aspect_ratio=aspect_ratio,
                    resolution=resolution,
                    duration_seconds=duration,
                    negative_prompt=negative_prompt,
                    person_generation=person_generation,
                )
        else:
            # Mode 3: Text-to-video only
            video_result = await veo.generate_from_text(
                prompt=optimized_prompt,
                aspect_ratio=aspect_ratio,
                resolution=resolution,
                duration_seconds=duration,
                negative_prompt=negative_prompt,
                person_generation=person_generation,
                seed=opts.seed if opts and hasattr(opts, 'seed') else None,
            )
        
        print("[Video API] Video generated, saving...")
        
        # Validate video result
        video_bytes = video_result.get("video_bytes")
        if not video_bytes or len(video_bytes) == 0:
            raise Exception(f"Video generation returned empty result. Check Veo API status.")
        
        print(f"[Video API] Video size: {len(video_bytes)} bytes")
        
        # Save the video
        result_path = f"outputs/videos/{job_id}/result.mp4"
        result_url = await storage.upload_bytes(
            video_bytes,
            result_path
        )
        
        # Calculate metrics
        end_time = datetime.utcnow()
        generation_time = (end_time - start_time).total_seconds()
        
        # Update job with success
        db_job.status = "success"
        db_job.result_url = result_url
        db_job.completed_at = end_time
        db_job.metrics = {
            "generation_time_seconds": generation_time,
            "prompt_used": optimized_prompt[:500],
            "video_duration": duration,
            "resolution": resolution,
            "aspect_ratio": aspect_ratio,
            "model": video_result.get("model", "veo-3.1"),
            "has_audio": True,
        }
        db.commit()
        
        print(f"[Video API] [OK] Video generated in {generation_time:.1f}s")
        
        return VideoGenerateResponse(
            job_id=job_id,
            status="success",
            message=f"Video generated in {generation_time:.1f}s",
            result_url=result_url,
            video_duration_seconds=duration,
            generation_time_seconds=generation_time,
            model_used=video_result.get("model", "veo-3.1"),
            has_audio=True,
        )
        
    except Exception as e:
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        
        # Log to file
        with open("error_log.txt", "a") as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"VIDEO ERROR at {datetime.utcnow()}\n")
            f.write(f"{'='*60}\n")
            f.write(error_msg)
            f.write(f"\n{'='*60}\n\n")
        
        print(f"[Video API] [ERROR] {error_msg}")
        
        db_job.status = "failed"
        db_job.error_message = str(e)
        db.commit()
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Video generation failed: {str(e)}"
        )


@router.post("/video/extend", response_model=VideoGenerateResponse, status_code=status.HTTP_202_ACCEPTED)
async def extend_video(
    request: VideoExtendRequest,
    db: Session = Depends(get_db),
):
    """
    Extend an existing video with additional content.
    
    Takes a previous video job ID and generates a continuation.
    """
    print(f"[Video API] Extend request for job: {request.job_id}")
    
    # Get original job
    original_job = db.query(Job).filter(Job.id == request.job_id).first()
    
    if not original_job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Original video job not found"
        )
    
    if original_job.status != "success":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Original video must be successfully generated first"
        )
    
    # Create new job for extension
    job_id = f"video_ext_{uuid.uuid4().hex[:12]}"
    
    db_job = Job(
        id=job_id,
        character_id=original_job.character_id,
        prompt=request.continuation_prompt or "Continue the video naturally",
        options={"type": "video_extension", "original_job": request.job_id},
        status="running",
    )
    db.add(db_job)
    db.commit()
    
    try:
        from app.services.veo_video import VeoVideoService
        from app.services.storage import StorageService
        
        start_time = datetime.utcnow()
        
        veo = VeoVideoService()
        storage = StorageService()
        
        # Load original video
        video_bytes = await storage.download_bytes(original_job.result_url)
        
        # Extend it
        video_result = await veo.extend_video(
            video_bytes=video_bytes,
            prompt=request.continuation_prompt,
            duration_seconds=request.duration_seconds,
        )
        
        # Save extended video
        result_path = f"outputs/videos/{job_id}/result.mp4"
        result_url = await storage.upload_bytes(
            video_result.get("video_bytes") or b"",
            result_path
        )
        
        end_time = datetime.utcnow()
        generation_time = (end_time - start_time).total_seconds()
        
        db_job.status = "success"
        db_job.result_url = result_url
        db_job.completed_at = end_time
        db_job.metrics = {
            "generation_time_seconds": generation_time,
            "extended_from": request.job_id,
            "extension_duration": request.duration_seconds,
        }
        db.commit()
        
        return VideoGenerateResponse(
            job_id=job_id,
            status="success",
            message=f"Video extended in {generation_time:.1f}s",
            result_url=result_url,
            video_duration_seconds=request.duration_seconds,
            generation_time_seconds=generation_time,
            has_audio=True,
        )
        
    except Exception as e:
        db_job.status = "failed"
        db_job.error_message = str(e)
        db.commit()
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Video extension failed: {str(e)}"
        )


@router.post("/video/transition", response_model=VideoGenerateResponse, status_code=status.HTTP_202_ACCEPTED)
async def generate_transition(
    request: VideoTransitionRequest,
    db: Session = Depends(get_db),
):
    """
    Generate a video transition between two character states.
    
    Creates a smooth video from first_frame_prompt to last_frame_prompt.
    Useful for pose changes, outfit changes, etc.
    """
    print(f"[Video API] Transition request for char: {request.character_id}")
    
    # Validate character
    character = db.query(Character).filter(Character.id == request.character_id).first()
    
    if not character:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Character not found"
        )
    
    # Create job
    job_id = f"video_trans_{uuid.uuid4().hex[:12]}"
    
    db_job = Job(
        id=job_id,
        character_id=request.character_id,
        prompt=request.transition_prompt,
        options={"type": "video_transition"},
        status="running",
    )
    db.add(db_job)
    db.commit()
    
    try:
        from app.services.veo_video import VeoVideoService
        from app.services.gemini_image import GeminiImageService
        from app.services.storage import StorageService
        
        start_time = datetime.utcnow()
        
        storage = StorageService()
        veo = VeoVideoService()
        gemini = GeminiImageService(storage_service=storage)
        
        # Build character data
        character_data = {
            "name": character.name,
            **(character.char_metadata or {})
        }
        
        opts = request.options or {}
        aspect_ratio = getattr(opts, 'aspect_ratio', "16:9") if opts else "16:9"
        resolution = getattr(opts, 'resolution', "720p") if opts else "720p"
        duration = getattr(opts, 'duration_seconds', 8) if opts else 8
        
        # Generate first frame
        print("[Video API] Generating first frame...")
        first_frame_bytes = await gemini.generate(
            prompt=request.first_frame_prompt,
            aspect_ratio=aspect_ratio,
            reference_image_url=character.base_image_url,
            character_data=character_data
        )
        
        # Generate last frame
        print("[Video API] Generating last frame...")
        last_frame_bytes = await gemini.generate(
            prompt=request.last_frame_prompt,
            aspect_ratio=aspect_ratio,
            reference_image_url=character.base_image_url,
            character_data=character_data
        )
        
        # Generate transition video
        print("[Video API] Generating transition video...")
        video_result = await veo.generate_with_frames(
            prompt=request.transition_prompt,
            first_frame_bytes=first_frame_bytes,
            last_frame_bytes=last_frame_bytes,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            duration_seconds=duration,
        )
        
        # Save video
        result_path = f"outputs/videos/{job_id}/result.mp4"
        result_url = await storage.upload_bytes(
            video_result.get("video_bytes") or b"",
            result_path
        )
        
        end_time = datetime.utcnow()
        generation_time = (end_time - start_time).total_seconds()
        
        db_job.status = "success"
        db_job.result_url = result_url
        db_job.completed_at = end_time
        db_job.metrics = {
            "generation_time_seconds": generation_time,
            "type": "transition",
            "first_frame_prompt": request.first_frame_prompt[:200],
            "last_frame_prompt": request.last_frame_prompt[:200],
        }
        db.commit()
        
        return VideoGenerateResponse(
            job_id=job_id,
            status="success",
            message=f"Transition video generated in {generation_time:.1f}s",
            result_url=result_url,
            video_duration_seconds=duration,
            generation_time_seconds=generation_time,
            has_audio=True,
        )
        
    except Exception as e:
        db_job.status = "failed"
        db_job.error_message = str(e)
        db.commit()
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Transition generation failed: {str(e)}"
        )


@router.get("/video/status/{job_id}", response_model=VideoJobStatus)
async def get_video_status(
    job_id: str,
    db: Session = Depends(get_db),
):
    """Get the status of a video generation job."""
    
    job = db.query(Job).filter(Job.id == job_id).first()
    
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )
    
    return VideoJobStatus(
        job_id=job.id,
        status=job.status,
        result_url=job.result_url,
        error_message=job.error_message,
    )
