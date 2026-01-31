"""
RQ Task Definitions
Defines the actual task functions that are executed by workers.
"""

import logging
import asyncio
from typing import Any, Dict, Optional
from datetime import datetime

from app.core.config import settings
from app.workers.base import with_retry, RetryableError, NonRetryableError

logger = logging.getLogger(__name__)


def _run_async(coro):
    """Helper to run async code in sync context (for RQ)."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


@with_retry(max_retries=2, retry_delay=5.0)
def run_image_generation_task(
    job_id: str,
    character_id: str,
    prompt: str,
    scene_number: Optional[int] = None,
    aspect_ratio: str = "16:9",
    **kwargs
) -> Dict[str, Any]:
    """
    RQ task for image generation pipeline.
    
    This runs the full generation workflow:
    1. Load character data
    2. Retrieve memory context
    3. Generate optimized prompt via Groq
    4. Generate image via Gemini
    5. Extract state and update episodic memory
    6. Compute IDR and trigger refinement if needed
    
    Args:
        job_id: Unique job identifier
        character_id: Character to generate for
        prompt: User prompt
        scene_number: Optional scene number for continuity
        aspect_ratio: Image aspect ratio
        
    Returns:
        Dict with job results
    """
    logger.info(f"[Task] Starting image generation: {job_id}")
    
    async def _generate():
        from app.core.database import SessionLocal
        from app.models.job import Job
        from app.models.character import Character
        from app.services.gemini_image import GeminiImageService
        from app.services.groq_llm import GroqLLMService
        from app.services.storage import StorageService
        from app.services.memory_engine import CharacterMemoryEngine
        from app.services.identity import IdentityService
        from app.workers.state import extract_state_task
        
        db = SessionLocal()
        
        try:
            # Update job status
            db_job = db.query(Job).filter(Job.id == job_id).first()
            if not db_job:
                raise NonRetryableError(f"Job not found: {job_id}")
            
            db_job.status = "running"
            db.commit()
            
            start_time = datetime.utcnow()
            
            # Load character
            character = db.query(Character).filter(Character.id == character_id).first()
            if not character:
                raise NonRetryableError(f"Character not found: {character_id}")
            
            # Initialize services
            storage = StorageService()
            gemini = GeminiImageService(storage_service=storage)
            groq = GroqLLMService()
            memory_engine = CharacterMemoryEngine(db)  # Fixed: pass db session
            identity_service = IdentityService(storage_service=storage)
            
            # Step 1: Get memory context
            logger.info(f"  [1/6] Loading memory context...")
            memory_context = memory_engine.build_prompt_context(character_id, prompt)  # Fixed: correct method name
            
            # Step 2: Generate optimized prompt
            logger.info(f"  [2/6] Generating optimized prompt...")
            character_data = {
                "name": character.name,
                **(character.char_metadata or {})
            }
            
            episodic_context = memory_context.get("episodic_summary", "")
            optimized_prompt = await groq.generate_prompt(
                character_data,
                prompt,
                [{"summary": episodic_context}] if episodic_context else []
            )
            
            # Step 3: Generate image
            logger.info(f"  [3/6] Generating image...")
            reference_image_url = character.base_image_url
            
            # Get additional reference images from character metadata
            char_metadata = character.char_metadata or {}
            additional_refs = char_metadata.get("reference_images", [])
            
            # Get semantic vector for identity preservation
            semantic_vector = None
            if character.semantic_vector_id:
                try:
                    from app.services.vectordb import VectorDBService
                    vectordb = VectorDBService()
                    semantic_vector = vectordb.get(character.semantic_vector_id)
                except Exception as vec_err:
                    logger.warning(f"Failed to retrieve semantic vector: {vec_err}")
            
            image_bytes = await gemini.generate(
                prompt=optimized_prompt,
                aspect_ratio=aspect_ratio,
                reference_image_url=reference_image_url,
                reference_images=additional_refs[:2],
                character_data=character_data,
                semantic_vector=semantic_vector
            )
            
            # Step 4: Store result
            logger.info(f"  [4/6] Storing result...")
            result_path = f"outputs/jobs/{job_id}/result.jpg"
            result_url = await storage.upload_bytes(image_bytes, result_path)
            
            # Step 5: Extract state
            logger.info(f"  [5/6] Extracting state...")
            scene_index = memory_engine.get_next_scene_index(character_id)
            state_result = await extract_state_task(
                job_id=job_id,
                image_url=result_url,
                character_id=character_id,
                scene_index=scene_index,
                prompt_used=optimized_prompt
            )
            
            idr_score = state_result.get("idr", 0) if state_result else 0
            
            # Step 6: Check IDR and trigger refinement if needed
            logger.info(f"  [6/6] Validating identity (IDR: {idr_score:.3f})...")
            final_url = result_url
            was_refined = False
            
            if idr_score > 0 and idr_score < settings.IDR_THRESHOLD:
                logger.info(f"    IDR {idr_score:.3f} < threshold {settings.IDR_THRESHOLD}, triggering refinement...")
                
                from app.workers.refiner import FaceRefiner
                refiner = FaceRefiner()
                
                refined_bytes, new_idr, was_refined = await refiner.refine_if_needed(
                    image_bytes=image_bytes,
                    character_id=character_id,
                    character_data=character_data,
                    current_idr=idr_score
                )
                
                if was_refined:
                    refined_path = f"outputs/jobs/{job_id}/result_refined.jpg"
                    final_url = await storage.upload_bytes(refined_bytes, refined_path)
                    idr_score = new_idr
                    logger.info(f"    Refinement complete. New IDR: {new_idr:.3f}")
            
            # Collect golden images for LoRA training (IDR > 0.85)
            try:
                from app.services.golden_collector import collect_golden_image
                collection_result = collect_golden_image(
                    character_id=character_id,
                    image_path=final_url,
                    idr_score=idr_score,
                    job_id=job_id,
                    prompt=optimized_prompt,
                    scene_index=scene_index
                )
                if collection_result.get("collected"):
                    logger.info(f"    🏆 Golden image collected (IDR: {idr_score:.3f})")
            except Exception as collect_err:
                logger.warning(f"    Golden collection skipped: {collect_err}")
            
            # Calculate metrics
            end_time = datetime.utcnow()
            generation_time = (end_time - start_time).total_seconds()
            
            # Update job
            db_job.status = "success"
            db_job.result_url = final_url
            db_job.completed_at = end_time
            db_job.metrics = {
                "generation_time_seconds": generation_time,
                "idr_score": idr_score,
                "was_refined": was_refined,
                "scene_index": scene_index,
                "prompt_used": optimized_prompt[:500]
            }
            db.commit()
            
            logger.info(f"[Task] Completed: {job_id} (IDR: {idr_score:.3f}, Refined: {was_refined})")
            
            return {
                "job_id": job_id,
                "status": "success",
                "result_url": final_url,
                "idr_score": idr_score,
                "was_refined": was_refined,
                "generation_time": generation_time
            }
            
        except NonRetryableError:
            db_job.status = "failed"
            db_job.error_message = str(e)
            db.commit()
            raise
            
        except Exception as e:
            logger.error(f"[Task] Failed: {job_id} - {e}")
            db_job.status = "failed"
            db_job.error_message = str(e)
            db.commit()
            raise RetryableError(str(e))
            
        finally:
            db.close()
    
    return _run_async(_generate())


@with_retry(max_retries=1, retry_delay=3.0)
def run_refinement_task(
    job_id: str,
    image_url: str,
    character_id: str,
    current_idr: float,
    character_data: dict,
    **kwargs
) -> Dict[str, Any]:
    """
    RQ task for face refinement.
    
    Args:
        job_id: Original job ID
        image_url: URL of image to refine
        character_id: Character ID
        current_idr: Current IDR score
        character_data: Character metadata
        
    Returns:
        Dict with refinement results
    """
    logger.info(f"[Task] Starting refinement: {job_id} (IDR: {current_idr:.3f})")
    
    async def _refine():
        from app.services.storage import StorageService
        from app.workers.refiner import FaceRefiner
        
        storage = StorageService()
        refiner = FaceRefiner()
        
        # Download image
        image_bytes = await storage.download_bytes(image_url)
        
        # Run refinement
        refined_bytes, new_idr, was_refined = await refiner.refine_if_needed(
            image_bytes=image_bytes,
            character_id=character_id,
            character_data=character_data,
            current_idr=current_idr
        )
        
        # Store refined result
        if was_refined:
            refined_path = f"outputs/jobs/{job_id}/result_refined.jpg"
            refined_url = await storage.upload_bytes(refined_bytes, refined_path)
        else:
            refined_url = image_url
        
        logger.info(f"[Task] Refinement complete: {job_id} (IDR: {current_idr:.3f} -> {new_idr:.3f})")
        
        return {
            "job_id": job_id,
            "original_idr": current_idr,
            "new_idr": new_idr,
            "was_refined": was_refined,
            "refined_url": refined_url
        }
    
    return _run_async(_refine())


@with_retry(max_retries=2, retry_delay=2.0)
def run_state_extraction_task(
    job_id: str,
    image_url: str,
    character_id: str,
    scene_index: int,
    prompt_used: str = "",
    **kwargs
) -> Dict[str, Any]:
    """
    RQ task for state extraction.
    
    Args:
        job_id: Generation job ID
        image_url: URL of generated image
        character_id: Character ID
        scene_index: Scene number
        prompt_used: The generation prompt
        
    Returns:
        Dict with extraction results
    """
    logger.info(f"[Task] Starting extraction: {job_id} (scene: {scene_index})")
    
    async def _extract():
        from app.workers.state import extract_state_task
        
        result = await extract_state_task(
            job_id=job_id,
            image_url=image_url,
            character_id=character_id,
            scene_index=scene_index,
            prompt_used=prompt_used
        )
        
        logger.info(f"[Task] Extraction complete: {job_id}")
        return result
    
    return _run_async(_extract())


# Export all
__all__ = [
    "run_image_generation_task",
    "run_refinement_task",
    "run_state_extraction_task"
]
