"""
Generator Worker
Handles image generation pipeline using Groq (LLM) + Gemini (Image).
"""

from datetime import datetime
from redis import Redis
from rq import Queue


def get_queue():
    """Get Redis queue."""
    return Queue(connection=Redis())


def queue_generation_job(job_id: str):
    """Queue a generation job."""
    q = get_queue()
    q.enqueue(run_generation_pipeline, job_id)


async def run_generation_pipeline(job_id: str):
    """
    Run the full generation pipeline:
    1. Retrieve character and semantic embedding
    2. Retrieve episodic memory
    3. Generate prompt via Groq LLM
    4. Generate image via Gemini
    5. Store result and update episodic memory via Groq summarization
    """
    from app.core.database import SessionLocal
    from app.models.job import Job
    from app.models.character import Character
    from app.services.vectordb import VectorDBService
    from app.services.groq_llm import GroqLLMService
    from app.services.gemini_image import GeminiImageService
    from app.services.storage import StorageService
    
    db = SessionLocal()
    
    try:
        # Get job
        job = db.query(Job).filter(Job.id == job_id).first()
        if not job:
            return
        
        job.status = "running"
        db.commit()
        
        start_time = datetime.utcnow()
        
        # Get character
        character = db.query(Character).filter(Character.id == job.character_id).first()
        
        # Initialize services
        vectordb = VectorDBService()
        groq = GroqLLMService()
        storage = StorageService()
        gemini = GeminiImageService(storage_service=storage)
        storage = StorageService()
        
        # Step 1: Retrieve embeddings from vector DB
        semantic = await vectordb.query_semantic(job.character_id)
        episodic = await vectordb.query_episodic(job.character_id)
        
        # Step 2: Generate optimized prompt using Groq
        character_data = {
            "name": character.name,
            **(character.char_metadata or {})
        }
        optimized_prompt = await groq.generate_prompt(
            character_data, 
            job.prompt, 
            episodic
        )
        
        # Step 3: Generate image using Gemini
        image_bytes = await gemini.generate(
            prompt=optimized_prompt,
            aspect_ratio="16:9",  # Default aspect ratio
            reference_image_url=character.base_image_url
        )
        
        # Step 4: Store result
        result_path = f"outputs/jobs/{job_id}/result.jpg"
        result_url = await storage.upload_bytes(image_bytes, result_path)
        
        # Step 5: Summarize image using Groq (for episodic memory)
        image_summary = await groq.summarize_image(optimized_prompt)
        
        # Calculate metrics
        end_time = datetime.utcnow()
        generation_time = (end_time - start_time).total_seconds()
        
        # Update job with success
        job.status = "success"
        job.result_url = result_url
        job.completed_at = end_time
        job.metrics = {
            "generation_time_seconds": generation_time,
            "prompt_used": optimized_prompt[:500],
            "image_tags": image_summary.get("tags", [])
        }
        db.commit()
        
        # Step 6: Update episodic memory (async - don't block)
        # TODO: Upsert to vector DB with new episodic state
        
    except Exception as e:
        job.status = "failed"
        job.error_message = str(e)
        db.commit()
    finally:
        db.close()
