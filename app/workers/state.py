"""
State Extractor Worker
Extracts state from generated images and updates episodic memory.
This is CRITICAL for character memory persistence!

Uses Gemini Vision for actual image analysis when available,
falls back to prompt-based inference.
"""

import uuid
from app.services.vectordb import VectorDBService
from app.services.identity import IdentityService
from app.services.storage import StorageService
from app.services.groq_llm import GroqLLMService


async def extract_state_task(job_id: str, image_url: str, character_id: str, scene_index: int, prompt_used: str = ""):
    """
    Extract state from generated image:
    1. Use Gemini Vision to analyze actual image (preferred)
    2. Fall back to LLM analysis of prompt if vision fails
    3. Compute generated face embedding
    4. Calculate IDR
    5. Update episodic memory with full state data
    
    This enables the character to "remember" their state between scenes.
    """
    from app.core.database import SessionLocal
    from app.models.job import Job
    from app.models.episodic import EpisodicState
    
    db = SessionLocal()
    vectordb = VectorDBService()
    storage = StorageService()
    identity = IdentityService(storage_service=storage)
    groq = GroqLLMService()
    
    try:
        # Get semantic embedding for IDR calculation
        semantic = await vectordb.query_semantic(character_id)
        
        # Compute IDR (Identity Retention score)
        idr = 0.0
        if semantic is not None:
            idr = await identity.compute_idr(semantic, image_url)
        
        # Update job metrics
        job = db.query(Job).filter(Job.id == job_id).first()
        if job:
            job.metrics = {**(job.metrics or {}), "idr": idr}
        
        # Try Gemini Vision for actual image analysis (PREFERRED)
        state_data = None
        try:
            from app.services.gemini_image import GeminiImageService
            gemini = GeminiImageService(storage_service=storage)
            state_data = await gemini.analyze_image(image_url=image_url)
            
            if state_data and "error" not in state_data:
                print(f"[OK] State extracted via Gemini Vision: {len(state_data.get('tags', []))} tags")
            else:
                state_data = None
        except Exception as e:
            print(f"[WARNING] Gemini Vision failed, falling back to prompt analysis: {e}")
            state_data = None
        
        # Fallback: Use Groq to analyze the prompt (less accurate but better than nothing)
        if state_data is None or (not state_data.get("clothing") and not state_data.get("tags")):
            print("[WARNING] Vision failed or found nothing, analyzing PROMPT for state...")
            image_context = f"Generated image from prompt: {prompt_used}"
            # Force extraction from the text prompt
            state_data = await groq.summarize_image(image_context)
            print(f"[OK] State extracted via prompt analysis: {state_data}")
        
        # Extract tags from state data
        tags = state_data.get("tags", [])
        
        # ensure clothing is in tags
        clothing = state_data.get("clothing", [])
        if clothing:
            tags.extend([f"wearing_{c.replace(' ', '_')}" for c in clothing])
            
        # ensure props are in tags
        props = state_data.get("props", [])
        if props:
            tags.extend([f"has_{p.replace(' ', '_')}" for p in props])
            
        if state_data.get("physical_state"):
            tags.extend(state_data.get("physical_state", []))
        
        # Deduplicate tags
        tags = list(set(tags))
        
        # Create comprehensive episodic state
        episodic_id = f"epi_{uuid.uuid4().hex[:12]}"
        episodic = EpisodicState(
            id=episodic_id,
            character_id=character_id,
            scene_index=scene_index,
            tags=tags,
            state_data={
                "clothing": state_data.get("clothing", []),
                "physical_state": state_data.get("physical_state", []),
                "props": state_data.get("props", []),
                "pose": state_data.get("pose", ""),
                "environment": state_data.get("environment", ""),
                "prompt_used": prompt_used[:500],
                "idr_score": idr
            },
            image_url=image_url,
            notes=f"Scene {scene_index}: {state_data.get('pose', 'Unknown pose')} in {state_data.get('environment', 'Unknown location')}"
        )
        db.add(episodic)
        db.commit()
        
        print(f"[OK] Episodic state saved: {len(tags)} tags, IDR={idr:.3f}")
        return {"success": True, "tags": tags, "idr": idr}
        
    except Exception as e:
        print(f"[ERROR] State extraction failed: {e}")
        db.rollback()
        return {"success": False, "error": str(e)}
    finally:
        db.close()


async def update_character_memory(character_id: str, new_traits: dict):
    """
    Update the character's semantic memory with new learned traits.
    Called when we detect consistent patterns across multiple episodes.
    """
    from app.core.database import SessionLocal
    from app.models.character import Character
    
    db = SessionLocal()
    try:
        character = db.query(Character).filter(Character.id == character_id).first()
        if character and character.char_metadata:
            # Merge new traits into existing metadata
            updated_metadata = {**character.char_metadata, **new_traits}
            character.char_metadata = updated_metadata
            db.commit()
            print(f"[OK] Character memory updated: {new_traits.keys()}")
    finally:
        db.close()
