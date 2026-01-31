"""
Generation API Routes
Handles image generation requests and job creation.
"""

import uuid
import traceback
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.models.character import Character
from app.models.job import Job
from app.schemas.generate import GenerateRequest, GenerateResponse

router = APIRouter()


@router.post("/generate", response_model=GenerateResponse, status_code=status.HTTP_202_ACCEPTED)
async def generate_image(
    request: GenerateRequest,
    db: Session = Depends(get_db),
):
    """
    Create a new image generation job.
    Runs synchronously (no Redis needed for testing).
    """
    print(f"Generate request for char: {request.character_id}")
    
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
    
    # VALIDATION CHECK: Ensure character has proper memory foundation
    print(f"[Validation] Checking memory health for {request.character_id}...")
    
    # Check semantic vector exists
    if not character.semantic_vector_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Character memory not initialized. Please wait for identity extraction to complete, or use /reextract-identity endpoint."
        )
    
    # Check char_metadata has basic identity data
    char_metadata = character.char_metadata or {}
    required_fields = ["face", "hair", "eyes"]
    missing_fields = [f for f in required_fields if not char_metadata.get(f)]
    
    if missing_fields:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Character identity incomplete. Missing: {', '.join(missing_fields)}. Use /reextract-identity to fix."
        )
    
    # Calculate health score
    health_score = 100
    
    # Deduct for missing optional fields
    optional_fields = ["distinctives", "build", "age_range", "skin_tone"]
    missing_optional = [f for f in optional_fields if not char_metadata.get(f)]
    health_score -= len(missing_optional) * 5
    
    # Deduct if no reference images
    if not char_metadata.get("reference_images"):
        health_score -= 20
    
    # Deduct if quality score is low
    quality_score = char_metadata.get("quality_score", 0)
    if quality_score < 0.5:
        health_score -= 15
    elif quality_score < 0.7:
        health_score -= 10
    
    # Block generation if health score is too low
    if health_score < 50:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Character memory health too low ({health_score}/100). Please use /reextract-identity to improve quality before generating images."
        )
    
    print(f"[Validation] Memory health: {health_score}/100 - OK")
    
    # Create job record
    job_id = f"job_{uuid.uuid4().hex[:12]}"
    
    print(f"DEBUG: Creating job {job_id}")
    print(f"DEBUG: Options type: {type(request.options)}")
    print(f"DEBUG: Options value: {request.options}")
    
    try:
        options_dict = request.options.model_dump() if request.options else {}
        print(f"DEBUG: Options dict: {options_dict}")
    except Exception as e:
        print(f"DEBUG: Error converting options: {e}")
        options_dict = {}
    
    db_job = Job(
        id=job_id,
        character_id=request.character_id,
        prompt=request.prompt,
        pose_image_url=request.pose_image_url,
        options=options_dict,
        status="running",
    )
    db.add(db_job)
    db.commit()
    print(f"DEBUG: Job created and committed")
    
    try:
        print("Starting synchronous generation...")
        
        # Import services here to avoid circular imports
        from app.services.groq_llm import GroqLLMService
        from app.services.gemini_image import GeminiImageService
        from app.services.storage import StorageService
        
        start_time = datetime.utcnow()
        
        # Initialize services
        groq = GroqLLMService()
        storage = StorageService()
        gemini = GeminiImageService(storage_service=storage)
        
        # Step 1: Use CharacterMemoryEngine to build complete prompt context
        # This properly merges semantic + episodic memory
        print("Building memory context with CharacterMemoryEngine...")
        from app.services.memory_engine import CharacterMemoryEngine
        
        memory_engine = CharacterMemoryEngine(db)
        memory_context = memory_engine.build_prompt_context(
            character_id=request.character_id,
            user_prompt=request.prompt
        )
        
        # Validate memory quality - warn if identity data is incomplete
        if memory_context.get("face") in [None, "", "Not specified", "Not analyzed"]:
            print(f"[WARNING] Character identity data incomplete!")
            print(f"[WARNING] Run POST /characters/{request.character_id}/reextract-identity to fix")
        
        # Check semantic vector exists
        if not character.semantic_vector_id:
            print(f"[WARNING] No semantic vector for {character.name}")
            print(f"[WARNING] IDR scoring will not work. Run reextract-identity to fix.")
        
        # CRITICAL: Retrieve semantic vector for identity preservation
        semantic_vector = None
        if character.semantic_vector_id:
            try:
                from app.services.vectordb import VectorDBService
                vectordb = VectorDBService()
                semantic_vector = vectordb.get(character.semantic_vector_id)
                if semantic_vector is not None:
                    print(f"[Identity] Retrieved semantic vector: shape={semantic_vector.shape}")
                else:
                    print(f"[WARNING] Semantic vector {character.semantic_vector_id} not found in vectordb")
            except Exception as vec_err:
                print(f"[WARNING] Failed to retrieve semantic vector: {vec_err}")
        
        # Build character_data from memory context for generation
        character_data = {
            "name": memory_context.get("name", character.name),
            "face": memory_context.get("face", ""),
            "hair": memory_context.get("hair", ""),
            "eyes": memory_context.get("eyes", ""),
            "distinctives": memory_context.get("distinctives", ""),
            "build": memory_context.get("build", ""),
            "age_range": memory_context.get("age_range", ""),
            "tags": memory_context.get("tags", []),
            "skin_tone": character.char_metadata.get("skin_tone", "") if character.char_metadata else "",
            "gender_presentation": character.char_metadata.get("gender_presentation", "") if character.char_metadata else "",
            "initial_outfit": character.char_metadata.get("initial_outfit", "") if character.char_metadata else "",
            "accessories": character.char_metadata.get("accessories", "") if character.char_metadata else "",
            "eyebrows": character.char_metadata.get("eyebrows", "") if character.char_metadata else "",
            "semantic_vector": semantic_vector,  # Pass for identity verification
        }
        
        # Get episodic states from memory context (already processed by engine)
        episodic_context = memory_context.get("recent_states", "No previous scenes")
        current_clothing = memory_context.get("current_clothing", [])
        current_state = memory_context.get("current_state", [])
        current_props = memory_context.get("current_props", [])
        episode_count = memory_context.get("episode_count", 0)
        
        print(f"Memory Context: {episode_count} previous episodes, confidence={memory_context.get('confidence', 0):.2f}")
        print(f"Identity: face='{character_data['face'][:30]}...', hair='{character_data['hair']}'")
        
        # Build episodic states for Groq (for backward compatibility)
        from app.models.episodic import EpisodicState
        from app.core.config import settings
        
        episodic_records = db.query(EpisodicState).filter(
            EpisodicState.character_id == request.character_id
        ).order_by(EpisodicState.scene_index.desc()).limit(
            settings.EPISODIC_TOP_K
        ).all()
        
        episodic_states = [
            {
                "scene_index": ep.scene_index,
                "tags": ep.tags or [],
                "notes": ep.notes,
                "state_data": ep.state_data or {}
            }
            for ep in reversed(episodic_records)
        ]
        print(f"Retrieved {len(episodic_states)} episodic states for memory")
        
        # Step 2: Generate optimized prompt using Groq with full memory context
        print("Calling Groq for prompt optimization...")
        optimized_prompt = await groq.generate_prompt(
            character_data, 
            request.prompt, 
            episodic_states
        )
        print(f"Optimized prompt: {optimized_prompt}")
        
        # Step 2: Generate image using Gemini
        print("Calling Gemini for image generation...")
        
        # Extract aspect ratio from options if provided, default to 16:9
        aspect_ratio = "16:9"
        if request.options and hasattr(request.options, 'aspect_ratio'):
            aspect_ratio = request.options.aspect_ratio
        
        # Get reference images for identity preservation
        # Priority: base_image_url first, then any additional reference images from metadata
        primary_ref = character.base_image_url
        additional_refs = character.char_metadata.get("reference_images", []) if character.char_metadata else []
        
        # If no base_image_url but we have reference_images in metadata, use the first one as primary
        if not primary_ref and additional_refs:
            primary_ref = additional_refs[0]
            additional_refs = additional_refs[1:]
        
        print(f"=== REFERENCE IMAGE DEBUG ===")
        print(f"Character ID: {character.id}")
        print(f"Character Name: {character.name}")
        print(f"Primary reference URL: {primary_ref}")
        print(f"Additional references: {len(additional_refs)}")
        print(f"=============================")
        
        # Generate with identity verification if we have semantic vector
        if semantic_vector is not None:
            print(f"[Identity] Generating with semantic vector guidance (embedding-based identity)")
        
        image_bytes = await gemini.generate(
            prompt=optimized_prompt,
            aspect_ratio=aspect_ratio,
            reference_image_url=primary_ref,
            reference_images=additional_refs[:2],  # Max 2 additional
            character_data=character_data,  # Pass data for strict trait enforcement!
            semantic_vector=semantic_vector  # CRITICAL: Pass for identity verification
        )
        print("Image generated successfully")
        
        # Step 4: Store result
        print("Uploading result...")
        result_path = f"outputs/jobs/{job_id}/result.jpg"
        result_url = await storage.upload_bytes(image_bytes, result_path)
        print(f"Result saved to: {result_url}")
        
        # Step 5: Extract state and update episodic memory (CRITICAL FOR MEMORY RETENTION!)
        print("Extracting state and updating episodic memory...")
        from app.workers.state import extract_state_task
        
        # memory_engine already initialized earlier in the request
        scene_index = memory_engine.get_next_scene_index(request.character_id)
        
        # Extract state from the generated image and store in episodic memory
        state_result = await extract_state_task(
            job_id=job_id,
            image_url=result_url,
            character_id=request.character_id,
            scene_index=scene_index,
            prompt_used=optimized_prompt
        )
        print(f"State extraction result: {state_result}")
        
        # Step 6: Face refinement (ENABLED - uses refiner worker)
        # If IDR < threshold, attempt to refine the face for better identity consistency
        idr_score = state_result.get("idr", 0) if state_result else 0
        final_url = result_url
        was_refined = False
        
        if idr_score > 0:
            print(f"IDR Score: {idr_score:.3f} (threshold: {settings.IDR_THRESHOLD})")
            
            # Trigger refinement if IDR is below threshold
            if idr_score < settings.IDR_THRESHOLD:
                print(f"[REFINE] IDR {idr_score:.3f} < threshold {settings.IDR_THRESHOLD}, attempting refinement...")
                try:
                    from app.workers.refiner import FaceRefiner
                    
                    refiner = FaceRefiner()
                    refined_bytes, new_idr, was_refined = await refiner.refine_if_needed(
                        image_bytes=image_bytes,
                        character_id=request.character_id,
                        character_data=character_data,
                        current_idr=idr_score
                    )
                    
                    if was_refined:
                        # Store refined result
                        refined_path = f"outputs/jobs/{job_id}/result_refined.jpg"
                        final_url = await storage.upload_bytes(refined_bytes, refined_path)
                        print(f"[REFINE] Success! IDR improved: {idr_score:.3f} -> {new_idr:.3f}")
                        idr_score = new_idr
                    else:
                        print(f"[REFINE] No improvement possible, using original")
                        
                except Exception as e:
                    print(f"[REFINE] Warning - Refinement failed: {e}")
                    # Continue with original image on refinement failure
        else:
            print("Note: IDR not computed (face not detected in generated image)")
        
        # Step 7: Trigger memory consolidation periodically (every 5 scenes)
        if scene_index % 5 == 0:
            print("Triggering periodic memory consolidation...")
            try:
                from app.services.memory_consolidation import consolidate_memory_task
                consolidation_result = await consolidate_memory_task(request.character_id)
                print(f"Memory consolidation: {consolidation_result.get('message', 'done')}")
            except Exception as e:
                print(f"[WARNING] Memory consolidation failed: {e}")
        
        # Calculate metrics
        end_time = datetime.utcnow()
        generation_time = (end_time - start_time).total_seconds()
        
        # Analyze memory quality for diagnostics
        memory_quality = memory_engine.analyze_memory_quality(request.character_id)
        
        # Update job with success
        db_job.status = "success"
        db_job.result_url = final_url
        db_job.completed_at = end_time
        db_job.metrics = {
            "generation_time_seconds": generation_time,
            "prompt_used": optimized_prompt[:500],
            "scene_index": scene_index,
            "episodic_count": len(episodic_states),
            "idr_score": idr_score,
            "was_refined": was_refined,
            "tags_extracted": state_result.get("tags", []) if state_result else [],
            "memory_quality": memory_quality.get("semantic_completeness", "unknown")
        }
        db.commit()
        
        return {
            "job_id": job_id,
            "status": "success",
            "message": f"Generation completed in {generation_time:.1f}s (Scene {scene_index})",
            "result_url": final_url,
            "scene_index": scene_index,
            "idr_score": idr_score,
            "was_refined": was_refined,
            "memory_context": f"{len(episodic_states)} previous scenes used for continuity"
        }
        
    except Exception as e:
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        
        # Log to file for debugging
        with open("error_log.txt", "a") as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"ERROR at {datetime.utcnow()}\n")
            f.write(f"{'='*60}\n")
            f.write(error_msg)
            f.write(f"\n{'='*60}\n\n")
        
        print(f"\n{'='*60}")
        print(f"GENERATION ERROR:")
        print(f"{'='*60}")
        print(error_msg)
        print(f"{'='*60}\n")
        
        db_job.status = "failed"
        db_job.error_message = str(e)
        db.commit()
        
        # Re-raise so FastAPI returns 500 but logs are printed
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Generation failed: {str(e)}"
        )
