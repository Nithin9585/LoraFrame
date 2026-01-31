"""
Characters API Routes
Handles character creation, retrieval, and deletion.
"""

import uuid
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form, Query
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.models.character import Character
from app.models.episodic import EpisodicState
from app.schemas.character import CharacterResponse, CharacterHistory, CharacterUpdate
from app.services.storage import StorageService

router = APIRouter()


@router.post("/test", response_model=CharacterResponse, status_code=status.HTTP_201_CREATED)
async def create_test_character(
    name: str = Query(default="Test Character"),
    description: str = Query(default="A test character"),
    db: Session = Depends(get_db),
):
    """Create a test character without file upload (for testing only)."""
    character_id = f"char_{uuid.uuid4().hex[:8]}"
    
    db_character = Character(
        id=character_id,
        name=name,
        description=description,
        base_image_url="https://placeholder.com/test.jpg",
        consent_given_at=datetime.utcnow(),
        char_metadata={"test": True, "hair": "brown", "eyes": "blue"}
    )
    db.add(db_character)
    db.commit()
    db.refresh(db_character)
    
    return db_character


@router.post("", response_model=CharacterResponse, status_code=status.HTTP_201_CREATED)
async def create_character(
    name: str = Form(...),
    description: str = Form(None),
    consent: bool = Form(...),
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db),
):
    """
    Create a new character from reference images.
    
    This endpoint:
    1. Uploads reference images to storage
    2. Extracts face embeddings using InsightFace
    3. Analyzes visual traits using Gemini Vision
    4. Stores semantic vector in VectorDB
    5. Populates char_metadata with identity data
    
    The extracted data enables:
    - IDR (Identity Retention) scoring
    - Character consistency in generation
    - Memory-augmented prompt generation
    """
    if not consent:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Consent is required to create a character"
        )
    
    if len(files) < 1 or len(files) > 5:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Please upload between 1 and 5 reference images"
        )
    
    # Generate character ID
    character_id = f"char_{uuid.uuid4().hex[:8]}"
    
    # Upload images to local storage
    storage = StorageService()
    image_urls = []
    for idx, file in enumerate(files):
        try:
            url = await storage.upload_file(
                file=file,
                path=f"characters/{character_id}/ref_{idx}.jpg"
            )
            image_urls.append(url)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to upload file: {str(e)}"
            )
    
    # Create initial database record
    db_character = Character(
        id=character_id,
        name=name,
        description=description,
        semantic_vector_id=None,  # Will be set after extraction
        base_image_url=image_urls[0] if image_urls else None,
        consent_given_at=datetime.utcnow(),
        char_metadata={"reference_images": image_urls}  # Initial metadata
    )
    db.add(db_character)
    db.commit()
    
    # ============================================================
    # CRITICAL: Extract identity and create semantic memory
    # ============================================================
    print(f"[Characters API] Starting identity extraction for {character_id}...")
    
    try:
        from app.workers.extractor import extract_identity_task
        
        extraction_result = await extract_identity_task(
            character_id=character_id,
            image_urls=image_urls,
            metadata={
                "name": name,
                "description": description or "",
            }
        )
        
        # Update character with extraction results
        if extraction_result.get("vector_id"):
            db_character.semantic_vector_id = extraction_result["vector_id"]
            print(f"[Characters API] [OK] Semantic vector created: {extraction_result['vector_id']}")
        else:
            print(f"[Characters API] [WARNING] No semantic vector created")
        
        # Merge extracted traits into char_metadata
        traits = extraction_result.get("traits", {})
        if traits:
            updated_metadata = {
                "reference_images": image_urls,
                "face": traits.get("face", "Not analyzed"),
                "hair": traits.get("hair", "Not analyzed"),
                "eyes": traits.get("eyes", "Not analyzed"),
                "distinctives": traits.get("distinctives", "None"),
                "build": traits.get("build", "Average"),
                "age_range": traits.get("age_range", "Unknown"),
                "skin_tone": traits.get("skin_tone", "Not specified"),
                "tags": traits.get("tags", []),
                "gender_presentation": traits.get("gender_presentation", ""),
                "faces_detected": extraction_result.get("faces_detected", 0),
                "quality_score": extraction_result.get("quality_score", 0.0),
            }
            db_character.char_metadata = updated_metadata
            print(f"[Characters API] [OK] char_metadata populated with {len(traits)} traits")
        
        db.commit()
        db.refresh(db_character)
        
        print(f"[Characters API] [OK] Character {character_id} created with full identity data")
        
    except Exception as e:
        # Log the error but don't fail character creation
        # The character exists, just without full memory
        print(f"[Characters API] [ERROR] Identity extraction failed: {e}")
        print(f"[Characters API] Character created but memory system may be incomplete")
        import traceback
        traceback.print_exc()
    
    return db_character


@router.get("", response_model=List[CharacterResponse])
async def list_characters(
    limit: int = 20,
    offset: int = 0,
    db: Session = Depends(get_db),
):
    """List all characters."""
    characters = db.query(Character).offset(offset).limit(limit).all()
    return characters


@router.get("/{character_id}", response_model=CharacterResponse)
async def get_character(
    character_id: str,
    db: Session = Depends(get_db),
):
    """Get character metadata by ID."""
    character = db.query(Character).filter(Character.id == character_id).first()
    
    if not character:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Character not found"
        )
    
    return character


@router.put("/{character_id}", response_model=CharacterResponse)
async def update_character(
    character_id: str,
    update_data: CharacterUpdate,
    db: Session = Depends(get_db),
):
    """Update character details."""
    character = db.query(Character).filter(Character.id == character_id).first()
    
    if not character:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Character not found"
        )
    
    update_dict = update_data.model_dump(exclude_unset=True)
    for field, value in update_dict.items():
        if value is not None:
            setattr(character, field, value)
    
    character.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(character)
    
    return character


@router.patch("/{character_id}", response_model=CharacterResponse)
async def patch_character(
    character_id: str,
    update_data: CharacterUpdate,
    db: Session = Depends(get_db),
):
    """Partially update character details."""
    return await update_character(character_id, update_data, db)


@router.delete("/{character_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_character(
    character_id: str,
    db: Session = Depends(get_db),
):
    """Delete a character and all associated data."""
    character = db.query(Character).filter(Character.id == character_id).first()
    
    if not character:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Character not found"
        )
    
    # Delete from local storage
    try:
        storage = StorageService()
        await storage.delete_folder(f"characters/{character_id}/")
    except Exception:
        pass  # Continue even if storage deletion fails
    
    # Delete episodic states
    db.query(EpisodicState).filter(EpisodicState.character_id == character_id).delete()
    
    # Delete character record
    db.delete(character)
    db.commit()
    
    return None


@router.get("/{character_id}/history", response_model=CharacterHistory)
async def get_character_history(
    character_id: str,
    db: Session = Depends(get_db),
):
    """Get character's episodic history and all generated images."""
    character = db.query(Character).filter(Character.id == character_id).first()
    
    if not character:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Character not found"
        )
    
    episodic_states = db.query(EpisodicState).filter(
        EpisodicState.character_id == character_id
    ).order_by(EpisodicState.scene_index.asc()).all()
    
    return {
        "character": character,
        "episodic_states": episodic_states,
        "total_scenes": len(episodic_states)
    }


@router.get("/{character_id}/memory-status")
async def get_memory_status(
    character_id: str,
    db: Session = Depends(get_db),
):
    """
    Check the health of a character's memory system.
    
    Returns detailed status of:
    - Semantic memory (identity vector)
    - Character metadata (visual traits)
    - Episodic memory (scene history)
    - Overall health score
    
    Use this to diagnose memory issues.
    """
    character = db.query(Character).filter(Character.id == character_id).first()
    
    if not character:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Character not found"
        )
    
    # Check semantic memory
    semantic_status = {
        "has_vector": character.semantic_vector_id is not None,
        "vector_id": character.semantic_vector_id,
        "status": "OK" if character.semantic_vector_id else "MISSING"
    }
    
    # Check metadata
    metadata = character.char_metadata or {}
    required_fields = ["face", "hair", "eyes", "distinctives"]
    populated_fields = [f for f in required_fields if metadata.get(f) and metadata.get(f) != "Not analyzed"]
    
    metadata_status = {
        "has_reference_images": bool(metadata.get("reference_images")),
        "reference_image_count": len(metadata.get("reference_images", [])),
        "populated_fields": populated_fields,
        "missing_fields": [f for f in required_fields if f not in populated_fields],
        "faces_detected": metadata.get("faces_detected", 0),
        "quality_score": metadata.get("quality_score", 0.0),
        "status": "OK" if len(populated_fields) >= 3 else "INCOMPLETE" if len(populated_fields) > 0 else "MISSING"
    }
    
    # Check episodic memory
    episodic_count = db.query(EpisodicState).filter(
        EpisodicState.character_id == character_id
    ).count()
    
    episodic_status = {
        "total_scenes": episodic_count,
        "status": "OK" if episodic_count > 0 else "EMPTY"
    }
    
    # Check learned traits (memory consolidation)
    learned_traits = metadata.get("learned_traits", {})
    has_learned = bool(learned_traits)
    learned_validated = learned_traits.get("quality_metrics", {}).get("validated", False) if has_learned else False
    
    learned_traits_status = {
        "has_learned_traits": has_learned,
        "episodes_analyzed": learned_traits.get("learned_from_episodes", 0),
        "validated": learned_validated,
        "high_confidence_patterns": learned_traits.get("quality_metrics", {}).get("high_confidence_patterns", 0),
        "manual_override": learned_traits.get("manual_override", False),
        "status": "VALIDATED" if learned_validated else "NEEDS_MORE_DATA" if episodic_count < 5 else "UNVALIDATED"
    }
    
    # Calculate overall health
    issues = []
    if not semantic_status["has_vector"]:
        issues.append("No semantic vector - identity cannot be preserved")
    if metadata_status["status"] == "MISSING":
        issues.append("No visual traits extracted - prompts will be generic")
    elif metadata_status["status"] == "INCOMPLETE":
        issues.append(f"Missing traits: {', '.join(metadata_status['missing_fields'])}")
    if has_learned and not learned_validated:
        issues.append("Learned traits exist but validation failed - low confidence patterns")
    
    health_score = 100
    if not semantic_status["has_vector"]:
        health_score -= 40
    if metadata_status["status"] == "MISSING":
        health_score -= 30
    elif metadata_status["status"] == "INCOMPLETE":
        health_score -= 15
    if metadata.get("faces_detected", 0) == 0:
        health_score -= 20
    if has_learned and not learned_validated:
        health_score -= 10
    
    return {
        "character_id": character_id,
        "character_name": character.name,
        "semantic_memory": semantic_status,
        "character_metadata": metadata_status,
        "episodic_memory": episodic_status,
        "learned_traits": learned_traits_status,
        "health_score": max(0, health_score),
        "health_status": "HEALTHY" if health_score >= 80 else "DEGRADED" if health_score >= 50 else "CRITICAL",
        "issues": issues,
        "recommendation": "Run POST /{character_id}/reextract-identity to fix" if issues else None
    }


@router.post("/{character_id}/reextract-identity", response_model=CharacterResponse)
async def reextract_identity(
    character_id: str,
    db: Session = Depends(get_db),
):
    """
    Re-extract identity for an existing character.
    
    Use this when:
    - Original extraction failed
    - Character is missing semantic vector
    - Visual traits are incomplete
    - Reference images were updated
    
    This will:
    1. Re-analyze reference images
    2. Create/update semantic vector
    3. Populate/update char_metadata with traits
    """
    character = db.query(Character).filter(Character.id == character_id).first()
    
    if not character:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Character not found"
        )
    
    # Get reference images
    metadata = character.char_metadata or {}
    image_urls = metadata.get("reference_images", [])
    
    if not image_urls and character.base_image_url:
        image_urls = [character.base_image_url]
    
    if not image_urls:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No reference images available for re-extraction"
        )
    
    # Convert old GCS URLs to API proxy URLs
    storage = StorageService()
    converted_urls = [storage.convert_gcs_url_to_api(url) for url in image_urls]
    print(f"[Characters API] Re-extracting identity for {character_id}...")
    print(f"[Characters API] Converted {len([u for u in image_urls if u.startswith('https://storage.googleapis.com')])} old GCS URLs")
    
    try:
        from app.workers.extractor import extract_identity_task
        
        extraction_result = await extract_identity_task(
            character_id=character_id,
            image_urls=converted_urls,  # Use converted URLs
            metadata={
                "name": character.name,
                "description": character.description or "",
                "reextraction": True
            }
        )
        
        # Update semantic vector
        if extraction_result.get("vector_id"):
            character.semantic_vector_id = extraction_result["vector_id"]
            print(f"[Characters API] [OK] Semantic vector updated: {extraction_result['vector_id']}")
        else:
            # No vector was created - check why
            faces_detected = extraction_result.get("faces_detected", 0)
            error_msg = extraction_result.get("error")
            
            if faces_detected == 0:
                print(f"[Characters API] [WARNING] No semantic vector created: No faces detected in images")
                print(f"[Characters API] Please upload clear photos with visible faces for identity extraction")
            elif error_msg:
                print(f"[Characters API] [WARNING] No semantic vector created: {error_msg}")
            else:
                print(f"[Characters API] [WARNING] No semantic vector created: Unknown reason")
        
        # Update char_metadata with new traits and converted URLs
        traits = extraction_result.get("traits", {})
        if traits:
            updated_metadata = {
                "reference_images": converted_urls,  # Store converted URLs
                "face": traits.get("face", metadata.get("face", "Not analyzed")),
                "hair": traits.get("hair", metadata.get("hair", "Not analyzed")),
                "eyes": traits.get("eyes", metadata.get("eyes", "Not analyzed")),
                "distinctives": traits.get("distinctives", metadata.get("distinctives", "None")),
                "build": traits.get("build", metadata.get("build", "Average")),
                "age_range": traits.get("age_range", metadata.get("age_range", "Unknown")),
                "skin_tone": traits.get("skin_tone", metadata.get("skin_tone", "Not specified")),
                "tags": traits.get("tags", metadata.get("tags", [])),
                "gender_presentation": traits.get("gender_presentation", ""),
                "faces_detected": extraction_result.get("faces_detected", 0),
                "quality_score": extraction_result.get("quality_score", 0.0),
                "last_extraction": datetime.utcnow().isoformat(),
            }
            character.char_metadata = updated_metadata
        
        # Update base_image_url if it's an old GCS URL
        if character.base_image_url and character.base_image_url.startswith("https://storage.googleapis.com/"):
            character.base_image_url = storage.convert_gcs_url_to_api(character.base_image_url)
            print(f"[Characters API] Converted base_image_url to API proxy URL")
        
        character.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(character)
        
        # Add helpful message if no vector was created
        if not character.semantic_vector_id:
            faces = extraction_result.get("faces_detected", 0)
            print(f"[Characters API] [OK] Character {character_id} updated, but semantic_vector_id is still null")
            print(f"[Characters API] Reason: {faces} faces detected in reference images")
            if faces == 0:
                print(f"[Characters API] Solution: Upload clear photos showing the person's face")
        else:
            print(f"[Characters API] [OK] Character {character_id} identity re-extracted successfully")
        
        return character
        
    except Exception as e:
        print(f"[Characters API] [ERROR] Re-extraction failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Identity re-extraction failed: {str(e)}"
        )


@router.post("/{character_id}/consolidate-memory")
async def consolidate_character_memory(
    character_id: str,
    db: Session = Depends(get_db)
):
    """
    Consolidate episodic memory into learned traits for a character.
    
    Analyzes all episodic states to extract patterns and update the
    character's learned_traits in metadata.
    """
    from app.services.memory_consolidation import MemoryConsolidationService
    
    character = db.query(Character).filter(Character.id == character_id).first()
    if not character:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Character not found"
        )
    
    try:
        service = MemoryConsolidationService(db)
        result = await service.consolidate_memory(character_id)
        
        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.get("message", "Memory consolidation failed")
            )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[Characters API] [ERROR] Memory consolidation failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Memory consolidation failed: {str(e)}"
        )


@router.put("/{character_id}/override-learned-traits")
async def override_learned_traits(
    character_id: str,
    learned_traits: dict,
    db: Session = Depends(get_db)
):
    """
    Manually override or correct learned traits for a character.
    
    Use this to fix incorrect patterns or set custom defaults.
    
    Example request body:
    {
        "default_clothing": ["leather_jacket", "jeans"],
        "signature_poses": ["confident_stance"],
        "common_props": ["sword"],
        "manual_override": true
    }
    """
    character = db.query(Character).filter(Character.id == character_id).first()
    if not character:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Character not found"
        )
    
    try:
        # Get current metadata
        current_metadata = character.char_metadata or {}
        
        # Update learned traits with manual override flag
        updated_traits = {
            **learned_traits,
            "manual_override": True,
            "last_updated": datetime.utcnow().isoformat(),
            "override_reason": "Manual correction by user"
        }
        
        current_metadata["learned_traits"] = updated_traits
        character.char_metadata = current_metadata
        character.updated_at = datetime.utcnow()
        
        db.commit()
        db.refresh(character)
        
        print(f"[Characters API] [OK] Learned traits manually overridden for {character_id}")
        
        return {
            "success": True,
            "character_id": character_id,
            "learned_traits": updated_traits,
            "message": "Learned traits successfully overridden"
        }
        
    except Exception as e:
        print(f"[Characters API] [ERROR] Override failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to override learned traits: {str(e)}"
        )


@router.post("/migrate-urls")
async def migrate_all_urls_to_api(
    db: Session = Depends(get_db),
):
    """
    Migrate all old GCS URLs to new API proxy URLs.
    
    This is a one-time migration endpoint to fix existing characters
    that have old https://storage.googleapis.com/ URLs.
    
    Updates:
    - base_image_url
    - char_metadata.reference_images
    
    Safe to run multiple times (idempotent).
    """
    try:
        storage = StorageService()
        characters = db.query(Character).all()
        
        updated_count = 0
        skipped_count = 0
        
        for character in characters:
            needs_update = False
            
            # Update base_image_url
            if character.base_image_url and character.base_image_url.startswith("https://storage.googleapis.com/"):
                old_url = character.base_image_url
                character.base_image_url = storage.convert_gcs_url_to_api(old_url)
                needs_update = True
                print(f"[Migration] Updated base_image_url for {character.id}")
            
            # Update reference_images in metadata
            metadata = character.char_metadata or {}
            ref_images = metadata.get("reference_images", [])
            
            if ref_images:
                new_ref_images = []
                for url in ref_images:
                    if url.startswith("https://storage.googleapis.com/"):
                        new_ref_images.append(storage.convert_gcs_url_to_api(url))
                        needs_update = True
                    else:
                        new_ref_images.append(url)
                
                if needs_update:
                    metadata["reference_images"] = new_ref_images
                    character.char_metadata = metadata
                    print(f"[Migration] Updated {len([u for u in ref_images if u.startswith('https://storage')])} reference URLs for {character.id}")
            
            if needs_update:
                character.updated_at = datetime.utcnow()
                updated_count += 1
            else:
                skipped_count += 1
        
        db.commit()
        
        return {
            "success": True,
            "total_characters": len(characters),
            "updated": updated_count,
            "skipped": skipped_count,
            "message": f"Successfully migrated {updated_count} characters to use API proxy URLs"
        }
        
    except Exception as e:
        db.rollback()
        print(f"[Migration] [ERROR] URL migration failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"URL migration failed: {str(e)}"
        )


# ============================================================================
# LoRA Training Endpoints
# ============================================================================

@router.get("/{character_id}/lora/status")
async def get_lora_training_status(
    character_id: str,
    db: Session = Depends(get_db)
):
    """
    Check if character is ready for LoRA training.
    
    Returns the number of golden images collected and training readiness.
    """
    # Verify character exists
    character = db.query(Character).filter(Character.id == character_id).first()
    if not character:
        raise HTTPException(status_code=404, detail="Character not found")
    
    from app.services.lora_cloud_trainer import LoraCloudTrainer
    trainer = LoraCloudTrainer()
    
    readiness = trainer.check_training_readiness(character_id)
    readiness["character_name"] = character.name
    
    return readiness


@router.post("/{character_id}/lora/train")
async def trigger_lora_training(
    character_id: str,
    training_steps: int = 500,
    learning_rate: float = 1e-4,
    rank: int = 16,
    db: Session = Depends(get_db)
):
    """
    Trigger LoRA training for a character.
    
    Requires:
    - At least 10 golden images (IDR > 0.85)
    - Cloud deployment with Vertex AI enabled
    
    Training runs on Vertex AI with GPU and takes ~15-30 minutes.
    """
    from app.core.config import settings
    
    # Verify character exists
    character = db.query(Character).filter(Character.id == character_id).first()
    if not character:
        raise HTTPException(status_code=404, detail="Character not found")
    
    # Check readiness
    from app.services.lora_cloud_trainer import LoraCloudTrainer
    trainer = LoraCloudTrainer()
    readiness = trainer.check_training_readiness(character_id)
    
    if not readiness.get("ready_for_training"):
        raise HTTPException(
            status_code=400,
            detail=f"Not ready for training: {readiness.get('message')}"
        )
    
    # Create LoRA model record
    from app.models.lora import LoraModel
    lora_model_id = f"lora_{uuid.uuid4().hex[:12]}"
    
    lora_model = LoraModel(
        id=lora_model_id,
        character_id=character_id,
        status="training",
        training_images_count=readiness.get("golden_images", 0),
        training_steps=training_steps,
        learning_rate=learning_rate,
        rank=rank
    )
    db.add(lora_model)
    db.commit()
    
    # Prepare dataset path
    dataset_gcs_path = f"gs://{settings.GCS_BUCKET_OUTPUTS}/golden_images/{character_id}"
    
    # Submit training job
    result = trainer.submit_training_job(
        character_id=character_id,
        lora_model_id=lora_model_id,
        dataset_gcs_path=dataset_gcs_path,
        training_steps=training_steps,
        learning_rate=learning_rate,
        rank=rank
    )
    
    if result.get("success"):
        # Update model with job info
        lora_model.vertex_job_id = result.get("job_id")
        lora_model.weights_path = result.get("output_path")
        db.commit()
    else:
        lora_model.status = "failed"
        lora_model.error_message = result.get("error")
        db.commit()
    
    return {
        "lora_model_id": lora_model_id,
        "character_id": character_id,
        **result
    }


@router.get("/{character_id}/lora/models")
async def list_character_lora_models(
    character_id: str,
    db: Session = Depends(get_db)
):
    """List all LoRA models trained for a character."""
    from app.models.lora import LoraModel
    
    models = db.query(LoraModel).filter(
        LoraModel.character_id == character_id
    ).order_by(LoraModel.created_at.desc()).all()
    
    return {
        "character_id": character_id,
        "models": [
            {
                "id": m.id,
                "status": m.status,
                "created_at": m.created_at.isoformat() if m.created_at else None,
                "training_images_count": m.training_images_count,
                "validation_idr": m.validation_idr,
                "is_active": m.is_active
            }
            for m in models
        ]
    }

