"""
Extractor Worker
Handles identity extraction from reference images.

This is the CRITICAL component that:
1. Extracts face embeddings using InsightFace (ArcFace)
2. Analyzes visual traits using Gemini Vision
3. Stores semantic vector in VectorDB
4. Returns extracted identity data for char_metadata

This must be called during character creation to enable:
- IDR (Identity Retention) scoring
- Character consistency in generation
- Memory-augmented prompt generation
"""

import traceback
from typing import Dict, Any, Optional, List
from app.services.identity import IdentityService
from app.services.vectordb import VectorDBService
from app.services.storage import StorageService
from app.core.config import settings


async def extract_identity_task(
    character_id: str, 
    image_urls: list, 
    metadata: dict
) -> Dict[str, Any]:
    """
    Extract identity from reference images and store in vector DB.
    
    Args:
        character_id: The character's unique ID
        image_urls: List of reference image URLs (local or remote)
        metadata: Additional metadata (name, description, etc.)
        
    Returns:
        Dict containing:
        - vector_id: The semantic vector ID in VectorDB
        - traits: Extracted visual traits (face, hair, eyes, etc.)
        - embedding_dim: Dimension of the stored embedding
        - faces_detected: Number of faces successfully detected
    """
    print(f"[Extractor] Starting identity extraction for {character_id}")
    print(f"[Extractor] Processing {len(image_urls)} reference images...")
    
    result = {
        "vector_id": None,
        "traits": {},
        "embedding_dim": 512,
        "faces_detected": 0,
        "quality_score": 0.0,
        "error": None
    }
    
    try:
        storage_service = StorageService()
        identity_service = IdentityService(storage_service=storage_service)
        vectordb = VectorDBService()
        
        # Step 1: Extract identity embedding using InsightFace
        print("[Extractor] Step 1: Extracting face embeddings...")
        embedding = await identity_service.extract_identity(image_urls)
        
        if embedding is None or embedding.sum() == 0:
            print("[Extractor] [WARNING] No valid face embeddings extracted")
            # Continue anyway - we'll still try to extract visual traits
        else:
            result["embedding_dim"] = len(embedding)
            print(f"[Extractor] [OK] Embedding extracted ({len(embedding)}-dim)")
        
        # Step 2: Analyze reference image quality
        print("[Extractor] Step 2: Analyzing reference image quality...")
        quality_result = await identity_service.analyze_identity_quality(image_urls)
        result["faces_detected"] = quality_result.get("faces_detected", 0)
        result["quality_score"] = quality_result.get("average_quality", 0.0)
        print(f"[Extractor] Faces detected: {result['faces_detected']}/{len(image_urls)}")
        
        # Step 3: Extract visual traits using Gemini Vision
        print("[Extractor] Step 3: Extracting visual traits with Gemini Vision...")
        traits = await extract_visual_traits(image_urls[0] if image_urls else None, metadata)
        result["traits"] = traits
        print(f"[Extractor] [OK] Visual traits extracted: {list(traits.keys())}")
        
        # Step 4: Store embedding in VectorDB with traits as metadata
        print("[Extractor] Step 4: Storing semantic vector in VectorDB...")
        enhanced_metadata = {
            **metadata,
            **traits,
            "faces_detected": result["faces_detected"],
            "quality_score": result["quality_score"],
        }
        
        # Only store vector if we have a valid embedding
        if embedding is not None and embedding.sum() != 0:
            vector_id = await vectordb.upsert_semantic(
                character_id=character_id,
                embedding=embedding,
                metadata=enhanced_metadata
            )
            
            result["vector_id"] = vector_id
            print(f"[Extractor] [OK] Semantic vector stored: {vector_id}")
        else:
            print("[Extractor] [WARNING] No valid embedding to store - skipping vector storage")
            print("[Extractor] Character will be created but semantic_vector_id will be null")
            result["vector_id"] = None
        
        return result
        
    except Exception as e:
        error_msg = f"Identity extraction failed: {str(e)}\n{traceback.format_exc()}"
        print(f"[Extractor] [ERROR] {error_msg}")
        result["error"] = str(e)
        return result


async def extract_visual_traits(
    primary_image_url: Optional[str],
    metadata: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Extract visual traits from reference image using Gemini Vision.
    
    This populates the critical char_metadata fields:
    - face: Facial structure description
    - hair: Hair color, style, length
    - eyes: Eye color, shape
    - distinctives: Scars, tattoos, birthmarks, etc.
    - build: Body type, height estimate
    - age_range: Estimated age range
    - skin_tone: Skin complexion
    - tags: Descriptive tags for the character
    
    Args:
        primary_image_url: URL of the primary reference image
        metadata: Existing metadata (name, description)
        
    Returns:
        Dict with extracted visual traits
    """
    default_traits = {
        "face": "Not analyzed",
        "hair": "Not analyzed",
        "eyes": "Not analyzed",
        "eyebrows": "Not analyzed",
        "distinctives": "None detected",
        "build": "Average",
        "age_range": "Unknown",
        "skin_tone": "Not specified",
        "gender_presentation": "Not specified",
        "facial_expression": "Not captured",
        "initial_outfit": "Not captured",
        "initial_background": "Not captured",
        "background_objects": "Not captured",
        "accessories": "None",
        "props_in_hands": "None",
        "visible_objects": "None",
        "pose": "Not captured",
        "hand_position": "Not captured",
        "camera_angle": "eye-level",
        "camera_distance": "medium",
        "subject_facing": "camera",
        "lighting": "Not captured",
        "color_palette": "Not captured",
        "image_composition": "Not captured",
        "tags": []
    }
    
    if not primary_image_url:
        print("[Extractor] ERROR: No image URL provided for trait extraction")
        return default_traits
    
    try:
        from app.services.gemini_image import GeminiImageService
        
        storage_service = StorageService()
        gemini = GeminiImageService(storage_service=storage_service)
        
        print(f"[Extractor] Loading image from: {primary_image_url}")
        
        # Create a detailed analysis prompt for character traits
        analysis_prompt = """Analyze this person's appearance for character consistency in AI image generation.

Extract the following details in JSON format:

{
    "face": "Detailed description of facial structure (face shape, nose shape and size, lips fullness and shape, jaw shape, forehead height, cheekbone prominence, chin shape)",
    "hair": "EXACT hair color (be specific: jet black, dark brown, chestnut, auburn, etc.), style (straight, wavy, curly, coily), length (pixie, ear-length, shoulder, mid-back, etc.), any highlights or unique features",
    "eyes": "EXACT eye color (brown, dark brown, hazel, green, blue, gray - be specific), shape (almond, round, hooded, monolid, etc.), any distinctive features (long lashes, etc.)",
    "eyebrows": "Shape, thickness, color, any distinctive features",
    "distinctives": "ALL unique marks visible: scars (location, size), moles (location), tattoos (design and location), birthmarks, piercings (type and location), freckles pattern",
    "build": "Body type (slim, athletic, muscular, average, curvy, plus-size) and apparent height (short, average, tall)",
    "age_range": "Estimated age range (e.g., '25-30', 'mid-30s')",
    "skin_tone": "Specific skin complexion (e.g., 'fair/porcelain', 'light olive', 'medium tan', 'deep brown', 'dark ebony')",
    "gender_presentation": "How the person presents (masculine, feminine, androgynous)",
    "facial_expression": "EXACT expression (smiling, neutral, serious, laughing, etc.) and emotion conveyed",
    "initial_outfit": "EXACT description of clothing (colors with specific shades, patterns, style, fabric type, how it fits)",
    "initial_background": "COMPLETE background description (location type, colors, textures, atmosphere, depth)",
    "background_objects": "LIST every object visible in background (furniture, plants, decorations, wall items, etc.)",
    "accessories": "ALL accessories (jewelry type/color/position, glasses style, watch, belt, etc.)",
    "props_in_hands": "What the person is holding or touching (phone, cup, book, nothing, etc.)",
    "visible_objects": "ALL objects visible in the image (on table, nearby, in frame)",
    "pose": "EXACT body pose (standing/sitting/leaning, weight distribution, body orientation)",
    "hand_position": "EXACT hand positions (in pockets, crossed, holding something, resting where)",
    "camera_angle": "Choose ONE: eye-level | high-angle | low-angle | birds-eye | worms-eye | dutch-angle | over-the-shoulder | pov | oblique",
    "camera_distance": "Choose ONE: extreme-close-up (face only) | close-up (head+shoulders) | medium-close (waist up) | medium (knees up) | medium-long (full body with space) | long-shot (full body, environment visible) | extreme-long (subject small in frame)",
    "subject_facing": "Where subject is looking: camera | left | right | up | down | away",
    "lighting": "DETAILED lighting (source direction, type, shadows, highlights, color temperature)",
    "color_palette": "Dominant colors in the entire image (list 5-7 main colors)",
    "image_composition": "How the image is framed (headshot, half-body, full-body, centered, rule of thirds)",
    "tags": ["list", "of", "15-20", "descriptive", "tags", "for", "every", "visual", "element"]
}

CRITICAL INSTRUCTIONS:
- This data will be used to recreate this EXACT image with the EXACT same person.
- EVERY tiny detail matters: small objects, shadows, reflections, textures.
- Colors must be EXACT (not 'blue' but 'navy blue with slight purple undertone').
- Positions must be PRECISE (not 'hand up' but 'right hand raised to shoulder height, palm facing camera').
- Background objects matter: if there's a plant, describe it. If there's a picture frame, describe it.
- NOTHING should be omitted - if you can see it, describe it.
Return ONLY the JSON object, no additional text."""
        
        # Load the image and analyze it
        image_bytes = await gemini._load_image_bytes(primary_image_url)
        
        if image_bytes is None:
            print(f"[Extractor] ERROR: Could not load image from: {primary_image_url}")
            print(f"[Extractor] Returning default traits")
            return default_traits
        
        print(f"[Extractor] Image loaded successfully: {len(image_bytes)} bytes")
        print(f"[Extractor] Calling Gemini Vision for analysis...")
        
        # Use Gemini Vision to analyze
        analysis_result = await gemini.analyze_image(
            image_bytes=image_bytes,
            analysis_prompt=analysis_prompt
        )
        
        if "error" in analysis_result:
            error_str = str(analysis_result.get('error', ''))
            print(f"[Extractor] ERROR: Vision analysis failed: {analysis_result['error']}")
            
            # If it's a rate limit error, try Groq vision as fallback
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                print(f"[Extractor] [WARNING] Gemini rate limited, trying Groq vision fallback...")
                try:
                    from groq import Groq
                    groq_client = Groq(api_key=settings.GROQ_API_KEY)
                    
                    # Convert image to base64
                    import base64
                    image_b64 = base64.b64encode(image_bytes).decode('utf-8')
                    
                    response = groq_client.chat.completions.create(
                        model="meta-llama/llama-4-scout-17b-16e-instruct",
                        messages=[{
                            "role": "user",
                            "content": [
                                {"type": "text", "text": analysis_prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{image_b64}"
                                    }
                                }
                            ]
                        }],
                        temperature=0.4,
                        max_tokens=1024
                    )
                    
                    result_text = response.choices[0].message.content
                    # Parse JSON from response
                    import json
                    cleaned = result_text.replace("```json", "").replace("```", "").strip()
                    analysis_result = json.loads(cleaned)
                    print(f"[Extractor] [OK] Groq vision extracted traits successfully")
                    
                except Exception as groq_error:
                    print(f"[Extractor] [ERROR] Groq fallback also failed: {groq_error}")
                    print(f"[Extractor] Returning default traits")
                    return default_traits
            else:
                print(f"[Extractor] Returning default traits")
                return default_traits
        
        print(f"[Extractor] Gemini analysis received: {len(str(analysis_result))} chars")
        
        # Merge with defaults (in case some fields are missing)
        traits = {**default_traits}
        
        for key in default_traits.keys():
            if key in analysis_result and analysis_result[key]:
                traits[key] = analysis_result[key]
        
        # No need for extra loop since all fields are now in default_traits
        
        print(f"[Extractor] Extracted traits: face='{traits['face'][:50]}...', hair='{traits['hair']}'")
        print(f"[Extractor] Initial outfit: {traits.get('initial_outfit', 'Not captured')[:50]}...")
        return traits
        
    except Exception as e:
        print(f"[Extractor] ERROR: Visual trait extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return default_traits


async def reextract_identity(
    character_id: str,
    image_urls: List[str],
    db_session = None
) -> Dict[str, Any]:
    """
    Re-extract identity for an existing character.
    
    Use this when:
    - New reference images are added
    - Original extraction failed
    - User wants to update character appearance
    
    Args:
        character_id: The character's ID
        image_urls: New or updated reference images
        db_session: Optional database session for updating character record
        
    Returns:
        Extraction result dict
    """
    print(f"[Extractor] Re-extracting identity for {character_id}...")
    
    result = await extract_identity_task(
        character_id=character_id,
        image_urls=image_urls,
        metadata={"reextraction": True}
    )
    
    # Update character record if db_session provided
    if db_session and result.get("vector_id"):
        from app.models.character import Character
        
        character = db_session.query(Character).filter(
            Character.id == character_id
        ).first()
        
        if character:
            # Update semantic vector ID
            character.semantic_vector_id = result["vector_id"]
            
            # Merge new traits with existing metadata
            existing_meta = character.char_metadata or {}
            existing_meta.update(result.get("traits", {}))
            character.char_metadata = existing_meta
            
            db_session.commit()
            print(f"[Extractor] [OK] Character record updated")
    
    return result
