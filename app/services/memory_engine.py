"""
Character Memory Engine
Core service for character memory management - the brain of the Character AI system.

This implements:
1. Semantic Memory: Who the character IS (permanent traits, identity)
2. Episodic Memory: What the character HAS DONE (scene-by-scene state)
3. Memory Merging: Weighted combination for prompt generation

Memory Architecture:
┌─────────────────────────────────────────────────────────────────┐
│                    CHARACTER MEMORY                             │
├──────────────────────┬──────────────────────────────────────────┤
│   SEMANTIC MEMORY    │           EPISODIC MEMORY                │
│   (Identity Core)    │         (Scene History)                   │
├──────────────────────┼──────────────────────────────────────────┤
│ • Face embedding     │ • Scene 1: tags, state, image            │
│ • Name, traits       │ • Scene 2: tags, state, image            │
│ • Distinctive marks  │ • Scene 3: tags, state, image (newest)   │
│ • Hair, eye color    │                                          │
│ • CLIP style vector  │ Recent scenes influence prompts more     │
└──────────────────────┴──────────────────────────────────────────┘
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import numpy as np
from sqlalchemy.orm import Session

from app.core.config import settings


@dataclass
class CharacterSheet:
    """Canonical character identity sheet."""
    name: str
    face_description: str
    hair: str
    eyes: str
    skin_tone: str
    distinctive_marks: str
    age_range: str
    build: str
    initial_outfit: str
    accessories: str
    semantic_tags: List[str]
    
    def to_prompt_block(self) -> str:
        """Convert to prompt template block."""
        return f"""[CHARACTER_SHEET]
Name: {self.name}
Face: {self.face_description}
Hair: {self.hair}
Eyes: {self.eyes}
Skin: {self.skin_tone}
Distinctives: {self.distinctive_marks}
Age: {self.age_range}
Build: {self.build}
Initial Outfit: {self.initial_outfit}
Accessories: {self.accessories}
Tags: {', '.join(self.semantic_tags)}"""


@dataclass
class EpisodicMemory:
    """Single episodic memory entry."""
    scene_index: int
    tags: List[str]
    clothing: List[str]
    physical_state: List[str]
    props: List[str]
    pose: str
    environment: str
    image_url: Optional[str]
    idr_score: float
    
    def to_prompt_line(self) -> str:
        """Convert to single prompt line for context."""
        tag_str = ', '.join(self.tags[:5]) if self.tags else 'untagged'
        return f"Scene {self.scene_index}: {tag_str} | {self.pose} | {self.environment}"


@dataclass
class MergedMemory:
    """Combined semantic + episodic memory for generation."""
    character_sheet: CharacterSheet
    recent_episodes: List[EpisodicMemory]
    current_clothing: List[str]  # From most recent episode
    current_state: List[str]     # Active physical states
    current_props: List[str]     # Props still in possession
    confidence_score: float      # Average IDR from recent episodes


class CharacterMemoryEngine:
    """
    Core Character Memory Engine.
    
    Responsible for:
    1. Building character sheets from metadata
    2. Retrieving and ranking episodic memories
    3. Merging semantic + episodic for prompt generation
    4. Tracking state continuity across scenes
    """
    
    def __init__(self, db: Session):
        self.db = db
        self.semantic_weight = settings.SEMANTIC_WEIGHT
        self.episodic_weight = settings.EPISODIC_WEIGHT
        self.episodic_decay = settings.EPISODIC_DECAY
        self.top_k = settings.EPISODIC_TOP_K
    
    def get_character_sheet(self, character_id: str) -> Optional[CharacterSheet]:
        """
        Build a CharacterSheet from database metadata.
        """
        from app.models.character import Character
        
        character = self.db.query(Character).filter(
            Character.id == character_id
        ).first()
        
        if not character:
            return None
        
        metadata = character.char_metadata or {}
        
        return CharacterSheet(
            name=character.name,
            face_description=metadata.get("face", "Not specified"),
            hair=metadata.get("hair", "Not specified"),
            eyes=metadata.get("eyes", "Not specified"),
            skin_tone=metadata.get("skin_tone", "Not specified"),
            distinctive_marks=metadata.get("distinctives", "None"),
            age_range=metadata.get("age_range", "Unknown"),
            build=metadata.get("build", "Average"),
            initial_outfit=metadata.get("initial_outfit", "Not captured"),
            accessories=metadata.get("accessories", "None"),
            semantic_tags=metadata.get("tags", [])
        )
    
    def get_recent_episodes(
        self, 
        character_id: str, 
        limit: Optional[int] = None
    ) -> List[EpisodicMemory]:
        """
        Retrieve recent episodic memories for a character.
        Returns in chronological order (oldest to newest).
        """
        from app.models.episodic import EpisodicState
        
        limit = limit or self.top_k
        
        episodes = self.db.query(EpisodicState).filter(
            EpisodicState.character_id == character_id
        ).order_by(
            EpisodicState.scene_index.desc()
        ).limit(limit).all()
        
        # Convert to EpisodicMemory objects in chronological order
        memories = []
        for ep in reversed(episodes):
            state_data = ep.state_data or {}
            memories.append(EpisodicMemory(
                scene_index=ep.scene_index,
                tags=ep.tags or [],
                clothing=state_data.get("clothing", []),
                physical_state=state_data.get("physical_state", []),
                props=state_data.get("props", []),
                pose=state_data.get("pose", ""),
                environment=state_data.get("environment", ""),
                image_url=ep.image_url,
                idr_score=state_data.get("idr_score", 0.0)
            ))
        
        return memories
    
    def merge_memory(self, character_id: str) -> Optional[MergedMemory]:
        """
        Merge semantic and episodic memory for prompt generation.
        
        Weighted merge algorithm:
        - Semantic (identity) gets weight w_s (default 0.6)
        - Episodic (scene state) gets weight w_e (default 0.4)
        - Recent episodes decay: alpha^(i-1) where i is recency rank
        
        For current state:
        - Clothing: from most recent episode
        - Physical state: accumulated from recent (injuries persist)
        - Props: from most recent (unless explicitly dropped)
        """
        sheet = self.get_character_sheet(character_id)
        if not sheet:
            return None
        
        episodes = self.get_recent_episodes(character_id)
        
        if not episodes:
            # No episodic memory yet - use only semantic
            return MergedMemory(
                character_sheet=sheet,
                recent_episodes=[],
                current_clothing=[],
                current_state=[],
                current_props=[],
                confidence_score=1.0
            )
        
        # Get current state from most recent episode
        latest = episodes[-1] if episodes else None
        
        # Accumulate physical state (some things persist like injuries)
        persistent_states = set()
        for ep in episodes:
            for state in ep.physical_state:
                # Injuries and conditions persist
                if any(kw in state.lower() for kw in 
                       ["injured", "wounded", "scar", "bruise", "tired", "wet"]):
                    persistent_states.add(state)
        
        # Calculate average IDR confidence
        idr_scores = [ep.idr_score for ep in episodes if ep.idr_score > 0]
        avg_idr = sum(idr_scores) / len(idr_scores) if idr_scores else 0.0
        
        return MergedMemory(
            character_sheet=sheet,
            recent_episodes=episodes,
            current_clothing=latest.clothing if latest else [],
            current_state=list(persistent_states),
            current_props=latest.props if latest else [],
            confidence_score=avg_idr
        )
    
    def build_prompt_context(self, character_id: str, user_prompt: str) -> Dict[str, Any]:
        """
        Build complete context dictionary for LLM prompt generation.
        
        Returns a structured dict that can be passed to GroqLLMService.
        """
        merged = self.merge_memory(character_id)
        
        if not merged:
            return {
                "error": "Character not found",
                "character_id": character_id
            }
        
        # Build recent states text for prompt
        recent_states_text = "\n".join([
            ep.to_prompt_line() for ep in merged.recent_episodes
        ]) if merged.recent_episodes else "No previous scenes"
        
        return {
            "name": merged.character_sheet.name,
            "face": merged.character_sheet.face_description,
            "hair": merged.character_sheet.hair,
            "eyes": merged.character_sheet.eyes,
            "distinctives": merged.character_sheet.distinctive_marks,
            "age_range": merged.character_sheet.age_range,
            "build": merged.character_sheet.build,
            "tags": merged.character_sheet.semantic_tags,
            "recent_states": recent_states_text,
            "current_clothing": merged.current_clothing,
            "current_state": merged.current_state,
            "current_props": merged.current_props,
            "confidence": merged.confidence_score,
            "episode_count": len(merged.recent_episodes),
            "user_prompt": user_prompt
        }
    
    def get_next_scene_index(self, character_id: str) -> int:
        """Get the next scene index for a character."""
        from app.models.episodic import EpisodicState
        
        latest = self.db.query(EpisodicState).filter(
            EpisodicState.character_id == character_id
        ).order_by(EpisodicState.scene_index.desc()).first()
        
        return (latest.scene_index + 1) if latest else 1
    
    def analyze_memory_quality(self, character_id: str) -> Dict[str, Any]:
        """
        Analyze the quality and completeness of character memory.
        Useful for debugging and improvements.
        """
        sheet = self.get_character_sheet(character_id)
        episodes = self.get_recent_episodes(character_id, limit=10)
        
        if not sheet:
            return {"error": "Character not found"}
        
        # Check semantic completeness
        semantic_fields = [
            sheet.face_description, sheet.hair, sheet.eyes,
            sheet.distinctive_marks, sheet.age_range, sheet.build
        ]
        semantic_complete = sum(1 for f in semantic_fields if f and f != "Not specified") / 6
        
        # Analyze episodic history
        avg_tags = sum(len(ep.tags) for ep in episodes) / len(episodes) if episodes else 0
        idr_scores = [ep.idr_score for ep in episodes if ep.idr_score > 0]
        
        return {
            "character_id": character_id,
            "character_name": sheet.name,
            "semantic_completeness": f"{semantic_complete * 100:.0f}%",
            "total_episodes": len(episodes),
            "average_tags_per_episode": f"{avg_tags:.1f}",
            "average_idr": f"{sum(idr_scores) / len(idr_scores) * 100:.1f}%" if idr_scores else "N/A",
            "has_distinctive_marks": sheet.distinctive_marks != "None",
            "recommendations": self._get_memory_recommendations(sheet, episodes)
        }
    
    def _get_memory_recommendations(
        self, 
        sheet: CharacterSheet, 
        episodes: List[EpisodicMemory]
    ) -> List[str]:
        """Generate recommendations to improve memory quality."""
        recs = []
        
        if sheet.face_description == "Not specified":
            recs.append("Add detailed face description for better identity retention")
        
        if sheet.distinctive_marks == "None":
            recs.append("Add distinctive marks (scars, tattoos, etc.) for uniqueness")
        
        if len(episodes) < 3:
            recs.append("Generate more scenes to build episodic history")
        
        if episodes:
            low_idr = [ep for ep in episodes if ep.idr_score > 0 and ep.idr_score < 0.7]
            if len(low_idr) > len(episodes) / 2:
                recs.append("Low IDR scores - consider uploading more reference images")
        
        if not recs:
            recs.append("Memory quality looks good! ✨")
        
        return recs
