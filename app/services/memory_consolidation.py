"""
Memory Consolidation Service
Learns patterns from episodic memory and updates semantic memory over time.

This implements long-term memory learning for character AI:
- Detects repeated patterns in episodic states
- Consolidates frequent states into "default" character traits
- Updates semantic memory with learned preferences

Memory Consolidation Architecture:
┌─────────────────────────────────────────────────────────────────────┐
│                    MEMORY CONSOLIDATION                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Episodic Memory (Many Scenes)                                     │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │ Scene 1: leather_jacket, determined_expression              │   │
│   │ Scene 2: leather_jacket, sword_prop                         │   │
│   │ Scene 3: leather_jacket, battle_stance                      │   │
│   │ Scene 4: formal_suit, confident_expression                  │   │
│   │ Scene 5: leather_jacket, motorcycle_prop                    │   │
│   │ ...                                                         │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                              ↓                                      │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │            Pattern Analysis Engine                          │   │
│   │  - Count tag frequencies                                    │   │
│   │  - Detect co-occurrence patterns                            │   │
│   │  - Identify "signature" traits                              │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                              ↓                                      │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │            Learned Character Traits                         │   │
│   │  - default_clothing: leather_jacket (80% of scenes)         │   │
│   │  - signature_poses: battle_stance, confident                │   │
│   │  - common_props: sword, motorcycle                          │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                              ↓                                      │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │            Updated Semantic Memory                          │   │
│   │  char_metadata.learned_traits = {...}                       │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
"""
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter
from dataclasses import dataclass
from sqlalchemy.orm import Session
from datetime import datetime


@dataclass
class LearnedTraits:
    """Traits learned from episodic memory patterns."""
    default_clothing: List[str]
    signature_poses: List[str]
    common_props: List[str]
    typical_environments: List[str]
    recurring_states: List[str]
    confidence_scores: Dict[str, float]
    total_episodes_analyzed: int


class MemoryConsolidationService:
    """
    Service for consolidating episodic memories into learned traits.
    
    This enables the character to develop consistent "habits" and
    "preferences" based on their history of generated scenes.
    """
    
    def __init__(self, db: Session):
        self.db = db
        self.min_episodes_for_learning = 5  # Minimum scenes before learning
        self.pattern_threshold = 0.3  # Minimum frequency to be considered a pattern
        self.strong_pattern_threshold = 0.6  # Threshold for "default" traits
        self.confidence_threshold = 0.4  # Minimum confidence to trust a learned trait
        self.high_confidence_threshold = 0.7  # High confidence threshold
    
    async def analyze_patterns(self, character_id: str) -> LearnedTraits:
        """
        Analyze episodic memory to extract patterns.
        
        Returns LearnedTraits with discovered patterns and their confidence.
        """
        from app.models.episodic import EpisodicState
        
        # Get all episodic states for character
        episodes = self.db.query(EpisodicState).filter(
            EpisodicState.character_id == character_id
        ).order_by(EpisodicState.scene_index.asc()).all()
        
        if len(episodes) < self.min_episodes_for_learning:
            return LearnedTraits(
                default_clothing=[],
                signature_poses=[],
                common_props=[],
                typical_environments=[],
                recurring_states=[],
                confidence_scores={},
                total_episodes_analyzed=len(episodes)
            )
        
        # Collect all state data
        clothing_counts = Counter()
        pose_counts = Counter()
        prop_counts = Counter()
        environment_counts = Counter()
        state_counts = Counter()
        
        for ep in episodes:
            state_data = ep.state_data or {}
            
            # Count clothing items
            for item in state_data.get("clothing", []):
                clothing_counts[item.lower()] += 1
            
            # Count poses
            pose = state_data.get("pose", "")
            if pose:
                # Extract key pose descriptors
                pose_words = self._extract_key_terms(pose)
                for word in pose_words:
                    pose_counts[word] += 1
            
            # Count props
            for prop in state_data.get("props", []):
                prop_counts[prop.lower()] += 1
            
            # Count environments
            env = state_data.get("environment", "")
            if env:
                env_words = self._extract_key_terms(env)
                for word in env_words:
                    environment_counts[word] += 1
            
            # Count physical states
            for state in state_data.get("physical_state", []):
                state_counts[state.lower()] += 1
            
            # Also count tags
            for tag in (ep.tags or []):
                # Categorize tags
                tag_lower = tag.lower()
                if tag_lower.startswith("wearing_"):
                    clothing_counts[tag_lower.replace("wearing_", "")] += 1
                elif tag_lower.startswith("has_"):
                    prop_counts[tag_lower.replace("has_", "")] += 1
                else:
                    state_counts[tag_lower] += 1
        
        total = len(episodes)
        
        # Extract patterns above threshold
        default_clothing = self._extract_patterns(clothing_counts, total, self.strong_pattern_threshold)
        signature_poses = self._extract_patterns(pose_counts, total, self.pattern_threshold)
        common_props = self._extract_patterns(prop_counts, total, self.pattern_threshold)
        typical_environments = self._extract_patterns(environment_counts, total, self.pattern_threshold)
        recurring_states = self._extract_patterns(state_counts, total, self.pattern_threshold)
        
        # Build confidence scores
        confidence_scores = {}
        for item, count in clothing_counts.most_common(5):
            confidence_scores[f"clothing:{item}"] = count / total
        for item, count in pose_counts.most_common(3):
            confidence_scores[f"pose:{item}"] = count / total
        for item, count in prop_counts.most_common(3):
            confidence_scores[f"prop:{item}"] = count / total
        
        return LearnedTraits(
            default_clothing=default_clothing,
            signature_poses=signature_poses,
            common_props=common_props,
            typical_environments=typical_environments,
            recurring_states=recurring_states,
            confidence_scores=confidence_scores,
            total_episodes_analyzed=total
        )
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key descriptive terms from text."""
        # Simple extraction - could use NLP for better results
        import re
        
        # Remove common filler words
        stop_words = {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'with', 'in', 'on', 'at', 'to', 'for', 'of', 'and', 'or'}
        
        words = re.findall(r'\b[a-z]+\b', text.lower())
        return [w for w in words if w not in stop_words and len(w) > 3]
    
    def _extract_patterns(
        self, 
        counts: Counter, 
        total: int, 
        threshold: float
    ) -> List[str]:
        """Extract items that appear above the threshold frequency."""
        return [
            item for item, count in counts.most_common(10)
            if count / total >= threshold
        ]
    
    async def consolidate_memory(self, character_id: str) -> Dict[str, Any]:
        """
        Consolidate episodic patterns into the character's semantic memory.
        
        Updates the character's metadata with learned traits.
        """
        from app.models.character import Character
        
        # Analyze patterns
        learned = await self.analyze_patterns(character_id)
        
        if learned.total_episodes_analyzed < self.min_episodes_for_learning:
            return {
                "success": False,
                "message": f"Not enough episodes ({learned.total_episodes_analyzed}). Need at least {self.min_episodes_for_learning}."
            }
        
        # Get character
        character = self.db.query(Character).filter(
            Character.id == character_id
        ).first()
        
        if not character:
            return {"success": False, "message": "Character not found"}
        
        # Build learned traits dict with validation
        learned_traits = {
            "learned_from_episodes": learned.total_episodes_analyzed,
            "last_consolidated": datetime.utcnow().isoformat(),
            "default_clothing": learned.default_clothing,
            "signature_poses": learned.signature_poses,
            "common_props": learned.common_props,
            "typical_environments": learned.typical_environments,
            "recurring_states": learned.recurring_states,
            "pattern_confidence": learned.confidence_scores,
            "quality_metrics": {
                "total_patterns_found": (
                    len(learned.default_clothing) + 
                    len(learned.signature_poses) + 
                    len(learned.common_props)
                ),
                "high_confidence_patterns": sum(
                    1 for conf in learned.confidence_scores.values() 
                    if conf >= self.high_confidence_threshold
                ),
                "validated": self._validate_learned_traits(learned)
            }
        }
        
        # Update character metadata
        current_metadata = character.char_metadata or {}
        current_metadata["learned_traits"] = learned_traits
        character.char_metadata = current_metadata
        
        self.db.commit()
        
        print(f"[OK] Memory consolidated for {character_id}")
        print(f"   Default clothing: {learned.default_clothing}")
        print(f"   Signature poses: {learned.signature_poses}")
        print(f"   Common props: {learned.common_props}")
        print(f"   Quality validated: {learned_traits['quality_metrics']['validated']}")
        
        return {
            "success": True,
            "learned_traits": learned_traits,
            "message": f"Consolidated patterns from {learned.total_episodes_analyzed} episodes"
        }
    
    def _validate_learned_traits(self, learned: LearnedTraits) -> bool:
        """
        Validate that learned traits meet minimum quality standards.
        
        Returns True if the learned traits are trustworthy.
        """
        # Check if we have enough data
        if learned.total_episodes_analyzed < self.min_episodes_for_learning:
            return False
        
        # Check if we have at least some high-confidence patterns
        high_conf_count = sum(
            1 for conf in learned.confidence_scores.values()
            if conf >= self.high_confidence_threshold
        )
        
        if high_conf_count == 0:
            return False
        
        # Check if at least one major category has patterns
        has_patterns = (
            len(learned.default_clothing) > 0 or
            len(learned.signature_poses) > 0 or
            len(learned.common_props) > 0
        )
        
        return has_patterns
    
    async def get_character_personality(self, character_id: str) -> Dict[str, Any]:
        """
        Get a summary of the character's learned personality/style.
        
        This can be used to inform prompt generation with learned preferences.
        """
        learned = await self.analyze_patterns(character_id)
        
        if learned.total_episodes_analyzed < 3:
            return {
                "personality_developed": False,
                "message": "Character needs more scenes to develop personality",
                "episodes": learned.total_episodes_analyzed
            }
        
        # Build personality summary
        personality = {
            "personality_developed": True,
            "episodes_analyzed": learned.total_episodes_analyzed,
            "style_summary": "",
            "details": {}
        }
        
        # Build style summary string
        style_parts = []
        
        if learned.default_clothing:
            style_parts.append(f"typically wears {', '.join(learned.default_clothing[:3])}")
            personality["details"]["clothing_preference"] = learned.default_clothing
        
        if learned.signature_poses:
            style_parts.append(f"often seen {', '.join(learned.signature_poses[:2])}")
            personality["details"]["signature_poses"] = learned.signature_poses
        
        if learned.common_props:
            style_parts.append(f"frequently has {', '.join(learned.common_props[:2])}")
            personality["details"]["common_props"] = learned.common_props
        
        if learned.typical_environments:
            style_parts.append(f"often found in {', '.join(learned.typical_environments[:2])}")
            personality["details"]["typical_environments"] = learned.typical_environments
        
        personality["style_summary"] = "; ".join(style_parts) if style_parts else "No distinct patterns yet"
        personality["confidence"] = learned.confidence_scores
        
        return personality
    
    async def suggest_scene_continuation(self, character_id: str) -> Dict[str, Any]:
        """
        Suggest scene elements based on character's established patterns.
        
        Useful for "auto-complete" style suggestions during generation.
        """
        learned = await self.analyze_patterns(character_id)
        
        suggestions = {
            "clothing": learned.default_clothing[:3] if learned.default_clothing else ["casual wear"],
            "props": learned.common_props[:2] if learned.common_props else [],
            "environment": learned.typical_environments[:2] if learned.typical_environments else ["neutral background"],
            "pose_style": learned.signature_poses[:2] if learned.signature_poses else ["natural pose"]
        }
        
        return {
            "based_on_episodes": learned.total_episodes_analyzed,
            "suggestions": suggestions,
            "hint": "These suggestions are based on the character's established visual style"
        }


async def consolidate_memory_task(character_id: str) -> Dict[str, Any]:
    """
    Background task for memory consolidation.
    
    Should be called periodically or after N new episodes.
    """
    from app.core.database import SessionLocal
    
    db = SessionLocal()
    try:
        service = MemoryConsolidationService(db)
        result = await service.consolidate_memory(character_id)
        return result
    finally:
        db.close()


async def get_personality_task(character_id: str) -> Dict[str, Any]:
    """
    Get character personality based on learned patterns.
    """
    from app.core.database import SessionLocal
    
    db = SessionLocal()
    try:
        service = MemoryConsolidationService(db)
        result = await service.get_character_personality(character_id)
        return result
    finally:
        db.close()
