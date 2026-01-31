"""
Groq LLM Service
Uses Groq (free tier) for prompt generation, summarization, and text processing.
"""

from groq import Groq
from app.core.config import settings


class GroqLLMService:
    """Service for Groq LLM operations."""
    
    PROMPT_TEMPLATE = """You are an expert at creating detailed image generation prompts for CONSISTENT character AI.

[CHARACTER_IDENTITY - ABSOLUTELY IMMUTABLE - COPY EXACTLY]
Name: {name}
Face Structure: {face_description}
Hair (EXACT): {hair}
Eyes (EXACT): {eyes}
Skin Tone: {skin_tone}
Distinctive Features: {distinctive_marks}
Age: {age_range}
Build: {build}
Character Tags: {semantic_tags}

[INITIAL APPEARANCE - Character's canonical look (PRESERVE UNLESS EXPLICITLY CHANGED)]
Original Outfit: {initial_outfit}
Original Background/Setting: {initial_background}
Original Pose: {initial_pose}
Original Lighting: {initial_lighting}
Accessories: {accessories}

[CURRENT STATE - Override only if scene EXPLICITLY changes it]
Currently wearing: {current_clothing}
Current physical state: {current_state}
Currently holding/has: {current_props}

[SCENE_HISTORY]
{recent_states}

[NEW_SCENE_REQUEST]
{user_prompt}

[TASK]
Create a PRECISE image generation prompt that:
1. COPIES VERBATIM the character's facial features, eye color, hair color/style, skin tone, and distinctive marks
2. MAINTAINS the character's outfit from [CURRENT STATE] - if "Not established yet", use [INITIAL APPEARANCE] outfit
3. ONLY changes clothing if the new scene EXPLICITLY requests a costume change
4. Includes ALL distinctive features (scars, moles, tattoos, piercings) in EXACT locations
5. Fulfills the scene request while keeping the character IDENTICAL

[MANDATORY RULES - VIOLATION IS FAILURE]
❌ NEVER change: eye color, hair color, skin tone, facial structure, distinctive marks
❌ NEVER change: body type, height, age appearance
❌ NEVER change: outfit UNLESS prompt explicitly says "wearing [X]" or "changed into [X]"
❌ NEVER change: background/environment UNLESS prompt explicitly says "in [new location]" or "at [new place]"
❌ NEVER change: pose UNLESS prompt explicitly requests a different pose
❌ NEVER change: lighting UNLESS prompt explicitly requests different lighting
✅ ALWAYS preserve the EXACT background from [INITIAL APPEARANCE] if no new location is specified
✅ ALWAYS include: EXACT eye color, EXACT hair description, ALL distinctive marks
✅ ALWAYS start the prompt with detailed character description BEFORE the scene
✅ Format: [Character] + [Same outfit unless changed] + [Same background unless changed] + [Same lighting unless changed]

[OUTPUT FORMAT]
A [gender] with [EXACT face description], [EXACT eye color and shape], [EXACT hair color/style/length], [EXACT skin tone], [ALL distinctive marks with locations]. Wearing [SAME outfit from INITIAL APPEARANCE unless prompt specifies new clothing]. [Pose - same unless changed]. In [SAME background from INITIAL APPEARANCE unless prompt specifies new location]. [SAME lighting unless changed]. Photorealistic, 8K, detailed skin texture.

CRITICAL: If the user prompt does NOT mention a new location/background, use the EXACT background from [INITIAL APPEARANCE].
CRITICAL: If the user prompt does NOT mention new clothes, use the EXACT outfit from [INITIAL APPEARANCE].

Generate ONLY the final prompt - no explanations:"""


    SUMMARIZE_TEMPLATE = """Analyze this generated image and extract:
1. Clothing/outfit details
2. Physical state (injuries, dirt, conditions)
3. Props or accessories visible
4. Pose and expression
5. Environment clues

Image description: {image_context}

Return a JSON object with these fields:
{{
    "clothing": ["item1", "item2"],
    "physical_state": ["state1", "state2"],
    "props": ["prop1"],
    "pose": "description",
    "environment": "description",
    "tags": ["tag1", "tag2", "tag3"]
}}"""

    def __init__(self):
        self.client = Groq(api_key=settings.GROQ_API_KEY)
        self.model = settings.GROQ_MODEL
    
    async def generate_prompt(
        self, 
        character_data: dict, 
        user_prompt: str, 
        episodic_states: list = None
    ) -> str:
        """
        Generate optimized image prompt using character data and episodic memory.
        
        This is the core of character memory - we merge:
        - Semantic memory (who the character IS)
        - Episodic memory (what they've DONE and their current state)
        """
        # Extract current state from most recent episode
        current_clothing = []
        current_state = []
        current_props = []
        
        if episodic_states and len(episodic_states) > 0:
            # Get state from most recent episode
            latest = episodic_states[-1]
            state_data = latest.get('state_data', {})
            current_clothing = state_data.get('clothing', [])
            current_props = state_data.get('props', [])
            
            # Accumulate persistent states (injuries persist across scenes)
            for ep in episodic_states:
                ep_state = ep.get('state_data', {})
                for state in ep_state.get('physical_state', []):
                    if any(kw in state.lower() for kw in 
                           ['injured', 'wounded', 'scar', 'bruise', 'tired', 'wet', 'dirty']):
                        if state not in current_state:
                            current_state.append(state)
        
        # Format recent states for context
        if episodic_states:
            recent_states = "\n".join([
                f"Scene {s.get('scene_index', '?')}: {', '.join(s.get('tags', [])[:5])} | {s.get('state_data', {}).get('environment', 'Unknown location')}"
                for s in episodic_states[-3:]
            ])
        else:
            recent_states = "No previous scenes - this is the character's first appearance"
        
        # Extract learned habits
        learned_traits = character_data.get("learned_traits", {})
        learned_style = "Not yet established"
        signature_poses = "None"
        
        if learned_traits:
            clothing = learned_traits.get("default_clothing", [])
            props = learned_traits.get("common_props", [])
            poses = learned_traits.get("signature_poses", [])
            
            style_parts = []
            if clothing: style_parts.append(f"Often wears: {', '.join(clothing)}")
            if props: style_parts.append(f"Often carries: {', '.join(props)}")
            
            learned_style = " | ".join(style_parts) if style_parts else "Not yet established"
            signature_poses = ", ".join(poses) if poses else "None"

        # Fill template with all memory context
        filled_prompt = self.PROMPT_TEMPLATE.format(
            name=character_data.get("name", "Unknown Character"),
            face_description=character_data.get("face", "Not specified"),
            hair=character_data.get("hair", "Not specified"),
            eyes=character_data.get("eyes", "Not specified"),
            skin_tone=character_data.get("skin_tone", "Not specified"),
            distinctive_marks=character_data.get("distinctives", "None"),
            age_range=character_data.get("age_range", "Adult"),
            build=character_data.get("build", "Average"),
            semantic_tags=", ".join(character_data.get("tags", [])) or "None specified",
            initial_outfit=character_data.get("initial_outfit", "Not captured"),
            initial_background=character_data.get("initial_background", "Not captured"),
            initial_pose=character_data.get("pose", "Not captured"),
            initial_lighting=character_data.get("lighting", "Not captured"),
            accessories=character_data.get("accessories", "None"),
            current_clothing=", ".join(current_clothing) if current_clothing else "Not established yet - use initial outfit",
            current_state=", ".join(current_state) if current_state else "Normal, healthy",
            current_props=", ".join(current_props) if current_props else "None",
            recent_states=recent_states,
            user_prompt=user_prompt
        )
        
        # Call Groq
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert image prompt engineer."},
                {"role": "user", "content": filled_prompt}
            ],
            temperature=settings.GROQ_TEMPERATURE,
            max_tokens=1024
        )
        
        return response.choices[0].message.content.strip()
    
    async def summarize_image(self, image_context: str) -> dict:
        """
        Summarize/analyze generated image to extract episodic state.
        """
        prompt = self.SUMMARIZE_TEMPLATE.format(image_context=image_context)
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an image analysis expert. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=512
        )
        
        import json
        try:
            return json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            return {"tags": [], "error": "Failed to parse response"}
    
    async def correct_prompt(self, raw_prompt: str) -> str:
        """
        Correct and enhance a raw user prompt for better image generation.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system", 
                    "content": "You are a prompt enhancement expert. Improve the given prompt for image generation while keeping the core intent. Add details about lighting, composition, and style. Return only the improved prompt."
                },
                {"role": "user", "content": f"Improve this prompt: {raw_prompt}"}
            ],
            temperature=0.3,
            max_tokens=512
        )
        
        return response.choices[0].message.content.strip()
    
    async def extract_character_traits(self, image_description: str) -> dict:
        """
        Extract character traits from image/description for character sheet.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": """Extract character traits from the description. Return JSON:
{
    "face": "facial features description",
    "hair": "hair color, style, length",
    "eyes": "eye color and shape",
    "distinctives": "scars, tattoos, unique features",
    "age_range": "estimated age range",
    "build": "body type description"
}"""
                },
                {"role": "user", "content": image_description}
            ],
            temperature=0.1,
            max_tokens=512
        )
        
        import json
        try:
            return json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            return {}
