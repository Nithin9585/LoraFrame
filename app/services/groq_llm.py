"""
Groq LLM Service
Uses Groq (free tier) for prompt generation, summarization, and text processing.
"""

from groq import Groq
from app.core.config import settings


class GroqLLMService:
    """Service for Groq LLM operations."""
    
    PROMPT_TEMPLATE = """You are an expert at creating PIXEL-PERFECT image generation prompts that preserve EVERY detail.

[CHARACTER_IDENTITY - ABSOLUTELY IMMUTABLE - COPY EXACTLY]
Name: {name}
Face Structure: {face_description}
Facial Expression: {facial_expression}
Hair (EXACT): {hair}
Eyes (EXACT): {eyes}
Skin Tone: {skin_tone}
Distinctive Features: {distinctive_marks}
Age: {age_range}
Build: {build}

[COMPLETE VISUAL STATE - PRESERVE EVERYTHING UNLESS EXPLICITLY CHANGED]
Outfit: {initial_outfit}
Accessories: {accessories}
Props in Hands: {props_in_hands}
Hand Position: {hand_position}
Pose/Body Angle: {initial_pose}
Body Angle/Camera: {body_angle}

[SCENE/ENVIRONMENT - PRESERVE EXACTLY UNLESS EXPLICITLY CHANGED]
Background: {initial_background}
Background Objects: {background_objects}
Visible Objects: {visible_objects}
Lighting: {initial_lighting}
Color Palette: {color_palette}
Image Composition: {image_composition}

[CURRENT STATE - From previous scenes]
Currently wearing: {current_clothing}
Current physical state: {current_state}
Currently holding/has: {current_props}

[SCENE_HISTORY]
{recent_states}

[NEW_SCENE_REQUEST]
{user_prompt}

[TASK]
Generate a prompt that recreates the EXACT SAME IMAGE with the EXACT SAME PERSON, changing ONLY what the user explicitly requests.

[ABSOLUTE RULES - EVERY DETAIL MATTERS]
❌ NEVER change ANY of these unless user EXPLICITLY mentions changing them:
   - Face, eyes, hair, skin, distinctive marks
   - Outfit, accessories, what's in hands
   - Background, objects in scene, lighting
   - Pose, hand position, body angle
   - Color palette, composition

✅ PRESERVE EXACTLY:
   - Every small object in the background
   - Hand positions and what they're holding
   - Facial expression
   - All accessories (jewelry, glasses, etc.)
   - Lighting direction and shadows
   - Color temperature and mood

[OUTPUT FORMAT]
[EXACT character description with ALL features] + [EXACT outfit unless changed] + [EXACT accessories] + [Holding EXACT same items unless changed] + [EXACT pose unless changed] + [In EXACT same background with ALL objects unless new location specified] + [EXACT same lighting unless changed]. [Same composition/framing]. Photorealistic, 8K.

CRITICAL: If user says "make them smile" - change ONLY the expression. Keep EVERYTHING else identical.
CRITICAL: If user says "different pose" - change ONLY the pose. Keep outfit, background, objects, lighting identical.
CRITICAL: If user doesn't mention background - the background must be EXACTLY the same including every small object.

Generate ONLY the final prompt:"""


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
            facial_expression=character_data.get("facial_expression", "Not captured"),
            hair=character_data.get("hair", "Not specified"),
            eyes=character_data.get("eyes", "Not specified"),
            skin_tone=character_data.get("skin_tone", "Not specified"),
            distinctive_marks=character_data.get("distinctives", "None"),
            age_range=character_data.get("age_range", "Adult"),
            build=character_data.get("build", "Average"),
            initial_outfit=character_data.get("initial_outfit", "Not captured"),
            initial_background=character_data.get("initial_background", "Not captured"),
            background_objects=character_data.get("background_objects", "Not captured"),
            visible_objects=character_data.get("visible_objects", "Not captured"),
            initial_pose=character_data.get("pose", "Not captured"),
            hand_position=character_data.get("hand_position", "Not captured"),
            body_angle=character_data.get("body_angle", "Not captured"),
            initial_lighting=character_data.get("lighting", "Not captured"),
            color_palette=character_data.get("color_palette", "Not captured"),
            image_composition=character_data.get("image_composition", "Not captured"),
            accessories=character_data.get("accessories", "None"),
            props_in_hands=character_data.get("props_in_hands", "Nothing"),
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
