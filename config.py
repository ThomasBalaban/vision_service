"""
Configuration for the Vision Service.
Handles screen / camera capture → Gemini 2.5 Flash analysis.
"""
import os
import sys

_THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_THIS_DIR)
for _p in (_THIS_DIR, _PARENT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

print(f"[config] sys.path[0:3] = {sys.path[:3]}", flush=True)

try:
    from api_keys import GEMINI_API_KEY
    print("[config] ✅ api_keys (GEMINI_API_KEY) loaded", flush=True)
except ImportError as e:
    print(f"[config] ❌ api_keys FAILED: {e}", flush=True)
    raise

API_KEY = GEMINI_API_KEY

# Vision
VIDEO_DEVICE_INDEX = 1          # Set to None to use screen-region capture
FPS                = 2
IMAGE_QUALITY      = 85
MAX_OUTPUT_TOKENS  = 600

CAPTURE_REGION = {
    "left": 14, "top": 154,
    "width": 1222, "height": 685,
}

SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT",        "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH",       "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

PROMPT = """You are the real-time perception engine (the "eyes and ears") for an interactive AI assistant. \
You are receiving a chronological sequence of frames spanning a 2-second window, along with the matching raw audio clip. \
The user is actively playing a video game, watching a video, or interacting with their screen. 

YOUR JOB: Provide a highly detailed, temporally-aware description of exactly what is happening so the downstream AI can react intelligently.

ANALYSIS RULES:
1. TRACK MOTION & PROGRESSION: You have multiple frames. Do not just describe a static picture. You MUST describe the action across time. What moves? What changes? (e.g., "The camera rapidly pans right," "The boss winds up a heavy attack," "A pop-up menu appears").
2. EXTRACT ACTIONABLE DETAILS: Pay close attention to the state of the screen. Identify UI elements (health bars, ammo, active menus, error messages), specific character actions, environments, and onscreen text.
3. WEAVE IN AUDIO CONTEXT: Connect the audio directly to the visual action. Describe sound effects (footsteps, explosions, UI clicks, gunfire), the mood/tempo of the music, and the emotional tone of any voices. 
4. DO NOT TRANSCRIBE: Focus on the *delivery* and *intent* of the audio (e.g., "A character yells in panic," "An upbeat narrator explains a concept") rather than writing down the exact words.
5. BE DENSE AND DESCRIPTIVE: Pack as much specific, granular detail as possible into your response. The reacting AI depends entirely on your description to understand the world.

OUTPUT FORMAT:
Write a dense, highly detailed summary synthesizing the chronological visual action, screen state, and audio cues.

EXAMPLE:
"Across the frames, the player's character (a sci-fi soldier) sprints forward and slides behind a concrete barrier as laser fire hits the wall above them. In the top-left UI, the shield bar drops to zero and flashes red. The audio features heavy, rapid footsteps, the high-pitched mechanical whine of incoming lasers, and a loud concrete impact sound. Tense, synth-heavy combat music is playing, and a robotic voice urgently announces a warning."

NOW ANALYZE THE CURRENT SCENE:"""

DEBUG_MODE = False

# Network
WEBSOCKET_PORT    = 8015
HTTP_CONTROL_PORT = 8016
HUB_URL           = "http://localhost:8002"

MIC_WS_URL   = "ws://localhost:8013"
SERVICE_NAME = "vision_service"