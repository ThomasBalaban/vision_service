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

PROMPT = """You are an expert scene analyzer providing real-time context for an AI assistant. \
You receive both video frames and a short audio clip of the current scene.

YOUR JOB: Combine what you SEE and what you HEAR into a unified description of what's happening.

ANALYSIS RULES:
1. MATCH AUDIO TO VISUALS: If you hear a sound effect (e.g., explosion, jump, notification) or background music, describe it in relation to what is happening on screen.
2. DO NOT TRANSCRIBE: Focus on the mood, sound effects, and the tone of any voices, rather than writing out exactly what is being said.
3. KEEP IT CONCISE: One short paragraph. Under 250 words.

OUTPUT FORMAT:
Natural paragraph combining visuals + notable audio. 

EXAMPLE:
"A character in a red suit jumps over a gap while an upbeat, fast-paced electronic track plays. A loud crashing sound effect is heard as the platform behind them collapses."

NOW ANALYZE THE CURRENT SCENE:"""

DEBUG_MODE = False

# Network
WEBSOCKET_PORT    = 8015
HTTP_CONTROL_PORT = 8016
HUB_URL           = "http://localhost:8002"

MIC_WS_URL   = "ws://localhost:8013"
SERVICE_NAME = "vision_service"