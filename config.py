"""
Configuration for the Vision Service.
Handles screen/camera capture + desktop audio → Gemini 2.5 Flash.
Outputs scene analysis on port 8015 and audio/transcript on port 8017.
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

# ── Vision ────────────────────────────────────────────────────────────────────
VIDEO_DEVICE_INDEX = 1          # Set to None to use screen-region capture
IMAGE_QUALITY      = 85
MAX_OUTPUT_TOKENS  = 800

CAPTURE_REGION = {
    "left": 14, "top": 154,
    "width": 1222, "height": 685,
}

# ── Audio capture ─────────────────────────────────────────────────────────────
_PREFERRED_DEVICE_NAME = "Cam Link 4K"
_FALLBACK_DEVICE_ID    = 4

def _find_device_by_name(name: str) -> int | None:
    try:
        import sounddevice as sd
        for i, dev in enumerate(sd.query_devices()):
            if dev["max_input_channels"] > 0 and name.lower() in dev["name"].lower():
                print(f"[config] ✅ Found audio device '{name}' → id={i} ({dev['name']})", flush=True)
                return i
    except Exception as e:
        print(f"[config] ⚠️  Audio device lookup failed: {e}", flush=True)
    return None

_detected = _find_device_by_name(_PREFERRED_DEVICE_NAME)
if _detected is None:
    print(f"[config] ⚠️  '{_PREFERRED_DEVICE_NAME}' not found — falling back to device_id={_FALLBACK_DEVICE_ID}", flush=True)
DESKTOP_AUDIO_DEVICE_ID = _detected if _detected is not None else _FALLBACK_DEVICE_ID

AUDIO_SAMPLE_RATE = 16000   # Target rate (resampled to this)
AUDIO_GAIN        = 1.5     # Pre-send gain
# How many seconds of ambient audio to always send for music/SFX detection
AMBIENT_WINDOW_S  = 2.0

# ── VAD ───────────────────────────────────────────────────────────────────────
VAD_AGGRESSIVENESS      = 2     # webrtcvad mode 0-3 (3 = most aggressive filtering)
VAD_FRAME_MS            = 30    # Frame duration in ms (10, 20, or 30)
VAD_SILENCE_DURATION_MS = 600   # Silence this long closes an utterance
VAD_MAX_UTTERANCE_S     = 8.0   # Force-close utterance after this long regardless

# ── Streaming ─────────────────────────────────────────────────────────────────
FPS            = 3      # Frame capture rate
BATCH_SIZE     = 6      # Frames per Gemini call
BATCH_INTERVAL = 2.0    # Seconds between Gemini dispatch ticks

# ── Safety ────────────────────────────────────────────────────────────────────
SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT",        "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH",       "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

# ── Prompt ────────────────────────────────────────────────────────────────────
# Gemini receives:
#   - 6 video frames
#   - Ambient audio clip (always present) — background sound, music, SFX
#   - Speech audio clip (only when VAD detected a complete utterance)
#
# Respond in exactly three tagged sections.
PROMPT = """You are an expert scene and audio analyzer for a live stream AI companion.

You receive:
  1. A sequence of video frames showing the current scene.
  2. An AMBIENT audio clip (always present) — this is the raw background audio
     captured from the stream, containing music, sound effects, and game audio.
  3. A SPEECH audio clip (only sometimes present) — this is an isolated segment
     of detected speech, cleanly captured with silence at both ends.

Respond using EXACTLY this format — no text outside the tags:

[SCENE]
One concise paragraph: who is on screen, what is happening, what game or content
is shown. Under 80 words.
[/SCENE]

[SPEECH]
If a speech audio clip was provided: transcribe it accurately. Attribute to the
speaker if they are visible on screen (use their name or a consistent label like
"Male Streamer"). One speaker per line. Preserve natural speech including filler
words if clearly audible.
If no speech clip was provided, write: none
[/SPEECH]

[AUDIO]
Describe what you hear in the ambient audio clip. Be specific and useful:
- Music: genre, mood, tempo (e.g. "tense orchestral combat music", "upbeat chiptune")
- Sound effects: what they are (e.g. "sword clashing SFX", "explosion", "UI click sounds")
- Game audio: describe notable in-game sounds
- If the ambient audio is silent or only low background hiss, write: none
[/AUDIO]"""

# ── Network ───────────────────────────────────────────────────────────────────
VISION_WS_PORT    = 8015    # Scene analysis (vision consumers unchanged)
AUDIO_WS_PORT     = 8017    # Transcript + audio events (replaces stream_audio_service)
HTTP_CONTROL_PORT = 8016
HUB_URL           = "http://localhost:8002"

SERVICE_NAME = "vision_service"
DEBUG_MODE   = False