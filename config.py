"""
Configuration for the Vision Service.
Handles screen / camera capture → Gemini 2.5 Flash analysis.
"""
import os
import sys
from pathlib import Path

_THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_THIS_DIR)
for _p in (_THIS_DIR, _PARENT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

print(f"[config] sys.path[0:3] = {sys.path[:3]}", flush=True)


def _load_sibling_secrets() -> None:
    secrets_dir = Path(__file__).resolve().parent.parent / "director_ui" / "secrets"
    if not secrets_dir.is_dir():
        return
    for path in sorted(secrets_dir.glob("*.env")):
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


if "GEMINI_API_KEY" not in os.environ:
    _load_sibling_secrets()

try:
    API_KEY = os.environ["GEMINI_API_KEY"]
except KeyError as e:
    raise RuntimeError(
        f"Missing required env var {e.args[0]}. "
        "Credentials live in director_ui/secrets/*.env. Start this service via "
        "the launcher, or export the variable manually."
    ) from None

# Vision
_PREFERRED_DEVICE_NAME = "Cam Link 4K"
_FALLBACK_DEVICE_ID    = 0
# Names containing any of these substrings are hidden from the picker and
# skipped by the auto-detect lookup. cv2/AVFoundation will happily try to
# connect to a Continuity Camera, which we never want.
_DEVICE_NAME_BLOCKLIST = ("iphone",)


def _enumerate_cameras() -> list[dict]:
    """Snapshot real macOS cameras at startup with cv2-compatible indexes.

    cv2.VideoCapture(N) on macOS enumerates AVFoundation devices sorted by
    `uniqueID` ascending — NOT the order shown by ffmpeg's `-list_devices`
    or by system_profiler's default output. Empirically:
      Studio Display Camera (uniqueID 0x2114...) → cv2 idx 0
      Cam Link 4K           (uniqueID 0x2132...) → cv2 idx 1
      MacBook Pro Camera    (uniqueID 6C70...)   → cv2 idx 2
      iPhone Camera         (uniqueID FD65...)   → cv2 idx 3

    So we read all cameras from system_profiler (which exposes uniqueIDs),
    sort by uniqueID, then assign positional ids that match cv2's indexing.
    iPhone is filtered out AFTER assigning indexes so the remaining ids
    still point at the correct cv2 indexes.
    """
    import json
    import subprocess
    try:
        out = subprocess.run(
            ["system_profiler", "SPCameraDataType", "-json"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout
        entries = json.loads(out).get("SPCameraDataType", [])
    except Exception as e:
        print(f"[config] ⚠️  system_profiler camera lookup failed: {e}", flush=True)
        return []

    # Sort by uniqueID to match cv2's AVFoundation enumeration order.
    entries_sorted = sorted(
        entries,
        key=lambda e: str(e.get("spcamera_unique-id", "")).lower(),
    )

    devices: list[dict] = []
    for cv2_idx, entry in enumerate(entries_sorted):
        name = str(entry.get("_name", f"Camera {cv2_idx}"))
        if any(blk in name.lower() for blk in _DEVICE_NAME_BLOCKLIST):
            continue
        devices.append({
            "id":        cv2_idx,
            "name":      name,
            "unique_id": entry.get("spcamera_unique-id"),
        })
    return devices


# Frozen snapshot of cameras taken at startup. We never re-enumerate at
# request time because (a) system_profiler is slow and (b) the ordering can
# drift when cv2 holds a camera, which corrupts the id mapping.
VIDEO_DEVICES: list[dict] = _enumerate_cameras()
for _d in VIDEO_DEVICES:
    print(f"[config] camera id={_d['id']} name={_d['name']!r}", flush=True)


def list_video_devices() -> list[dict]:
    """Return the cached startup snapshot."""
    return list(VIDEO_DEVICES)


def _find_device_by_name(name: str) -> int | None:
    for dev in VIDEO_DEVICES:
        if name.lower() in dev["name"].lower():
            print(f"[config] ✅ Found '{name}' → device_id={dev['id']} ({dev['name']})", flush=True)
            return dev["id"]
    return None


_detected = _find_device_by_name(_PREFERRED_DEVICE_NAME)
if _detected is None:
    print(f"[config] ⚠️  '{_PREFERRED_DEVICE_NAME}' not found — falling back to device_id={_FALLBACK_DEVICE_ID}", flush=True)
VIDEO_DEVICE_INDEX = _detected if _detected is not None else _FALLBACK_DEVICE_ID

FPS                = 2
IMAGE_QUALITY      = 85
MAX_OUTPUT_TOKENS  = 1500

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