"""
VisionService — merged vision + audio with VAD-gated speech
Captures frames + desktop audio → single Gemini call every 2 seconds.

Audio flow:
  AudioCapture → ambient buffer (always)  → [AUDIO] section (music/SFX)
  AudioCapture → AudioVAD → utterance queue → [SPEECH] section (only when complete)

Vision output    → ws://localhost:8015
Audio/transcript → ws://localhost:8017
"""

import asyncio
import re
import threading
import time
import traceback
from datetime import datetime

import socketio

from audio_capture import AudioCapture
from audio_vad import AudioVAD
from config import (
    API_KEY, AUDIO_SAMPLE_RATE, CAPTURE_REGION, DEBUG_MODE,
    DESKTOP_AUDIO_DEVICE_ID, FPS, HUB_URL, IMAGE_QUALITY,
    MAX_OUTPUT_TOKENS, PROMPT, SERVICE_NAME, VIDEO_DEVICE_INDEX,
)
from gemini_client import GeminiClient
from screen_capture import ScreenCapture
from streaming_manager import StreamingManager
from websocket_server import WebSocketServer

_BUFFER_MAX_CHARS = 3000


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] [{SERVICE_NAME}] {msg}", flush=True)


def _extract_section(text: str, tag: str) -> str | None:
    pattern = rf"\[{tag}\](.*?)\[/{tag}\]"
    m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if m:
        content = m.group(1).strip()
        return content if content.lower() != "none" else None
    return None


class VisionService:
    def __init__(self):
        log("👁️🎧 Initializing (vision + audio + VAD) …")
        self._shutting_down      = False
        self._shutdown_lock      = threading.Lock()
        self._hub_emit_count     = 0
        self._vision_ws_count    = 0
        self._audio_ws_count     = 0
        self._gemini_resp_count  = 0

        # ── WebSocket servers (8015 + 8017) ───────────────────────────────────
        log("Creating dual WebSocket servers (vision:8015, audio:8017) …")
        self.ws_server = WebSocketServer()

        # ── Socket.IO hub client ──────────────────────────────────────────────
        self.sio      = socketio.AsyncClient(reconnection=True, reconnection_delay=5)
        self.hub_loop: asyncio.AbstractEventLoop | None = None
        self._register_hub_events()

        # ── Screen capture ────────────────────────────────────────────────────
        log(f"Initializing ScreenCapture (VIDEO_DEVICE_INDEX={VIDEO_DEVICE_INDEX}) …")
        try:
            self.screen_capture = ScreenCapture(
                image_quality=IMAGE_QUALITY,
                video_index=VIDEO_DEVICE_INDEX,
            )
            if VIDEO_DEVICE_INDEX is None and CAPTURE_REGION:
                self.screen_capture.set_capture_region(CAPTURE_REGION)
                log(f"Capture region: {CAPTURE_REGION}")
            log("✅ ScreenCapture ready")
        except Exception as e:
            log(f"❌ ScreenCapture init FAILED: {e}")
            log(traceback.format_exc())
            raise

        # ── Audio VAD ─────────────────────────────────────────────────────────
        log("Initializing AudioVAD …")
        try:
            self.audio_vad = AudioVAD()
            log("✅ AudioVAD ready")
        except Exception as e:
            log(f"⚠️  AudioVAD init FAILED (continuing without VAD): {e}")
            self.audio_vad = None

        # ── Audio capture ─────────────────────────────────────────────────────
        log(f"Initializing AudioCapture (device_id={DESKTOP_AUDIO_DEVICE_ID}) …")
        try:
            self.audio_capture = AudioCapture(
                device_id   = DESKTOP_AUDIO_DEVICE_ID,
                sample_rate = AUDIO_SAMPLE_RATE,
            )
            if self.audio_vad:
                self.audio_capture.set_vad(self.audio_vad)
                log("  ↳ VAD registered with AudioCapture")
            self.audio_capture.set_volume_callback(self._on_volume)
            log("✅ AudioCapture ready")
        except Exception as e:
            log(f"⚠️  AudioCapture init FAILED (continuing without audio): {e}")
            self.audio_capture = None

        # ── Gemini client ─────────────────────────────────────────────────────
        log("Initializing GeminiClient …")
        try:
            self.gemini_client = GeminiClient(
                api_key           = API_KEY,
                system_prompt     = PROMPT,
                response_callback = self._on_gemini_chunk,
                error_callback    = self._on_gemini_error,
                max_output_tokens = MAX_OUTPUT_TOKENS,
                debug_mode        = DEBUG_MODE,
            )
            log("✅ GeminiClient ready")
        except Exception as e:
            log(f"❌ GeminiClient init FAILED: {e}")
            log(traceback.format_exc())
            raise

        # ── Streaming manager ─────────────────────────────────────────────────
        self.streaming_manager = StreamingManager(
            screen_capture = self.screen_capture,
            gemini_client  = self.gemini_client,
            audio_capture  = self.audio_capture,
            audio_vad      = self.audio_vad,
            debug_mode     = DEBUG_MODE,
        )
        self.streaming_manager.set_error_callback(self._on_streaming_error)
        self.streaming_manager.set_dispatch_callback(self._on_batch_dispatched)

        self._response_buffer      = ""
        self._response_buffer_lock = threading.Lock()

        log("✅ VisionService initialized")

    # ── Public ────────────────────────────────────────────────────────────────

    def run(self):
        self.hub_loop = asyncio.new_event_loop()
        threading.Thread(
            target=self._hub_thread, args=(self.hub_loop,),
            daemon=True, name="VisionHub",
        ).start()
        log("Hub thread started")

        self.ws_server.start()
        log("WebSocket servers started")

        threading.Thread(target=self._delayed_start, daemon=True).start()
        log("Delayed start thread launched …")

        log("✅ Running — waiting for stream to begin …")
        try:
            while not self._shutting_down:
                time.sleep(0.5)
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        with self._shutdown_lock:
            if self._shutting_down:
                return
            self._shutting_down = True
        log(f"🛑 Shutting down (hub: {self._hub_emit_count}, "
            f"vision WS: {self._vision_ws_count}, "
            f"audio WS: {self._audio_ws_count}, "
            f"gemini: {self._gemini_resp_count})")
        try:
            self.streaming_manager.stop_streaming()
        except Exception as e:
            log(f"Error stopping stream manager: {e}")
        if self.audio_capture:
            try:
                self.audio_capture.stop()
            except Exception as e:
                log(f"Error stopping audio capture: {e}")
        try:
            self.screen_capture.release()
        except Exception as e:
            log(f"Error releasing screen capture: {e}")
        try:
            self.ws_server.stop()
        except Exception as e:
            log(f"Error stopping WS servers: {e}")
        if self.hub_loop and self.sio.connected:
            try:
                asyncio.run_coroutine_threadsafe(self.sio.disconnect(), self.hub_loop)
                time.sleep(0.5)
            except Exception as e:
                log(f"Error disconnecting hub: {e}")
        if self.hub_loop:
            self.hub_loop.call_soon_threadsafe(self.hub_loop.stop)
        log("🛑 Stopped.")

    # ── Hub ───────────────────────────────────────────────────────────────────

    def _register_hub_events(self):
        @self.sio.event
        async def connect():
            log(f"✅ Hub CONNECTED → {HUB_URL}")

        @self.sio.event
        async def disconnect():
            log("⚠️  Hub DISCONNECTED")

        @self.sio.event
        async def connect_error(data):
            log(f"❌ Hub CONNECTION ERROR: {data}")

    def _hub_thread(self, loop: asyncio.AbstractEventLoop):
        asyncio.set_event_loop(loop)
        loop.create_task(self._hub_connection_loop())
        loop.run_forever()

    async def _hub_connection_loop(self):
        while not self._shutting_down:
            if not self.sio.connected:
                try:
                    log(f"Attempting hub connect → {HUB_URL} …")
                    await self.sio.connect(HUB_URL)
                except Exception as e:
                    log(f"⚠️  Hub connect failed: {e} — retry in 5s")
                    await asyncio.sleep(5)
            await asyncio.sleep(2)

    def _emit_to_hub(self, event: str, data: dict):
        if not self.sio.connected or not self.hub_loop:
            return
        try:
            asyncio.run_coroutine_threadsafe(self.sio.emit(event, data), self.hub_loop)
            self._hub_emit_count += 1
            log(f"→ HUB [{event}] {str(data)[:160]}")
        except Exception as e:
            log(f"❌ HUB EMIT ERROR [{event}]: {e}")

    # ── Gemini response handling ──────────────────────────────────────────────

    def _on_batch_dispatched(self):
        """Reset response buffer just before a new Gemini request fires."""
        with self._response_buffer_lock:
            if self._response_buffer.strip():
                log(f"⚠️  Discarding stale buffer ({len(self._response_buffer)} chars)")
            self._response_buffer = ""

    def _on_gemini_chunk(self, text_chunk: str):
        with self._response_buffer_lock:
            self._response_buffer += text_chunk
            response_complete = "[/AUDIO]" in self._response_buffer
            overflowed        = len(self._response_buffer) > _BUFFER_MAX_CHARS

            if not (response_complete or overflowed):
                return

            full_response         = self._response_buffer.strip()
            self._response_buffer = ""

        if not full_response:
            return

        self._gemini_resp_count += 1
        timestamp = datetime.now().isoformat()

        if overflowed and not response_complete:
            log(f"⚠️  Buffer force-flushed ({len(full_response)} chars — no [/AUDIO])")

        log(f"🤖 GEMINI RESPONSE #{self._gemini_resp_count} ({len(full_response)} chars)")
        self._dispatch_response(full_response, timestamp)

    def _dispatch_response(self, full_response: str, timestamp: str):
        scene  = _extract_section(full_response, "SCENE")
        speech = _extract_section(full_response, "SPEECH")
        audio  = _extract_section(full_response, "AUDIO")

        log(f"  ↳ SCENE:  {repr((scene  or 'none')[:80])}")
        log(f"  ↳ SPEECH: {repr((speech or 'none')[:80])}")
        log(f"  ↳ AUDIO:  {repr((audio  or 'none')[:80])}")

        # Scene → vision WS (8015) + Hub
        if scene:
            try:
                self.ws_server.broadcast_vision({
                    "type":      "vision_analysis",
                    "content":   scene,
                    "timestamp": timestamp,
                })
                self._vision_ws_count += 1
            except Exception as e:
                log(f"❌ Vision WS error: {e}")
            self._emit_to_hub("vision_context", {"context": scene, "timestamp": timestamp})
            self._emit_to_hub("text_update",    {"type": "text_update", "content": scene, "timestamp": timestamp})

        # Speech → audio WS (8017) + Hub
        if speech:
            try:
                self.ws_server.broadcast_audio({
                    "type":      "transcript_enriched",
                    "source":    "desktop",
                    "text":      speech,
                    "enriched":  True,
                    "timestamp": timestamp,
                })
                self._audio_ws_count += 1
            except Exception as e:
                log(f"❌ Audio WS error (speech): {e}")
            self._emit_to_hub("audio_context", {
                "context":    speech,
                "is_partial": False,
                "timestamp":  timestamp,
                "metadata":   {"source": "desktop"},
            })

        # Background audio events → audio WS (8017)
        if audio:
            try:
                self.ws_server.broadcast_audio({
                    "type":      "audio_event",
                    "source":    "desktop",
                    "text":      audio,
                    "timestamp": timestamp,
                })
                self._audio_ws_count += 1
            except Exception as e:
                log(f"❌ Audio WS error (audio event): {e}")

    # ── Other callbacks ───────────────────────────────────────────────────────

    def _on_volume(self, level: float):
        self.ws_server.broadcast_audio({
            "type":   "volume",
            "source": "desktop",
            "level":  level,
        })

    def _on_gemini_error(self, msg: str):
        log(f"❌ GEMINI ERROR: {msg}")
        self.ws_server.broadcast_vision({"type": "error", "source": "gemini", "message": msg})

    def _on_streaming_error(self, msg: str):
        log(f"❌ STREAMING ERROR: {msg}")

    # ── Startup ───────────────────────────────────────────────────────────────

    def _delayed_start(self):
        time.sleep(2)

        log("Testing capture source …")
        if not self.screen_capture.is_ready():
            log("❌ Capture source NOT READY")
            return
        log("✅ Capture source ready")

        log("Testing Gemini API connection …")
        try:
            ok, msg = self.gemini_client.test_connection()
        except Exception as e:
            log(f"❌ Gemini test threw exception: {e}")
            log(traceback.format_exc())
            return

        if not ok:
            log(f"❌ Gemini API FAILED: {msg}")
            return

        log(f"✅ Gemini API OK: {msg}")

        if self.audio_capture:
            self.audio_capture.start()
            log("🎧 AudioCapture started")
            time.sleep(0.5)  # Let buffer fill before first dispatch

        self.streaming_manager.start_streaming()
        log("🎬 Streaming started")