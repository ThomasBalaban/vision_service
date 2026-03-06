"""
VisionService — full verbose logging
Stateless Gemini calls: 6 frames per batch, dispatched every 2 seconds.
"""
import asyncio
import threading
import time
import traceback
from datetime import datetime

import socketio

from config import (
    API_KEY, CAPTURE_REGION, DEBUG_MODE, FPS,
    HUB_URL, IMAGE_QUALITY, MAX_OUTPUT_TOKENS,
    PROMPT, SERVICE_NAME, VIDEO_DEVICE_INDEX,
)
from gemini_client import GeminiClient
from screen_capture import ScreenCapture
from streaming_manager import StreamingManager
from websocket_server import WebSocketServer

# Flush response buffer if it exceeds this many chars without hitting a sentence end
_BUFFER_FLUSH_CHARS = 800


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] [{SERVICE_NAME}] {msg}", flush=True)


class VisionService:
    def __init__(self):
        log("👁️  Initializing …")
        self._shutting_down         = False
        self._shutdown_lock         = threading.Lock()
        self._hub_emit_count        = 0
        self._ws_broadcast_count    = 0
        self._gemini_response_count = 0
        self._audio_ctx_received    = 0

        # ── WebSocket server ──────────────────────────────────────────────────
        log("Creating WebSocket broadcast server …")
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
                log(f"Capture region set: {CAPTURE_REGION}")
            log("✅ ScreenCapture ready")
        except Exception as e:
            log(f"❌ ScreenCapture init FAILED: {e}")
            log(traceback.format_exc())
            raise

        # ── Gemini client ─────────────────────────────────────────────────────
        log("Initializing GeminiClient …")
        try:
            self.gemini_client = GeminiClient(
                api_key           = API_KEY,
                system_prompt     = PROMPT.replace("{audio_transcripts}", ""),
                response_callback = self._on_gemini_response,
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
            screen_capture  = self.screen_capture,
            gemini_client   = self.gemini_client,
            debug_mode      = DEBUG_MODE,
        )
        self.streaming_manager.set_error_callback(self._on_streaming_error)

        self._response_buffer = ""

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
        log("WebSocket server started")

        threading.Thread(target=self._delayed_start, daemon=True).start()
        log("Delayed start thread launched …")

        log("✅ Running — waiting for Gemini stream to begin …")
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
        log(f"🛑 Shutting down (hub emits: {self._hub_emit_count}, "
            f"WS: {self._ws_broadcast_count}, "
            f"Gemini responses: {self._gemini_response_count})")
        try:
            self.streaming_manager.stop_streaming()
        except Exception as e:
            log(f"Error stopping stream manager: {e}")
        try:
            self.screen_capture.release()
        except Exception as e:
            log(f"Error releasing screen capture: {e}")
        try:
            self.ws_server.stop()
        except Exception as e:
            log(f"Error stopping WS server: {e}")
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

        @self.sio.on("audio_context")
        async def on_audio_context(data):
            self._audio_ctx_received += 1
            ctx = data.get("context", "")
            src = data.get("metadata", {}).get("source", "audio")
            log(f"📥 HUB audio_context #{self._audio_ctx_received} "
                f"from [{src.upper()}]: {repr(ctx[:80])}")
            if ctx:
                self.streaming_manager.add_transcript(f"[{src.upper()}]: {ctx}")

        @self.sio.on("spoken_word_context")
        async def on_spoken_word(data):
            ctx = data.get("context", "")
            log(f"📥 HUB spoken_word_context: {repr(ctx[:80])}")
            if ctx:
                self.streaming_manager.add_transcript(f"[MIC]: {ctx}")

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
        if not self.sio.connected:
            log(f"⚠️  SKIPPED hub emit (not connected): {event}")
            return
        if not self.hub_loop:
            log(f"⚠️  SKIPPED hub emit (no loop): {event}")
            return
        try:
            asyncio.run_coroutine_threadsafe(self.sio.emit(event, data), self.hub_loop)
            self._hub_emit_count += 1
            log(f"→ HUB [{event}] {str(data)[:160]}")
        except Exception as e:
            log(f"❌ HUB EMIT ERROR [{event}]: {e}")
            log(traceback.format_exc())

    # ── Gemini callbacks ──────────────────────────────────────────────────────

    def _on_gemini_response(self, text_chunk: str):
        self._response_buffer += text_chunk

        stripped = self._response_buffer.strip()

        # Only flush on a genuine sentence boundary:
        #   - plain end:             ...word.  ...word!  ...word?
        #   - sentence inside quote: ..."word."  ..."word!"  ..."word?"
        #   - newline
        # A bare " mid-sentence (opening quote) no longer triggers a flush.
        at_sentence_end = (
            stripped.endswith((".", "!", "?"))
            or stripped.endswith(('."', '!"', '?"'))
            or stripped.endswith("\n")
        )
        buffer_overflowed = len(self._response_buffer) > _BUFFER_FLUSH_CHARS

        if not (at_sentence_end or buffer_overflowed):
            return

        final_text = stripped
        if not final_text:
            self._response_buffer = ""
            return

        self._response_buffer = ""
        self._gemini_response_count += 1
        timestamp = datetime.now().isoformat()

        if buffer_overflowed and not at_sentence_end:
            log(f"⚠️  Buffer force-flushed at {len(final_text)} chars (no sentence end found)")

        log(f"🤖 GEMINI RESPONSE #{self._gemini_response_count}: {repr(final_text[:120])}")

        # WebSocket broadcast
        try:
            self.ws_server.broadcast({
                "type":      "vision_analysis",
                "content":   final_text,
                "timestamp": timestamp,
            })
            self._ws_broadcast_count += 1
            log(f"→ WS broadcast #{self._ws_broadcast_count} sent")
        except Exception as e:
            log(f"❌ WS BROADCAST ERROR: {e}")
            log(traceback.format_exc())

        # Hub
        self._emit_to_hub("vision_context", {"context": final_text, "timestamp": timestamp})
        self._emit_to_hub("text_update",    {"type": "text_update", "content": final_text, "timestamp": timestamp})

    def _on_gemini_error(self, msg: str):
        log(f"❌ GEMINI ERROR: {msg}")
        self.ws_server.broadcast({"type": "error", "source": "gemini", "message": msg})

    def _on_streaming_error(self, msg: str):
        log(f"❌ STREAMING ERROR: {msg}")

    # ── Startup ───────────────────────────────────────────────────────────────

    def _delayed_start(self):
        time.sleep(2)
        log("Testing capture source …")
        if not self.screen_capture.is_ready():
            log(f"❌ Capture source NOT READY "
                f"(VIDEO_DEVICE_INDEX={VIDEO_DEVICE_INDEX}, CAPTURE_REGION={CAPTURE_REGION})")
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

        log(f"✅ Gemini API OK: {msg} — starting stream …")
        self.streaming_manager.start_streaming()
        log("🎬 Streaming started")