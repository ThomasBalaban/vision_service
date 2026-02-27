"""
VisionService
─────────────
• Captures screen / camera frames
• Sends frames + buffered audio-context to Gemini 2.5 Flash
• Subscribes to Hub audio_context events (from mic & desktop audio services)
  so Gemini always gets the combined picture + audio context
• Broadcasts vision analysis results to:
    - WebSocket clients on port 8015
    - The central Hub (vision_context, text_update)
"""

import asyncio
import threading
import time
from datetime import datetime

import socketio

from config import (
    API_KEY,
    CAPTURE_REGION,
    DEBUG_MODE,
    FPS,
    HUB_URL,
    IMAGE_QUALITY,
    MAX_OUTPUT_TOKENS,
    PROMPT,
    SERVICE_NAME,
    VIDEO_DEVICE_INDEX,
)
from gemini_client import GeminiClient
from screen_capture import ScreenCapture
from streaming_manager import StreamingManager
from websocket_server import WebSocketServer


class VisionService:
    def __init__(self):
        print(f"👁️  [{SERVICE_NAME}] Initializing …")

        self._shutting_down  = False
        self._shutdown_lock  = threading.Lock()

        # ── WebSocket broadcast server ────────────────────────────────────────
        self.ws_server = WebSocketServer()

        # ── Socket.IO hub client ──────────────────────────────────────────────
        self.sio      = socketio.AsyncClient(reconnection=True, reconnection_delay=5)
        self.hub_loop: asyncio.AbstractEventLoop | None = None
        self._register_hub_events()

        # ── Screen capture ────────────────────────────────────────────────────
        self.screen_capture = ScreenCapture(
            image_quality=IMAGE_QUALITY,
            video_index=VIDEO_DEVICE_INDEX,
        )
        if VIDEO_DEVICE_INDEX is None and CAPTURE_REGION:
            self.screen_capture.set_capture_region(CAPTURE_REGION)

        # ── Gemini response buffer ────────────────────────────────────────────
        self._response_buffer      = ""
        self._last_gemini_context  = ""

        # ── Gemini client ─────────────────────────────────────────────────────
        self.gemini_client = GeminiClient(
            api_key           = API_KEY,
            system_prompt     = PROMPT.replace("{audio_transcripts}", ""),
            response_callback = self._on_gemini_response,
            error_callback    = self._on_gemini_error,
            max_output_tokens = MAX_OUTPUT_TOKENS,
            debug_mode        = DEBUG_MODE,
        )

        # ── Streaming manager ─────────────────────────────────────────────────
        self.streaming_manager = StreamingManager(
            screen_capture   = self.screen_capture,
            gemini_client    = self.gemini_client,
            target_fps       = FPS,
            restart_interval = 1500,
            debug_mode       = DEBUG_MODE,
        )
        self.streaming_manager.set_error_callback(self._on_streaming_error)

    # ── Public ───────────────────────────────────────────────────────────────

    def run(self):
        """Start all components; blocks until stop() is called."""
        # Hub event-loop
        self.hub_loop = asyncio.new_event_loop()
        threading.Thread(
            target=self._hub_thread, args=(self.hub_loop,), daemon=True, name="VisionHub"
        ).start()

        # WebSocket server
        self.ws_server.start()

        # Brief pause to let hub connect, then verify Gemini + start streaming
        threading.Thread(target=self._delayed_start, daemon=True).start()

        print(f"✅ [{SERVICE_NAME}] Running. Press Ctrl-C to stop.")

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

        print(f"🛑 [{SERVICE_NAME}] Shutting down …")

        try:
            self.streaming_manager.stop_streaming()
        except Exception:
            pass

        try:
            self.screen_capture.release()
        except Exception:
            pass

        try:
            self.ws_server.stop()
        except Exception:
            pass

        if self.hub_loop:
            self.hub_loop.call_soon_threadsafe(self.hub_loop.stop)

        print(f"🛑 [{SERVICE_NAME}] Stopped.")

    # ── Hub: connection + event registration ─────────────────────────────────

    def _register_hub_events(self):
        @self.sio.on("audio_context")
        async def on_audio_context(data):
            """Feed incoming audio transcripts into the Gemini context buffer."""
            ctx = data.get("context", "")
            if ctx:
                src = data.get("metadata", {}).get("source", "audio")
                self.streaming_manager.add_transcript(f"[{src.upper()}]: {ctx}")

        @self.sio.on("spoken_word_context")
        async def on_spoken_word(data):
            ctx = data.get("context", "")
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
                    await self.sio.connect(HUB_URL)
                    print(f"✅ [{SERVICE_NAME}] Hub connected: {HUB_URL}")
                except Exception as e:
                    print(f"⚠️  [{SERVICE_NAME}] Hub connect failed: {e}. Retrying …")
                    await asyncio.sleep(5)
            await asyncio.sleep(2)

    def _emit_to_hub(self, event: str, data: dict):
        if self.sio.connected and self.hub_loop:
            try:
                asyncio.run_coroutine_threadsafe(
                    self.sio.emit(event, data), self.hub_loop
                )
            except Exception as e:
                print(f"❌ [{SERVICE_NAME}] Hub emit error: {e}")

    # ── Gemini callbacks ──────────────────────────────────────────────────────

    def _on_gemini_response(self, text_chunk: str):
        """Accumulate chunks; flush on sentence-ending punctuation."""
        self._response_buffer += text_chunk

        if self._response_buffer.strip().endswith((".", "!", "?", '"', "\n")):
            final_text = self._response_buffer.strip()
            self._last_gemini_context = final_text
            self._response_buffer = ""

            timestamp = datetime.now().isoformat()

            # ── WebSocket clients ────────────────────────────────────────────
            self.ws_server.broadcast({
                "type":      "vision_analysis",
                "content":   final_text,
                "timestamp": timestamp,
            })

            # ── Hub ──────────────────────────────────────────────────────────
            self._emit_to_hub("vision_context",  {"context": final_text})
            self._emit_to_hub("text_update",     {"type": "text_update", "content": final_text})

            print(f"👁️  [{SERVICE_NAME}] → {final_text[:120]}")

    def _on_gemini_error(self, msg: str):
        print(f"❌ [{SERVICE_NAME}] Gemini error: {msg}")
        self.ws_server.broadcast({"type": "error", "source": "gemini", "message": msg})

    def _on_streaming_error(self, msg: str):
        print(f"❌ [{SERVICE_NAME}] Streaming error: {msg}")

    # ── Startup helpers ───────────────────────────────────────────────────────

    def _delayed_start(self):
        """Test Gemini connection then kick off streaming."""
        time.sleep(2)  # Give hub a moment to connect
        if not self.screen_capture.is_ready():
            print(f"⚠️  [{SERVICE_NAME}] Capture source not ready. Check VIDEO_DEVICE_INDEX / CAPTURE_REGION.")
            return

        print(f"👁️  [{SERVICE_NAME}] Testing Gemini API …")
        ok, msg = self.gemini_client.test_connection()
        if not ok:
            print(f"❌ [{SERVICE_NAME}] Gemini API failed: {msg}")
            return

        print(f"✅ [{SERVICE_NAME}] Gemini OK. Starting stream at {FPS} FPS …")
        self.streaming_manager.start_streaming()