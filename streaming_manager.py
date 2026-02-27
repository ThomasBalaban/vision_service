"""
StreamingManager for the Vision Service.
Captures frames at a target FPS and sends them to Gemini with any buffered
audio-context transcripts received from the Hub or sibling services.
No GUI dependencies.
"""

import threading
import time
import traceback
from datetime import datetime


class StreamingManager:
    def __init__(
        self,
        screen_capture,
        gemini_client,
        target_fps: float = 2.0,
        restart_interval: int = 1500,
        debug_mode: bool = False,
    ):
        self.screen_capture     = screen_capture
        self.gemini_client      = gemini_client
        self.target_fps         = target_fps
        self.restart_interval   = restart_interval
        self.debug_mode         = debug_mode

        self.streaming_active   = False
        self.frame_count        = 0
        self.stop_event         = threading.Event()
        self.stream_thread: threading.Thread | None = None

        # Callbacks
        self.status_callback  = None
        self.error_callback   = None
        self.restart_callback = None

        # Audio transcript buffer (fed by Hub subscription)
        self.transcript_buffer: list[str] = []
        self.buffer_lock = threading.Lock()

    # ── Callbacks ─────────────────────────────────────────────────────────

    def set_status_callback(self, cb):    self.status_callback  = cb
    def set_error_callback(self, cb):     self.error_callback   = cb
    def set_restart_callback(self, cb):   self.restart_callback = cb

    # ── Transcript feed (called by VisionService on hub audio_context events) ─

    def add_transcript(self, text: str):
        with self.buffer_lock:
            ts    = datetime.now().strftime("%H:%M:%S")
            entry = f"[{ts}] {text}"
            self.transcript_buffer.append(entry)
            if self.debug_mode:
                print(f"[StreamMgr] Buffered: {entry}")

    # ── Control ───────────────────────────────────────────────────────────

    def start_streaming(self):
        if self.streaming_active:
            return
        print("[StreamMgr] Starting …")
        self.streaming_active = True
        self.stop_event.clear()
        self.frame_count      = 0
        self.stream_thread    = threading.Thread(
            target=self._stream_loop, daemon=True, name="VisionStream"
        )
        self.stream_thread.start()

    def stop_streaming(self):
        if not self.streaming_active:
            return
        print("[StreamMgr] Stopping …")
        self.streaming_active = False
        self.stop_event.set()
        if self.stream_thread:
            self.stream_thread.join(timeout=2)
            self.stream_thread = None

    # ── Internal ─────────────────────────────────────────────────────────

    def _stream_loop(self):
        delay = 1.0 / self.target_fps

        while self.streaming_active and not self.stop_event.is_set():
            t0 = time.time()
            try:
                frame = self.screen_capture.capture_frame()
                if frame:
                    self.frame_count += 1
                    self._send_to_gemini(frame)

                    if self.restart_interval and self.frame_count % self.restart_interval == 0:
                        if self.restart_callback:
                            self.restart_callback()
                else:
                    if self.error_callback:
                        self.error_callback("Failed to capture frame")

            except Exception as e:
                traceback.print_exc()
                if self.error_callback:
                    self.error_callback(f"Stream loop error: {e}")

            sleep_time = max(0, delay - (time.time() - t0))
            time.sleep(sleep_time)

    def _send_to_gemini(self, frame):
        try:
            text_part = ""
            with self.buffer_lock:
                if self.transcript_buffer:
                    text_part = "\n\nRECENT AUDIO LOGS:\n" + "\n".join(self.transcript_buffer)
                    self.transcript_buffer.clear()

            self.gemini_client.send_message(
                frame,
                text_prompt=text_part.strip() or None,
            )
        except Exception as e:
            print(f"[StreamMgr] Send error: {e}")
            if self.error_callback:
                self.error_callback(str(e))