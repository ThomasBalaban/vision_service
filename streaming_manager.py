"""
StreamingManager for the Vision Service.

Two-thread design:
  - Capture thread: grabs frames at CAPTURE_FPS, stores in a rolling buffer.
  - Dispatch thread: grabs buffered frames + raw WAV audio and fires a stateless Gemini request.
"""

import threading
import time
import traceback
from collections import deque
from datetime import datetime

from audio_capture import AudioCapture

# How many frames to collect before sending
BATCH_SIZE     = 6
# How often (seconds) to fire the batch off to Gemini
BATCH_INTERVAL = 2.0
# How fast to capture frames into the buffer
CAPTURE_FPS    = 3.0


class StreamingManager:
    def __init__(
        self,
        screen_capture,
        gemini_client,
        target_fps: float = CAPTURE_FPS,
        batch_size: int = BATCH_SIZE,
        batch_interval: float = BATCH_INTERVAL,
        restart_interval: int = 0,
        debug_mode: bool = False,
    ):
        self.screen_capture  = screen_capture
        self.gemini_client   = gemini_client
        self.target_fps      = target_fps
        self.batch_size      = batch_size
        self.batch_interval  = batch_interval
        self.debug_mode      = debug_mode

        self.streaming_active = False
        self.frame_count      = 0
        self.batch_count      = 0
        self.stop_event       = threading.Event()

        self._capture_thread: threading.Thread | None = None
        self._dispatch_thread: threading.Thread | None = None

        # Rolling frame buffer
        self._frame_buffer: deque = deque(maxlen=batch_size)
        self._frame_lock = threading.Lock()

        # Raw audio rolling buffer
        self.audio_capture = AudioCapture(duration=int(batch_interval))

        # Callbacks
        self.status_callback  = None
        self.error_callback   = None
        self.restart_callback = None

    # ── Callbacks ─────────────────────────────────────────────────────────

    def set_status_callback(self, cb):    self.status_callback  = cb
    def set_error_callback(self, cb):     self.error_callback   = cb
    def set_restart_callback(self, cb):   self.restart_callback = cb

    # ── Transcript feed (Legacy/No-op to prevent service.py crashes) ──────

    def add_transcript(self, text: str):
        # We are now using raw audio bytes instead of text transcripts from the Hub.
        # This is intentionally left blank so the Socket.IO event handler doesn't crash.
        pass

    # ── Control ───────────────────────────────────────────────────────────

    def start_streaming(self):
        if self.streaming_active:
            return
        print(f"[StreamMgr] Starting — capture at {self.target_fps} FPS, "
              f"dispatch every {self.batch_interval}s, batch size {self.batch_size}")
        self.streaming_active = True
        self.stop_event.clear()

        # Start audio recording
        self.audio_capture.start()

        self._capture_thread = threading.Thread(
            target=self._capture_loop, daemon=True, name="VisionCapture"
        )
        self._dispatch_thread = threading.Thread(
            target=self._dispatch_loop, daemon=True, name="VisionDispatch"
        )
        self._capture_thread.start()
        self._dispatch_thread.start()

    def stop_streaming(self):
        if not self.streaming_active:
            return
        print("[StreamMgr] Stopping …")
        self.streaming_active = False
        self.stop_event.set()
        
        self.audio_capture.stop()
        
        if self._capture_thread:
            self._capture_thread.join(timeout=2)
        if self._dispatch_thread:
            self._dispatch_thread.join(timeout=3)
        self._capture_thread  = None
        self._dispatch_thread = None

    # ── Capture loop (fills the buffer) ──────────────────────────────────

    def _capture_loop(self):
        delay = 1.0 / self.target_fps
        while self.streaming_active and not self.stop_event.is_set():
            t0 = time.time()
            try:
                frame = self.screen_capture.capture_frame()
                if frame is not None:
                    with self._frame_lock:
                        self._frame_buffer.append(frame)
                    self.frame_count += 1
                    if self.debug_mode:
                        print(f"[StreamMgr] Captured frame #{self.frame_count} "
                              f"(buffer: {len(self._frame_buffer)}/{self.batch_size})")
                else:
                    if self.error_callback:
                        self.error_callback("Failed to capture frame")
            except Exception as e:
                traceback.print_exc()
                if self.error_callback:
                    self.error_callback(f"Capture loop error: {e}")

            elapsed = time.time() - t0
            time.sleep(max(0.0, delay - elapsed))

    # ── Dispatch loop (fires batches at Gemini) ───────────────────────────

    def _dispatch_loop(self):
        while self.streaming_active and not self.stop_event.is_set():
            self.stop_event.wait(timeout=self.batch_interval)
            if not self.streaming_active:
                break

            with self._frame_lock:
                frames = list(self._frame_buffer)
                self._frame_buffer.clear()

            if not frames:
                if self.debug_mode:
                    print("[StreamMgr] Dispatch tick — no frames, skipping")
                continue

            # Grab the raw WAV bytes from the audio capture buffer
            audio_bytes = self.audio_capture.get_recent_wav_bytes()

            self.batch_count += 1
            print(f"[StreamMgr] Dispatching batch #{self.batch_count}: "
                  f"{len(frames)} frames, audio={'yes' if audio_bytes else 'no'}")

            try:
                # Dispatch to Gemini with visuals + audio
                self.gemini_client.send_frames(frames, text_prompt=None, audio_bytes=audio_bytes)
            except Exception as e:
                traceback.print_exc()
                if self.error_callback:
                    self.error_callback(f"Dispatch error: {e}")