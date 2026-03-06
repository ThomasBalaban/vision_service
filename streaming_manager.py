"""
StreamingManager for the Vision Service.

Two-thread design:
  - Capture thread: grabs frames at CAPTURE_FPS (default 3), stores in a
    rolling buffer capped at BATCH_SIZE (default 6).
  - Dispatch thread: every BATCH_INTERVAL seconds (default 2), takes the
    buffered frames and fires a single stateless Gemini request.

This means one API call per 2 seconds containing up to 6 frames, giving
Gemini temporal context while keeping token usage flat and predictable.
"""

import threading
import time
import traceback
from collections import deque
from datetime import datetime


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
        restart_interval: int = 0,      # kept for API compat, unused (stateless now)
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

        # Rolling frame buffer — only keeps the most recent `batch_size` frames
        self._frame_buffer: deque = deque(maxlen=batch_size)
        self._frame_lock = threading.Lock()

        # Audio transcript buffer
        self.transcript_buffer: list[str] = []
        self._transcript_lock = threading.Lock()

        # Callbacks
        self.status_callback  = None
        self.error_callback   = None
        self.restart_callback = None  # kept for API compat, unused (stateless)

    # ── Callbacks ─────────────────────────────────────────────────────────

    def set_status_callback(self, cb):    self.status_callback  = cb
    def set_error_callback(self, cb):     self.error_callback   = cb
    def set_restart_callback(self, cb):   self.restart_callback = cb  # no-op now

    # ── Transcript feed ───────────────────────────────────────────────────

    def add_transcript(self, text: str):
        with self._transcript_lock:
            ts    = datetime.now().strftime("%H:%M:%S")
            entry = f"[{ts}] {text}"
            self.transcript_buffer.append(entry)
            if self.debug_mode:
                print(f"[StreamMgr] Buffered transcript: {entry}")

    # ── Control ───────────────────────────────────────────────────────────

    def start_streaming(self):
        if self.streaming_active:
            return
        print(f"[StreamMgr] Starting — capture at {self.target_fps} FPS, "
              f"dispatch every {self.batch_interval}s, batch size {self.batch_size}")
        self.streaming_active = True
        self.stop_event.clear()

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
            # Wait for the batch interval
            self.stop_event.wait(timeout=self.batch_interval)
            if not self.streaming_active:
                break

            # Grab current frames
            with self._frame_lock:
                frames = list(self._frame_buffer)
                self._frame_buffer.clear()

            if not frames:
                if self.debug_mode:
                    print("[StreamMgr] Dispatch tick — no frames, skipping")
                continue

            # Grab and clear audio transcripts
            with self._transcript_lock:
                transcripts     = list(self.transcript_buffer)
                self.transcript_buffer.clear()

            text_part = None
            if transcripts:
                text_part = "RECENT AUDIO LOGS:\n" + "\n".join(transcripts)

            self.batch_count += 1
            print(f"[StreamMgr] Dispatching batch #{self.batch_count}: "
                  f"{len(frames)} frames, audio={'yes' if text_part else 'no'}")

            try:
                self.gemini_client.send_frames(frames, text_prompt=text_part)
            except Exception as e:
                traceback.print_exc()
                if self.error_callback:
                    self.error_callback(f"Dispatch error: {e}")