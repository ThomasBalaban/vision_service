"""
StreamingManager for the Vision Service.

Two-thread design:
  - Capture thread: grabs frames at FPS, stores in rolling deque(maxlen=BATCH_SIZE).
  - Dispatch thread: every BATCH_INTERVAL seconds:
      · Takes current frame batch
      · Drains ambient audio buffer  (always → [AUDIO] section)
      · Drains VAD utterance queue   (only when complete speech ready → [SPEECH] section)
      · Fires one stateless Gemini call with all of the above

Audio and video are decoupled — speech is only included when VAD has a
complete utterance. Ambient audio is always included for music/SFX detection.
"""

import threading
import time
import traceback
from collections import deque

from config import AUDIO_SAMPLE_RATE, BATCH_INTERVAL, BATCH_SIZE, DEBUG_MODE, FPS


class StreamingManager:
    def __init__(
        self,
        screen_capture,
        gemini_client,
        audio_capture=None,
        audio_vad=None,
        target_fps: float    = FPS,
        batch_size: int      = BATCH_SIZE,
        batch_interval: float = BATCH_INTERVAL,
        debug_mode: bool     = DEBUG_MODE,
    ):
        self.screen_capture  = screen_capture
        self.gemini_client   = gemini_client
        self.audio_capture   = audio_capture
        self.audio_vad       = audio_vad
        self.target_fps      = target_fps
        self.batch_size      = batch_size
        self.batch_interval  = batch_interval
        self.debug_mode      = debug_mode

        self.streaming_active = False
        self.frame_count      = 0
        self.batch_count      = 0
        self.stop_event       = threading.Event()

        self._capture_thread: threading.Thread | None  = None
        self._dispatch_thread: threading.Thread | None = None

        self._frame_buffer: deque = deque(maxlen=batch_size)
        self._frame_lock = threading.Lock()

        self.error_callback    = None
        self.dispatch_callback = None

    # ── Callbacks ─────────────────────────────────────────────────────────

    def set_error_callback(self, cb):
        self.error_callback = cb

    def set_dispatch_callback(self, cb):
        """Called just before each Gemini send — used to reset the response buffer."""
        self.dispatch_callback = cb

    # ── Control ───────────────────────────────────────────────────────────

    def start_streaming(self):
        if self.streaming_active:
            return
        print(
            f"[StreamMgr] Starting — {self.target_fps} FPS capture, "
            f"dispatch every {self.batch_interval}s, batch={self.batch_size} frames, "
            f"ambient_audio={'yes' if self.audio_capture else 'no'}, "
            f"vad={'yes' if self.audio_vad else 'no'}"
        )
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

    # ── Capture loop ──────────────────────────────────────────────────────

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
                        print(f"[StreamMgr] Frame #{self.frame_count} "
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

    # ── Dispatch loop ─────────────────────────────────────────────────────

    def _dispatch_loop(self):
        while self.streaming_active and not self.stop_event.is_set():
            self.stop_event.wait(timeout=self.batch_interval)
            if not self.streaming_active:
                break

            # Grab frames
            with self._frame_lock:
                frames = list(self._frame_buffer)
                self._frame_buffer.clear()

            if not frames:
                if self.debug_mode:
                    print("[StreamMgr] Dispatch tick — no frames, skipping")
                continue

            # Ambient audio — always sent when capture is running (for music/SFX)
            ambient_audio = None
            if self.audio_capture:
                ambient_audio = self.audio_capture.drain_ambient()

            # Speech audio — only sent when VAD has a complete utterance
            speech_audio = None
            if self.audio_vad:
                speech_audio = self.audio_vad.get_ready_utterances()

            self.batch_count += 1
            print(
                f"[StreamMgr] Dispatch #{self.batch_count}: "
                f"{len(frames)} frames | "
                f"ambient={'yes' if ambient_audio else 'no'} | "
                f"speech={'yes' if speech_audio else 'no'}"
            )

            # Notify service to reset response buffer before new chunks arrive
            if self.dispatch_callback:
                try:
                    self.dispatch_callback()
                except Exception as e:
                    print(f"[StreamMgr] dispatch_callback error: {e}")

            try:
                self.gemini_client.send_frames(
                    frames,
                    ambient_audio = ambient_audio,
                    speech_audio  = speech_audio,
                    sample_rate   = AUDIO_SAMPLE_RATE,
                )
            except Exception as e:
                traceback.print_exc()
                if self.error_callback:
                    self.error_callback(f"Dispatch error: {e}")