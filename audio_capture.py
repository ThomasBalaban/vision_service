"""
AudioCapture for the Vision Service.

Captures desktop audio via sounddevice, resamples to AUDIO_SAMPLE_RATE,
and serves two consumers:

  1. AudioVAD  — receives every PCM chunk for speech detection.
                 Registered via set_vad(vad_instance).

  2. Ambient buffer — a rolling window of the last AMBIENT_WINDOW_S seconds.
                      Drained by StreamingManager every dispatch tick for
                      music/SFX detection.
"""

import threading
import time

import numpy as np
import sounddevice as sd
from scipy import signal as scipy_signal

from config import (
    AMBIENT_WINDOW_S, AUDIO_GAIN, AUDIO_SAMPLE_RATE,
    DEBUG_MODE, DESKTOP_AUDIO_DEVICE_ID,
)


class AudioCapture:
    def __init__(
        self,
        device_id: int   = DESKTOP_AUDIO_DEVICE_ID,
        sample_rate: int = AUDIO_SAMPLE_RATE,
        gain: float      = AUDIO_GAIN,
    ):
        self.device_id   = device_id
        self.sample_rate = sample_rate
        self.gain        = gain

        self._native_rate: int = sample_rate
        self._vad = None   # Set via set_vad()

        # Ambient rolling buffer: keeps last AMBIENT_WINDOW_S of resampled float32
        self._ambient_max  = int(sample_rate * AMBIENT_WINDOW_S)
        self._ambient_buf: list[np.ndarray] = []
        self._ambient_lock = threading.Lock()

        self._running = False
        self._stream: sd.InputStream | None = None
        self._thread: threading.Thread | None = None

        self.volume_callback = None

    # ── Public ────────────────────────────────────────────────────────────────

    def set_vad(self, vad):
        """Register an AudioVAD instance to receive resampled PCM chunks."""
        self._vad = vad

    def set_volume_callback(self, cb):
        self.volume_callback = cb

    def start(self):
        self._running = True
        self._thread  = threading.Thread(
            target=self._run, daemon=True, name="AudioCapture"
        )
        self._thread.start()

    def stop(self):
        self._running = False
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None

    def drain_ambient(self) -> bytes | None:
        """
        Return the last AMBIENT_WINDOW_S of audio as int16 PCM bytes,
        with DC removal and gain applied. Does NOT clear the buffer —
        ambient audio is a rolling window, not a consume-once queue.
        Returns None if the buffer is nearly empty (< 0.5s).
        """
        with self._ambient_lock:
            if not self._ambient_buf:
                return None
            combined = np.concatenate(self._ambient_buf)

        min_samples = int(self.sample_rate * 0.5)
        if len(combined) < min_samples:
            return None

        combined = combined - np.mean(combined)
        combined = np.clip(combined * self.gain, -1.0, 1.0)
        return (combined * 32767).astype(np.int16).tobytes()

    def is_ready(self) -> bool:
        return self._running and self._stream is not None

    # ── Internal ─────────────────────────────────────────────────────────────

    def _resample(self, chunk: np.ndarray) -> np.ndarray:
        if self._native_rate == self.sample_rate:
            return chunk
        n = int(len(chunk) * self.sample_rate / self._native_rate)
        return scipy_signal.resample(chunk, n)

    def _audio_callback(self, indata, frames, time_info, status):
        if not self._running:
            return

        # Convert int16 → float32 and resample to target rate
        raw   = indata.flatten().astype(np.float32) / 32768.0
        chunk = self._resample(raw)

        # Feed to VAD (operates on resampled float32)
        if self._vad is not None:
            try:
                self._vad.feed(chunk)
            except Exception:
                pass

        # Maintain ambient rolling buffer
        with self._ambient_lock:
            self._ambient_buf.append(chunk)
            # Trim to max window size
            total = sum(len(c) for c in self._ambient_buf)
            while total > self._ambient_max and self._ambient_buf:
                removed = self._ambient_buf.pop(0)
                total  -= len(removed)

        # Volume meter
        if self.volume_callback:
            rms = float(np.sqrt(np.mean(chunk ** 2)))
            self.volume_callback(min(1.0, rms * 10))

    def _run(self):
        retry, max_retry = 0, 5
        while self._running and retry < max_retry:
            try:
                dev_info          = sd.query_devices(self.device_id, "input")
                self._native_rate = int(dev_info["default_samplerate"])
                name              = dev_info["name"]

                print(
                    f"🎧 [AudioCapture] Device: {name} | "
                    f"native={self._native_rate} Hz → target={self.sample_rate} Hz",
                    flush=True,
                )

                # webrtcvad requires exact frame sizes; use 30ms chunks at native rate,
                # the callback will resample before feeding VAD
                chunk_samples = int(self._native_rate * 0.03)

                self._stream = sd.InputStream(
                    device    = self.device_id,
                    channels  = 1,
                    samplerate= self._native_rate,
                    callback  = self._audio_callback,
                    blocksize = chunk_samples,
                    dtype     = "int16",
                    latency   = "low",
                )
                self._stream.start()
                print("✅ [AudioCapture] Stream active", flush=True)

                while self._running:
                    time.sleep(0.1)
                break

            except Exception as e:
                retry += 1
                print(f"⚠️  [AudioCapture] Error (attempt {retry}/{max_retry}): {e}", flush=True)
                if self._stream:
                    try:
                        self._stream.stop()
                        self._stream.close()
                    except Exception:
                        pass
                    self._stream = None
                time.sleep(2.0)

        if retry >= max_retry:
            print("❌ [AudioCapture] Failed to start after max retries", flush=True)