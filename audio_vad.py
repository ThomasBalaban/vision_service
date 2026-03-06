"""
AudioVAD for the Vision Service.

Receives resampled float32 PCM chunks from AudioCapture and runs them
through webrtcvad to detect speech boundaries. Completed utterances
(speech followed by VAD_SILENCE_DURATION_MS of silence, or hitting
VAD_MAX_UTTERANCE_S) are placed in a queue for StreamingManager to pick up.

State machine:
    SILENCE  →  (speech frame detected)  →  SPEAKING
    SPEAKING →  (silence ≥ 600ms)        →  queue utterance, → SILENCE
    SPEAKING →  (length ≥ 8s)            →  queue utterance, → SILENCE (force close)
"""

import queue
import threading
from collections import deque

import numpy as np
import webrtcvad

from config import (
    AUDIO_SAMPLE_RATE,
    VAD_AGGRESSIVENESS,
    VAD_FRAME_MS,
    VAD_MAX_UTTERANCE_S,
    VAD_SILENCE_DURATION_MS,
    DEBUG_MODE,
)

# Number of silent frames needed to close an utterance
_SILENCE_FRAMES_NEEDED = VAD_SILENCE_DURATION_MS // VAD_FRAME_MS

# Max utterance length in samples
_MAX_UTT_SAMPLES = int(AUDIO_SAMPLE_RATE * VAD_MAX_UTTERANCE_S)

# VAD frame size in samples (must match VAD_FRAME_MS exactly)
_FRAME_SAMPLES = int(AUDIO_SAMPLE_RATE * VAD_FRAME_MS / 1000)


class AudioVAD:
    def __init__(self):
        self._vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)

        # Internal working buffer — accumulates resampled float32 chunks
        # until we have a full VAD frame to process
        self._work_buf  = np.array([], dtype=np.float32)
        self._work_lock = threading.Lock()

        # Speech accumulation
        self._speech_buf: list[np.ndarray] = []
        self._in_speech       = False
        self._silent_frames   = 0

        # Completed utterances ready for StreamingManager
        self._utterance_queue: queue.Queue = queue.Queue()

    # ── Public ────────────────────────────────────────────────────────────────

    def feed(self, chunk: np.ndarray):
        """
        Receive a float32 resampled chunk from AudioCapture.
        Internally buffers until full VAD frames are available.
        """
        with self._work_lock:
            self._work_buf = np.concatenate([self._work_buf, chunk])
            while len(self._work_buf) >= _FRAME_SAMPLES:
                frame            = self._work_buf[:_FRAME_SAMPLES]
                self._work_buf   = self._work_buf[_FRAME_SAMPLES:]
                self._process_frame(frame)

    def get_ready_utterances(self) -> bytes | None:
        """
        Drain all completed utterances from the queue, concatenate them,
        and return as raw int16 PCM bytes at AUDIO_SAMPLE_RATE.
        Returns None if nothing is ready.
        """
        chunks = []
        while True:
            try:
                chunks.append(self._utterance_queue.get_nowait())
            except queue.Empty:
                break

        if not chunks:
            return None

        combined = np.concatenate(chunks)
        combined = combined - np.mean(combined)
        combined = np.clip(combined, -1.0, 1.0)
        return (combined * 32767).astype(np.int16).tobytes()

    def has_ready_utterances(self) -> bool:
        return not self._utterance_queue.empty()

    # ── Internal ─────────────────────────────────────────────────────────────

    def _process_frame(self, frame: np.ndarray):
        # Convert float32 → int16 PCM bytes for webrtcvad
        pcm_bytes = (frame * 32767).astype(np.int16).tobytes()

        try:
            is_speech = self._vad.is_speech(pcm_bytes, AUDIO_SAMPLE_RATE)
        except Exception:
            # webrtcvad can raise if frame size is wrong — skip silently
            return

        if is_speech:
            self._in_speech     = True
            self._silent_frames = 0
            self._speech_buf.append(frame)

            # Force-close if utterance has grown too long
            total_samples = sum(len(f) for f in self._speech_buf)
            if total_samples >= _MAX_UTT_SAMPLES:
                if DEBUG_MODE:
                    print(f"[VAD] Force-closing utterance at max length "
                          f"({VAD_MAX_UTTERANCE_S}s)", flush=True)
                self._close_utterance()

        else:
            if self._in_speech:
                self._silent_frames += 1
                # Keep buffering during short pauses — include silence frames
                # so the audio clip has natural trailing silence
                self._speech_buf.append(frame)

                if self._silent_frames >= _SILENCE_FRAMES_NEEDED:
                    if DEBUG_MODE:
                        print(f"[VAD] Utterance closed after "
                              f"{self._silent_frames * VAD_FRAME_MS}ms silence", flush=True)
                    self._close_utterance()

    def _close_utterance(self):
        if not self._speech_buf:
            return

        utterance = np.concatenate(self._speech_buf)

        if DEBUG_MODE:
            dur = len(utterance) / AUDIO_SAMPLE_RATE
            print(f"[VAD] Queuing utterance: {dur:.2f}s", flush=True)

        self._utterance_queue.put(utterance)

        # Reset state
        self._speech_buf    = []
        self._in_speech     = False
        self._silent_frames = 0