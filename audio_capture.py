"""
AudioCapture for the Vision Service.
Maintains a rolling buffer of the last N seconds of audio.
"""
import io
import wave
import threading
from collections import deque
import pyaudio

class AudioCapture:
    def __init__(self, duration: int = 2, rate: int = 16000, chunk: int = 1024):
        self.duration = duration
        self.rate = rate
        self.chunk = chunk
        self.format = pyaudio.paInt16
        self.channels = 1
        
        self.p = pyaudio.PyAudio()
        
        # Determine how many chunks make up our target duration
        self.max_chunks = int((self.rate / self.chunk) * self.duration)
        self.buffer = deque(maxlen=self.max_chunks)
        
        self.is_recording = False
        self._thread = None
        self._lock = threading.Lock()

    def start(self):
        if self.is_recording: return
        self.is_recording = True
        self.stream = self.p.open(format=self.format,
                                  channels=self.channels,
                                  rate=self.rate,
                                  input=True,
                                  frames_per_buffer=self.chunk)
        self._thread = threading.Thread(target=self._record_loop, daemon=True)
        self._thread.start()
        print(f"[AudioCapture] Started recording ({self.rate}Hz, {self.duration}s buffer)")

    def _record_loop(self):
        while self.is_recording:
            try:
                data = self.stream.read(self.chunk, exception_on_overflow=False)
                with self._lock:
                    self.buffer.append(data)
            except Exception as e:
                print(f"[AudioCapture] Error: {e}")

    def get_recent_wav_bytes(self) -> bytes | None:
        """Returns the rolling buffer as a WAV file in memory."""
        with self._lock:
            if not self.buffer:
                return None
            frames = list(self.buffer)
            
        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.p.get_sample_size(self.format))
            wf.setframerate(self.rate)
            wf.writeframes(b''.join(frames))
            
        return buf.getvalue()

    def stop(self):
        self.is_recording = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()