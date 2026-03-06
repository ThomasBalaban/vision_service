"""
GeminiClient for the Vision Service.
Stateless single calls — no chat history, no context growth.

Each call receives:
  - Up to 6 video frames
  - Ambient audio (always, for music/SFX detection)
  - Speech audio (optional, only when VAD has a complete utterance)
"""

import io
import threading
import wave

import cv2
from google import genai
from google.genai import types
from PIL import Image


class GeminiClient:
    def __init__(
        self,
        api_key: str,
        system_prompt: str,
        response_callback,
        error_callback,
        max_output_tokens: int = 800,
        debug_mode: bool = False,
    ):
        self.api_key           = api_key
        self.system_prompt     = system_prompt
        self.response_callback = response_callback
        self.error_callback    = error_callback
        self.max_output_tokens = max_output_tokens
        self.debug_mode        = debug_mode

        self.client     = genai.Client(api_key=self.api_key)
        self.model_name = "gemini-2.5-flash"

        self._is_processing = False
        self._lock          = threading.Lock()

        self._safety_settings = [
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT",        threshold="BLOCK_NONE"),
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH",       threshold="BLOCK_NONE"),
            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
        ]

    def test_connection(self):
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents="Reply with 'OK' if you receive this.",
            )
            if response and response.text:
                return True, "Connection Successful"
            return False, "No response text received"
        except Exception as e:
            return False, str(e)

    def send_frames(
        self,
        frames: list,
        ambient_audio: bytes | None = None,
        speech_audio: bytes | None = None,
        sample_rate: int = 16000,
    ):
        """
        Send a batch of frames + audio as a single stateless request.

        ambient_audio: raw int16 PCM bytes — always sent when available,
                       Gemini uses for music/SFX detection.
        speech_audio:  raw int16 PCM bytes — only sent when VAD has a
                       complete utterance, Gemini uses for transcription.

        Skips if a request is already in flight.
        """
        with self._lock:
            if self._is_processing:
                if self.debug_mode:
                    print(f"[Gemini] Skipped batch of {len(frames)} frames (API busy)")
                return
            self._is_processing = True

        threading.Thread(
            target=self._process_request,
            args=(frames, ambient_audio, speech_audio, sample_rate),
            daemon=True,
        ).start()

    def _frame_to_jpeg_bytes(self, frame) -> bytes:
        if hasattr(frame, "shape"):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)
        else:
            pil_image = frame

        max_size = 512
        if pil_image.width > max_size or pil_image.height > max_size:
            pil_image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

        buf = io.BytesIO()
        pil_image.save(buf, format="JPEG", quality=75)
        return buf.getvalue()

    def _pcm_to_wav(self, pcm_bytes: bytes, sample_rate: int) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)          # int16 = 2 bytes
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_bytes)
        return buf.getvalue()

    def _process_request(
        self,
        frames: list,
        ambient_audio: bytes | None,
        speech_audio: bytes | None,
        sample_rate: int,
    ):
        try:
            parts = []

            # 1. Video frames
            for i, frame in enumerate(frames):
                img_bytes = self._frame_to_jpeg_bytes(frame)
                parts.append(types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"))
                if self.debug_mode:
                    print(f"[Gemini] Frame {i+1}/{len(frames)} ({len(img_bytes)} bytes)")

            # 2. Ambient audio (always present when capture is running)
            if ambient_audio:
                wav = self._pcm_to_wav(ambient_audio, sample_rate)
                parts.append(types.Part.from_bytes(data=wav, mime_type="audio/wav"))
                parts.append(types.Part.from_text(
                    text="The above audio is the AMBIENT background audio clip."
                ))
                if self.debug_mode:
                    print(f"[Gemini] Ambient audio: {len(wav)} bytes WAV")

            # 3. Speech audio (only when VAD detected a complete utterance)
            if speech_audio:
                wav = self._pcm_to_wav(speech_audio, sample_rate)
                parts.append(types.Part.from_bytes(data=wav, mime_type="audio/wav"))
                parts.append(types.Part.from_text(
                    text="The above audio is the SPEECH clip — an isolated utterance "
                         "detected by voice activity detection. Transcribe it accurately."
                ))
                if self.debug_mode:
                    print(f"[Gemini] Speech audio: {len(wav)} bytes WAV")

            # Stateless call
            response_stream = self.client.models.generate_content_stream(
                model=self.model_name,
                contents=[types.Content(role="user", parts=parts)],
                config=types.GenerateContentConfig(
                    system_instruction=self.system_prompt,
                    temperature=0.7,
                    max_output_tokens=self.max_output_tokens,
                    safety_settings=self._safety_settings,
                ),
            )

            for chunk in response_stream:
                if chunk.text and self.response_callback:
                    self.response_callback(chunk.text)

        except Exception as e:
            print(f"[Gemini] Error: {e}")
            if self.error_callback:
                self.error_callback(str(e))
        finally:
            with self._lock:
                self._is_processing = False