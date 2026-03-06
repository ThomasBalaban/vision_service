"""
GeminiClient for the Vision Service.
Stateless single calls — no chat history, no context growth.
Accepts a batch of frames (up to 6) per request for temporal awareness.
"""

import io
import threading

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
        max_output_tokens: int = 600,
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

    def send_frames(self, frames: list, text_prompt: str | None = None):
        """
        Send a batch of frames as a single stateless request.
        frames: list of PIL.Image or numpy arrays (up to 6).
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
            args=(frames, text_prompt),
            daemon=True,
        ).start()

    def _frame_to_jpeg_bytes(self, frame) -> bytes:
        if hasattr(frame, "shape"):
            # numpy / cv2 frame
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)
        else:
            pil_image = frame

        # Resize to max 512px on longest side — cuts token cost vs 800px
        max_size = 512
        if pil_image.width > max_size or pil_image.height > max_size:
            pil_image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

        buf = io.BytesIO()
        pil_image.save(buf, format="JPEG", quality=75)
        return buf.getvalue()

    def _process_request(self, frames: list, text_prompt: str | None):
        try:
            parts = []

            for i, frame in enumerate(frames):
                img_bytes = self._frame_to_jpeg_bytes(frame)
                parts.append(types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"))
                if self.debug_mode:
                    print(f"[Gemini] Frame {i+1}/{len(frames)} encoded ({len(img_bytes)} bytes)")

            if text_prompt:
                parts.append(types.Part.from_text(text=text_prompt))

            # Stateless — no chat session, no growing history, flat token cost every call
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