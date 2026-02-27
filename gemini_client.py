"""
GeminiClient for the Vision Service.
Wraps google-genai chat API with streaming support.
Stateful chat session — call reset_chat() to clear history.
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

        self._init_chat()

    def _init_chat(self):
        try:
            self.chat = self.client.chats.create(
                model=self.model_name,
                config=types.GenerateContentConfig(
                    system_instruction=self.system_prompt,
                    temperature=0.7,
                    max_output_tokens=self.max_output_tokens,
                    safety_settings=[
                        types.SafetySetting(category="HARM_CATEGORY_HARASSMENT",        threshold="BLOCK_NONE"),
                        types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH",       threshold="BLOCK_NONE"),
                        types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
                        types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
                    ],
                ),
            )
        except Exception as e:
            if self.error_callback:
                self.error_callback(f"Failed to init chat: {e}")

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

    def send_message(self, frame, text_prompt: str | None = None):
        if self._is_processing:
            if self.debug_mode:
                print("[Gemini] Skipped frame (API busy)")
            return
        threading.Thread(
            target=self._process_request, args=(frame, text_prompt), daemon=True
        ).start()

    def _process_request(self, frame, text_prompt: str | None):
        with self._lock:
            self._is_processing = True
        try:
            # Convert frame to JPEG bytes
            if hasattr(frame, "shape"):
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb)
            else:
                pil_image = frame

            buf = io.BytesIO()
            pil_image.save(buf, format="JPEG", quality=80)
            img_bytes = buf.getvalue()

            parts = [types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg")]
            if text_prompt:
                parts.append(types.Part.from_text(text=text_prompt))

            response_stream = self.chat.send_message_stream(message=parts)
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

    def reset_chat(self):
        self._init_chat()
        if self.debug_mode:
            print("[Gemini] Chat history reset.")