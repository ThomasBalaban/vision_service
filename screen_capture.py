"""
ScreenCapture for the Vision Service.
Supports both direct video (camera/capture-card via OpenCV) and MSS screen region capture.
No GUI dependencies.
"""

import base64
from io import BytesIO

import cv2
import mss
import numpy as np
from PIL import Image


class ScreenCapture:
    def __init__(self, image_quality: int = 85, video_index: int | None = None):
        self.image_quality = image_quality
        self.video_index   = video_index
        self.cap           = None
        self.sct           = None
        self.capture_region: dict | None = None

        if video_index is not None:
            print(f"[ScreenCapture] Initializing camera on index {video_index} …")
            self.cap = cv2.VideoCapture(video_index)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            if not self.cap.isOpened():
                print(f"[ScreenCapture] ⚠️  Could not open video device {video_index}")
        else:
            print("[ScreenCapture] Initializing screen capture (MSS) …")
            self.sct = mss.mss()

    def set_capture_region(self, region: dict):
        self.capture_region = region

    def is_ready(self) -> bool:
        if self.cap and self.cap.isOpened():
            return True
        if self.sct and self.capture_region:
            return True
        return False

    def capture_frame(self) -> Image.Image | None:
        # ── Camera mode ────────────────────────────────────────────────────
        if self.cap:
            ret, frame = self.cap.read()
            if not ret:
                print("[ScreenCapture] Failed to read camera frame")
                return None
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            # Resize if huge
            max_size = 1024
            if img.width > max_size:
                ratio      = max_size / img.width
                img        = img.resize((max_size, int(img.height * ratio)), Image.Resampling.LANCZOS)
            return img

        # ── Screen region mode ─────────────────────────────────────────────
        if self.sct and self.capture_region:
            try:
                shot = self.sct.grab(self.capture_region)
                img  = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
                max_size = 800
                if img.width > max_size or img.height > max_size:
                    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                return img
            except Exception as e:
                print(f"[ScreenCapture] Capture error: {e}")
                return None

        return None

    def image_to_base64(self, image: Image.Image) -> str:
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=self.image_quality)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def release(self):
        if self.cap:
            self.cap.release()