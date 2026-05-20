#!/usr/bin/env python3
"""
Vision Service — Entry Point
=============================
Captures screen / camera → Gemini 2.5 Flash analysis → broadcasts results.

Subscribes to Hub audio_context events so Gemini receives the full
picture + audio context from microphone_audio_service and stream_audio_service.

WebSocket clients:  ws://localhost:8015
Health check:       GET  http://localhost:8016/health
List devices:       GET  http://localhost:8016/devices
Set device:         POST http://localhost:8016/set-device  {"device_id": N}
Shutdown:           POST http://localhost:8016/shutdown
"""

import os
import signal
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(_THIS_DIR)

import http_control
from service import VisionService

_service: VisionService | None = None


def _shutdown(*_):
    global _service
    if _service:
        _service.stop()
    sys.exit(0)


def _swap_device(device_id: int):
    global _service
    if _service:
        _service.swap_device(device_id)


def main():
    global _service
    _service = VisionService()

    http_control.start(
        shutdown_callback   = _shutdown,
        set_device_callback = _swap_device,
    )

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    _service.run()


if __name__ == "__main__":
    main()