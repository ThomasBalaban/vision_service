#!/usr/bin/env python3
"""
Vision Service — Entry Point (merged vision + audio)
=====================================================
Captures screen/camera + desktop audio → Gemini 2.5 Flash → dual WebSocket output.

Scene analysis:    ws://localhost:8015
Audio/transcript:  ws://localhost:8017
Health check:      GET  http://localhost:8016/health
Shutdown:          POST http://localhost:8016/shutdown
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


def main():
    global _service
    _service = VisionService()

    http_control.start(shutdown_callback=_shutdown)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT,  _shutdown)

    _service.run()


if __name__ == "__main__":
    main()