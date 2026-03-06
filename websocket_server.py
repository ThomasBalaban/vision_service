"""
Dual WebSocket broadcast server for the Vision Service.

  Port 8015 — scene analysis  (vision_ws)  → replaces old vision_service WS
  Port 8017 — audio/transcript (audio_ws)  → replaces stream_audio_service WS

Both servers share the same queue/broadcast pattern.
External consumers connect to the same ports as before — nothing changes on their end.
"""

import asyncio
import json
import queue
import threading
import time

import websockets

from config import AUDIO_WS_PORT, VISION_WS_PORT

_QUEUE_MAXSIZE = 100


class _WSServer:
    """Single WebSocket broadcast server on a given port."""

    def __init__(self, port: int, service_label: str):
        self.port              = port
        self.service_label     = service_label
        self.connected_clients: set = set()
        self.message_queue     = queue.Queue(maxsize=_QUEUE_MAXSIZE)
        self.loop: asyncio.AbstractEventLoop | None = None
        self.running           = True

    def start(self):
        t = threading.Thread(
            target=self._run_in_thread, daemon=True,
            name=f"WS-{self.port}"
        )
        t.start()
        print(f"🔌 [{self.service_label}] WebSocket server starting on ws://localhost:{self.port}")

    def stop(self):
        self.running = False
        if self.loop and self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)

    def broadcast(self, data: dict):
        try:
            self.message_queue.put_nowait(data)
        except queue.Full:
            try:
                self.message_queue.get_nowait()
                self.message_queue.put_nowait(data)
            except Exception:
                pass

    def _run_in_thread(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_until_complete(self._serve())
        finally:
            self.loop.close()

    async def _serve(self):
        asyncio.ensure_future(self._queue_processor())
        asyncio.ensure_future(self._heartbeat())
        async with websockets.serve(self._handler, "0.0.0.0", self.port):
            await asyncio.Future()

    async def _handler(self, ws, path=None):
        self.connected_clients.add(ws)
        print(f"🔌 [{self.service_label}] Client connected (total: {len(self.connected_clients)})")
        try:
            await ws.send(json.dumps({
                "type":      "connection_established",
                "service":   self.service_label,
                "timestamp": time.time(),
            }))
            async for msg in ws:
                try:
                    data = json.loads(msg)
                    if data.get("type") == "ping":
                        await ws.send(json.dumps({"type": "pong", "timestamp": time.time()}))
                except Exception:
                    pass
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.connected_clients.discard(ws)
            print(f"🔌 [{self.service_label}] Client disconnected (total: {len(self.connected_clients)})")

    async def _queue_processor(self):
        while self.running:
            try:
                data = self.message_queue.get_nowait()
                await self._do_broadcast(data)
            except queue.Empty:
                pass
            except Exception:
                pass
            await asyncio.sleep(0.01)

    async def _heartbeat(self):
        while self.running:
            await asyncio.sleep(5)
            if self.connected_clients:
                await self._do_broadcast({"type": "heartbeat", "timestamp": time.time()})

    async def _do_broadcast(self, data: dict):
        if not self.connected_clients:
            return
        msg  = json.dumps(data)
        dead = set()
        for client in self.connected_clients.copy():
            try:
                await client.send(msg)
            except Exception:
                dead.add(client)
        for c in dead:
            self.connected_clients.discard(c)


class WebSocketServer:
    """
    Facade that owns both WS servers.
    Callers use broadcast_vision() or broadcast_audio() to route to the right port.
    """

    def __init__(self):
        self._vision = _WSServer(VISION_WS_PORT, "VisionWS")
        self._audio  = _WSServer(AUDIO_WS_PORT,  "AudioWS")

    def start(self):
        self._vision.start()
        self._audio.start()

    def stop(self):
        self._vision.stop()
        self._audio.stop()

    def broadcast_vision(self, data: dict):
        self._vision.broadcast(data)

    def broadcast_audio(self, data: dict):
        self._audio.broadcast(data)