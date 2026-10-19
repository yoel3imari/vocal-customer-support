"""Headless audio-loop smoke test (no browser, no models).

Connects to a running server, starts a call, streams a 440 Hz tone as PCM16
frames, and verifies echoed binary frames + state/transcript messages come back.

    uv run python tools/smoke_ws.py
"""

from __future__ import annotations

import asyncio
import json
import math
import struct

import websockets

URL = "ws://127.0.0.1:8000/ws"
RATE = 16_000
FRAME = 1024


def tone(seconds: float, freq: float = 440.0) -> bytes:
    samples = int(RATE * seconds)
    out = bytearray()
    for i in range(samples):
        out += struct.pack("<h", int(0.2 * 32767 * math.sin(2 * math.pi * freq * i / RATE)))
    return bytes(out)


async def main() -> None:
    payload = tone(1.0)
    async with websockets.connect(URL, max_size=2**24) as ws:
        await ws.send(json.dumps({"type": "start"}))
        audio_frames = 0
        states: list[str] = []
        sent = 0
        for offset in range(0, len(payload), FRAME):
            await ws.send(payload[offset:offset + FRAME])
            sent += 1
            await asyncio.sleep(FRAME / (RATE * 2))
            while True:  # drain anything the server already sent us
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=0.001)
                except asyncio.TimeoutError:
                    break
                if isinstance(msg, bytes):
                    audio_frames += 1
                else:
                    data = json.loads(msg)
                    if data.get("state"):
                        states.append(data["state"])
        await asyncio.sleep(1.0)
        while True:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=0.5)
            except asyncio.TimeoutError:
                break
            if isinstance(msg, bytes):
                audio_frames += 1
            else:
                data = json.loads(msg)
                if data.get("state"):
                    states.append(data["state"])
        await ws.send(json.dumps({"type": "end"}))
        print(f"sent {sent} mic frames, received {audio_frames} agent audio frames")
        print(f"states seen: {states}")


if __name__ == "__main__":
    asyncio.run(main())
