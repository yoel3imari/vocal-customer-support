"""End-to-end call simulator: speak a question into the WebSocket like a browser would.

    uv run python tools/simulate_call.py "How much is the Smart Plus plan?"

Requires espeak-ng + ffmpeg (caller audio only). The agent's spoken reply is saved to
/tmp/voxdesk_reply.wav so you can listen to it.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import tempfile
import wave
from pathlib import Path

import websockets

URL = "ws://127.0.0.1:8000/ws"
RATE = 16_000
FRAME = 1024  # bytes = 32 ms


def synthesize_question(text: str) -> bytes:
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "question.wav"
        raw = Path(tmp) / "question.raw"
        subprocess.run(
            ["espeak-ng", "-v", "en-us", "-s", "155", "-w", str(wav), text],
            check=True,
        )
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav),
             "-ar", str(RATE), "-ac", "1", "-f", "s16le", str(raw)],
            check=True,
        )
        return raw.read_bytes()


async def listen(ws, agent_pcm: list[bytes], done: asyncio.Event, state: dict) -> None:
    try:
        async for message in ws:
            if isinstance(message, bytes):
                agent_pcm.append(message)
                continue
            data = json.loads(message)
            kind = data.get("type")
            if kind == "state":
                state["state"] = data["state"]
                if data.get("sampleRate"):
                    state["rate"] = data["sampleRate"]
                print(f"  · state: {data['state']} {data.get('detail') or ''}".rstrip())
                if data["state"] == "listening" and state.get("waiting_for_reply"):
                    done.set()
                if data["state"] == "speaking" and state.get("waiting_for_reply"):
                    state["spoke"] = True
            elif kind == "transcript":
                print(f"  · {data['role']}: {data['text']}{' [interrupted]' if data.get('interrupted') else ''}")
            elif kind == "metrics":
                print(f"  · metric {data['stage']}: {data['ms']} ms")
            elif kind == "error":
                print(f"  ! error: {data['message']}")
            elif kind == "call_end":
                done.set()
    except websockets.ConnectionClosed:
        done.set()


async def wait_for(ws, state: dict, target: str, timeout: float = 30.0) -> None:
    for _ in range(int(timeout / 0.1)):
        if state.get("state") == target:
            return
        await asyncio.sleep(0.1)


async def main(text: str, url: str, save: Path) -> None:
    pcm = synthesize_question(text)
    seconds = len(pcm) / (RATE * 2)
    print(f"caller question: {text!r} ({seconds:.1f}s of audio)")

    agent_pcm: list[bytes] = []
    state: dict = {}
    done = asyncio.Event()

    async with websockets.connect(url, max_size=2**24) as ws:
        await ws.send(json.dumps({"type": "start"}))
        listener = asyncio.create_task(listen(ws, agent_pcm, done, state))

        await wait_for(ws, state, "speaking", timeout=10)   # greeting
        await wait_for(ws, state, "listening", timeout=20)  # greeting done
        state["spoke"] = False
        state["waiting_for_reply"] = True
        print("streaming caller audio…")
        for offset in range(0, len(pcm), FRAME):
            await ws.send(pcm[offset:offset + FRAME])
            await asyncio.sleep(FRAME / (RATE * 2))
        # Real mics keep streaming silence; that's what lets the server endpoint.
        silence = b"\x00\x00" * (RATE // 2)
        for offset in range(0, len(silence), FRAME):
            await ws.send(silence[offset:offset + FRAME])
            await asyncio.sleep(FRAME / (RATE * 2))
        await asyncio.sleep(0.2)

        try:
            await asyncio.wait_for(done.wait(), timeout=45)
        except asyncio.TimeoutError:
            print("! timed out waiting for the agent")

        await ws.send(json.dumps({"type": "end"}))
        await asyncio.sleep(0.3)
        listener.cancel()

    rate = state.get("rate", 22050)
    if agent_pcm:
        audio = b"".join(agent_pcm)
        with wave.open(str(save), "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(rate)
            out.writeframes(audio)
        print(f"agent reply saved → {save} ({len(audio) / (rate * 2):.1f}s @ {rate} Hz)")
    else:
        print("no agent audio received")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("text", help="what the caller says")
    parser.add_argument("--url", default=URL)
    parser.add_argument("--save", default="/tmp/voxdesk_reply.wav", type=Path)
    args = parser.parse_args()
    asyncio.run(main(args.text, args.url, args.save))
