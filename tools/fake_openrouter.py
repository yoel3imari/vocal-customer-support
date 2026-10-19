#!/usr/bin/env python3
"""Minimal OpenAI-compatible chat endpoint for offline development.

Simulates OpenRouter's /api/v1/chat/completions (streaming and non-streaming)
with a canned support answer, so the full voice loop can be exercised without
an API key or network access:

    uv run python tools/fake_openrouter.py &          # listens on :8123
    OPENROUTER_API_KEY=fake \\
    OPENROUTER_BASE_URL=http://127.0.0.1:8123/api/v1 \\
        make run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI(title="fake-openrouter")

ANSWER = (
    "Happy to help with that. The Smart Plus plan is forty-five dollars a month and "
    "includes unlimited 5G data plus a free streaming subscription. "
    "Would you like me to walk you through signing up?"
)


def _last_user_message(messages: list[dict]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return str(message.get("content", ""))
    return ""


def _chunks(text: str, model: str):
    words = text.split()
    for index, word in enumerate(words):
        yield {
            "id": "fake-chunk-1",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": word if index == 0 else f" {word}"},
                    "finish_reason": None,
                }
            ],
        }


@app.post("/api/v1/chat/completions")
async def completions(request: Request):
    body = await request.json()
    model = str(body.get("model", "fake"))
    reply = f'You asked: "{_last_user_message(body.get("messages", []))[:80]}" ' + ANSWER

    if body.get("stream"):
        async def events():
            for chunk in _chunks(reply, model):
                yield f"data: {json.dumps(chunk)}\n\n"
                await asyncio.sleep(0.04)  # pretend to type
            yield "data: [DONE]\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    return JSONResponse(
        {
            "id": "fake-cmpl-1",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": reply},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8123)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
