from __future__ import annotations

"""VoxDesk — FastAPI entrypoint: call console + audio WebSocket."""

import asyncio
import contextlib
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.config import Settings, load_settings
from src.pipeline.session import CallSession
from src.protocol import error as error_msg
from src.runtime import Runtime

log = logging.getLogger("voxdesk")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

settings: Settings = load_settings()
runtime: Runtime | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global runtime
    log.info("starting VoxDesk (tenant=%s, model=%s)", settings.tenant.slug, settings.llm_model)
    runtime = await asyncio.to_thread(Runtime.build, settings)
    try:
        yield
    finally:
        await runtime.aclose()
        runtime = None


app = FastAPI(title="VoxDesk", docs_url=None, redoc_url=None, lifespan=lifespan)
app.mount("/static", StaticFiles(directory=settings.web_dir), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(settings.web_dir / "index.html")


@app.get("/healthz")
async def healthz() -> dict:
    return {
        "ok": runtime is not None,
        "tenant": settings.tenant.slug,
        "model": settings.llm_model,
        "rag_chunks": len(runtime.knowledge.chunks) if runtime else 0,
        "llm_configured": bool(settings.openrouter_api_key),
    }


@app.get("/api/calls")
async def recent_calls() -> dict:
    if runtime is None:
        return {"calls": [], "stats": {}}
    return {
        "calls": runtime.store.recent_calls(20),
        "stats": runtime.store.stats(),
    }


@app.websocket("/ws")
async def ws_audio(websocket: WebSocket) -> None:
    if runtime is None:
        await websocket.close(code=1013, reason="Server starting")
        return
    await websocket.accept()
    session = CallSession(settings, runtime)

    async def send_json(payload: dict) -> None:
        await websocket.send_json(payload)

    async def send_audio(frame: bytes) -> None:
        await websocket.send_bytes(frame)

    await session.connect(send_json, send_audio)
    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break
            if (data := message.get("bytes")) is not None:
                await session.on_audio(data)
            elif (text := message.get("text")) is not None:
                await session.on_control(text)
    except WebSocketDisconnect:
        pass
    except Exception:
        log.exception("call crashed")
        with contextlib.suppress(Exception):
            await websocket.send_json(error_msg("Internal error — call ended."))
    finally:
        await session.close()
