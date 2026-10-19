# AGENT.md

VoxDesk is a browser-based AI call-center agent. One WebSocket per call streams mic audio
(16 kHz PCM16) through endpointing → local Whisper STT → docs RAG → OpenRouter LLM →
local Piper TTS. A tenant is just a folder under `data/` (`brand.txt` + `docs/`).

## Commands

```bash
make setup                 # Python 3.12 via uv + dependencies
make test                  # 35 unit tests, no models/network/API key needed
make run                   # server on 127.0.0.1:8000
make dev                   # same, auto-reload
make smoke                 # transport smoke test (server must be running)
make call TEXT="..."       # simulated end-to-end call
make fake-llm              # offline OpenAI-compatible LLM on :8123
make reset-db              # delete data/calls.db
make clean                 # remove caches
```

Use `uv run ...` for direct Python commands. Python 3.12 only. Run `uv sync` after changing
dependencies; commit the updated `uv.lock`. `make run TENANT=shopverse PORT=9000` overrides
defaults. See `README.md` for full configuration and wire protocol details.

## Layout

- `src/main.py` — FastAPI app, `/ws` WebSocket, startup/lifespan.
- `src/config.py` — env-based `Settings` and tenant loading. Add settings here, not ad hoc.
- `src/runtime.py` — process-wide models/services, built once at startup in a thread.
- `src/pipeline/session.py` — per-call state machine, turn loop, barge-in, metrics.
- `src/pipeline/{llm,prompts,stt,tts,vad}.py` — components, each independently testable.
- `src/rag/{ingest,retriever}.py` — heading-aware chunking and in-memory cosine index.
- `src/store.py` — SQLite calls/turns/metrics. `src/protocol.py` — wire messages.
- `web/` — vanilla JS call console (`app.js`, `pcm-worklet.js`, `index.html`, `styles.css`).
- `data/<tenant>/` — `brand.txt` persona + `docs/` (`.md`, `.txt`, `.pdf`).
- `tests/` — fakes/`SimpleNamespace` runtimes; `tools/` — smoke, simulated calls, fake LLM.

## Conventions and invariants

- Python 3.12, `from __future__ import annotations`, type hints, dataclasses, stdlib
  `logging`. No linter/formatter is configured; match the surrounding style (4 spaces).
- Keep pipeline/RAG modules decoupled from FastAPI and sessions. All configuration comes
  from `src/config.py`; document new env vars in `.env.example` and the README table.
- Never block the event loop: run STT, TTS, embedding, and retrieval work via
  `asyncio.to_thread(...)`.
- Binary frames are raw PCM16 little-endian mono with no WAV header. Input is always
  16 kHz; output uses Piper's rate announced once in the `state` message (`sampleRate`).
- Protocol changes require updating `src/protocol.py`, `web/app.js`, the README wire
  table, and tests together.
- Barge-in is a core behavior: caller speech cancels in-flight turn/greeting tasks, sends
  `interrupt`, flushes playback, and records the agent turn as interrupted. Preserve the
  cancellation paths and state transitions when editing the session.
- Grounding is strict: answer only from retrieved excerpts; if retrieval is empty or
  irrelevant, say so and offer a specialist. Do not loosen prompt or retrieval rules
  without tests.
- Tenant docs are re-ingested in memory on every startup — no persistent index. Audio is
  never persisted; `data/calls.db` is the only state.
- Tests must not download models, need network/API keys, or load real Whisper/Piper/
  fastembed. Use fake components like those in `tests/test_session.py`.

## Verification

- Run `make test` for every change.
- For transport/protocol edits, start the server and run `make smoke`.
- For end-to-end behavior, use `make fake-llm` plus `make call TEXT="..."` so no API key
  is required.
