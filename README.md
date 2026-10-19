# 📞 VoxDesk — Portable AI Call-Center Voice Agent

Talk to a company's support line in your browser — a real streaming voice loop, grounded in
that company's own documentation. Point `AGENT_TENANT` at a different folder and the same
codebase becomes a different company's support agent.

<!-- architecture diagram -->
```text
Browser "call console"                          FastAPI (localhost:8000)
┌──────────────────────────┐   WebSocket    ┌───────────────────────────────────────┐
│ mic ── AudioWorklet ─────┼── PCM16 16 kHz │ endpointing ── Whisper (local CPU)    │
│  (512-sample frames)     │     binary     │       │                               │
│                          │                │       ▼                               │
│ playback queue ◄─────────┼── PCM16 (Piper │  docs RAG (fastembed, top-3 chunks)   │
│  (barge-in flush)        │    22.05 kHz)  │       │                               │
│                          │◄── JSON state, │       ▼                               │
│ live transcript          │    transcript, │  OpenRouter LLM (streamed tokens)     │
│                          │    metrics     │       │                               │
└──────────────────────────┘                │       ▼                               │
                                            │  Piper TTS (sentence by sentence)     │
                                            │       │                               │
                                            │  SQLite: calls / turns / latency      │
                                            └───────────────────────────────────────┘
```

## Why it feels like a phone call

- **Streaming everywhere** — STT at end-of-speech, LLM tokens streamed, Piper speaks the
  first sentence while the rest is still generating.
- **Barge-in** — the mic stays open during playback; caller speech cancels the LLM/TTS
  work, flushes the browser audio queue, and the agent stops mid-sentence.
- **End-of-speech endpointing** — Silero VAD + 650 ms of silence, with 320 ms of pre-roll
  so the first word is never clipped.
- **Grounded answers only** — top-3 doc chunks are injected into the prompt; if retrieval
  finds nothing, the agent says so and offers a specialist instead of hallucinating.
- **Latency mask** — if the LLM hasn't produced a token in 800 ms, it says
  “Let me check that for you…” while it waits.

### LLM layer

Chat transport is **LangChain v1** (`langchain-openai`'s `ChatOpenAI`) pointed at
OpenRouter's OpenAI-compatible API. `src/pipeline/llm.py` wraps it in a small
`stream()` / `complete()` interface: a one-shot fallback model, fail-fast retries
(no hidden stalls — a voice turn would rather switch models), and plain-dict messages
that are converted to LangChain messages at the boundary. Change `LLM_MODEL`,
`LLM_FALLBACK_MODEL`, or `OPENROUTER_BASE_URL` (any OpenAI-compatible endpoint, e.g.
a local Ollama server) without touching the pipeline.

## Quick start

```bash
# 1. Python 3.12 env + dependencies (uv manages the interpreter)
make setup            # or: uv python install 3.12 && uv sync

# 2. Configure the LLM key
cp .env.example .env
# edit .env → OPENROUTER_API_KEY=sk-or-v1-...

# 3. Run
make run              # or: uv run uvicorn src.main:app --host 127.0.0.1 --port 8000

# 4. Open the call console
#    http://localhost:8000  (microphone requires localhost or HTTPS)
```

`make help` lists everything:

| Target | What it does |
|---|---|
| `make setup` | install Python 3.12 + sync dependencies |
| `make run` / `make dev` | start the server (dev = auto-reload) |
| `make test` | unit test suite (30 tests, no models needed) |
| `make smoke` | transport smoke test against a running server |
| `make call TEXT="…"` | simulated end-to-end call (espeak-ng + ffmpeg) |
| `make fake-llm` | offline fake OpenAI-compatible LLM on :8123 (no API key needed) |
| `make stop` | stop a running server |
| `make reset-db` | delete call history |
| `make clean` | remove caches |

Override defaults per command: `make run TENANT=shopverse PORT=9000 WHISPER=tiny.en`.

First run downloads local models once (internet required): Piper voice (~60 MB), Silero VAD
(~2 MB), fastembed ONNX model (~130 MB). Whisper weights come from Hugging Face on first use.
Model warm-up + doc indexing takes a few seconds at boot.

Use **headphones** — otherwise speaker audio can leak into the mic and trigger barge-in.

## Switch companies

A tenant is just a folder:

```text
data/
├── acme-telecom/            ← NovaCell, a mobile carrier
│   ├── brand.txt            name, greeting, persona, rules, escalation paths
│   └── docs/                plans.md, billing.md, outages.md, support.md
└── shopverse/               ← ShopVerse, an online store
    ├── brand.txt
    └── docs/                shipping.md, returns.md, orders.md
```

```bash
AGENT_TENANT=shopverse uv run uvicorn src.main:app
```

`brand.txt` supports optional header lines:

```text
name: NovaCell Support
greeting: Thanks for calling NovaCell. How can I help you today?
```

Everything after the headers is the system persona. Docs can be `.md`, `.txt`, or `.pdf`;
the index is rebuilt in memory at startup (small corpora, no admin UI, no stale cache).

## Wire protocol

One WebSocket per call at `/ws`. Text frames are JSON control messages; binary frames are
raw PCM16 little-endian mono audio with no WAV headers.

| Direction | Message | Meaning |
|---|---|---|
| → server | `{"type":"start"}` | begin call; agent speaks the greeting |
| → server | `{"type":"end"}` | hang up |
| → server | `{"type":"mute","muted":true}` | mic gating |
| → server | `{"type":"interrupt"}` | client-side barge-in (flush playback) |
| → server | binary | mic audio, 16 kHz PCM16, 512-sample frames |
| ← client | `{"type":"state","state":"listening…"}` | idle / listening / thinking / speaking |
| ← client | `{"type":"transcript","role":"caller\|agent","final":…}` | live transcript |
| ← client | `{"type":"metrics","stage":"stt","ms":412}` | per-stage latency |
| ← client | `{"type":"interrupt"}` | flush queued playback now |
| ← client | binary | agent speech at `sampleRate` announced in `state` |

The sample rate is announced once in the `state` message (Piper's rate, usually 22050 Hz)
so the browser never has to guess.

## Configuration

All settings are env vars; see `.env.example`. The useful ones:

| Variable | Default | Notes |
|---|---|---|
| `AGENT_TENANT` | `acme-telecom` | folder under `data/` |
| `OPENROUTER_API_KEY` | — | required for answers |
| `LLM_MODEL` | `cohere/north-mini-code:free` | any OpenRouter model |
| `LLM_FALLBACK_MODEL` | `cohere/command-r7b-12-2024` | used if the free model fails before output |
| `WHISPER_MODEL` | `base.en` | `tiny.en` is faster, less accurate |
| `VAD_BACKEND` | `silero` | `energy` fallback, no download |
| `TTS_VOICE` | `en_US-lessac-medium` | any Piper `en_US-*-*` voice |
| `RAG_TOP_K` | `3` | doc chunks per turn |
| `RAG_MIN_SCORE` | `0.6` | cosine cutoff (tuned: relevant ~0.67–0.85, noise <0.65) |

Point `OPENROUTER_BASE_URL` at a local Ollama/OpenAI-compatible server for private
deployments — transcripts and audio then never leave the machine.

## Performance on this class of hardware

Measured target on an i5-8257U, 4 cores, no GPU, 7.6 GB RAM:

| Stage | Typical |
|---|---|
| Endpoint silence wait | 650 ms (by design) |
| Whisper `base.en` int8 | 0.4–0.9 s |
| RAG query | 0.03 s |
| OpenRouter time-to-first-token | 0.4–1.2 s (free tier queues) |
| Piper first sentence | 0.15–0.4 s |
| **Median full turn** | **~2.8 s** |

One live call uses ~1.2–1.8 GB RAM and bursts 4 CPU cores during transcription. Realistic
concurrency is 1–2 calls; more calls queue Whisper transcriptions.

## Data & privacy

- `data/calls.db` stores call metadata, transcripts, per-stage latency, and rolling
  summaries. Delete the file for a clean slate.
- Audio is never persisted — only streamed.
- Local STT/TTS/RAG; only the prompt (persona + retrieved excerpts + recent turns) goes to
  OpenRouter. Use a local OpenAI-compatible endpoint to keep everything on-device.
- No cross-call memory: every call starts clean, like a real call center.

## Development

```bash
make test                                   # chunking, VAD, prompts, LLM parsing, session flow
make smoke                                  # transport smoke test against a running server
make call TEXT="How much is the Smart Plus plan?"
```

`tools/simulate_call.py` (behind `make call`) synthesizes a caller question with
`espeak-ng`, streams it into the WebSocket in real time, and prints the transcript/metrics
the way a browser would.

No OpenRouter key handy? Run the full loop against the bundled fake endpoint — the
canned answer still travels through the real LangChain streaming path:

```bash
make fake-llm &                                    # terminal 1: fake LLM on :8123
OPENROUTER_API_KEY=fake \
OPENROUTER_BASE_URL=http://127.0.0.1:8123/api/v1 \
    make run                                       # terminal 2: agent against the fake
make call TEXT="How much is the Smart Plus plan?"  # terminal 3
```

## Roadmap

- Milestones 1–6 are implemented: audio transport, endpointing + STT, streaming LLM,
  Piper TTS with barge-in, docs RAG, and call records/latency logging.
- **Optional next:** Twilio media streams (mulaw 8 kHz) need a public URL and a Twilio
  number; the pipeline already sits behind the WebSocket transport.

## License

MIT — use it, fork it, give it your own company's docs.
