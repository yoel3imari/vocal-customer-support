"""Call session state machine.

idle -> listening -> thinking -> speaking -> listening ...
                         \\-> (barge-in) -> listening

The mic stays open while the agent speaks; endpointed caller speech cancels the
in-flight LLM/TTS work, flushes the browser playback queue and starts a new turn.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Awaitable, Callable

from src.config import Settings
from src import protocol
from src.pipeline.prompts import build_messages
from src.pipeline.tts import SentenceBuffer
from src.pipeline.vad import FRAME_BYTES, EndpointEvent, EndpointDetector
from src.runtime import Runtime

log = logging.getLogger("voxdesk.session")

SendJson = Callable[[dict], Awaitable[None]]
SendAudio = Callable[[bytes], Awaitable[None]]

OUTPUT_FRAME_BYTES = 4096  # ~93 ms @ 22.05 kHz
MAX_PARTIAL_CHARS = 45


class CallSession:
    def __init__(self, settings: Settings, runtime: Runtime) -> None:
        self.settings = settings
        self.runtime = runtime
        self.call_id = f"call_{uuid.uuid4().hex[:12]}"
        self.state = "idle"
        self.muted = False
        self.frames_in = 0
        self.started_at: float | None = None
        self.history: list[dict] = []
        self.summary: str | None = None
        self.turn_count = 0

        self._send_json: SendJson | None = None
        self._send_audio: SendAudio | None = None
        self._detector = EndpointDetector(
            runtime.vad_factory(),
            threshold=settings.vad_threshold,
            silence_ms=settings.vad_silence_ms,
            min_speech_ms=settings.vad_min_speech_ms,
        )
        self._pcm = bytearray()
        self._queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._worker_task: asyncio.Task | None = None
        self._turn_task: asyncio.Task | None = None
        self._greeting_task: asyncio.Task | None = None
        self._summary_task: asyncio.Task | None = None
        self._tts_lock = asyncio.Lock()
        self._audio_end = 0.0
        self._agent_text = ""
        self._agent_partial_len = 0
        self._turn_sources: list[str] = []
        self._tts_first_reported = False
        self._interrupted = False
        self._started = False
        self._ended = False
        self._record_open = False
        self._closing = False

    # ------------------------------------------------------------------ setup

    async def connect(self, send_json: SendJson, send_audio: SendAudio) -> None:
        self._send_json = send_json
        self._send_audio = send_audio
        await self._json(
            protocol.state(
                "idle",
                call_id=self.call_id,
                tenant=self.settings.tenant.slug,
                tenant_name=self.settings.tenant.name,
                sample_rate=self.runtime.speaker.sample_rate,
                voice=self.settings.tts_voice,
                detail="Press Call to start",
            )
        )

    async def close(self) -> None:
        self._closing = True
        for task in (self._greeting_task, self._summary_task):
            if task and not task.done():
                task.cancel()
        if self._turn_task and not self._turn_task.done():
            self._turn_task.cancel()
        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
        self._finish_call_record()
        log.info(
            "call %s closed (%d frames in, %d turns)",
            self.call_id, self.frames_in, self.turn_count,
        )

    # --------------------------------------------------------------- inbound

    async def on_control(self, raw: str) -> None:
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            await self._json(protocol.error("Malformed control message"))
            return

        kind = msg.get("type")
        if kind == "start":
            await self.start_call()
        elif kind == "end":
            await self.end_call()
        elif kind == "mute":
            self.muted = bool(msg.get("muted"))
            if self.muted:
                self._pcm.clear()
                self._detector.reset()
        elif kind == "interrupt":
            await self._barge_in()
        elif kind == "ping":
            await self._json({"type": "pong", "t": msg.get("t")})
        else:
            await self._json(protocol.error(f"Unknown message type: {kind!r}"))

    async def on_audio(self, data: bytes) -> None:
        """Arbitrary PCM16 mono 16 kHz chunks from the browser, reframed for VAD."""
        if not self._started or self.muted or self._ended:
            return
        self.frames_in += 1
        self._pcm.extend(data)
        while len(self._pcm) >= FRAME_BYTES:
            frame = bytes(self._pcm[:FRAME_BYTES])
            del self._pcm[:FRAME_BYTES]
            for event in self._detector.feed(frame):
                await self._on_vad(event)

    async def _on_vad(self, event: EndpointEvent) -> None:
        if event.kind == "speech_start":
            log.info("speech start (state=%s)", self.state)
            await self._barge_in()
        elif event.kind == "speech_end" and event.audio:
            if not self._started or self._ended:
                return
            self._interrupted = False
            self._queue.put_nowait(event.audio)

    # --------------------------------------------------------------- actions

    async def start_call(self) -> None:
        if self._started:
            return
        self._started = True
        self._record_open = True
        self.started_at = time.time()
        self.frames_in = 0
        self._pcm.clear()
        self._detector.reset()
        self.runtime.store.start_call(
            self.call_id, self.settings.tenant.slug, self.settings.llm_model
        )
        greeting = self.settings.tenant.greeting
        self._tts_first_reported = False
        self.history.append({"role": "assistant", "text": greeting})
        self.runtime.store.add_turn(self.call_id, "assistant", greeting)
        self._worker_task = asyncio.create_task(self._worker())
        await self._set_state("listening", detail="Call in progress")
        await self._json(protocol.transcript("agent", greeting))
        self._greeting_task = asyncio.create_task(self._say_greeting(greeting))
        log.info("call %s started (tenant=%s)", self.call_id, self.settings.tenant.slug)

    async def end_call(self) -> None:
        if self._ended:
            return
        self._ended = True
        self._started = False
        for task in (self._turn_task, self._greeting_task, self._worker_task):
            if task and not task.done():
                task.cancel()
        self._finish_call_record()
        await self._set_state("idle", detail="Call ended")
        await self._json(protocol.call_end(self.call_id))
        log.info("call %s ended (%d turns)", self.call_id, self.turn_count)

    async def _barge_in(self) -> None:
        """Caller started speaking: stop whatever the agent is doing right now."""
        active = None
        for task in (self._greeting_task, self._turn_task):
            if task and not task.done():
                active = task
        if active is None:
            return
        self._interrupted = True
        active.cancel()
        await self._json(protocol.interrupt())
        self._audio_end = 0.0
        await self._set_state("listening", detail="Caller is speaking")

    # ----------------------------------------------------------------- turns

    async def _worker(self) -> None:
        while True:
            pcm = await self._queue.get()
            if not pcm:
                return
            if self._ended:
                return
            self._turn_task = asyncio.create_task(self._run_turn(pcm))
            try:
                await self._turn_task
            except asyncio.CancelledError:
                if self._closing or self._ended:
                    raise
                log.info("turn cancelled by barge-in")
            except Exception:  # noqa: BLE001 - one bad turn must not kill the call
                log.exception("turn failed")
                await self._set_state("listening")
            finally:
                self._turn_task = None

    async def _run_turn(self, pcm: bytes) -> None:
        self._interrupted = False
        self._agent_text = ""
        self._agent_partial_len = 0
        self._turn_sources = []
        self._tts_first_reported = False
        turn_start = time.perf_counter()
        await self._set_state("thinking")

        try:
            t0 = time.perf_counter()
            text = await asyncio.to_thread(self.runtime.transcriber.transcribe, pcm)
            await self._metric("stt", time.perf_counter() - t0)
            if not text:
                await self._set_state("listening", detail="Didn't catch that")
                return

            log.info("caller: %s", text)
            self.turn_count += 1
            await self._json(protocol.transcript("caller", text))
            self.runtime.store.add_turn(self.call_id, "caller", text)
            self.history.append({"role": "user", "text": text})

            previous = self._previous_caller_text(exclude_last=True)
            query = f"{previous} {text}".strip() if previous else text
            t0 = time.perf_counter()
            retrieved = await asyncio.to_thread(self.runtime.knowledge.search, query)
            await self._metric("retrieval", time.perf_counter() - t0)
            self._turn_sources = [r.source for r in retrieved]

            messages = build_messages(
                persona=self.settings.tenant.persona_prompt(),
                history=self.history[:-1],
                summary=self.summary,
                retrieved=retrieved,
                user_text=text,
                max_history=self.settings.max_history_messages,
            )
            await self._stream_answer(messages, turn_start)
            await self._wait_for_playback()
            await self._set_state("listening")
        except asyncio.CancelledError:
            self._interrupted = True
            raise
        except Exception as exc:  # noqa: BLE001
            log.exception("turn error")
            apology = "Sorry, I'm having trouble on my end. Could you say that again?"
            try:
                await self._speak_text(apology)
                self._agent_text = (self._agent_text + " " + apology).strip()
                await self._wait_for_playback()
            except Exception:  # noqa: BLE001
                pass
            await self._json(protocol.error(str(exc)))
            await self._set_state("listening")
        finally:
            if self._agent_text.strip():
                await self._record_agent_turn(self._agent_text, interrupted=self._interrupted)
            await self._metric("turn_total", time.perf_counter() - turn_start)
            if self._interrupted:
                await self._set_state("listening", detail="Go ahead")
            self._schedule_summary()

    async def _stream_answer(self, messages: list[dict], turn_start: float) -> None:
        buf = SentenceBuffer()
        first_token = asyncio.Event()
        filler = asyncio.create_task(self._filler(first_token))
        llm_start = time.perf_counter()
        got_token = False
        try:
            async for token in self.runtime.llm.stream(messages):
                if not got_token:
                    got_token = True
                    first_token.set()
                    await self._metric("llm_ttft", time.perf_counter() - llm_start)
                self._agent_text += token
                if len(self._agent_text) - self._agent_partial_len >= MAX_PARTIAL_CHARS:
                    self._agent_partial_len = len(self._agent_text)
                    await self._json(protocol.transcript("agent", self._agent_text, final=False))
                sentences = buf.feed(token)
                if sentences and len(self._agent_text) > self._agent_partial_len:
                    self._agent_partial_len = len(self._agent_text)
                    await self._json(protocol.transcript("agent", self._agent_text, final=False))
                for sentence in sentences:
                    await self._speak_text(sentence)
            tail = buf.flush()
            if tail:
                await self._speak_text(tail)
            await self._metric("llm_total", time.perf_counter() - llm_start)
            await self._metric("turn_response", time.perf_counter() - turn_start)
        finally:
            first_token.set()
            if not filler.done():
                filler.cancel()

    async def _filler(self, first_token: asyncio.Event) -> None:
        try:
            await asyncio.wait_for(
                first_token.wait(), timeout=self.settings.filler_delay_ms / 1000
            )
            return
        except asyncio.TimeoutError:
            pass
        if not self._interrupted:
            log.info("LLM slow — speaking filler")
            await self._speak_text("Let me check that for you.")

    # ------------------------------------------------------------------ speech

    async def _say_greeting(self, text: str) -> None:
        try:
            await self._speak_text(text)
            await self._wait_for_playback()
            await self._set_state("listening")
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            log.exception("greeting failed")
            await self._set_state("listening")

    async def _speak_text(self, sentence: str) -> None:
        sentence = sentence.strip()
        if not sentence or self._interrupted:
            return
        async with self._tts_lock:
            if self._interrupted:
                return
            if self.state != "speaking":
                await self._set_state("speaking")
            synth_start = time.perf_counter()
            generator = self.runtime.speaker.synthesize(sentence)
            while True:
                chunk = await asyncio.to_thread(next, generator, None)
                if chunk is None or self._interrupted:
                    break
                if not self._tts_first_reported:
                    self._tts_first_reported = True
                    await self._metric("tts_first", time.perf_counter() - synth_start)
                await self._emit_audio(chunk)

    async def _emit_audio(self, pcm: bytes) -> None:
        rate = self.runtime.speaker.sample_rate
        loop = asyncio.get_running_loop()
        self._audio_end = max(loop.time(), self._audio_end) + len(pcm) / (rate * 2)
        for offset in range(0, len(pcm), OUTPUT_FRAME_BYTES):
            if self._interrupted:
                return
            await self._audio(pcm[offset:offset + OUTPUT_FRAME_BYTES])

    async def _wait_for_playback(self) -> None:
        loop = asyncio.get_running_loop()
        delay = self._audio_end - loop.time()
        if delay > 0:
            await asyncio.sleep(delay)

    # --------------------------------------------------------------- memory

    def _previous_caller_text(self, *, exclude_last: bool) -> str:
        turns = self.history[:-1] if exclude_last else self.history
        for turn in reversed(turns):
            if turn["role"] == "user":
                return turn["text"]
        return ""

    async def _record_agent_turn(self, text: str, *, interrupted: bool) -> None:
        display = text.strip()
        stored = display + (" (cut off by caller)" if interrupted else "")
        self.history.append({"role": "assistant", "text": stored})
        self.runtime.store.add_turn(
            self.call_id, "assistant", stored, interrupted=interrupted,
            sources=self._turn_sources,
        )
        await self._json(protocol.transcript("agent", display, interrupted=interrupted))

    def _schedule_summary(self) -> None:
        if len(self.history) < self.settings.summary_trigger_messages:
            return
        if self._summary_task and not self._summary_task.done():
            return
        self._summary_task = asyncio.create_task(self._summarize())

    async def _summarize(self) -> None:
        keep = self.settings.max_history_messages
        older = self.history[:-keep]
        if len(older) < 4:
            return
        transcript = "\n".join(f"{t['role']}: {t['text']}" for t in older)
        messages = [
            {
                "role": "system",
                "content": "Summarize the call so far in at most 60 words. Keep names, "
                           "numbers, decisions and unresolved requests. Plain text, no bullets.",
            },
            {"role": "user", "content": transcript},
        ]
        try:
            t0 = time.perf_counter()
            summary = await self.runtime.llm.complete(messages, max_tokens=120)
            if summary:
                self.summary = (self.summary + " " + summary).strip() if self.summary else summary
                self.history = self.history[-keep:]
                await self._metric("summary", time.perf_counter() - t0)
                log.info("rolling summary updated: %s", summary)
        except Exception as exc:  # noqa: BLE001 - summaries are best-effort
            log.warning("summary failed: %s", exc)

    # ------------------------------------------------------------- plumbing

    async def _metric(self, stage: str, seconds: float) -> None:
        ms = seconds * 1000
        self.runtime.store.add_metric(self.call_id, stage, ms)
        await self._json(protocol.metrics(stage, ms))

    async def _set_state(self, state: str, detail: str | None = None) -> None:
        self.state = state
        await self._json(
            protocol.state(state, call_id=self.call_id, detail=detail)
        )

    async def _json(self, payload: dict) -> None:
        if self._send_json is not None:
            await self._send_json(payload)

    async def _audio(self, frame: bytes) -> None:
        if self._send_audio is not None:
            await self._send_audio(frame)

    def _finish_call_record(self) -> None:
        if not self._record_open or self.started_at is None:
            return
        self._record_open = False
        self.runtime.store.end_call(
            self.call_id,
            duration_s=time.time() - self.started_at,
            summary=self.summary,
        )
