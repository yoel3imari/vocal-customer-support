from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from src.config import load_settings
from src.pipeline.session import CallSession


class FakeVad:
    sample_rate = 16_000

    def __init__(self) -> None:
        self.loud = False

    def reset(self) -> None:
        self.loud = False

    def prob(self, frame) -> float:
        return 1.0 if self.loud else 0.0


class FakeTranscriber:
    text = "How much is the Smart Plus plan?"

    def transcribe(self, pcm: bytes) -> str:
        return self.text


class FakeSpeaker:
    sample_rate = 16_000

    def __init__(self) -> None:
        self.spoken: list[str] = []

    def synthesize(self, text: str):
        self.spoken.append(text)
        yield b"\x00\x00" * 1600  # 100 ms


class FakeLLM:
    def __init__(self, tokens: list[str] | None = None, delay: float = 0.0) -> None:
        self.tokens = tokens or ["It costs twenty nine dollars per month."]
        self.delay = delay
        self.prompts: list[list[dict]] = []

    async def stream(self, messages):
        self.prompts.append(messages)
        for token in self.tokens:
            await asyncio.sleep(self.delay)
            yield token

    async def complete(self, messages, *, max_tokens=220):
        return ""


class FakeKnowledge:
    chunks: list = []

    def search(self, query, **kwargs):
        return []


class FakeStore:
    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.turns: list[tuple] = []
        self.metrics: list[tuple] = []

    def start_call(self, *args):
        self.calls.append(("start", *args))

    def add_turn(self, *args, **kwargs):
        self.turns.append((args, kwargs))

    def add_metric(self, call_id, stage, ms):
        self.metrics.append((stage, ms))

    def end_call(self, *args, **kwargs):
        self.calls.append(("end", *args))


def make_runtime(vad: FakeVad, llm: FakeLLM | None = None) -> tuple[SimpleNamespace, FakeSpeaker]:
    speaker = FakeSpeaker()
    runtime = SimpleNamespace(
        transcriber=FakeTranscriber(),
        speaker=speaker,
        knowledge=FakeKnowledge(),
        llm=llm or FakeLLM(),
        store=FakeStore(),
        vad_factory=lambda: vad,
    )
    return runtime, speaker


async def until(predicate, timeout: float = 3.0, label: str = "condition") -> None:
    for _ in range(int(timeout / 0.01)):
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"timed out waiting for {label}")


async def feed(session: CallSession, frames: int) -> None:
    for _ in range(frames):
        await session.on_audio(b"\x00\x00" * 512)


async def start_and_greet(session: CallSession, events: list[dict]) -> None:
    await session.connect(_json_sender(events), _noop_audio)
    await session.start_call()
    await until(
        lambda: any(e.get("type") == "transcript" and e.get("role") == "agent" for e in events),
        label="greeting transcript",
    )
    await until(lambda: session.state == "listening", label="greeting playback")


def _json_sender(events: list[dict]):
    async def send(payload: dict) -> None:
        events.append(payload)
    return send


async def _noop_audio(frame: bytes) -> None:
    return None


@pytest.mark.asyncio
async def test_full_turn_transcribes_answers_and_returns_to_listening():
    settings = load_settings()
    vad = FakeVad()
    runtime, speaker = make_runtime(vad)
    events: list[dict] = []
    session = CallSession(settings, runtime)
    await start_and_greet(session, events)

    vad.loud = True
    await feed(session, 9)          # onset + enough voiced frames
    vad.loud = False
    await feed(session, 21)         # 650 ms of silence -> endpoint

    await until(
        lambda: any(
            e.get("type") == "transcript" and e.get("role") == "caller"
            and "Smart Plus" in e.get("text", "")
            for e in events
        ),
        label="caller transcript",
    )
    await until(
        lambda: any(
            e.get("type") == "transcript" and e.get("role") == "agent"
            and e.get("final") and "twenty nine" in e.get("text", "")
            for e in events
        ),
        label="agent transcript",
    )
    await until(lambda: session.state == "listening", label="turn completion")

    states = [e["state"] for e in events if e.get("type") == "state"]
    assert "thinking" in states
    assert "speaking" in states
    assert runtime.store.turns, "turns should be persisted"
    assert any("twenty nine" in text for text in speaker.spoken)
    assert runtime.llm.prompts, "LLM should be called"


@pytest.mark.asyncio
async def test_barge_in_cancels_agent_and_marks_transcript_interrupted():
    settings = load_settings()
    vad = FakeVad()
    llm = FakeLLM(
        tokens=["This is a deliberately long answer. ", "It keeps going. ", "And going."],
        delay=0.12,
    )
    runtime, _speaker = make_runtime(vad, llm)
    events: list[dict] = []
    session = CallSession(settings, runtime)
    await start_and_greet(session, events)

    vad.loud = True
    await feed(session, 9)
    vad.loud = False
    await feed(session, 21)

    await until(lambda: session.state == "speaking", label="agent speaking")
    vad.loud = True
    await feed(session, 3)  # caller barges in

    await until(
        lambda: any(e.get("type") == "interrupt" for e in events),
        label="interrupt message",
    )
    await until(
        lambda: any(
            e.get("type") == "transcript" and e.get("role") == "agent"
            and e.get("interrupted")
            for e in events
        ),
        label="interrupted transcript",
    )
    await until(lambda: session.state == "listening", label="listening after barge-in")
    assert any(e.get("type") == "interrupt" for e in events)
    await session.close()
