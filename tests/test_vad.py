from __future__ import annotations

import numpy as np

from src.pipeline.vad import FRAME_BYTES, EndpointDetector


class FakeModel:
    sample_rate = 16_000

    def __init__(self, pattern: list[float]) -> None:
        self.pattern = pattern
        self.i = 0

    def reset(self) -> None:
        self.i = 0

    def prob(self, frame: np.ndarray) -> float:
        value = self.pattern[min(self.i, len(self.pattern) - 1)]
        self.i += 1
        return value


def frame() -> bytes:
    return b"\x00\x00" * 512


def test_onset_and_endpoint():
    # 3 onset + 6 voiced (>=250 ms) then 22 silent frames (>=650 ms)
    pattern = [0.9] * 9 + [0.0] * 30
    detector = EndpointDetector(FakeModel(pattern), onset_frames=3, silence_ms=650, min_speech_ms=250)
    events = []
    for _ in range(len(pattern)):
        events.extend(detector.feed(frame()))
    kinds = [e.kind for e in events]
    assert kinds == ["speech_start", "speech_end"]
    audio = events[1].audio
    assert audio is not None
    assert len(audio) % FRAME_BYTES == 0
    assert len(audio) >= 9 * FRAME_BYTES


def test_cough_is_dropped():
    # 3 voiced frames start the detector, then silence before min speech
    pattern = [0.9] * 3 + [0.0] * 30
    detector = EndpointDetector(FakeModel(pattern), onset_frames=3, silence_ms=650, min_speech_ms=250)
    events = []
    for _ in range(len(pattern)):
        events.extend(detector.feed(frame()))
    assert [e.kind for e in events] == ["speech_start"]  # no bogus turn


def test_short_noise_does_not_start():
    pattern = [0.9, 0.0, 0.9, 0.0, 0.0]
    detector = EndpointDetector(FakeModel(pattern), onset_frames=3)
    events = []
    for _ in range(len(pattern)):
        events.extend(detector.feed(frame()))
    assert events == []


def test_max_utterance_forces_endpoint():
    pattern = [0.9] * 200
    detector = EndpointDetector(FakeModel(pattern), onset_frames=3, max_utterance_ms=640)
    events = []
    for _ in range(len(pattern)):
        events.extend(detector.feed(frame()))
    assert [e.kind for e in events][:2] == ["speech_start", "speech_end"]
