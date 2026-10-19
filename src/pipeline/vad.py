"""Voice activity detection + endpointing.

Silero VAD v5 runs directly on onnxruntime (no torch dependency) in 32 ms frames.
If the model can't be downloaded, an adaptive energy detector keeps the demo alive.
"""

from __future__ import annotations

import logging
import threading
import urllib.request
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from src.config import INPUT_SAMPLE_RATE, Settings

log = logging.getLogger("voxdesk.vad")

FRAME_SAMPLES = 512  # 32 ms @ 16 kHz — Silero v5 chunk size
FRAME_BYTES = FRAME_SAMPLES * 2
FRAME_MS = 32

SILERO_URL = (
    "https://raw.githubusercontent.com/snakers4/silero-vad/master/"
    "src/silero_vad/data/silero_vad.onnx"
)


class ProbModel(Protocol):
    sample_rate: int

    def reset(self) -> None: ...

    def prob(self, frame: np.ndarray) -> float: ...


class SileroVAD:
    """Minimal ONNX wrapper for Silero VAD v5 (16 kHz, 512-sample frames)."""

    sample_rate = INPUT_SAMPLE_RATE
    context_size = 64  # samples of left context expected by the 16 kHz model

    def __init__(self, model_path: Path) -> None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(model_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        input_names = {i.name for i in self.session.get_inputs()}
        output_names = [o.name for o in self.session.get_outputs()]
        if not {"input", "state", "sr"} <= input_names:
            raise RuntimeError(f"Unexpected Silero VAD inputs: {sorted(input_names)}")
        self._output_state = "stateN" if "stateN" in output_names else output_names[-1]
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, self.context_size), dtype=np.float32)

    def prob(self, frame: np.ndarray) -> float:
        x = frame.astype(np.float32).reshape(1, -1) / 32768.0
        with self._lock:
            inp = np.concatenate([self._context, x], axis=1)
            out, self._state = self.session.run(
                None,
                {
                    "input": inp,
                    "state": self._state,
                    "sr": np.array(self.sample_rate, dtype=np.int64),
                },
            )
            self._context = inp[:, -self.context_size:]
        return float(np.asarray(out).reshape(-1)[0])


class EnergyVAD:
    """Adaptive RMS fallback — decent in quiet rooms, no downloads needed."""

    sample_rate = INPUT_SAMPLE_RATE

    def __init__(self, threshold: float = 0.012) -> None:
        self.threshold = threshold
        self.noise = 0.005
        self.reset()

    def reset(self) -> None:
        self.noise = 0.005

    def prob(self, frame: np.ndarray) -> float:
        x = frame.astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(x * x) + 1e-9))
        speaking = rms > max(self.threshold, self.noise * 3.0)
        if not speaking:
            self.noise = 0.99 * self.noise + 0.01 * rms
        return 1.0 if speaking else 0.0


def load_vad_model(settings: Settings) -> ProbModel:
    """Build the configured VAD, downloading Silero once if necessary."""
    if settings.vad_backend == "energy":
        log.info("VAD: energy-based (configured)")
        return EnergyVAD()

    model_path = Path(settings.vad_model_path)
    try:
        if not model_path.is_file() or model_path.stat().st_size < 100_000:
            model_path.parent.mkdir(parents=True, exist_ok=True)
            log.info("downloading Silero VAD model → %s", model_path)
            with urllib.request.urlopen(SILERO_URL, timeout=30) as resp:
                data = resp.read()
            if len(data) < 100_000:
                raise RuntimeError("downloaded Silero model looks truncated")
            model_path.write_bytes(data)
        model = SileroVAD(model_path)
        log.info("VAD: Silero ONNX (%s)", model_path.name)
        return model
    except Exception as exc:  # noqa: BLE001 - fallback is intentional
        log.warning("Silero VAD unavailable (%s); falling back to energy VAD", exc)
        return EnergyVAD()


@dataclass
class EndpointEvent:
    kind: str  # "speech_start" | "speech_end"
    audio: bytes | None = None


class EndpointDetector:
    """Turns a stream of probabilities into utterances with ~650 ms end-of-speech."""

    def __init__(
        self,
        model: ProbModel,
        *,
        threshold: float = 0.5,
        silence_ms: int = 650,
        min_speech_ms: int = 250,
        max_utterance_ms: int = 20_000,
        onset_frames: int = 3,
        preroll_ms: int = 320,
        on_event=None,
    ) -> None:
        self.model = model
        self.threshold = threshold
        self.silence_frames_needed = max(1, silence_ms // FRAME_MS)
        self.min_speech_frames = max(1, min_speech_ms // FRAME_MS)
        self.max_frames = max(1, max_utterance_ms // FRAME_MS)
        self.onset_frames_needed = max(1, onset_frames)
        self.on_event = on_event
        self.preroll: deque[bytes] = deque(maxlen=max(1, preroll_ms // FRAME_MS))
        self.reset()

    def reset(self) -> None:
        self.model.reset()
        self.speaking = False
        self._onset = 0
        self._silence = 0
        self._voiced = 0
        self._frames = 0
        self._utterance = bytearray()
        self.preroll.clear()

    def feed(self, frame: bytes) -> list[EndpointEvent]:
        """Feed exactly 1024 bytes (512 PCM16 samples). Extra bytes are ignored by caller."""
        if len(frame) != FRAME_BYTES:
            raise ValueError(f"VAD frame must be {FRAME_BYTES} bytes, got {len(frame)}")
        events: list[EndpointEvent] = []
        samples = np.frombuffer(frame, dtype=np.int16)
        p = self.model.prob(samples)

        if not self.speaking:
            self.preroll.append(frame)
            if p >= self.threshold:
                self._onset += 1
                if self._onset >= self.onset_frames_needed:
                    self.speaking = True
                    self._onset = 0
                    self._silence = 0
                    self._voiced = self.onset_frames_needed
                    self._frames = 0
                    self._utterance = bytearray(b"".join(self.preroll))
                    self._frames = len(self._utterance) // FRAME_BYTES
                    events.append(EndpointEvent("speech_start"))
            else:
                self._onset = 0
            return events

        self._utterance.extend(frame)
        self._frames += 1
        if p >= self.threshold:
            self._voiced += 1
            self._silence = 0
        else:
            self._silence += 1

        if self._frames > self.max_frames:
            log.warning("utterance hit max length (%d ms) — forcing endpoint", self.max_frames * FRAME_MS)
            events.append(self._finish())
        elif self._silence >= self.silence_frames_needed and self._voiced >= self.min_speech_frames:
            events.append(self._finish())
        elif self._silence >= self.silence_frames_needed and self._voiced < self.min_speech_frames:
            # cough / click: drop it, don't produce a turn
            self.speaking = False
            self._silence = 0
            self._voiced = 0
            self._frames = 0
            self._utterance.clear()
            self.preroll.clear()
        return events

    def _finish(self) -> EndpointEvent:
        audio = bytes(self._utterance)
        log.info("endpoint: %.2fs utterance", len(audio) / (INPUT_SAMPLE_RATE * 2))
        self.speaking = False
        self._silence = 0
        self._voiced = 0
        self._frames = 0
        self._utterance.clear()
        self.preroll.clear()
        return EndpointEvent("speech_end", audio)
