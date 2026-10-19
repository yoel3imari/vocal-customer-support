"""Local speech-to-text with faster-whisper (CPU, int8)."""

from __future__ import annotations

import logging
import re
import threading

import numpy as np

from src.config import INPUT_SAMPLE_RATE, Settings

log = logging.getLogger("voxdesk.stt")

_ARTIFACTS = re.compile(r"^\s*(\[.*?\]|\(.*?\)|music|applause|silence)\s*$", re.IGNORECASE)


class Transcriber:
    def __init__(self, settings: Settings) -> None:
        from faster_whisper import WhisperModel

        self.settings = settings
        self._lock = threading.Lock()
        log.info(
            "loading Whisper %s (%s/%s, %d threads)",
            settings.whisper_model, settings.whisper_device,
            settings.whisper_compute_type, settings.whisper_threads,
        )
        self.model = WhisperModel(
            settings.whisper_model,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
            cpu_threads=settings.whisper_threads,
        )

    def warmup(self) -> None:
        # Load weights + run one silent pass so the first real turn isn't slow.
        self.transcribe(b"\x00\x00" * (INPUT_SAMPLE_RATE // 2))

    def transcribe(self, pcm16: bytes) -> str:
        if len(pcm16) < INPUT_SAMPLE_RATE // 2:  # < 0.25 s
            return ""
        audio = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
        with self._lock:
            segments, _info = self.model.transcribe(
                audio,
                language="en",
                beam_size=1,
                best_of=1,
                vad_filter=False,
                condition_on_previous_text=False,
                without_timestamps=True,
                no_speech_threshold=0.6,
            )
            texts = []
            for seg in segments:
                text = seg.text.strip()
                if not text or _ARTIFACTS.match(text):
                    continue
                texts.append(text)
        result = " ".join(texts).strip()
        # Whisper occasionally echoes these stock lines on silence/noise.
        if result.lower() in {"thank you.", "thanks for watching!", "you"}:
            return ""
        return result
