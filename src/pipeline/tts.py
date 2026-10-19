"""Local text-to-speech with Piper (CPU), streamed per sentence."""

from __future__ import annotations

import logging
import threading
import urllib.request
import wave
import io
from pathlib import Path
from typing import Iterator

from src.config import Settings

log = logging.getLogger("voxdesk.tts")

HF_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main"
MIN_MODEL_BYTES = 5_000_000


def _supports_synthesis_config() -> bool:
    try:
        from piper import SynthesisConfig  # noqa: F401
    except ImportError:
        return False
    return True


def voice_download_urls(voice: str) -> tuple[str, str]:
    """en_US-lessac-medium -> (model url, config url)."""
    parts = voice.split("-")
    if len(parts) != 3:
        raise ValueError(
            f"TTS_VOICE must look like 'en_US-lessac-medium', got {voice!r}"
        )
    locale, name, quality = parts
    lang = locale.split("_")[0]
    base = f"{HF_BASE}/{lang}/{locale}/{name}/{quality}/{voice}"
    return f"{base}.onnx", f"{base}.onnx.json"


class Speaker:
    """Piper wrapper. `synthesize()` yields raw PCM16 bytes at `sample_rate`."""

    def __init__(self, settings: Settings) -> None:
        from piper import PiperVoice

        self.settings = settings
        self._lock = threading.Lock()
        model_path = settings.voice_path()
        config_path = model_path.with_suffix(".onnx.json")
        if not model_path.is_file() or model_path.stat().st_size < MIN_MODEL_BYTES:
            self._download(model_path, config_path)
        log.info("loading Piper voice %s", settings.tts_voice)
        self.voice = PiperVoice.load(str(model_path), config_path=str(config_path) if config_path.is_file() else None)
        self.sample_rate = int(self._voice_sample_rate())
        self._raw_stream = hasattr(self.voice, "synthesize_stream_raw")
        self._has_synth_config = _supports_synthesis_config()
        log.info("TTS ready: %s @ %d Hz", settings.tts_voice, self.sample_rate)

    # -- setup -------------------------------------------------------------

    def _download(self, model_path: Path, config_path: Path) -> None:
        model_url, config_url = voice_download_urls(self.settings.tts_voice)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        for url, dest, minimum in ((model_url, model_path, MIN_MODEL_BYTES),
                                   (config_url, config_path, 100)):
            if dest.is_file() and dest.stat().st_size >= minimum:
                continue
            log.info("downloading Piper voice file → %s", dest.name)
            try:
                with urllib.request.urlopen(url, timeout=120) as resp, open(dest, "wb") as fh:
                    while chunk := resp.read(1 << 16):
                        fh.write(chunk)
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    f"Could not download Piper voice {self.settings.tts_voice}: {exc}. "
                    "Connect to the internet once, or place the .onnx and .onnx.json "
                    f"files in {model_path.parent} manually."
                ) from exc
            if dest.stat().st_size < minimum:
                raise RuntimeError(f"Downloaded file looks truncated: {dest}")

    def _voice_sample_rate(self) -> int:
        config = getattr(self.voice, "config", None)
        if config is not None and getattr(config, "sample_rate", None):
            return int(config.sample_rate)
        return 22050

    def warmup(self) -> None:
        try:
            next(self.synthesize("Hello."), None)
        except Exception as exc:  # noqa: BLE001 - warmup is best-effort
            log.warning("TTS warmup failed: %s", exc)

    # -- synthesis ---------------------------------------------------------

    def _locked(self, generator):
        """Pull one chunk at a time under the Piper lock; never hold it while sending."""
        while True:
            with self._lock:
                try:
                    chunk = next(generator)
                except StopIteration:
                    return
            yield chunk

    def synthesize(self, text: str) -> Iterator[bytes]:
        """Yield raw PCM16 little-endian chunks (no WAV headers) for `text`."""
        text = text.strip()
        if not text:
            return
        length_scale = 1.0 / self.settings.tts_speed

        if self._raw_stream:
            yield from self._locked(
                self.voice.synthesize_stream_raw(text, length_scale=length_scale)
            )
            return

        if self._has_synth_config:
            from piper import SynthesisConfig

            syn_config = SynthesisConfig(length_scale=length_scale)
            for chunk in self._locked(self.voice.synthesize(text, syn_config=syn_config)):
                pcm = getattr(chunk, "audio_int16_bytes", None)
                if pcm:
                    yield pcm
            return

        # Last resort: synthesize into an in-memory WAV, then strip the header.
        with self._lock:
            buffer = io.BytesIO()
            with wave.open(buffer, "wb") as wav_file:
                self.voice.synthesize(text, wav_file)
        buffer.seek(0)
        with wave.open(buffer, "rb") as wav_file:
            yield wav_file.readframes(wav_file.getnframes())


def split_sentences() -> "SentenceBuffer":
    return SentenceBuffer()


class SentenceBuffer:
    """Incrementally turns an LLM token stream into speakable sentences."""

    def __init__(self, min_chars: int = 2, max_chars: int = 240) -> None:
        self.buffer = ""
        self.min_chars = min_chars
        self.max_chars = max_chars

    def feed(self, token: str) -> list[str]:
        self.buffer += token
        out: list[str] = []
        while True:
            cut = self._find_boundary()
            if cut is None:
                break
            sentence = self.buffer[:cut].strip()
            self.buffer = self.buffer[cut:].lstrip()
            if sentence:
                out.append(sentence)
        while len(self.buffer) > self.max_chars:
            window = self.buffer[: self.max_chars]
            cut = max(window.rfind(", "), window.rfind(" "))
            if cut < self.max_chars // 2:
                cut = self.max_chars
            sentence = self.buffer[:cut].strip()
            self.buffer = self.buffer[cut:].lstrip()
            if sentence:
                out.append(sentence)
        return [s for s in out if s]

    def _find_boundary(self) -> int | None:
        # A sentence ends at . ! ? followed by whitespace (avoids $19.99, e.g.)
        for i, ch in enumerate(self.buffer):
            if ch in ".!?" and i + 1 < len(self.buffer):
                nxt = self.buffer[i + 1]
                if nxt.isspace() and i + 1 >= self.min_chars:
                    # don't split common abbreviations
                    word = self.buffer[: i + 1].rsplit(" ", 1)[-1].lower()
                    if word in {"mr.", "mrs.", "ms.", "dr.", "st.", "e.g.", "i.e.", "vs."}:
                        continue
                    return i + 1
            elif ch in ".!?" and i + 1 == len(self.buffer):
                # token ended exactly on punctuation: wait one more token for whitespace
                continue
            elif ch == "\n":
                return i + 1
        return None

    def flush(self) -> str | None:
        text = self.buffer.strip()
        self.buffer = ""
        return text or None
