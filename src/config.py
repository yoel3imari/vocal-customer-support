"""Configuration: environment variables + per-tenant folder loading."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

INPUT_SAMPLE_RATE = 16_000  # mic / STT / VAD


def _env(name: str, default: str | None = None) -> str:
    value = os.getenv(name, default)
    if value is None:
        raise RuntimeError(f"Missing required env var {name} (see .env.example)")
    return value


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw else default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw else default


@dataclass(frozen=True)
class Tenant:
    """One company: persona/rules + a docs folder. The entire 'portability' trick."""

    slug: str
    root: Path
    brand: str  # raw brand.txt contents
    name: str  # display name extracted from brand.txt (`name:` line) or slug
    greeting: str  # optional `greeting:` line, else a sensible default

    @property
    def docs_dir(self) -> Path:
        return self.root / "docs"

    def persona_prompt(self) -> str:
        """brand.txt minus `name:` / `greeting:` header lines, used as the system persona."""
        lines = [
            ln for ln in self.brand.splitlines()
            if not ln.lower().startswith(("name:", "greeting:"))
        ]
        return "\n".join(lines).strip()


@dataclass(frozen=True)
class Settings:
    tenant: Tenant
    data_dir: Path
    db_path: Path
    web_dir: Path

    # LLM
    llm_model: str
    llm_fallback_model: str
    openrouter_api_key: str
    openrouter_base_url: str
    llm_max_tokens: int
    llm_temperature: float

    # STT
    whisper_model: str
    whisper_device: str
    whisper_compute_type: str
    whisper_threads: int

    # VAD
    vad_backend: str  # "silero" | "energy"
    vad_silence_ms: int
    vad_min_speech_ms: int
    vad_threshold: float
    vad_model_path: Path

    # TTS
    tts_voice: str
    tts_speed: float
    voices_dir: Path
    filler_delay_ms: int

    # RAG
    embedding_model: str
    rag_top_k: int
    rag_min_score: float

    # Session policy
    max_history_messages: int
    summary_trigger_messages: int

    def voice_path(self) -> Path:
        return self.voices_dir / f"{self.tts_voice}.onnx"


def _load_tenant(data_dir: Path, slug: str) -> Tenant:
    root = data_dir / slug
    if not root.is_dir():
        raise RuntimeError(
            f"Unknown tenant '{slug}': expected folder {root}. "
            f"Available: {', '.join(sorted(p.name for p in data_dir.iterdir() if p.is_dir())) or 'none'}"
        )
    brand_file = root / "brand.txt"
    if not brand_file.is_file():
        raise RuntimeError(f"Tenant '{slug}' is missing {brand_file}")
    brand = brand_file.read_text(encoding="utf-8")
    name = slug
    greeting = ""
    for line in brand.splitlines():
        lowered = line.lower()
        if lowered.startswith("name:"):
            name = line.split(":", 1)[1].strip()
        elif lowered.startswith("greeting:"):
            greeting = line.split(":", 1)[1].strip()
    greeting = greeting or f"Thanks for calling {name}. How can I help you today?"
    return Tenant(slug=slug, root=root, brand=brand, name=name, greeting=greeting)


def load_settings() -> Settings:
    data_dir = Path(os.getenv("DATA_DIR", ROOT / "data")).expanduser().resolve()
    slug = _env("AGENT_TENANT", "acme-telecom")
    return Settings(
        tenant=_load_tenant(data_dir, slug),
        data_dir=data_dir,
        db_path=Path(os.getenv("DB_PATH", data_dir / "calls.db")),
        web_dir=ROOT / "web",
        llm_model=_env("LLM_MODEL", "cohere/north-mini-code:free"),
        llm_fallback_model=_env("LLM_FALLBACK_MODEL", "cohere/command-r7b-12-2024"),
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),
        openrouter_base_url=_env("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        llm_max_tokens=_env_int("LLM_MAX_TOKENS", 160),
        llm_temperature=_env_float("LLM_TEMPERATURE", 0.4),
        whisper_model=_env("WHISPER_MODEL", "base.en"),
        whisper_device=_env("WHISPER_DEVICE", "cpu"),
        whisper_compute_type=_env("WHISPER_COMPUTE_TYPE", "int8"),
        whisper_threads=_env_int("WHISPER_THREADS", 4),
        vad_backend=_env("VAD_BACKEND", "silero"),
        vad_silence_ms=_env_int("VAD_SILENCE_MS", 650),
        vad_min_speech_ms=_env_int("VAD_MIN_SPEECH_MS", 250),
        vad_threshold=_env_float("VAD_THRESHOLD", 0.5),
        vad_model_path=Path(
            os.getenv("VAD_MODEL_PATH", ROOT / "models" / "silero_vad.onnx")
        ).expanduser().resolve(),
        tts_voice=_env("TTS_VOICE", "en_US-lessac-medium"),
        tts_speed=_env_float("TTS_SPEED", 1.0),
        voices_dir=Path(os.getenv("VOICES_DIR", ROOT / "voices")).expanduser().resolve(),
        filler_delay_ms=_env_int("FILLER_DELAY_MS", 800),
        embedding_model=_env("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
        rag_top_k=_env_int("RAG_TOP_K", 3),
        rag_min_score=_env_float("RAG_MIN_SCORE", 0.6),
        max_history_messages=_env_int("MAX_HISTORY_MESSAGES", 8),
        summary_trigger_messages=_env_int("SUMMARY_TRIGGER_MESSAGES", 12),
    )
