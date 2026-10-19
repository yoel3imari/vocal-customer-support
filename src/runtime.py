"""Process-wide model/services bundle, built once at startup (CPU-heavy, in a thread)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

from src.config import Settings
from src.pipeline.llm import LLMClient
from src.pipeline.stt import Transcriber
from src.pipeline.tts import Speaker
from src.pipeline.vad import ProbModel, load_vad_model
from src.rag.retriever import KnowledgeBase
from src.store import CallStore

log = logging.getLogger("voxdesk.runtime")


@dataclass
class Runtime:
    settings: Settings
    transcriber: Transcriber
    speaker: Speaker
    knowledge: KnowledgeBase
    llm: LLMClient
    store: CallStore
    vad_factory: Callable[[], ProbModel]

    @classmethod
    def build(cls, settings: Settings) -> "Runtime":
        log.info("warming up models for tenant %s", settings.tenant.slug)
        store = CallStore(settings.db_path)
        speaker = Speaker(settings)
        transcriber = Transcriber(settings)
        transcriber.warmup()
        speaker.warmup()
        knowledge = KnowledgeBase(settings)
        knowledge.build()
        llm = LLMClient(settings)
        log.info(
            "runtime ready: tenant=%s stt=%s tts=%s@%dHz rag_chunks=%d model=%s",
            settings.tenant.slug, settings.whisper_model, settings.tts_voice,
            speaker.sample_rate, len(knowledge.chunks), settings.llm_model,
        )
        return cls(
            settings=settings,
            transcriber=transcriber,
            speaker=speaker,
            knowledge=knowledge,
            llm=llm,
            store=store,
            vad_factory=lambda: load_vad_model(settings),
        )

    async def aclose(self) -> None:
        await self.llm.aclose()
        self.store.close()
