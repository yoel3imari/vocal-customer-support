"""Local embedding index (fastembed + numpy cosine search)."""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import numpy as np

from src.config import ROOT, Settings
from src.rag.ingest import Chunk, build_chunks
from src.pipeline.prompts import Retrieved

log = logging.getLogger("voxdesk.retriever")


class KnowledgeBase:
    """Per-tenant, in-memory index. Rebuilt at startup (small corpus, seconds)."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.docs_dir = settings.tenant.docs_dir
        self.chunks: list[Chunk] = []
        self._matrix: np.ndarray | None = None
        self._model = None
        self._lock = threading.Lock()
        self.enabled = True

    # -- lifecycle ---------------------------------------------------------

    def build(self) -> int:
        chunks = build_chunks(self.docs_dir)
        self.chunks = chunks
        if not chunks:
            log.warning("no documents in %s — agent will run without grounding", self.docs_dir)
            self._matrix = None
            return 0
        try:
            self._model = self._load_model()
            vectors = np.array(
                list(self._model.embed([self._render(c) for c in chunks], batch_size=16)),
                dtype=np.float32,
            )
            self._matrix = self._normalize(vectors)
            log.info("indexed %d chunks from %s", len(chunks), self.docs_dir.name)
            return len(chunks)
        except Exception as exc:  # noqa: BLE001 - demo should still start
            self.enabled = False
            self._matrix = None
            log.warning("embeddings unavailable (%s) — running without RAG", exc)
            return 0

    def _load_model(self):
        from fastembed import TextEmbedding

        cache_dir = Path(ROOT / "models" / "fastembed")
        return TextEmbedding(
            model_name=self.settings.embedding_model,
            cache_dir=str(cache_dir),
            threads=self.settings.whisper_threads,
        )

    @staticmethod
    def _render(chunk: Chunk) -> str:
        """Text shown to the model and embedded: heading carries plan names, prices, section context."""
        return f"{chunk.heading}\n{chunk.text}".strip() if chunk.heading else chunk.text

    @staticmethod
    def _normalize(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9
        return matrix / norms

    # -- query -------------------------------------------------------------

    def search(self, query: str, *, k: int | None = None, min_score: float | None = None) -> list[Retrieved]:
        query = query.strip()
        if not query or self._matrix is None or self._model is None or not self.enabled:
            return []
        k = k or self.settings.rag_top_k
        min_score = self.settings.rag_min_score if min_score is None else min_score
        with self._lock:
            query_vec = np.array(next(iter(self._model.query_embed([query]))), dtype=np.float32)
            query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-9)
            scores = self._matrix @ query_vec
        order = np.argsort(scores)[::-1][:k]
        results: list[Retrieved] = []
        seen: set[str] = set()
        for idx in order:
            score = float(scores[idx])
            if score < min_score:
                continue
            chunk = self.chunks[int(idx)]
            if chunk.chunk_id in seen:
                continue
            seen.add(chunk.chunk_id)
            results.append(Retrieved(source=chunk.source, text=self._render(chunk), score=round(score, 3)))
        log.info("retrieval %r -> %s", query[:80], [(r.source, r.score) for r in results])
        return results
