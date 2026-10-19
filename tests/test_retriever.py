from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from src.rag.ingest import Chunk
from src.rag.retriever import KnowledgeBase


class FakeEmbedding:
    def __init__(self, mapping: dict[str, np.ndarray]) -> None:
        self.mapping = mapping

    def query_embed(self, query):
        for text in query:
            yield self.mapping[text]


def _kb(docs_dir, *, top_k: int = 3, min_score: float = 0.35) -> KnowledgeBase:
    settings = SimpleNamespace(
        tenant=SimpleNamespace(docs_dir=docs_dir),
        embedding_model="fake",
        whisper_threads=1,
        rag_top_k=top_k,
        rag_min_score=min_score,
    )
    return KnowledgeBase(settings)


def test_search_returns_best_and_applies_threshold(tmp_path):
    kb = _kb(tmp_path, top_k=2, min_score=0.5)
    kb.chunks = [
        Chunk.make("billing policy", "billing.md"),
        Chunk.make("outage policy", "outages.md"),
        Chunk.make("random noise", "noise.md"),
    ]
    kb._model = FakeEmbedding(
        {
            "billing question": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        }
    )
    kb._matrix = kb._normalize(
        np.array(
            [
                [0.95, 0.05, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
    )
    results = kb.search("billing question")
    assert [r.source for r in results] == ["billing.md"]
    assert results[0].score >= 0.5


def test_search_includes_heading_in_retrieved_text(tmp_path):
    """Plan names/prices live in headings; the model must see them (regression)."""
    kb = _kb(tmp_path, top_k=1, min_score=0.0)
    chunk = Chunk.make("- 25 GB of 5G data.", "plans.md", "Smart Plus — $29/month")
    kb.chunks = [chunk]
    kb._model = FakeEmbedding({"price question": np.array([1.0, 0.0], dtype=np.float32)})
    kb._matrix = kb._normalize(np.array([[1.0, 0.0]], dtype=np.float32))
    results = kb.search("price question")
    assert results[0].text.startswith("Smart Plus — $29/month")
    assert KnowledgeBase._render(chunk) == results[0].text


def test_search_empty_query_and_empty_index(tmp_path):
    kb = _kb(tmp_path)
    assert kb.search("") == []
    assert kb.search("anything") == []  # no matrix built


def test_build_without_docs_disables_index(tmp_path):
    kb = _kb(tmp_path)
    assert kb.build() == 0
    assert kb.search("hello") == []
