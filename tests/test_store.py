from __future__ import annotations

from src.store import CallStore


def test_call_lifecycle_is_persisted(tmp_path):
    store = CallStore(tmp_path / "calls.db")
    store.start_call("call_1", "acme-telecom", "test/model")
    store.add_turn("call_1", "user", "how much is the plan?")
    store.add_turn("call_1", "assistant", "It is $19.", sources=["plans.md"])
    store.add_metric("call_1", "stt", 412.4)
    store.add_metric("call_1", "stt", 300.0)
    store.end_call("call_1", duration_s=42.5, summary="Caller asked about pricing.")

    calls = store.recent_calls()
    assert calls[0]["id"] == "call_1"
    assert calls[0]["tenant"] == "acme-telecom"
    assert calls[0]["turns"] == 2
    assert calls[0]["duration_s"] == 42.5

    stats = store.stats()
    assert stats["calls"] == 1
    assert stats["turns"] == 2
    assert stats["avg_ms"]["stt"] == 356.2
    store.close()
