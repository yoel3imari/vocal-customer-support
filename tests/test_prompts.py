from __future__ import annotations

from src.pipeline.prompts import Retrieved, build_messages, knowledge_block


def _history(n: int) -> list[dict]:
    turns = []
    for i in range(n):
        turns.append({"role": "user" if i % 2 == 0 else "assistant", "text": f"turn {i}"})
    return turns


def test_history_is_capped():
    messages = build_messages(
        persona="You are a support agent.",
        history=_history(40),
        summary=None,
        retrieved=[],
        user_text="current question",
        max_history=8,
    )
    # system + 8 history + knowledge system + current user
    assert len(messages) == 11
    assert messages[0]["role"] == "system"
    assert messages[-1] == {"role": "user", "content": "current question"}
    assert messages[1]["content"] == "turn 32"


def test_summary_is_included_when_present():
    messages = build_messages(
        persona="persona",
        history=[],
        summary="Caller wants a refund for order 123.",
        retrieved=[],
        user_text="hello",
    )
    assert any("Caller wants a refund" in m["content"] for m in messages)


def test_retrieved_context_is_injected():
    messages = build_messages(
        persona="persona",
        history=[],
        summary=None,
        retrieved=[Retrieved(source="billing.md", text="Late fee is $7.", score=0.71)],
        user_text="why is there a fee?",
    )
    block = messages[-2]["content"]
    assert messages[-2]["role"] == "system"
    assert "Late fee is $7." in block
    assert "billing.md" in block


def test_no_context_policy_is_explicit():
    block = knowledge_block([])
    assert "none found" in block
    assert "Do not guess" in block


def test_chunks_are_truncated_to_budget():
    block = knowledge_block([Retrieved(source="big.md", text="x " * 2000)])
    assert len(block) < 1600
