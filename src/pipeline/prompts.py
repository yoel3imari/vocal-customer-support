"""Prompt composition and context budgeting for one call turn."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable, Sequence

MAX_CHUNK_CHARS = 700


@dataclass(frozen=True)
class Retrieved:
    source: str
    text: str
    score: float = 0.0


SYSTEM_RULES = """\
You are speaking on a live phone call. Your words are converted to speech, so:
- Reply in 1-2 short sentences. Be warm, direct, and human.
- Plain spoken language only: no lists, markdown, emoji, URLs, or stage directions.
- Ask at most one question at a time.
- Never say you are an AI, a language model, or reading a knowledge base.

Grounding rules:
- Factual claims must come from the knowledge base excerpts below. Never invent prices,
  dates, policies, account details or delivery estimates.
- If the excerpts do not answer the caller's question, say you don't have that
  information and offer to connect them to a specialist.
- If the caller asks to speak to a human, hand off gracefully.
- Today's date is {today}.
"""


def knowledge_block(retrieved: Sequence[Retrieved]) -> str:
    if not retrieved:
        return (
            "KNOWLEDGE BASE EXCERPTS: none found for this question.\n"
            "You do not have grounded information for it — say so honestly and offer a "
            "specialist handoff. Do not guess."
        )
    parts = ["KNOWLEDGE BASE EXCERPTS (the only source of truth for this call):"]
    for i, chunk in enumerate(retrieved, 1):
        text = chunk.text.strip().replace("\n\n", "\n")
        if len(text) > MAX_CHUNK_CHARS:
            text = text[:MAX_CHUNK_CHARS].rsplit(" ", 1)[0] + "…"
        parts.append(f"[{i}] {chunk.source}\n{text}")
    parts.append(
        "Use only the excerpts above, and only if they actually answer the caller's "
        "question — ignore anything unrelated. If none of them answers it, say you don't "
        "have that information and offer a specialist."
    )
    return "\n\n".join(parts)


def build_messages(
    *,
    persona: str,
    history: Iterable[dict],
    summary: str | None,
    retrieved: Sequence[Retrieved],
    user_text: str,
    max_history: int = 8,
) -> list[dict[str, str]]:
    """Compose the chat messages for one turn, capped and bounded."""
    messages: list[dict[str, str]] = [
        {"role": "system", "content": persona.strip() + "\n\n" + SYSTEM_RULES.format(today=date.today().isoformat())}
    ]
    if summary:
        messages.append({"role": "system", "content": f"Call so far (summary): {summary}"})

    recent = list(history)[-max_history:]
    messages.extend({"role": turn["role"], "content": turn["text"]} for turn in recent)

    messages.append({"role": "system", "content": knowledge_block(retrieved)})
    messages.append({"role": "user", "content": user_text.strip()})
    return messages
