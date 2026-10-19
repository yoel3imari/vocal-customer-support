from __future__ import annotations

from src.pipeline.tts import SentenceBuffer


def _feed_all(tokens: list[str]) -> tuple[list[str], str | None]:
    buf = SentenceBuffer()
    out: list[str] = []
    for token in tokens:
        out.extend(buf.feed(token))
    return out, buf.flush()


def test_sentences_split_at_punctuation_and_space():
    out, tail = _feed_all(
        ["Thanks", " for", " calling", " NovaCell", ".", " How", " can", " I", " help", "?", " I", " can", " check", "."]
    )
    assert out == ["Thanks for calling NovaCell.", "How can I help?"]
    assert tail == "I can check."


def test_decimal_prices_are_not_split():
    out, tail = _feed_all(["It", " costs", " $19", ".", "99", " per", " month", "."])
    assert out == []
    assert tail == "It costs $19.99 per month."


def test_abbreviation_is_not_a_boundary():
    out, tail = _feed_all(["Mr", ".", " Smith", " called", ".", " Anything", " else", "?"])
    assert out == ["Mr. Smith called."]
    assert tail == "Anything else?"


def test_long_text_is_chunked_for_low_latency():
    buf = SentenceBuffer()
    out = buf.feed("word " * 100)
    assert out
    assert len(out[0]) <= 240
    assert buf.flush()
