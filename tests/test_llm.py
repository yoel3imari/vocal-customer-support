from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.pipeline.llm import LLMClient, LLMError, message_text, to_langchain_messages


def _settings(**overrides) -> SimpleNamespace:
    values = dict(
        openrouter_api_key="test-key",
        openrouter_base_url="https://example.test/api/v1",
        llm_model="primary/model:free",
        llm_fallback_model="fallback/model",
        llm_temperature=0.4,
        llm_max_tokens=160,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _chunk(text: str) -> str:
    payload = {
        "id": "chunk-1",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "mock",
        "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
    }
    return f"data: {json.dumps(payload)}\n\n"


def _sse(*chunks: str) -> bytes:
    return ("".join(_chunk(c) for c in chunks) + "data: [DONE]\n\n").encode()


def _completion(text: str) -> bytes:
    payload = {
        "id": "cmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": "mock",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    return json.dumps(payload).encode()


def _client_with(handler) -> tuple[LLMClient, httpx.AsyncClient]:
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return LLMClient(_settings(), http_async_client=http), http


def _stream_response(content: bytes) -> httpx.Response:
    return httpx.Response(200, content=content, headers={"content-type": "text/event-stream"})


@pytest.mark.asyncio
async def test_stream_parses_sse_tokens_through_langchain():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer test-key"
        body = json.loads(request.content)
        assert body["model"] == "primary/model:free"
        assert body["stream"] is True
        # langchain-openai renames max_tokens -> max_completion_tokens (OpenAI's new name)
        assert body["max_completion_tokens"] == 160
        return _stream_response(_sse("Hello", " world"))

    client, http = _client_with(handler)
    tokens = [token async for token in client.stream([{"role": "user", "content": "hi"}])]
    assert tokens == ["Hello", " world"]
    await http.aclose()
    await client.aclose()


@pytest.mark.asyncio
async def test_stream_falls_back_before_first_token():
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        model = json.loads(request.content)["model"]
        seen.append(model)
        if model == "primary/model:free":
            return httpx.Response(
                429,
                json={"error": {"message": "rate limited", "type": "rate_limit_error"}},
            )
        return _stream_response(_sse("fallback answer"))

    client, http = _client_with(handler)
    tokens = [token async for token in client.stream([{"role": "user", "content": "hi"}])]
    assert tokens == ["fallback answer"]
    assert seen == ["primary/model:free", "fallback/model"]
    await http.aclose()
    await client.aclose()


@pytest.mark.asyncio
async def test_empty_response_falls_back():
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        model = json.loads(request.content)["model"]
        seen.append(model)
        return _stream_response(_sse() if model == "primary/model:free" else _sse("ok"))

    client, http = _client_with(handler)
    tokens = [token async for token in client.stream([{"role": "user", "content": "hi"}])]
    assert tokens == ["ok"]
    assert seen == ["primary/model:free", "fallback/model"]
    await http.aclose()
    await client.aclose()


@pytest.mark.asyncio
async def test_complete_uses_non_streaming_call():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["stream"] is False
        assert body["max_completion_tokens"] == 120
        return httpx.Response(200, json=json.loads(_completion("Caller asked about a refund.")))

    client, http = _client_with(handler)
    summary = await client.complete([{"role": "user", "content": "transcript"}], max_tokens=120)
    assert summary == "Caller asked about a refund."
    await http.aclose()
    await client.aclose()


@pytest.mark.asyncio
async def test_missing_api_key_is_a_clear_error():
    client = LLMClient(_settings(openrouter_api_key=""))
    with pytest.raises(LLMError, match="OPENROUTER_API_KEY"):
        async for _ in client.stream([{"role": "user", "content": "hi"}]):
            pass
    await client.aclose()


def test_model_is_a_langchain_chatopenai_pointed_at_openrouter():
    client = LLMClient(_settings())
    model = client.model("primary/model:free")
    assert isinstance(model, ChatOpenAI)
    assert model.model_name == "primary/model:free"
    assert str(model.openai_api_base) == "https://example.test/api/v1"
    assert model.max_tokens == 160


def test_messages_are_converted_to_langchain_types():
    converted = to_langchain_messages(
        [
            {"role": "system", "content": "persona"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
    )
    assert [type(m) for m in converted] == [SystemMessage, HumanMessage, AIMessage]
    assert [m.content for m in converted] == ["persona", "hello", "hi there"]


def test_message_text_handles_str_and_block_content():
    assert message_text(AIMessage(content="plain")) == "plain"
    blocks = AIMessage(content=[{"type": "text", "text": "part one "}, {"type": "text", "text": "part two"}])
    assert message_text(blocks) == "part one part two"
