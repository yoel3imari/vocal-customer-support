"""OpenRouter chat client built on LangChain (ChatOpenAI) with a model fallback.

The rest of the pipeline only sees ``stream(messages) -> AsyncIterator[str]`` and
``complete(messages) -> str``; LangChain handles the OpenAI-compatible transport,
SSE parsing and retries. Messages are plain dicts at this boundary and converted
to LangChain messages here.
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Sequence

import httpx
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.config import Settings

log = logging.getLogger("voxdesk.llm")

DEFAULT_TIMEOUT = httpx.Timeout(60.0, connect=10.0)

_ROLE_MAP = {"system": SystemMessage, "user": HumanMessage, "assistant": AIMessage}


class LLMError(RuntimeError):
    pass


def to_langchain_messages(messages: Sequence[dict]) -> list[BaseMessage]:
    converted: list[BaseMessage] = []
    for message in messages:
        factory = _ROLE_MAP.get(message.get("role", "user"), HumanMessage)
        converted.append(factory(content=message.get("content", "")))
    return converted


def message_text(message: Any) -> str:
    """Extract plain text from a LangChain message/chunk (str, blocks, or list content)."""
    text = getattr(message, "text", None)
    if isinstance(text, str):
        return text
    if callable(text):  # older langchain-core exposed .text() as a method
        return text()
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    return ""


class LLMClient:
    def __init__(self, settings: Settings, *, http_async_client: httpx.AsyncClient | None = None) -> None:
        self.settings = settings
        self._http = http_async_client or httpx.AsyncClient(timeout=DEFAULT_TIMEOUT)
        self._owns_http = http_async_client is None
        self._models: dict[tuple[str, int], ChatOpenAI] = {}

    async def aclose(self) -> None:
        if self._owns_http:
            await self._http.aclose()

    # -- model plumbing ----------------------------------------------------

    def _requirements_ok(self) -> None:
        if not self.settings.openrouter_api_key:
            raise LLMError(
                "OPENROUTER_API_KEY is not set. Copy .env.example to .env and add your key."
            )

    def _model_names(self) -> list[str]:
        names = [self.settings.llm_model]
        fallback = self.settings.llm_fallback_model
        if fallback and fallback != self.settings.llm_model:
            names.append(fallback)
        return names

    def model(self, name: str, *, max_tokens: int | None = None) -> ChatOpenAI:
        """LangChain chat model pointed at OpenRouter (cached per model + token budget)."""
        key = (name, max_tokens or self.settings.llm_max_tokens)
        if key not in self._models:
            self._models[key] = ChatOpenAI(
                model=name,
                api_key=self.settings.openrouter_api_key,
                base_url=self.settings.openrouter_base_url,
                temperature=self.settings.llm_temperature,
                max_tokens=key[1],
                timeout=DEFAULT_TIMEOUT,
                # Fail fast: a voice turn would rather switch models than stall on retries.
                max_retries=0,
                default_headers={
                    "HTTP-Referer": "https://github.com/voxdesk",
                    "X-Title": "VoxDesk",
                },
                http_async_client=self._http,
            )
        return self._models[key]

    # -- generation --------------------------------------------------------

    async def stream(self, messages: Sequence[dict]) -> AsyncIterator[str]:
        """Yield content deltas, falling back once if the primary model fails before output."""
        self._requirements_ok()
        lc_messages = to_langchain_messages(messages)
        last_error: Exception | None = None
        names = self._model_names()
        for index, name in enumerate(names):
            produced = False
            try:
                async for chunk in self.model(name).astream(lc_messages):
                    text = message_text(chunk)
                    if text:
                        produced = True
                        yield text
                if not produced:
                    raise LLMError(f"empty response from {name}")
                return
            except Exception as exc:  # noqa: BLE001 - surfaced as LLMError below
                last_error = exc
                if produced:
                    raise LLMError(f"stream broke mid-answer ({name}): {exc}") from exc
                log.warning("LLM model %s failed: %s", name, exc)
                if index + 1 < len(names):
                    log.info("retrying with fallback model %s", names[index + 1])
                    continue
        raise LLMError(str(last_error) if last_error else "LLM request failed")

    async def complete(self, messages: Sequence[dict], *, max_tokens: int = 220) -> str:
        """Non-streaming completion (used for rolling call summaries, off the critical path)."""
        self._requirements_ok()
        lc_messages = to_langchain_messages(messages)
        last_error: Exception | None = None
        for name in self._model_names():
            try:
                reply = await self.model(name, max_tokens=max_tokens).ainvoke(lc_messages)
                return message_text(reply).strip()
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                log.warning("summary model %s failed: %s", name, exc)
        raise LLMError(str(last_error) if last_error else "summary failed")
