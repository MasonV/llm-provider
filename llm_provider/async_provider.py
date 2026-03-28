"""Async variants of the LLM provider abstraction.

Mirrors the sync API in ``provider.py`` with ``async``/``await`` support.
All three SDK dependencies (anthropic, openai, httpx) provide async clients,
so no new runtime dependencies are needed.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, TypeVar

from .provider import (
    ClaudeProvider,
    CompletionResult,
    CompletionStream,
    CompletionUsage,
    OllamaProvider,
    OpenAIProvider,
    _extract_json_object,
    _parse_json_response,
    _validate_model_class,
)

T = TypeVar("T")

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Async retry
# ---------------------------------------------------------------------------

async def _async_retry(
    fn: Any,
    *,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    retryable: tuple[type[Exception], ...],
) -> Any:
    """Call awaitable *fn()* with exponential backoff on retryable exceptions."""
    for attempt in range(max_attempts):
        try:
            return await fn()
        except retryable:
            if attempt + 1 == max_attempts:
                raise
            delay = base_delay * 2**attempt
            _log.warning("Attempt %d failed, retrying in %.1fs", attempt + 1, delay)
            await asyncio.sleep(delay)


# ---------------------------------------------------------------------------
# Async streaming
# ---------------------------------------------------------------------------

class AsyncCompletionStream:
    """Async iterator of text chunks that accumulates into a :class:`CompletionResult`.

    After the stream is fully consumed, call :meth:`get_result` to obtain a
    :class:`CompletionResult` with the concatenated text and any metadata.

    Unlike the sync :class:`CompletionStream` which exposes a ``result``
    property, this class uses an async method because auto-consuming the
    stream requires awaiting.
    """

    def __init__(
        self,
        chunks: AsyncIterator[str],
        *,
        model: str = "",
        finalizer: Any | None = None,
    ) -> None:
        self._chunks = chunks
        self._model = model
        self._finalizer = finalizer  # callable(text) -> CompletionResult
        self._text_parts: list[str] = []
        self._result: CompletionResult | None = None

    def __aiter__(self) -> AsyncCompletionStream:
        return self

    async def __anext__(self) -> str:
        try:
            chunk = await self._chunks.__anext__()
            self._text_parts.append(chunk)
            return chunk
        except StopAsyncIteration:
            text = "".join(self._text_parts)
            if self._finalizer:
                self._result = self._finalizer(text)
            else:
                self._result = CompletionResult(text, model=self._model)
            raise

    async def get_result(self) -> CompletionResult:
        """Full result with metadata.  Available after the stream is consumed."""
        if self._result is None:
            async for _ in self:
                pass
        assert self._result is not None
        return self._result


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class AsyncAIProvider(ABC):
    @abstractmethod
    async def complete(self, system: str, user: str) -> str: ...

    async def stream(self, system: str, user: str) -> AsyncCompletionStream:
        """Stream the response as text chunks.

        The default implementation wraps :meth:`complete` in a single-chunk
        stream.  Providers with native streaming support override this.
        """
        result = await self.complete(system, user)

        async def _single_chunk() -> AsyncIterator[str]:
            yield result

        return AsyncCompletionStream(_single_chunk())

    async def complete_json(self, system: str, user: str) -> dict:
        """Call complete() and parse the result as JSON.

        Uses a multi-stage approach to handle messy LLM output:
        1. Strip markdown code fences
        2. Try json.loads on the cleaned text
        3. Fall back to brace-depth extraction for responses with preamble
        """
        return _parse_json_response(await self.complete(system, user))

    async def complete_model(self, system: str, user: str, model_class: type[T]) -> T:
        """Call complete_json() and validate the result as a Pydantic BaseModel.

        Requires pydantic: pip install llm-provider[pydantic]

        Raises TypeError if model_class is not a Pydantic BaseModel subclass.
        Raises ImportError if pydantic is not installed.
        """
        _validate_model_class(model_class)
        data = await self.complete_json(system, user)
        return model_class.model_validate(data)

    async def close(self) -> None:
        """Release resources held by this provider. No-op by default."""

    async def __aenter__(self) -> AsyncAIProvider:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()


# ---------------------------------------------------------------------------
# Async providers
# ---------------------------------------------------------------------------

class AsyncClaudeProvider(AsyncAIProvider):
    def __init__(
        self,
        api_key: str = "",
        model: str = "claude-haiku-4-5-20251001",
        max_tokens: int = 512,
        max_retries: int = 3,
    ) -> None:
        import anthropic

        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not resolved_key:
            raise ValueError(
                "Anthropic API key required — pass api_key or set ANTHROPIC_API_KEY"
            )
        self._client = anthropic.AsyncAnthropic(api_key=resolved_key)
        self._model = model
        self._max_tokens = max_tokens
        self._max_retries = max_retries

    async def complete(self, system: str, user: str) -> str:
        import anthropic

        _log.debug("AsyncClaude request: model=%s, max_tokens=%d", self._model, self._max_tokens)

        async def _call() -> CompletionResult:
            msg = await self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return ClaudeProvider._make_result(msg)

        if self._max_retries <= 0:
            result = await _call()
        else:
            result = await _async_retry(
                _call,
                max_attempts=self._max_retries,
                retryable=(anthropic.RateLimitError, anthropic.InternalServerError),
            )
        _log.debug("AsyncClaude response: %d chars", len(result))
        return result

    async def stream(self, system: str, user: str) -> AsyncCompletionStream:
        _log.debug("AsyncClaude stream: model=%s, max_tokens=%d", self._model, self._max_tokens)

        stream_ctx = self._client.messages.stream(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        managed_stream = await stream_ctx.__aenter__()

        async def _chunks() -> AsyncIterator[str]:
            try:
                async for text in managed_stream.text_stream:
                    yield text
            finally:
                await stream_ctx.__aexit__(None, None, None)

        def _finalizer(text: str) -> CompletionResult:
            msg = managed_stream.get_final_message()
            return ClaudeProvider._make_result(msg)

        return AsyncCompletionStream(_chunks(), model=self._model, finalizer=_finalizer)


class AsyncOllamaProvider(AsyncAIProvider):
    def __init__(
        self, base_url: str = "", model: str = "", max_retries: int = 3
    ) -> None:
        import httpx

        self._base_url = base_url or os.environ.get(
            "OLLAMA_BASE_URL", "http://localhost:11434"
        )
        self._model = model or os.environ.get("OLLAMA_MODEL", "llama3")
        self._client: httpx.AsyncClient | None = httpx.AsyncClient(timeout=60.0)
        self._max_retries = max_retries

    async def complete(self, system: str, user: str) -> str:
        import httpx

        _log.debug("AsyncOllama request: model=%s, url=%s", self._model, self._base_url)
        prompt = f"{system}\n\n{user}"

        async def _call() -> CompletionResult:
            try:
                response = await self._client.post(
                    f"{self._base_url}/api/generate",
                    json={"model": self._model, "prompt": prompt, "stream": False},
                )
            except httpx.ConnectError:
                raise ConnectionError(
                    f"Cannot reach Ollama at {self._base_url}"
                ) from None
            response.raise_for_status()
            data = response.json()
            return OllamaProvider._make_result(data["response"], data, self._model)

        if self._max_retries <= 0:
            result = await _call()
        else:
            result = await _async_retry(
                _call,
                max_attempts=self._max_retries,
                retryable=(ConnectionError, httpx.HTTPStatusError),
            )
        _log.debug("AsyncOllama response: %d chars", len(result))
        return result

    async def stream(self, system: str, user: str) -> AsyncCompletionStream:
        import httpx

        _log.debug("AsyncOllama stream: model=%s, url=%s", self._model, self._base_url)
        prompt = f"{system}\n\n{user}"
        final_data: dict[str, Any] = {}

        async def _chunks() -> AsyncIterator[str]:
            try:
                async with self._client.stream(
                    "POST",
                    f"{self._base_url}/api/generate",
                    json={"model": self._model, "prompt": prompt, "stream": True},
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        data = json.loads(line)
                        if data.get("response"):
                            yield data["response"]
                        if data.get("done"):
                            final_data.update(data)
            except httpx.ConnectError:
                raise ConnectionError(
                    f"Cannot reach Ollama at {self._base_url}"
                ) from None

        def _finalizer(text: str) -> CompletionResult:
            return OllamaProvider._make_result(text, final_data, self._model)

        return AsyncCompletionStream(_chunks(), model=self._model, finalizer=_finalizer)

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


class AsyncOpenAIProvider(AsyncAIProvider):
    def __init__(
        self,
        api_key: str = "",
        model: str = "gpt-4o-mini",
        max_tokens: int = 512,
        max_retries: int = 3,
    ) -> None:
        import openai

        resolved_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not resolved_key:
            raise ValueError(
                "OpenAI API key required — pass api_key or set OPENAI_API_KEY"
            )
        self._client = openai.AsyncOpenAI(api_key=resolved_key)
        self._model = model
        self._max_tokens = max_tokens
        self._max_retries = max_retries

    async def complete(self, system: str, user: str) -> str:
        import openai

        _log.debug("AsyncOpenAI request: model=%s, max_tokens=%d", self._model, self._max_tokens)

        async def _call() -> CompletionResult:
            response = await self._client.chat.completions.create(
                model=self._model,
                max_tokens=self._max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            return OpenAIProvider._make_result(response)

        if self._max_retries <= 0:
            result = await _call()
        else:
            result = await _async_retry(
                _call,
                max_attempts=self._max_retries,
                retryable=(openai.RateLimitError, openai.InternalServerError),
            )
        _log.debug("AsyncOpenAI response: %d chars", len(result))
        return result

    async def stream(self, system: str, user: str) -> AsyncCompletionStream:
        _log.debug("AsyncOpenAI stream: model=%s, max_tokens=%d", self._model, self._max_tokens)

        final_usage: dict[str, int] = {}

        async def _chunks() -> AsyncIterator[str]:
            response = await self._client.chat.completions.create(
                model=self._model,
                max_tokens=self._max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                stream=True,
                stream_options={"include_usage": True},
            )
            async for chunk in response:
                if chunk.usage:
                    final_usage["prompt_tokens"] = chunk.usage.prompt_tokens
                    final_usage["completion_tokens"] = chunk.usage.completion_tokens
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        def _finalizer(text: str) -> CompletionResult:
            usage = CompletionUsage(**final_usage) if final_usage else None
            return CompletionResult(text, usage=usage, model=self._model)

        return AsyncCompletionStream(_chunks(), model=self._model, finalizer=_finalizer)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_async_provider(
    provider: str | None = None,
    *,
    api_key: str = "",
    ollama_base_url: str = "",
    ollama_model: str = "",
    openai_model: str = "",
) -> AsyncAIProvider:
    """Return an AsyncAIProvider instance.

    Configuration priority (for each setting):
      1. Keyword arguments passed to this function
      2. Environment variables: AI_PROVIDER, ANTHROPIC_API_KEY,
         OPENAI_API_KEY, OLLAMA_BASE_URL, OLLAMA_MODEL
      3. Built-in defaults
    """
    name = provider or os.environ.get("AI_PROVIDER", "claude")
    _log.debug("Using async provider %r", name)
    if name == "claude":
        return AsyncClaudeProvider(api_key=api_key)
    if name == "openai":
        return AsyncOpenAIProvider(api_key=api_key, model=openai_model or "gpt-4o-mini")
    if name == "ollama":
        return AsyncOllamaProvider(base_url=ollama_base_url, model=ollama_model)
    raise ValueError(
        f"Unknown AI provider: {name!r}. Expected 'claude', 'openai', or 'ollama'."
    )
