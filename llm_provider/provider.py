from __future__ import annotations

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterator, TypeVar

T = TypeVar("T")

_log = logging.getLogger(__name__)


def _retry(
    fn: Any,
    *,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    retryable: tuple[type[Exception], ...],
) -> Any:
    """Call *fn()* with exponential backoff on retryable exceptions."""
    for attempt in range(max_attempts):
        try:
            return fn()
        except retryable:
            if attempt + 1 == max_attempts:
                raise
            delay = base_delay * 2**attempt
            _log.warning("Attempt %d failed, retrying in %.1fs", attempt + 1, delay)
            time.sleep(delay)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

@dataclass
class Prompt:
    """A bundled system + user prompt pair.

    Defining prompts as named objects keeps them out of call sites and makes
    them easy to reuse, test, and version independently of the provider call.
    """

    system: str
    user: str


# ---------------------------------------------------------------------------
# Response metadata
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CompletionUsage:
    """Token counts for a completion request."""
    prompt_tokens: int
    completion_tokens: int


class CompletionResult(str):
    """A string that also carries completion metadata.

    Behaves exactly like ``str`` for existing callers.  Metadata is
    available via ``.usage``, ``.model``, and ``.stop_reason``.
    """
    usage: CompletionUsage | None
    model: str
    stop_reason: str | None

    def __new__(
        cls,
        text: str,
        *,
        usage: CompletionUsage | None = None,
        model: str = "",
        stop_reason: str | None = None,
    ) -> CompletionResult:
        instance = super().__new__(cls, text)
        instance.usage = usage
        instance.model = model
        instance.stop_reason = stop_reason
        return instance


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------

class CompletionStream:
    """Iterator of text chunks that accumulates into a :class:`CompletionResult`.

    After the stream is fully consumed, the ``.result`` property returns
    a :class:`CompletionResult` with the concatenated text and any metadata.
    """

    def __init__(
        self,
        chunks: Iterator[str],
        *,
        model: str = "",
        finalizer: Any | None = None,
    ) -> None:
        self._chunks = chunks
        self._model = model
        self._finalizer = finalizer  # callable(text) -> CompletionResult
        self._text_parts: list[str] = []
        self._result: CompletionResult | None = None

    def __iter__(self) -> CompletionStream:
        return self

    def __next__(self) -> str:
        try:
            chunk = next(self._chunks)
            self._text_parts.append(chunk)
            return chunk
        except StopIteration:
            text = "".join(self._text_parts)
            if self._finalizer:
                self._result = self._finalizer(text)
            else:
                self._result = CompletionResult(text, model=self._model)
            raise

    @property
    def result(self) -> CompletionResult:
        """Full result with metadata.  Available after the stream is consumed."""
        if self._result is None:
            for _ in self:
                pass
        assert self._result is not None
        return self._result


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class AIProvider(ABC):
    @abstractmethod
    def complete(self, system: str, user: str) -> str: ...

    def stream(self, system: str, user: str) -> CompletionStream:
        """Stream the response as text chunks.

        The default implementation wraps :meth:`complete` in a single-chunk
        stream.  Providers with native streaming support override this.
        """
        result = self.complete(system, user)
        return CompletionStream(iter([result]))

    def complete_json(self, system: str, user: str) -> dict:
        """Call complete() and parse the result as JSON.

        Strips markdown code fences (```json ... ```) if present,
        since some models wrap JSON in fences even when asked not to.
        """
        raw = self.complete(system, user).strip()
        if raw.startswith("```"):
            # Drop opening fence line (e.g. "```json")
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw
            # Drop closing fence
            raw = raw.rsplit("```", 1)[0]
        return json.loads(raw.strip())

    def complete_model(self, system: str, user: str, model_class: type[T]) -> T:
        """Call complete_json() and validate the result as a Pydantic BaseModel.

        Requires pydantic: pip install llm-provider[pydantic]

        Raises TypeError if model_class is not a Pydantic BaseModel subclass.
        Raises ImportError if pydantic is not installed.
        """
        try:
            from pydantic import BaseModel
        except ImportError:
            raise ImportError(
                "Pydantic is required for complete_model — install with: "
                "pip install 'llm-provider[pydantic]'"
            ) from None
        if not (isinstance(model_class, type) and issubclass(model_class, BaseModel)):
            raise TypeError(
                f"model_class must be a Pydantic BaseModel subclass, got {model_class!r}"
            )
        data = self.complete_json(system, user)
        return model_class.model_validate(data)

    def close(self) -> None:
        """Release resources held by this provider. No-op by default."""

    def __enter__(self) -> AIProvider:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class ClaudeProvider(AIProvider):
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
        self._client = anthropic.Anthropic(api_key=resolved_key)
        self._model = model
        self._max_tokens = max_tokens
        self._max_retries = max_retries

    def _make_result(self, msg: Any) -> CompletionResult:
        usage = CompletionUsage(
            prompt_tokens=msg.usage.input_tokens,
            completion_tokens=msg.usage.output_tokens,
        )
        return CompletionResult(
            msg.content[0].text,
            usage=usage,
            model=msg.model,
            stop_reason=msg.stop_reason,
        )

    def complete(self, system: str, user: str) -> str:
        import anthropic

        _log.debug("Claude request: model=%s, max_tokens=%d", self._model, self._max_tokens)

        def _call() -> CompletionResult:
            msg = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return self._make_result(msg)

        if self._max_retries <= 0:
            result = _call()
        else:
            result = _retry(
                _call,
                max_attempts=self._max_retries,
                retryable=(anthropic.RateLimitError, anthropic.InternalServerError),
            )
        _log.debug("Claude response: %d chars", len(result))
        return result

    def stream(self, system: str, user: str) -> CompletionStream:
        _log.debug("Claude stream: model=%s, max_tokens=%d", self._model, self._max_tokens)

        stream_ctx = self._client.messages.stream(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        managed_stream = stream_ctx.__enter__()

        def _chunks() -> Iterator[str]:
            try:
                yield from managed_stream.text_stream
            finally:
                stream_ctx.__exit__(None, None, None)

        def _finalizer(text: str) -> CompletionResult:
            msg = managed_stream.get_final_message()
            return self._make_result(msg)

        return CompletionStream(_chunks(), model=self._model, finalizer=_finalizer)


class OllamaProvider(AIProvider):
    def __init__(
        self, base_url: str = "", model: str = "", max_retries: int = 3
    ) -> None:
        import httpx

        self._base_url = base_url or os.environ.get(
            "OLLAMA_BASE_URL", "http://localhost:11434"
        )
        self._model = model or os.environ.get("OLLAMA_MODEL", "llama3")
        self._client: httpx.Client | None = httpx.Client(timeout=60.0)
        self._max_retries = max_retries

    def _make_result(self, text: str, data: dict) -> CompletionResult:
        prompt_tokens = data.get("prompt_eval_count", 0)
        completion_tokens = data.get("eval_count", 0)
        usage = CompletionUsage(prompt_tokens, completion_tokens) if prompt_tokens or completion_tokens else None
        return CompletionResult(
            text,
            usage=usage,
            model=data.get("model", self._model),
            stop_reason=data.get("done_reason"),
        )

    def complete(self, system: str, user: str) -> str:
        import httpx

        _log.debug("Ollama request: model=%s, url=%s", self._model, self._base_url)
        prompt = f"{system}\n\n{user}"

        def _call() -> CompletionResult:
            try:
                response = self._client.post(
                    f"{self._base_url}/api/generate",
                    json={"model": self._model, "prompt": prompt, "stream": False},
                )
            except httpx.ConnectError:
                raise ConnectionError(
                    f"Cannot reach Ollama at {self._base_url}"
                ) from None
            response.raise_for_status()
            data = response.json()
            return self._make_result(data["response"], data)

        if self._max_retries <= 0:
            result = _call()
        else:
            result = _retry(
                _call,
                max_attempts=self._max_retries,
                retryable=(ConnectionError, httpx.HTTPStatusError),
            )
        _log.debug("Ollama response: %d chars", len(result))
        return result

    def stream(self, system: str, user: str) -> CompletionStream:
        import httpx

        _log.debug("Ollama stream: model=%s, url=%s", self._model, self._base_url)
        prompt = f"{system}\n\n{user}"
        final_data: dict[str, Any] = {}

        def _chunks() -> Iterator[str]:
            try:
                with self._client.stream(
                    "POST",
                    f"{self._base_url}/api/generate",
                    json={"model": self._model, "prompt": prompt, "stream": True},
                ) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
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
            return self._make_result(text, final_data)

        return CompletionStream(_chunks(), model=self._model, finalizer=_finalizer)

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def __del__(self) -> None:
        self.close()


class OpenAIProvider(AIProvider):
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
        self._client = openai.OpenAI(api_key=resolved_key)
        self._model = model
        self._max_tokens = max_tokens
        self._max_retries = max_retries

    def _make_result(self, response: Any) -> CompletionResult:
        choice = response.choices[0]
        usage = None
        if response.usage:
            usage = CompletionUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
            )
        return CompletionResult(
            choice.message.content,
            usage=usage,
            model=response.model,
            stop_reason=choice.finish_reason,
        )

    def complete(self, system: str, user: str) -> str:
        import openai

        _log.debug("OpenAI request: model=%s, max_tokens=%d", self._model, self._max_tokens)

        def _call() -> CompletionResult:
            response = self._client.chat.completions.create(
                model=self._model,
                max_tokens=self._max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            return self._make_result(response)

        if self._max_retries <= 0:
            result = _call()
        else:
            result = _retry(
                _call,
                max_attempts=self._max_retries,
                retryable=(openai.RateLimitError, openai.InternalServerError),
            )
        _log.debug("OpenAI response: %d chars", len(result))
        return result

    def stream(self, system: str, user: str) -> CompletionStream:
        _log.debug("OpenAI stream: model=%s, max_tokens=%d", self._model, self._max_tokens)

        final_usage: dict[str, int] = {}

        def _chunks() -> Iterator[str]:
            response = self._client.chat.completions.create(
                model=self._model,
                max_tokens=self._max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                stream=True,
                stream_options={"include_usage": True},
            )
            for chunk in response:
                if chunk.usage:
                    final_usage["prompt_tokens"] = chunk.usage.prompt_tokens
                    final_usage["completion_tokens"] = chunk.usage.completion_tokens
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        def _finalizer(text: str) -> CompletionResult:
            usage = CompletionUsage(**final_usage) if final_usage else None
            return CompletionResult(text, usage=usage, model=self._model)

        return CompletionStream(_chunks(), model=self._model, finalizer=_finalizer)


def get_provider(
    provider: str | None = None,
    *,
    api_key: str = "",
    ollama_base_url: str = "",
    ollama_model: str = "",
    openai_model: str = "",
) -> AIProvider:
    """Return an AIProvider instance.

    Configuration priority (for each setting):
      1. Keyword arguments passed to this function
      2. Environment variables: AI_PROVIDER, ANTHROPIC_API_KEY,
         OPENAI_API_KEY, OLLAMA_BASE_URL, OLLAMA_MODEL
      3. Built-in defaults
    """
    name = provider or os.environ.get("AI_PROVIDER", "claude")
    _log.debug("Using provider %r", name)
    if name == "claude":
        return ClaudeProvider(api_key=api_key)
    if name == "openai":
        return OpenAIProvider(api_key=api_key, model=openai_model or "gpt-4o-mini")
    if name == "ollama":
        return OllamaProvider(base_url=ollama_base_url, model=ollama_model)
    raise ValueError(
        f"Unknown AI provider: {name!r}. Expected 'claude', 'openai', or 'ollama'."
    )
