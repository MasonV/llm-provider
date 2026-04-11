from __future__ import annotations

import json
import logging
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterator, TypeVar

from .events import CompletionCallback, CompletionEvent

T = TypeVar("T")

_log = logging.getLogger(__name__)


def _retry(
    fn: Any,
    *,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    retryable: tuple[type[Exception], ...],
    callback: CompletionCallback | None = None,
    provider_name: str = "",
    model_name: str = "",
    context: dict | None = None,
) -> Any:
    """Call *fn()* with exponential backoff on retryable exceptions."""
    for attempt in range(max_attempts):
        try:
            return fn()
        except retryable as exc:
            if attempt + 1 == max_attempts:
                raise
            delay = base_delay * 2**attempt
            _log.warning(
                "Attempt %d failed (%s), retrying in %.1fs",
                attempt + 1, exc, delay,
            )
            if callback:
                callback(CompletionEvent(
                    phase="retry",
                    provider=provider_name,
                    model=model_name,
                    error=str(exc),
                    attempt=attempt + 1,
                    context=context,
                ))
            time.sleep(delay)


def _parse_json_response(raw: str) -> dict:
    """Parse a potentially messy LLM response into a dict.

    Strips markdown code fences, tries json.loads, falls back to brace-depth
    extraction.  Shared by both sync and async complete_json implementations.
    """
    raw = raw.strip()
    # Strip markdown fences — handles mid-response fences, not just leading
    cleaned = re.sub(
        r"^```(?:json)?\n?|```$", "", raw, flags=re.MULTILINE
    ).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as first_err:
        # Fall back: extract the first balanced {...} block
        block = _extract_json_object(cleaned)
        if block:
            try:
                return json.loads(block)
            except json.JSONDecodeError:
                pass

        _log.warning(
            "complete_json: all parse attempts failed; raw=%.300r", raw
        )
        raise first_err


def _validate_model_class(model_class: type) -> None:
    """Validate that model_class is a Pydantic BaseModel subclass.

    Raises ImportError if pydantic is not installed.
    Raises TypeError if model_class is not a BaseModel subclass.
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


def _extract_json_object(text: str) -> str | None:
    """Extract the first top-level {...} block using brace-depth tracking.

    Handles nested braces, quoted strings (with escaped quotes), and
    multi-line values — unlike a simple regex which breaks on nested objects.
    """
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


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
    _callback: CompletionCallback | None = None
    _context: dict | None = None

    def set_callback(
        self,
        callback: CompletionCallback | None,
        context: dict | None = None,
    ) -> None:
        """Register an optional event callback for observability.

        Args:
            callback: Function called with a :class:`CompletionEvent` at each
                lifecycle point (start, end, error, retry).  Pass ``None`` to
                clear.
            context: Arbitrary dict passed through unchanged on every event.
                Useful for attaching caller-specific identifiers.
        """
        self._callback = callback
        self._context = context

    def _emit(self, event: CompletionEvent) -> None:
        """Dispatch *event* to the registered callback, if any."""
        if self._callback:
            self._callback(event)

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

        Uses a multi-stage approach to handle messy LLM output:
        1. Strip markdown code fences
        2. Try json.loads on the cleaned text
        3. Fall back to brace-depth extraction for responses with preamble
        """
        return _parse_json_response(self.complete(system, user))

    def complete_model(self, system: str, user: str, model_class: type[T]) -> T:
        """Call complete_json() and validate the result as a Pydantic BaseModel.

        Requires pydantic: pip install llm-provider[pydantic]

        Raises TypeError if model_class is not a Pydantic BaseModel subclass.
        Raises ImportError if pydantic is not installed.
        """
        _validate_model_class(model_class)
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

    @staticmethod
    def _make_result(msg: Any) -> CompletionResult:
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

        _log.info("Claude request: model=%s, max_tokens=%d", self._model, self._max_tokens)
        self._emit(CompletionEvent(
            phase="start", provider="claude", model=self._model,
            context=self._context,
        ))
        t0 = time.monotonic()

        def _call() -> CompletionResult:
            msg = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return ClaudeProvider._make_result(msg)

        try:
            if self._max_retries <= 0:
                result = _call()
            else:
                result = _retry(
                    _call,
                    max_attempts=self._max_retries,
                    retryable=(anthropic.RateLimitError, anthropic.InternalServerError),
                    callback=self._callback,
                    provider_name="claude",
                    model_name=self._model,
                    context=self._context,
                )
        except Exception as exc:
            elapsed = (time.monotonic() - t0) * 1000
            _log.warning("Claude request failed after %.0fms: %s", elapsed, exc)
            self._emit(CompletionEvent(
                phase="error", provider="claude", model=self._model,
                elapsed_ms=elapsed, error=str(exc), context=self._context,
            ))
            raise

        elapsed = (time.monotonic() - t0) * 1000
        _log.info(
            "Claude response: %d chars, %.0fms, %s/%s tokens",
            len(result), elapsed,
            result.usage.prompt_tokens if result.usage else "?",
            result.usage.completion_tokens if result.usage else "?",
        )
        self._emit(CompletionEvent(
            phase="end", provider="claude", model=self._model,
            elapsed_ms=elapsed,
            prompt_tokens=result.usage.prompt_tokens if result.usage else None,
            completion_tokens=result.usage.completion_tokens if result.usage else None,
            context=self._context,
        ))
        return result

    def stream(self, system: str, user: str) -> CompletionStream:
        _log.info("Claude stream: model=%s, max_tokens=%d", self._model, self._max_tokens)

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
            return ClaudeProvider._make_result(msg)

        return CompletionStream(_chunks(), model=self._model, finalizer=_finalizer)


class OllamaProvider(AIProvider):
    def __init__(
        self,
        base_url: str = "",
        model: str = "",
        max_retries: int = 3,
        timeout: float = 120.0,
        think: bool | None = None,
    ) -> None:
        import httpx

        self._base_url = base_url or os.environ.get(
            "OLLAMA_BASE_URL", "http://localhost:11434"
        )
        self._model = model or os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")
        self._timeout = timeout
        self._think = think
        self._client: httpx.Client | None = httpx.Client(timeout=timeout)
        self._max_retries = max_retries

    @staticmethod
    def _make_result(text: str, data: dict, fallback_model: str = "") -> CompletionResult:
        prompt_tokens = data.get("prompt_eval_count", 0)
        completion_tokens = data.get("eval_count", 0)
        usage = CompletionUsage(prompt_tokens, completion_tokens) if prompt_tokens or completion_tokens else None
        return CompletionResult(
            text,
            usage=usage,
            model=data.get("model", fallback_model),
            stop_reason=data.get("done_reason"),
        )

    def _chat_payload(self, system: str, user: str, stream: bool) -> dict[str, Any]:
        """Build the /api/chat request body."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "stream": stream,
        }
        if self._think is not None:
            payload["think"] = self._think
        return payload

    @staticmethod
    def _extract_content(message: dict) -> str:
        """Extract the visible content from a chat response message."""
        return message.get("content", "")

    def complete(self, system: str, user: str) -> str:
        import httpx

        _log.info("Ollama request: model=%s, url=%s", self._model, self._base_url)
        self._emit(CompletionEvent(
            phase="start", provider="ollama", model=self._model,
            context=self._context,
        ))
        t0 = time.monotonic()

        def _call() -> CompletionResult:
            try:
                response = self._client.post(
                    f"{self._base_url}/api/chat",
                    json=self._chat_payload(system, user, stream=False),
                )
            except httpx.ConnectError:
                raise ConnectionError(
                    f"Cannot reach Ollama at {self._base_url}"
                ) from None
            response.raise_for_status()
            data = response.json()
            text = OllamaProvider._extract_content(data.get("message", {}))
            return OllamaProvider._make_result(text, data, self._model)

        try:
            if self._max_retries <= 0:
                result = _call()
            else:
                result = _retry(
                    _call,
                    max_attempts=self._max_retries,
                    retryable=(ConnectionError, httpx.HTTPStatusError, httpx.TimeoutException),
                    callback=self._callback,
                    provider_name="ollama",
                    model_name=self._model,
                    context=self._context,
                )
        except Exception as exc:
            elapsed = (time.monotonic() - t0) * 1000
            _log.warning("Ollama request failed after %.0fms: %s", elapsed, exc)
            self._emit(CompletionEvent(
                phase="error", provider="ollama", model=self._model,
                elapsed_ms=elapsed, error=str(exc), context=self._context,
            ))
            raise

        elapsed = (time.monotonic() - t0) * 1000
        _log.info(
            "Ollama response: %d chars, %.0fms, %s/%s tokens",
            len(result), elapsed,
            result.usage.prompt_tokens if result.usage else "?",
            result.usage.completion_tokens if result.usage else "?",
        )
        self._emit(CompletionEvent(
            phase="end", provider="ollama", model=self._model,
            elapsed_ms=elapsed,
            prompt_tokens=result.usage.prompt_tokens if result.usage else None,
            completion_tokens=result.usage.completion_tokens if result.usage else None,
            context=self._context,
        ))
        return result

    def stream(self, system: str, user: str) -> CompletionStream:
        import httpx

        _log.info("Ollama stream: model=%s, url=%s", self._model, self._base_url)
        final_data: dict[str, Any] = {}

        def _chunks() -> Iterator[str]:
            try:
                with self._client.stream(
                    "POST",
                    f"{self._base_url}/api/chat",
                    json=self._chat_payload(system, user, stream=True),
                ) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        data = json.loads(line)
                        msg = data.get("message", {})
                        content = msg.get("content", "")
                        if content:
                            yield content
                        if data.get("done"):
                            final_data.update(data)
            except httpx.ConnectError:
                raise ConnectionError(
                    f"Cannot reach Ollama at {self._base_url}"
                ) from None

        def _finalizer(text: str) -> CompletionResult:
            return OllamaProvider._make_result(text, final_data, self._model)

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

    @staticmethod
    def _make_result(response: Any) -> CompletionResult:
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

        _log.info("OpenAI request: model=%s, max_tokens=%d", self._model, self._max_tokens)
        self._emit(CompletionEvent(
            phase="start", provider="openai", model=self._model,
            context=self._context,
        ))
        t0 = time.monotonic()

        def _call() -> CompletionResult:
            response = self._client.chat.completions.create(
                model=self._model,
                max_tokens=self._max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            return OpenAIProvider._make_result(response)

        try:
            if self._max_retries <= 0:
                result = _call()
            else:
                result = _retry(
                    _call,
                    max_attempts=self._max_retries,
                    retryable=(openai.RateLimitError, openai.InternalServerError),
                    callback=self._callback,
                    provider_name="openai",
                    model_name=self._model,
                    context=self._context,
                )
        except Exception as exc:
            elapsed = (time.monotonic() - t0) * 1000
            _log.warning("OpenAI request failed after %.0fms: %s", elapsed, exc)
            self._emit(CompletionEvent(
                phase="error", provider="openai", model=self._model,
                elapsed_ms=elapsed, error=str(exc), context=self._context,
            ))
            raise

        elapsed = (time.monotonic() - t0) * 1000
        _log.info(
            "OpenAI response: %d chars, %.0fms, %s/%s tokens",
            len(result), elapsed,
            result.usage.prompt_tokens if result.usage else "?",
            result.usage.completion_tokens if result.usage else "?",
        )
        self._emit(CompletionEvent(
            phase="end", provider="openai", model=self._model,
            elapsed_ms=elapsed,
            prompt_tokens=result.usage.prompt_tokens if result.usage else None,
            completion_tokens=result.usage.completion_tokens if result.usage else None,
            context=self._context,
        ))
        return result

    def stream(self, system: str, user: str) -> CompletionStream:
        _log.info("OpenAI stream: model=%s, max_tokens=%d", self._model, self._max_tokens)

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
    ollama_timeout: float = 0,
    ollama_think: bool | None = None,
    openai_model: str = "",
    callback: CompletionCallback | None = None,
) -> AIProvider:
    """Return an AIProvider instance.

    Configuration priority (for each setting):
      1. Keyword arguments passed to this function
      2. Environment variables: LLM, AI_PROVIDER, ANTHROPIC_API_KEY,
         OPENAI_API_KEY, OLLAMA_BASE_URL, OLLAMA_MODEL
      3. Built-in defaults

    ``LLM`` is the preferred env var for provider selection (e.g. ``LLM=ollama``).
    ``AI_PROVIDER`` is a supported alias for backwards compatibility.

    Args:
        callback: Optional event callback for observability.  Receives a
            :class:`CompletionEvent` at start, end, error, and retry points.
    """
    name = provider or os.environ.get("LLM") or os.environ.get("AI_PROVIDER", "claude")
    _log.info("Using provider %r", name)
    if name == "claude":
        inst = ClaudeProvider(api_key=api_key)
    elif name == "openai":
        inst = OpenAIProvider(api_key=api_key, model=openai_model or "gpt-4o-mini")
    elif name == "ollama":
        kwargs: dict[str, Any] = {"base_url": ollama_base_url, "model": ollama_model}
        if ollama_timeout > 0:
            kwargs["timeout"] = ollama_timeout
        if ollama_think is not None:
            kwargs["think"] = ollama_think
        inst = OllamaProvider(**kwargs)
    else:
        raise ValueError(
            f"Unknown AI provider: {name!r}. Expected 'claude', 'openai', or 'ollama'."
        )
    if callback:
        inst.set_callback(callback)
    return inst
