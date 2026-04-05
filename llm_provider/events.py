"""Structured observability events for LLM provider operations.

Consumers can register a callback to receive ``CompletionEvent`` instances
before, after, and on failure of each LLM call.  The callback is synchronous
(a quick dict-like emission, not I/O) so it works in both sync and async
contexts without requiring an async variant.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class CompletionEvent:
    """Immutable event emitted at key points during an LLM completion.

    Attributes:
        phase: Lifecycle stage — ``"start"``, ``"end"``, ``"error"``, or ``"retry"``.
        provider: Provider name (``"claude"``, ``"openai"``, ``"ollama"``).
        model: Model identifier used for the request.
        elapsed_ms: Wall-clock time in milliseconds (set on ``"end"``).
        prompt_tokens: Input token count (set on ``"end"`` when available).
        completion_tokens: Output token count (set on ``"end"`` when available).
        error: Error description (set on ``"error"`` / ``"retry"``).
        attempt: Retry attempt number, 1-indexed (set on ``"retry"``).
        context: Arbitrary consumer-supplied dict passed through unchanged.
            Lets callers attach identifiers (e.g. ``{"video_id": "abc"}``).
    """

    phase: str
    provider: str
    model: str
    elapsed_ms: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    error: str | None = None
    attempt: int | None = None
    context: dict | None = None


CompletionCallback = Callable[[CompletionEvent], None]
"""Signature for the optional event callback accepted by providers."""
