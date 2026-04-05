"""Tests for the CompletionEvent callback system."""
from __future__ import annotations

import time
from dataclasses import FrozenInstanceError
from unittest.mock import MagicMock, patch

import pytest

from llm_provider import (
    AIProvider,
    CompletionCallback,
    CompletionEvent,
    CompletionResult,
    CompletionUsage,
    get_provider,
)
from llm_provider.provider import _retry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _StubProvider(AIProvider):
    """Provider that records calls and returns a fixed CompletionResult."""

    def __init__(self, response: str = "ok", *, usage: CompletionUsage | None = None) -> None:
        self._response = response
        self._usage = usage

    def complete(self, system: str, user: str) -> str:
        result = CompletionResult(
            self._response,
            usage=self._usage,
            model="stub-model",
            stop_reason="end_turn",
        )
        return result


class _TimedStubProvider(AIProvider):
    """Provider that emits start/end events with timing via the base class hooks."""

    def __init__(self) -> None:
        self._usage = CompletionUsage(prompt_tokens=10, completion_tokens=20)

    def complete(self, system: str, user: str) -> str:
        self._emit(CompletionEvent(
            phase="start", provider="stub", model="stub-model",
            context=self._context,
        ))
        result = CompletionResult(
            "response",
            usage=self._usage,
            model="stub-model",
            stop_reason="end_turn",
        )
        self._emit(CompletionEvent(
            phase="end", provider="stub", model="stub-model",
            elapsed_ms=42.0,
            prompt_tokens=10,
            completion_tokens=20,
            context=self._context,
        ))
        return result


# ---------------------------------------------------------------------------
# CompletionEvent construction and immutability
# ---------------------------------------------------------------------------

class TestCompletionEvent:
    def test_required_fields(self) -> None:
        event = CompletionEvent(phase="start", provider="claude", model="haiku")
        assert event.phase == "start"
        assert event.provider == "claude"
        assert event.model == "haiku"

    def test_optional_defaults_are_none(self) -> None:
        event = CompletionEvent(phase="start", provider="claude", model="haiku")
        assert event.elapsed_ms is None
        assert event.prompt_tokens is None
        assert event.completion_tokens is None
        assert event.error is None
        assert event.attempt is None
        assert event.context is None

    def test_all_fields_set(self) -> None:
        ctx = {"video_id": "abc123"}
        event = CompletionEvent(
            phase="end",
            provider="ollama",
            model="qwen2.5:3b",
            elapsed_ms=1500.0,
            prompt_tokens=100,
            completion_tokens=200,
            error=None,
            attempt=None,
            context=ctx,
        )
        assert event.elapsed_ms == 1500.0
        assert event.prompt_tokens == 100
        assert event.completion_tokens == 200
        assert event.context is ctx

    def test_frozen_immutability(self) -> None:
        event = CompletionEvent(phase="start", provider="claude", model="haiku")
        with pytest.raises(FrozenInstanceError):
            event.phase = "end"  # type: ignore[misc]

    def test_retry_event(self) -> None:
        event = CompletionEvent(
            phase="retry",
            provider="claude",
            model="haiku",
            error="rate limit exceeded",
            attempt=2,
        )
        assert event.phase == "retry"
        assert event.error == "rate limit exceeded"
        assert event.attempt == 2

    def test_error_event(self) -> None:
        event = CompletionEvent(
            phase="error",
            provider="openai",
            model="gpt-4o-mini",
            elapsed_ms=500.0,
            error="connection refused",
        )
        assert event.phase == "error"
        assert event.elapsed_ms == 500.0
        assert event.error == "connection refused"


# ---------------------------------------------------------------------------
# Callback registration and invocation
# ---------------------------------------------------------------------------

class TestCallbackRegistration:
    def test_no_callback_by_default(self) -> None:
        provider = _StubProvider("hello")
        assert provider._callback is None
        assert provider._context is None

    def test_set_callback(self) -> None:
        provider = _StubProvider("hello")
        cb = MagicMock()
        provider.set_callback(cb, context={"key": "val"})
        assert provider._callback is cb
        assert provider._context == {"key": "val"}

    def test_clear_callback(self) -> None:
        provider = _StubProvider("hello")
        cb = MagicMock()
        provider.set_callback(cb)
        provider.set_callback(None)
        assert provider._callback is None
        assert provider._context is None

    def test_emit_calls_callback(self) -> None:
        provider = _StubProvider("hello")
        cb = MagicMock()
        provider.set_callback(cb)
        event = CompletionEvent(phase="start", provider="test", model="m1")
        provider._emit(event)
        cb.assert_called_once_with(event)

    def test_emit_noop_without_callback(self) -> None:
        """Emitting without a callback should not raise."""
        provider = _StubProvider("hello")
        event = CompletionEvent(phase="start", provider="test", model="m1")
        provider._emit(event)  # should not raise


class TestCallbackInvocation:
    def test_timed_provider_emits_start_and_end(self) -> None:
        events: list[CompletionEvent] = []
        provider = _TimedStubProvider()
        provider.set_callback(events.append, context={"step": "summarize"})

        provider.complete("sys", "user")

        assert len(events) == 2
        assert events[0].phase == "start"
        assert events[0].context == {"step": "summarize"}
        assert events[1].phase == "end"
        assert events[1].elapsed_ms == 42.0
        assert events[1].prompt_tokens == 10
        assert events[1].completion_tokens == 20
        assert events[1].context == {"step": "summarize"}

    def test_context_passthrough(self) -> None:
        """Context dict is passed through unchanged on every event."""
        events: list[CompletionEvent] = []
        ctx = {"video_id": "abc", "run_id": 42}
        provider = _TimedStubProvider()
        provider.set_callback(events.append, context=ctx)

        provider.complete("sys", "user")

        for event in events:
            assert event.context is ctx


# ---------------------------------------------------------------------------
# _retry emits retry events
# ---------------------------------------------------------------------------

class TestRetryEvents:
    def test_retry_emits_callback_events(self) -> None:
        events: list[CompletionEvent] = []
        call_count = 0

        def _flaky() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("fail")
            return "ok"

        result = _retry(
            _flaky,
            max_attempts=3,
            base_delay=0.01,
            retryable=(ConnectionError,),
            callback=events.append,
            provider_name="ollama",
            model_name="test-model",
            context={"key": "val"},
        )

        assert result == "ok"
        assert len(events) == 2
        assert events[0].phase == "retry"
        assert events[0].attempt == 1
        assert events[0].provider == "ollama"
        assert events[0].model == "test-model"
        assert events[0].error == "fail"
        assert events[0].context == {"key": "val"}
        assert events[1].attempt == 2

    def test_retry_no_callback_still_works(self) -> None:
        """Retry without callback should work as before."""
        call_count = 0

        def _flaky() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("fail")
            return "ok"

        result = _retry(
            _flaky,
            max_attempts=3,
            base_delay=0.01,
            retryable=(ConnectionError,),
        )
        assert result == "ok"

    def test_retry_exhausted_raises(self) -> None:
        events: list[CompletionEvent] = []

        def _always_fail() -> str:
            raise ConnectionError("fail")

        with pytest.raises(ConnectionError):
            _retry(
                _always_fail,
                max_attempts=2,
                base_delay=0.01,
                retryable=(ConnectionError,),
                callback=events.append,
                provider_name="test",
                model_name="m1",
            )

        # Only 1 retry event (attempt 1 fails, retry, attempt 2 fails and raises)
        assert len(events) == 1
        assert events[0].attempt == 1


# ---------------------------------------------------------------------------
# get_provider with callback
# ---------------------------------------------------------------------------

class TestGetProviderCallback:
    def test_callback_wired_to_ollama_provider(self) -> None:
        cb = MagicMock()
        with patch.dict("os.environ", {"AI_PROVIDER": "ollama", "OLLAMA_MODEL": "test"}):
            provider = get_provider(callback=cb)
        assert provider._callback is cb

    def test_no_callback_default(self) -> None:
        with patch.dict("os.environ", {"AI_PROVIDER": "ollama", "OLLAMA_MODEL": "test"}):
            provider = get_provider()
        assert provider._callback is None
