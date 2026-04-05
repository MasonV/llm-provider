from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_provider import (
    AsyncAIProvider,
    AsyncClaudeProvider,
    AsyncCompletionStream,
    AsyncOllamaProvider,
    AsyncOpenAIProvider,
    CompletionResult,
    CompletionUsage,
    get_async_provider,
)
from llm_provider.async_provider import _async_retry
from llm_provider.provider import _parse_json_response, _validate_model_class


def _has_openai() -> bool:
    try:
        import openai  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _AsyncStubProvider(AsyncAIProvider):
    """Concrete async provider that returns a fixed string for testing."""

    def __init__(self, response: str) -> None:
        self._response = response

    async def complete(self, system: str, user: str) -> str:
        return self._response


async def _async_iter(items: list[str]):
    """Helper to create an async iterator from a list."""
    for item in items:
        yield item


# ---------------------------------------------------------------------------
# _parse_json_response (shared helper)
# ---------------------------------------------------------------------------

class TestParseJsonResponse:
    def test_plain_json(self) -> None:
        assert _parse_json_response('{"key": "value"}') == {"key": "value"}

    def test_fenced_json(self) -> None:
        assert _parse_json_response('```json\n{"key": "value"}\n```') == {"key": "value"}

    def test_preamble_then_json(self) -> None:
        assert _parse_json_response('Here:\n{"key": "value"}') == {"key": "value"}

    def test_whitespace(self) -> None:
        assert _parse_json_response('  \n{"key": "value"}\n  ') == {"key": "value"}

    def test_invalid_raises(self) -> None:
        with pytest.raises(json.JSONDecodeError):
            _parse_json_response("not json")


# ---------------------------------------------------------------------------
# _validate_model_class (shared helper)
# ---------------------------------------------------------------------------

class TestValidateModelClass:
    def test_valid_model(self) -> None:
        from pydantic import BaseModel

        class Foo(BaseModel):
            x: int

        # Should not raise
        _validate_model_class(Foo)

    def test_non_basemodel_raises(self) -> None:
        class NotAModel:
            pass

        with pytest.raises(TypeError, match="Pydantic BaseModel subclass"):
            _validate_model_class(NotAModel)

    def test_non_type_raises(self) -> None:
        with pytest.raises(TypeError, match="Pydantic BaseModel subclass"):
            _validate_model_class("not a type")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# AsyncCompletionStream
# ---------------------------------------------------------------------------

class TestAsyncCompletionStream:
    @pytest.mark.asyncio
    async def test_yields_chunks(self) -> None:
        stream = AsyncCompletionStream(_async_iter(["hello ", "world"]))
        chunks = [chunk async for chunk in stream]
        assert chunks == ["hello ", "world"]

    @pytest.mark.asyncio
    async def test_result_after_exhaustion(self) -> None:
        stream = AsyncCompletionStream(_async_iter(["a", "b", "c"]), model="test-model")
        async for _ in stream:
            pass
        r = await stream.get_result()
        assert r == "abc"
        assert r.model == "test-model"

    @pytest.mark.asyncio
    async def test_get_result_auto_consumes(self) -> None:
        stream = AsyncCompletionStream(_async_iter(["hello"]))
        r = await stream.get_result()
        assert r == "hello"

    @pytest.mark.asyncio
    async def test_finalizer(self) -> None:
        def _fin(text: str) -> CompletionResult:
            return CompletionResult(text, model="fin-model", stop_reason="stop")

        stream = AsyncCompletionStream(_async_iter(["x", "y"]), finalizer=_fin)
        async for _ in stream:
            pass
        r = await stream.get_result()
        assert r.model == "fin-model"
        assert r.stop_reason == "stop"

    @pytest.mark.asyncio
    async def test_default_stream_from_abc(self) -> None:
        p = _AsyncStubProvider("full text")
        stream = await p.stream("s", "u")
        chunks = [chunk async for chunk in stream]
        assert chunks == ["full text"]
        r = await stream.get_result()
        assert r == "full text"


# ---------------------------------------------------------------------------
# _async_retry
# ---------------------------------------------------------------------------

class TestAsyncRetry:
    @pytest.mark.asyncio
    async def test_succeeds_on_second_attempt(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("llm_provider.async_provider.asyncio.sleep", AsyncMock())
        calls = 0

        async def _fn() -> str:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise ValueError("transient")
            return "ok"

        result = await _async_retry(_fn, max_attempts=3, base_delay=1.0, retryable=(ValueError,))
        assert result == "ok"
        assert calls == 2

    @pytest.mark.asyncio
    async def test_exhausted_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("llm_provider.async_provider.asyncio.sleep", AsyncMock())

        async def _fn() -> str:
            raise ValueError("always fails")

        with pytest.raises(ValueError, match="always fails"):
            await _async_retry(_fn, max_attempts=2, base_delay=1.0, retryable=(ValueError,))

    @pytest.mark.asyncio
    async def test_non_retryable_raises_immediately(self) -> None:
        calls = 0

        async def _fn() -> str:
            nonlocal calls
            calls += 1
            raise TypeError("not retryable")

        with pytest.raises(TypeError):
            await _async_retry(_fn, max_attempts=3, base_delay=1.0, retryable=(ValueError,))
        assert calls == 1


# ---------------------------------------------------------------------------
# get_async_provider factory
# ---------------------------------------------------------------------------

class TestGetAsyncProvider:
    @patch("llm_provider.async_provider.AsyncClaudeProvider")
    def test_defaults_to_claude(self, mock_cls: MagicMock) -> None:
        get_async_provider(api_key="sk-test")
        mock_cls.assert_called_once_with(api_key="sk-test")

    @patch("llm_provider.async_provider.AsyncOllamaProvider")
    def test_ollama(self, mock_cls: MagicMock) -> None:
        get_async_provider("ollama", ollama_base_url="http://host:1234", ollama_model="m1")
        mock_cls.assert_called_once_with(base_url="http://host:1234", model="m1")

    @patch("llm_provider.async_provider.AsyncOllamaProvider")
    def test_ollama_with_timeout(self, mock_cls: MagicMock) -> None:
        get_async_provider("ollama", ollama_base_url="http://host:1234", ollama_model="m1", ollama_timeout=300.0)
        mock_cls.assert_called_once_with(base_url="http://host:1234", model="m1", timeout=300.0)

    @patch("llm_provider.async_provider.AsyncOllamaProvider")
    def test_ollama_with_think(self, mock_cls: MagicMock) -> None:
        get_async_provider("ollama", ollama_model="m1", ollama_think=False)
        mock_cls.assert_called_once_with(base_url="", model="m1", think=False)

    @patch("llm_provider.async_provider.AsyncOpenAIProvider")
    def test_openai(self, mock_cls: MagicMock) -> None:
        get_async_provider("openai", api_key="sk-test", openai_model="gpt-4o")
        mock_cls.assert_called_once_with(api_key="sk-test", model="gpt-4o")

    @patch("llm_provider.async_provider.AsyncOpenAIProvider")
    def test_openai_default_model(self, mock_cls: MagicMock) -> None:
        get_async_provider("openai", api_key="sk-test")
        mock_cls.assert_called_once_with(api_key="sk-test", model="gpt-4o-mini")

    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown AI provider"):
            get_async_provider("nope")

    @patch("llm_provider.async_provider.AsyncOllamaProvider")
    def test_from_env(self, mock_cls: MagicMock, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AI_PROVIDER", "ollama")
        get_async_provider()
        mock_cls.assert_called_once()


# ---------------------------------------------------------------------------
# AsyncClaudeProvider
# ---------------------------------------------------------------------------

class TestAsyncClaudeProvider:
    def test_empty_api_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(ValueError, match="API key required"):
            AsyncClaudeProvider(api_key="")

    @patch("llm_provider.async_provider.anthropic", create=True)
    @pytest.mark.asyncio
    async def test_complete(self, mock_anthropic: MagicMock) -> None:
        mock_client = MagicMock()
        mock_anthropic.AsyncAnthropic.return_value = mock_client

        mock_msg = MagicMock()
        mock_msg.content = [MagicMock(text="hello")]
        mock_msg.usage.input_tokens = 5
        mock_msg.usage.output_tokens = 3
        mock_msg.model = "claude-haiku-4-5-20251001"
        mock_msg.stop_reason = "end_turn"
        mock_client.messages.create = AsyncMock(return_value=mock_msg)

        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            p = AsyncClaudeProvider(api_key="sk-test")
            result = await p.complete("system", "user")

        assert result == "hello"
        assert result.usage.prompt_tokens == 5
        assert result.usage.completion_tokens == 3
        assert result.model == "claude-haiku-4-5-20251001"
        assert result.stop_reason == "end_turn"
        mock_client.messages.create.assert_called_once()

    @patch("llm_provider.async_provider.anthropic", create=True)
    @pytest.mark.asyncio
    async def test_stream(self, mock_anthropic: MagicMock) -> None:
        mock_client = MagicMock()
        mock_anthropic.AsyncAnthropic.return_value = mock_client

        # Mock async stream context manager
        mock_stream = MagicMock()
        mock_stream.text_stream = _async_iter(["hel", "lo"])
        mock_final = MagicMock()
        mock_final.content = [MagicMock(text="hello")]
        mock_final.usage.input_tokens = 4
        mock_final.usage.output_tokens = 2
        mock_final.model = "claude-haiku-4-5-20251001"
        mock_final.stop_reason = "end_turn"
        mock_stream.get_final_message.return_value = mock_final

        mock_ctx = MagicMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_stream)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_client.messages.stream.return_value = mock_ctx

        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            p = AsyncClaudeProvider(api_key="sk-test")
            stream = await p.stream("system", "user")
            chunks = [chunk async for chunk in stream]

        assert chunks == ["hel", "lo"]
        r = await stream.get_result()
        assert r == "hello"
        assert r.usage.prompt_tokens == 4


# ---------------------------------------------------------------------------
# AsyncOllamaProvider
# ---------------------------------------------------------------------------

class TestAsyncOllamaProvider:
    def _make_provider(self, monkeypatch: pytest.MonkeyPatch, mock_client: MagicMock) -> AsyncOllamaProvider:
        import httpx

        monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: mock_client)
        return AsyncOllamaProvider(base_url="http://localhost:11434", model="llama3")

    @pytest.mark.asyncio
    async def test_complete(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "message": {"role": "assistant", "content": "world"},
            "model": "llama3",
            "prompt_eval_count": 8,
            "eval_count": 5,
            "done_reason": "stop",
        }
        mock_resp.raise_for_status = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_resp)

        p = self._make_provider(monkeypatch, mock_client)
        result = await p.complete("sys", "usr")

        assert result == "world"
        assert result.usage.prompt_tokens == 8
        assert result.usage.completion_tokens == 5
        assert result.model == "llama3"
        assert result.stop_reason == "stop"
        mock_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_complete_sends_chat_messages(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "message": {"role": "assistant", "content": "ok"},
            "model": "llama3",
        }
        mock_resp.raise_for_status = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_resp)

        p = self._make_provider(monkeypatch, mock_client)
        await p.complete("system prompt", "user prompt")

        call_args = mock_client.post.call_args
        assert "/api/chat" in call_args[0][0]
        payload = call_args[1]["json"]
        assert payload["messages"] == [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "user prompt"},
        ]

    @pytest.mark.asyncio
    async def test_complete_no_token_counts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "message": {"role": "assistant", "content": "hi"},
            "model": "llama3",
        }
        mock_resp.raise_for_status = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_resp)

        p = self._make_provider(monkeypatch, mock_client)
        result = await p.complete("sys", "usr")

        assert result == "hi"
        assert result.usage is None

    @pytest.mark.asyncio
    async def test_unreachable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import httpx

        mock_client = MagicMock()
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("refused"))

        p = self._make_provider(monkeypatch, mock_client)
        with pytest.raises(ConnectionError, match="Cannot reach Ollama"):
            await p.complete("sys", "usr")

    @pytest.mark.asyncio
    async def test_context_manager(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_client = MagicMock()
        mock_client.aclose = AsyncMock()

        p = self._make_provider(monkeypatch, mock_client)
        async with p:
            pass

        mock_client.aclose.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_idempotent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_client = MagicMock()
        mock_client.aclose = AsyncMock()

        p = self._make_provider(monkeypatch, mock_client)
        await p.close()
        await p.close()
        mock_client.aclose.assert_called_once()

    def test_custom_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import httpx

        captured_kwargs: dict = {}

        def _capture_client(**kw: object) -> MagicMock:
            captured_kwargs.update(kw)
            return MagicMock()

        monkeypatch.setattr(httpx, "AsyncClient", _capture_client)
        p = AsyncOllamaProvider(base_url="http://localhost:11434", model="llama3", timeout=300.0)
        assert captured_kwargs["timeout"] == 300.0
        assert p._timeout == 300.0

    def test_default_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import httpx

        captured_kwargs: dict = {}

        def _capture_client(**kw: object) -> MagicMock:
            captured_kwargs.update(kw)
            return MagicMock()

        monkeypatch.setattr(httpx, "AsyncClient", _capture_client)
        AsyncOllamaProvider(base_url="http://localhost:11434", model="llama3")
        assert captured_kwargs["timeout"] == 120.0

    @pytest.mark.asyncio
    async def test_timeout_is_retryable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import httpx

        monkeypatch.setattr("llm_provider.async_provider.asyncio.sleep", AsyncMock())
        mock_client = MagicMock()
        call_count = 0

        async def _post(*args: object, **kwargs: object) -> MagicMock:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise httpx.ReadTimeout("timed out")
            resp = MagicMock()
            resp.json.return_value = {
                "message": {"role": "assistant", "content": "ok"},
                "model": "llama3",
            }
            resp.raise_for_status = MagicMock()
            return resp

        mock_client.post = _post
        monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: mock_client)
        p = AsyncOllamaProvider(base_url="http://localhost:11434", model="llama3")
        result = await p.complete("sys", "usr")
        assert result == "ok"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_think_parameter(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "message": {"role": "assistant", "content": "ok"},
            "model": "llama3",
        }
        mock_resp.raise_for_status = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_resp)

        monkeypatch.setattr("httpx.AsyncClient", lambda **kw: mock_client)
        p = AsyncOllamaProvider(base_url="http://localhost:11434", model="llama3", think=False)
        await p.complete("sys", "usr")

        payload = mock_client.post.call_args[1]["json"]
        assert payload["think"] is False

    @pytest.mark.asyncio
    async def test_think_omitted_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "message": {"role": "assistant", "content": "ok"},
            "model": "llama3",
        }
        mock_resp.raise_for_status = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_resp)

        monkeypatch.setattr("httpx.AsyncClient", lambda **kw: mock_client)
        p = AsyncOllamaProvider(base_url="http://localhost:11434", model="llama3")
        await p.complete("sys", "usr")

        payload = mock_client.post.call_args[1]["json"]
        assert "think" not in payload


# ---------------------------------------------------------------------------
# AsyncOpenAIProvider
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not _has_openai(),
    reason="openai package not installed",
)
class TestAsyncOpenAIProvider:
    def test_empty_api_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="API key required"):
            AsyncOpenAIProvider(api_key="")

    @patch("llm_provider.async_provider.openai", create=True)
    @pytest.mark.asyncio
    async def test_complete(self, mock_openai: MagicMock) -> None:
        mock_client = MagicMock()
        mock_openai.AsyncOpenAI.return_value = mock_client

        mock_choice = MagicMock()
        mock_choice.message.content = "hi there"
        mock_choice.finish_reason = "stop"
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_resp.model = "gpt-4o-mini"
        mock_resp.usage.prompt_tokens = 12
        mock_resp.usage.completion_tokens = 7
        mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)

        with patch.dict("sys.modules", {"openai": mock_openai}):
            p = AsyncOpenAIProvider(api_key="sk-test")
            result = await p.complete("system", "user")

        assert result == "hi there"
        assert result.usage.prompt_tokens == 12
        assert result.usage.completion_tokens == 7
        assert result.model == "gpt-4o-mini"
        assert result.stop_reason == "stop"
        mock_client.chat.completions.create.assert_called_once()

    @patch("llm_provider.async_provider.openai", create=True)
    @pytest.mark.asyncio
    async def test_complete_no_usage(self, mock_openai: MagicMock) -> None:
        mock_client = MagicMock()
        mock_openai.AsyncOpenAI.return_value = mock_client

        mock_choice = MagicMock()
        mock_choice.message.content = "hi"
        mock_choice.finish_reason = "stop"
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_resp.model = "gpt-4o-mini"
        mock_resp.usage = None
        mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)

        with patch.dict("sys.modules", {"openai": mock_openai}):
            p = AsyncOpenAIProvider(api_key="sk-test")
            result = await p.complete("system", "user")

        assert result == "hi"
        assert result.usage is None


# ---------------------------------------------------------------------------
# Async complete_json / complete_model
# ---------------------------------------------------------------------------

class TestAsyncCompleteJson:
    @pytest.mark.asyncio
    async def test_plain_json(self) -> None:
        p = _AsyncStubProvider('{"key": "value"}')
        assert await p.complete_json("s", "u") == {"key": "value"}

    @pytest.mark.asyncio
    async def test_fenced_json(self) -> None:
        p = _AsyncStubProvider('```json\n{"key": "value"}\n```')
        assert await p.complete_json("s", "u") == {"key": "value"}

    @pytest.mark.asyncio
    async def test_invalid_json(self) -> None:
        p = _AsyncStubProvider("not json")
        with pytest.raises(json.JSONDecodeError):
            await p.complete_json("s", "u")

    @pytest.mark.asyncio
    async def test_preamble_then_json(self) -> None:
        p = _AsyncStubProvider('Sure, here is the JSON:\n{"key": "value"}')
        assert await p.complete_json("s", "u") == {"key": "value"}


class TestAsyncCompleteModel:
    @pytest.mark.asyncio
    async def test_valid_model(self) -> None:
        from pydantic import BaseModel

        class Book(BaseModel):
            title: str
            author: str

        p = _AsyncStubProvider('{"title": "Dune", "author": "Herbert"}')
        result = await p.complete_model("s", "u", Book)
        assert result.title == "Dune"
        assert result.author == "Herbert"

    @pytest.mark.asyncio
    async def test_non_basemodel_raises(self) -> None:
        class NotAModel:
            pass

        p = _AsyncStubProvider("{}")
        with pytest.raises(TypeError, match="Pydantic BaseModel subclass"):
            await p.complete_model("s", "u", NotAModel)  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_validation_error_propagates(self) -> None:
        from pydantic import BaseModel, ValidationError

        class Strict(BaseModel):
            count: int

        p = _AsyncStubProvider('{"count": "not-an-int"}')
        with pytest.raises(ValidationError):
            await p.complete_model("s", "u", Strict)
