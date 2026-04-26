from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from llm_provider import (
    AIProvider,
    ClaudeProvider,
    CompletionResult,
    CompletionStream,
    CompletionUsage,
    OllamaProvider,
    OpenAIProvider,
    Prompt,
    get_provider,
)
from llm_provider.provider import _extract_json_object, _retry


def _has_openai() -> bool:
    try:
        import openai  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _StubProvider(AIProvider):
    """Concrete provider that returns a fixed string for testing."""

    def __init__(self, response: str) -> None:
        self._response = response

    def complete(self, system: str, user: str) -> str:
        return self._response


# ---------------------------------------------------------------------------
# CompletionResult / CompletionUsage
# ---------------------------------------------------------------------------

class TestCompletionResult:
    def test_is_str(self) -> None:
        r = CompletionResult("hello")
        assert isinstance(r, str)

    def test_equality(self) -> None:
        assert CompletionResult("hello") == "hello"

    def test_metadata_accessible(self) -> None:
        usage = CompletionUsage(prompt_tokens=10, completion_tokens=20)
        r = CompletionResult("hi", usage=usage, model="m1", stop_reason="end_turn")
        assert r.usage == usage
        assert r.usage.prompt_tokens == 10
        assert r.usage.completion_tokens == 20
        assert r.model == "m1"
        assert r.stop_reason == "end_turn"

    def test_defaults(self) -> None:
        r = CompletionResult("hi")
        assert r.usage is None
        assert r.model == ""
        assert r.stop_reason is None

    def test_string_operations(self) -> None:
        r = CompletionResult("hello world", model="m1")
        assert r.upper() == "HELLO WORLD"
        assert r.split() == ["hello", "world"]
        assert len(r) == 11


# ---------------------------------------------------------------------------
# CompletionStream
# ---------------------------------------------------------------------------

class TestCompletionStream:
    def test_yields_chunks(self) -> None:
        stream = CompletionStream(iter(["hello ", "world"]))
        chunks = list(stream)
        assert chunks == ["hello ", "world"]

    def test_result_after_exhaustion(self) -> None:
        stream = CompletionStream(iter(["a", "b", "c"]), model="test-model")
        list(stream)  # exhaust
        r = stream.result
        assert r == "abc"
        assert r.model == "test-model"

    def test_result_auto_consumes(self) -> None:
        stream = CompletionStream(iter(["hello"]))
        r = stream.result
        assert r == "hello"

    def test_finalizer(self) -> None:
        def _fin(text: str) -> CompletionResult:
            return CompletionResult(text, model="fin-model", stop_reason="stop")

        stream = CompletionStream(iter(["x", "y"]), finalizer=_fin)
        list(stream)
        assert stream.result.model == "fin-model"
        assert stream.result.stop_reason == "stop"

    def test_default_stream_from_abc(self) -> None:
        p = _StubProvider("full text")
        stream = p.stream("s", "u")
        chunks = list(stream)
        assert chunks == ["full text"]
        assert stream.result == "full text"


# ---------------------------------------------------------------------------
# get_provider factory
# ---------------------------------------------------------------------------

class TestGetProvider:
    @patch("llm_provider.provider.ClaudeProvider")
    def test_defaults_to_claude(self, mock_cls: MagicMock) -> None:
        get_provider(api_key="sk-test")
        mock_cls.assert_called_once_with(api_key="sk-test")

    @patch("llm_provider.provider.OllamaProvider")
    def test_ollama(self, mock_cls: MagicMock) -> None:
        get_provider("ollama", ollama_base_url="http://host:1234", ollama_model="m1")
        mock_cls.assert_called_once_with(base_url="http://host:1234", model="m1")

    @patch("llm_provider.provider.OllamaProvider")
    def test_ollama_with_timeout(self, mock_cls: MagicMock) -> None:
        get_provider("ollama", ollama_base_url="http://host:1234", ollama_model="m1", ollama_timeout=300.0)
        mock_cls.assert_called_once_with(base_url="http://host:1234", model="m1", timeout=300.0)

    @patch("llm_provider.provider.OllamaProvider")
    def test_ollama_with_think(self, mock_cls: MagicMock) -> None:
        get_provider("ollama", ollama_model="m1", ollama_think=False)
        mock_cls.assert_called_once_with(base_url="", model="m1", think=False)

    @patch("llm_provider.provider.OpenAIProvider")
    def test_openai(self, mock_cls: MagicMock) -> None:
        get_provider("openai", api_key="sk-test", openai_model="gpt-4o")
        mock_cls.assert_called_once_with(api_key="sk-test", model="gpt-4o")

    @patch("llm_provider.provider.OpenAIProvider")
    def test_openai_default_model(self, mock_cls: MagicMock) -> None:
        get_provider("openai", api_key="sk-test")
        mock_cls.assert_called_once_with(api_key="sk-test", model="gpt-4o-mini")

    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown AI provider"):
            get_provider("nope")

    @patch("llm_provider.provider.OllamaProvider")
    def test_from_env(self, mock_cls: MagicMock, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AI_PROVIDER", "ollama")
        get_provider()
        mock_cls.assert_called_once()

    @patch("llm_provider.provider.OllamaProvider")
    def test_from_llm_env(self, mock_cls: MagicMock, monkeypatch: pytest.MonkeyPatch) -> None:
        # LLM env var is the preferred key; AI_PROVIDER is an alias
        monkeypatch.setenv("LLM", "ollama")
        get_provider()
        mock_cls.assert_called_once()

    @patch("llm_provider.provider.OllamaProvider")
    def test_llm_env_takes_precedence_over_ai_provider(self, mock_cls: MagicMock, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM", "ollama")
        monkeypatch.setenv("AI_PROVIDER", "claude")
        get_provider()
        mock_cls.assert_called_once()


# ---------------------------------------------------------------------------
# ClaudeProvider
# ---------------------------------------------------------------------------

class TestClaudeProvider:
    def test_empty_api_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(ValueError, match="API key required"):
            ClaudeProvider(api_key="")

    @patch("llm_provider.provider.anthropic", create=True)
    def test_complete(self, mock_anthropic: MagicMock, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_msg = MagicMock()
        mock_msg.content = [MagicMock(text="hello")]
        mock_msg.usage.input_tokens = 5
        mock_msg.usage.output_tokens = 3
        mock_msg.model = "claude-haiku-4-5-20251001"
        mock_msg.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_msg

        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            p = ClaudeProvider(api_key="sk-test")
            result = p.complete("system", "user")

        assert result == "hello"
        assert result.usage.prompt_tokens == 5
        assert result.usage.completion_tokens == 3
        assert result.model == "claude-haiku-4-5-20251001"
        assert result.stop_reason == "end_turn"
        mock_client.messages.create.assert_called_once()

    @patch("llm_provider.provider.anthropic", create=True)
    def test_stream(self, mock_anthropic: MagicMock, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client

        # Mock stream context manager
        mock_stream = MagicMock()
        mock_stream.text_stream = iter(["hel", "lo"])
        mock_final = MagicMock()
        mock_final.content = [MagicMock(text="hello")]
        mock_final.usage.input_tokens = 4
        mock_final.usage.output_tokens = 2
        mock_final.model = "claude-haiku-4-5-20251001"
        mock_final.stop_reason = "end_turn"
        mock_stream.get_final_message.return_value = mock_final

        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_stream)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_client.messages.stream.return_value = mock_ctx

        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            p = ClaudeProvider(api_key="sk-test")
            stream = p.stream("system", "user")
            chunks = list(stream)

        assert chunks == ["hel", "lo"]
        assert stream.result == "hello"
        assert stream.result.usage.prompt_tokens == 4


# ---------------------------------------------------------------------------
# OllamaProvider
# ---------------------------------------------------------------------------

class TestOllamaProvider:
    def _make_provider(self, monkeypatch: pytest.MonkeyPatch, mock_client: MagicMock) -> OllamaProvider:
        import httpx

        monkeypatch.setattr(httpx, "Client", lambda **kw: mock_client)
        return OllamaProvider(base_url="http://localhost:11434", model="llama3")

    def test_complete(self, monkeypatch: pytest.MonkeyPatch) -> None:
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
        mock_client.post.return_value = mock_resp

        p = self._make_provider(monkeypatch, mock_client)
        result = p.complete("sys", "usr")

        assert result == "world"
        assert result.usage.prompt_tokens == 8
        assert result.usage.completion_tokens == 5
        assert result.model == "llama3"
        assert result.stop_reason == "stop"
        mock_client.post.assert_called_once()

    def test_complete_sends_chat_messages(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "message": {"role": "assistant", "content": "ok"},
            "model": "llama3",
        }
        mock_resp.raise_for_status = MagicMock()
        mock_client.post.return_value = mock_resp

        p = self._make_provider(monkeypatch, mock_client)
        p.complete("system prompt", "user prompt")

        call_args = mock_client.post.call_args
        assert "/api/chat" in call_args[0][0]
        payload = call_args[1]["json"]
        assert payload["messages"] == [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "user prompt"},
        ]

    def test_complete_no_token_counts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "message": {"role": "assistant", "content": "hi"},
            "model": "llama3",
        }
        mock_resp.raise_for_status = MagicMock()
        mock_client.post.return_value = mock_resp

        p = self._make_provider(monkeypatch, mock_client)
        result = p.complete("sys", "usr")

        assert result == "hi"
        assert result.usage is None

    def test_unreachable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import httpx

        mock_client = MagicMock()
        mock_client.post.side_effect = httpx.ConnectError("refused")

        p = self._make_provider(monkeypatch, mock_client)
        with pytest.raises(ConnectionError, match="Cannot reach Ollama"):
            p.complete("sys", "usr")

    def test_context_manager(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_client = MagicMock()
        p = self._make_provider(monkeypatch, mock_client)

        with p:
            pass

        mock_client.close.assert_called_once()

    def test_close_idempotent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_client = MagicMock()
        p = self._make_provider(monkeypatch, mock_client)

        p.close()
        p.close()
        mock_client.close.assert_called_once()

    def test_custom_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import httpx

        captured_kwargs: dict = {}
        original_client = httpx.Client

        def _capture_client(**kw: object) -> MagicMock:
            captured_kwargs.update(kw)
            return MagicMock()

        monkeypatch.setattr(httpx, "Client", _capture_client)
        p = OllamaProvider(base_url="http://localhost:11434", model="llama3", timeout=300.0)
        assert captured_kwargs["timeout"] == 300.0
        assert p._timeout == 300.0

    def test_default_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import httpx

        captured_kwargs: dict = {}

        def _capture_client(**kw: object) -> MagicMock:
            captured_kwargs.update(kw)
            return MagicMock()

        monkeypatch.setattr(httpx, "Client", _capture_client)
        OllamaProvider(base_url="http://localhost:11434", model="llama3")
        assert captured_kwargs["timeout"] == 120.0

    def test_timeout_is_retryable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import httpx

        monkeypatch.setattr("llm_provider.provider.time.sleep", lambda _: None)
        mock_client = MagicMock()
        call_count = 0

        def _post(*args: object, **kwargs: object) -> MagicMock:
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

        mock_client.post.side_effect = _post
        monkeypatch.setattr(httpx, "Client", lambda **kw: mock_client)
        p = OllamaProvider(base_url="http://localhost:11434", model="llama3")
        result = p.complete("sys", "usr")
        assert result == "ok"
        assert call_count == 3

    def test_think_parameter(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "message": {"role": "assistant", "content": "ok"},
            "model": "llama3",
        }
        mock_resp.raise_for_status = MagicMock()
        mock_client.post.return_value = mock_resp

        monkeypatch.setattr("httpx.Client", lambda **kw: mock_client)
        p = OllamaProvider(base_url="http://localhost:11434", model="llama3", think=False)
        p.complete("sys", "usr")

        payload = mock_client.post.call_args[1]["json"]
        assert payload["think"] is False

    def test_think_omitted_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "message": {"role": "assistant", "content": "ok"},
            "model": "llama3",
        }
        mock_resp.raise_for_status = MagicMock()
        mock_client.post.return_value = mock_resp

        monkeypatch.setattr("httpx.Client", lambda **kw: mock_client)
        p = OllamaProvider(base_url="http://localhost:11434", model="llama3")
        p.complete("sys", "usr")

        payload = mock_client.post.call_args[1]["json"]
        assert "think" not in payload


# ---------------------------------------------------------------------------
# OpenAIProvider
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not _has_openai(),
    reason="openai package not installed",
)
class TestOpenAIProvider:
    def test_empty_api_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="API key required"):
            OpenAIProvider(api_key="")

    @patch("llm_provider.provider.openai", create=True)
    def test_complete(self, mock_openai: MagicMock, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_client

        mock_choice = MagicMock()
        mock_choice.message.content = "hi there"
        mock_choice.finish_reason = "stop"
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_resp.model = "gpt-4o-mini"
        mock_resp.usage.prompt_tokens = 12
        mock_resp.usage.completion_tokens = 7
        mock_client.chat.completions.create.return_value = mock_resp

        with patch.dict("sys.modules", {"openai": mock_openai}):
            p = OpenAIProvider(api_key="sk-test")
            result = p.complete("system", "user")

        assert result == "hi there"
        assert result.usage.prompt_tokens == 12
        assert result.usage.completion_tokens == 7
        assert result.model == "gpt-4o-mini"
        assert result.stop_reason == "stop"
        mock_client.chat.completions.create.assert_called_once()

    @patch("llm_provider.provider.openai", create=True)
    def test_complete_no_usage(self, mock_openai: MagicMock, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_client

        mock_choice = MagicMock()
        mock_choice.message.content = "hi"
        mock_choice.finish_reason = "stop"
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_resp.model = "gpt-4o-mini"
        mock_resp.usage = None
        mock_client.chat.completions.create.return_value = mock_resp

        with patch.dict("sys.modules", {"openai": mock_openai}):
            p = OpenAIProvider(api_key="sk-test")
            result = p.complete("system", "user")

        assert result == "hi"
        assert result.usage is None


# ---------------------------------------------------------------------------
# Prompt dataclass
# ---------------------------------------------------------------------------

class TestPrompt:
    def test_fields(self) -> None:
        p = Prompt(system="sys", user="usr")
        assert p.system == "sys"
        assert p.user == "usr"

    def test_usable_as_kwargs(self) -> None:
        p = Prompt(system="sys", user="usr")
        stub = _StubProvider("response")
        result = stub.complete(p.system, p.user)
        assert result == "response"


# ---------------------------------------------------------------------------
# complete_model / Pydantic parsing
# ---------------------------------------------------------------------------

class TestCompleteModel:
    def test_valid_model(self) -> None:
        from pydantic import BaseModel

        class Book(BaseModel):
            title: str
            author: str

        p = _StubProvider('{"title": "Dune", "author": "Herbert"}')
        result = p.complete_model("s", "u", Book)
        assert result.title == "Dune"
        assert result.author == "Herbert"

    def test_fenced_json_model(self) -> None:
        from pydantic import BaseModel

        class Item(BaseModel):
            value: int

        p = _StubProvider('```json\n{"value": 42}\n```')
        result = p.complete_model("s", "u", Item)
        assert result.value == 42

    def test_non_basemodel_raises(self) -> None:
        class NotAModel:
            pass

        p = _StubProvider("{}")
        with pytest.raises(TypeError, match="Pydantic BaseModel subclass"):
            p.complete_model("s", "u", NotAModel)  # type: ignore[arg-type]

    def test_validation_error_propagates(self) -> None:
        from pydantic import BaseModel, ValidationError

        class Strict(BaseModel):
            count: int

        p = _StubProvider('{"count": "not-an-int"}')
        with pytest.raises(ValidationError):
            p.complete_model("s", "u", Strict)


# ---------------------------------------------------------------------------
# complete_json / fence stripping
# ---------------------------------------------------------------------------

class TestCompleteJson:
    def test_plain_json(self) -> None:
        p = _StubProvider('{"key": "value"}')
        assert p.complete_json("s", "u") == {"key": "value"}

    def test_fenced_json(self) -> None:
        p = _StubProvider('```json\n{"key": "value"}\n```')
        assert p.complete_json("s", "u") == {"key": "value"}

    def test_fenced_no_lang(self) -> None:
        p = _StubProvider('```\n{"key": "value"}\n```')
        assert p.complete_json("s", "u") == {"key": "value"}

    def test_invalid_json(self) -> None:
        p = _StubProvider("not json")
        with pytest.raises(json.JSONDecodeError):
            p.complete_json("s", "u")

    def test_whitespace_padding(self) -> None:
        p = _StubProvider('  \n{"key": "value"}\n  ')
        assert p.complete_json("s", "u") == {"key": "value"}

    def test_preamble_then_json(self) -> None:
        p = _StubProvider('Sure, here is the JSON:\n{"key": "value"}')
        assert p.complete_json("s", "u") == {"key": "value"}

    def test_preamble_then_fenced(self) -> None:
        p = _StubProvider('Here you go:\n```json\n{"key": "value"}\n```')
        assert p.complete_json("s", "u") == {"key": "value"}

    def test_nested_braces(self) -> None:
        p = _StubProvider('{"task": "test", "meta": {"source": "ai"}}')
        result = p.complete_json("s", "u")
        assert result["task"] == "test"
        assert result["meta"]["source"] == "ai"

    def test_multiline_notes_in_fences(self) -> None:
        """The real-world case: bullet-point notes inside fenced JSON."""
        payload = {
            "task": "Fix the sink",
            "notes": "- Turn off water supply\n- Check washer\n- Have plumber's tape ready",
        }
        raw = f"Sure, here is the JSON:\n```json\n{json.dumps(payload, indent=2)}\n```"
        p = _StubProvider(raw)
        result = p.complete_json("s", "u")
        assert result["task"] == "Fix the sink"
        assert "washer" in result["notes"]

    def test_unbalanced_braces_raises(self) -> None:
        p = _StubProvider('{"key": "value"')
        with pytest.raises(json.JSONDecodeError):
            p.complete_json("s", "u")


# ---------------------------------------------------------------------------
# _extract_json_object
# ---------------------------------------------------------------------------


class TestExtractJsonObject:
    def test_simple(self) -> None:
        assert _extract_json_object('{"key": "value"}') == '{"key": "value"}'

    def test_nested(self) -> None:
        text = 'Here is the result: {"task": "Buy groceries", "meta": {"source": "ai"}}'
        result = _extract_json_object(text)
        parsed = json.loads(result)
        assert parsed["task"] == "Buy groceries"
        assert parsed["meta"]["source"] == "ai"

    def test_escaped_quotes(self) -> None:
        text = r'{"notes": "Use \"organic\" if possible"}'
        result = _extract_json_object(text)
        assert result is not None
        parsed = json.loads(result)
        assert "organic" in parsed["notes"]

    def test_no_json(self) -> None:
        assert _extract_json_object("no json here") is None


# ---------------------------------------------------------------------------
# _retry
# ---------------------------------------------------------------------------

class TestRetry:
    def test_succeeds_on_second_attempt(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("llm_provider.provider.time.sleep", lambda _: None)
        calls = 0

        def _fn() -> str:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise ValueError("transient")
            return "ok"

        result = _retry(_fn, max_attempts=3, base_delay=1.0, retryable=(ValueError,))
        assert result == "ok"
        assert calls == 2

    def test_exhausted_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("llm_provider.provider.time.sleep", lambda _: None)

        def _fn() -> str:
            raise ValueError("always fails")

        with pytest.raises(ValueError, match="always fails"):
            _retry(_fn, max_attempts=2, base_delay=1.0, retryable=(ValueError,))

    def test_non_retryable_raises_immediately(self) -> None:
        calls = 0

        def _fn() -> str:
            nonlocal calls
            calls += 1
            raise TypeError("not retryable")

        with pytest.raises(TypeError):
            _retry(_fn, max_attempts=3, base_delay=1.0, retryable=(ValueError,))
        assert calls == 1

    def test_ollama_retry_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import httpx

        mock_client = MagicMock()
        mock_client.post.side_effect = httpx.ConnectError("refused")
        monkeypatch.setattr(httpx, "Client", lambda **kw: mock_client)

        p = OllamaProvider(base_url="http://localhost:11434", model="llama3", max_retries=0)
        with pytest.raises(ConnectionError, match="Cannot reach Ollama"):
            p.complete("sys", "usr")
        assert mock_client.post.call_count == 1


# ---------------------------------------------------------------------------
# list_models
# ---------------------------------------------------------------------------

from llm_provider.provider import CLAUDE_MODELS, OPENAI_MODELS, list_models


class TestListModels:
    def test_claude_returns_list(self) -> None:
        models = list_models("claude")
        assert isinstance(models, list)
        assert "claude-sonnet-4-6" in models
        assert "claude-opus-4-7" in models
        assert "claude-haiku-4-5-20251001" in models

    def test_claude_returns_copy(self) -> None:
        # mutating the return value must not affect the constant
        models = list_models("claude")
        models.clear()
        assert len(CLAUDE_MODELS) > 0

    def test_openai_returns_list(self) -> None:
        models = list_models("openai")
        assert isinstance(models, list)
        assert "gpt-4o-mini" in models
        assert "gpt-5.4" in models

    def test_openai_returns_copy(self) -> None:
        models = list_models("openai")
        models.clear()
        assert len(OPENAI_MODELS) > 0

    def test_ollama_returns_names_on_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import httpx

        fake_response = MagicMock()
        fake_response.json.return_value = {
            "models": [{"name": "qwen2.5:3b"}, {"name": "llama3:8b"}]
        }
        monkeypatch.setattr(httpx, "get", lambda *a, **kw: fake_response)
        result = list_models("ollama")
        assert result == ["qwen2.5:3b", "llama3:8b"]

    def test_ollama_returns_empty_on_connection_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import httpx

        def _raise(*a, **kw):
            raise httpx.ConnectError("refused")

        monkeypatch.setattr(httpx, "get", _raise)
        result = list_models("ollama")
        assert result == []

    def test_ollama_uses_env_base_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import httpx

        captured: list[str] = []
        fake_response = MagicMock()
        fake_response.json.return_value = {"models": []}

        def _fake_get(url: str, **kw):
            captured.append(url)
            return fake_response

        monkeypatch.setenv("OLLAMA_BASE_URL", "http://myhost:9999")
        monkeypatch.setattr(httpx, "get", _fake_get)
        list_models("ollama")
        assert captured[0].startswith("http://myhost:9999")

    def test_ollama_kwarg_overrides_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import httpx

        captured: list[str] = []
        fake_response = MagicMock()
        fake_response.json.return_value = {"models": []}

        def _fake_get(url: str, **kw):
            captured.append(url)
            return fake_response

        monkeypatch.setenv("OLLAMA_BASE_URL", "http://envhost:8888")
        monkeypatch.setattr(httpx, "get", _fake_get)
        list_models("ollama", ollama_base_url="http://override:1234")
        assert captured[0].startswith("http://override:1234")

    def test_unknown_provider_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown provider"):
            list_models("nope")
