from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from llm_provider import AIProvider, ClaudeProvider, CompletionResult, OllamaProvider, get_provider
from llm_provider.provider import _retry


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

    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown AI provider"):
            get_provider("nope")

    @patch("llm_provider.provider.OllamaProvider")
    def test_from_env(self, mock_cls: MagicMock, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AI_PROVIDER", "ollama")
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
        mock_client.messages.create.return_value = mock_msg

        # Patch the import inside __init__
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            p = ClaudeProvider(api_key="sk-test")
            result = p.complete("system", "user")

        assert result == "hello"
        mock_client.messages.create.assert_called_once()


# ---------------------------------------------------------------------------
# OllamaProvider
# ---------------------------------------------------------------------------

class TestOllamaProvider:
    def _make_provider(self, monkeypatch: pytest.MonkeyPatch, mock_client: MagicMock) -> OllamaProvider:
        import httpx

        monkeypatch.setattr(httpx, "Client", lambda **kw: mock_client)
        return OllamaProvider(base_url="http://localhost:11434", model="llama3")

    _OLLAMA_RESPONSE = {
        "response": "world",
        "model": "llama3",
        "prompt_eval_count": 10,
        "eval_count": 5,
        "done_reason": "stop",
    }

    def test_complete(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = self._OLLAMA_RESPONSE
        mock_resp.raise_for_status = MagicMock()
        mock_client.post.return_value = mock_resp

        p = self._make_provider(monkeypatch, mock_client)
        result = p.complete("sys", "usr")

        assert result == "world"
        mock_client.post.assert_called_once()

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
# complete_with_metadata
# ---------------------------------------------------------------------------

class TestCompleteWithMetadata:
    def test_stub_provider_fallback(self) -> None:
        p = _StubProvider("hello")
        result = p.complete_with_metadata("s", "u")
        assert isinstance(result, CompletionResult)
        assert result.text == "hello"
        assert result.model == "unknown"
        assert result.usage == {}

    @patch("llm_provider.provider.anthropic", create=True)
    def test_claude_metadata(self, mock_anthropic: MagicMock) -> None:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_msg = MagicMock()
        mock_msg.content = [MagicMock(text="hello")]
        mock_msg.model = "claude-haiku-4-5-20251001"
        mock_msg.usage.input_tokens = 12
        mock_msg.usage.output_tokens = 5
        mock_msg.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_msg

        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            p = ClaudeProvider(api_key="sk-test")
            result = p.complete_with_metadata("system", "user")

        assert result.text == "hello"
        assert result.model == "claude-haiku-4-5-20251001"
        assert result.usage == {"input_tokens": 12, "output_tokens": 5}
        assert result.stop_reason == "end_turn"

    def test_ollama_metadata(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import httpx

        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "response": "world",
            "model": "llama3",
            "prompt_eval_count": 10,
            "eval_count": 5,
            "done_reason": "stop",
        }
        mock_resp.raise_for_status = MagicMock()
        mock_client.post.return_value = mock_resp
        monkeypatch.setattr(httpx, "Client", lambda **kw: mock_client)

        p = OllamaProvider(base_url="http://localhost:11434", model="llama3")
        result = p.complete_with_metadata("sys", "usr")

        assert result.text == "world"
        assert result.model == "llama3"
        assert result.usage == {"prompt_tokens": 10, "completion_tokens": 5}
        assert result.stop_reason == "stop"
