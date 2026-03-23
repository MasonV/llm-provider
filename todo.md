# llm-provider — Development TODO

Extracted from Pillars as a minimal pluggable LLM abstraction (Claude + OpenAI + Ollama).
Current state: core API works with three providers, response metadata, and streaming support.

---

## High Priority

- [x] **Add a README** — usage examples, env var reference, how to add a custom provider
- [x] **Write tests** — unit tests with mocked API calls for both providers + `get_provider()` factory + `complete_json()` parsing/fence-stripping
- [x] **Input validation** — check for empty API key before calling Anthropic, surface clear errors when Ollama is unreachable
- [x] **Close httpx client properly** — context manager, `close()`, and `__del__` fallback on OllamaProvider

## Medium Priority

- [x] **Streaming support** — `stream()` method returning `CompletionStream` iterator with `.result` property
- [x] **Retry / rate-limit handling** — exponential backoff via `_retry()`, configurable `max_retries` param
- [x] **Response metadata** — `CompletionResult(str)` subclass exposing `.usage`, `.model`, `.stop_reason`
- [ ] **Async variants** — `async complete()` for use in async codebases
- [x] **Logging** — `logging.getLogger(__name__)` for provider selection, requests, and retries

## Low Priority / Nice-to-Have

- [x] **Additional providers** — OpenAI via `OpenAIProvider`
- [ ] **More providers** — Google Gemini, local llama.cpp, etc.
- [ ] **Token counting** — pre-flight token estimation so callers can check limits
- [ ] **CI pipeline** — GitHub Actions for lint (ruff), type check (mypy), and tests on push
- [ ] **Publish to PyPI** — automate release from tags
- [ ] **Example scripts** — small runnable demos showing common usage patterns
