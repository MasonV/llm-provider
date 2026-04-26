# llm-provider

Minimal pluggable LLM provider abstraction for Python. Supports Claude (Anthropic), OpenAI, and Ollama out of the box. Also provides a unified interface for CLI-based coding agents (Claude Code, OpenAI Codex).

## Install

```bash
# Claude only
pip install "llm-provider[claude] @ git+https://github.com/MasonV/llm-provider.git"

# OpenAI only
pip install "llm-provider[openai] @ git+https://github.com/MasonV/llm-provider.git"

# Ollama only
pip install "llm-provider[ollama] @ git+https://github.com/MasonV/llm-provider.git"

# All providers
pip install "llm-provider[all] @ git+https://github.com/MasonV/llm-provider.git"
```

## Quick Start

```python
from llm_provider import get_provider

provider = get_provider()  # uses AI_PROVIDER env var, defaults to "claude"
result = provider.complete("Return a greeting.", "Say hello.")
print(result)  # "Hello!"

# JSON parsing with fence stripping
data = provider.complete_json("Return JSON with a greeting field.", "Say hello.")
print(data)  # {"greeting": "Hello!"}
```

## Response Metadata

Built-in providers return a `CompletionResult` — a `str` subclass that also
carries token usage, model name, and stop reason:

```python
result = provider.complete("system prompt", "user message")
print(result)              # works as a normal string
print(result.model)        # "claude-haiku-4-5-20251001"
print(result.stop_reason)  # "end_turn"
print(result.usage)        # CompletionUsage(prompt_tokens=12, completion_tokens=8)
```

## Streaming

All providers support streaming via the `stream()` method, which returns a
`CompletionStream` — an iterator of text chunks:

```python
stream = provider.stream("system prompt", "user message")
for chunk in stream:
    print(chunk, end="", flush=True)

# After the stream is consumed, metadata is available:
print(stream.result.usage)
```

## Async Support

All providers have async variants for use with `asyncio`:

```python
import asyncio
from llm_provider import get_async_provider

async def main():
    async with get_async_provider() as provider:
        result = await provider.complete("Return a greeting.", "Say hello.")
        print(result)

        # JSON parsing works the same way
        data = await provider.complete_json("Return JSON.", "Say hello.")
        print(data)

asyncio.run(main())
```

### Async Streaming

```python
stream = await provider.stream("system prompt", "user message")
async for chunk in stream:
    print(chunk, end="", flush=True)

# Metadata available after consumption
result = await stream.get_result()
print(result.usage)
```

### Custom Async Provider

Subclass `AsyncAIProvider` and implement `complete()`:

```python
from llm_provider import AsyncAIProvider

class MyAsyncProvider(AsyncAIProvider):
    async def complete(self, system: str, user: str) -> str:
        # call your async LLM client here
        return "response text"
```

## Models

### Claude (provider: `"claude"`)

| Model ID | Notes |
|---|---|
| `claude-haiku-4-5-20251001` | Fast, cheap — default |
| `claude-sonnet-4-6` | Balanced |
| `claude-opus-4-7` | Most capable |

### OpenAI (provider: `"openai"`)

| Model ID | Notes |
|---|---|
| `gpt-4o-mini` | Fast, cheap — default |
| `gpt-4o` | Balanced |
| `gpt-4.1-nano` | Lightweight |
| `gpt-4.1-mini` | |
| `gpt-4.1` | |
| `o4-mini` | Reasoning |
| `o3` | Reasoning |
| `gpt-5.2` | |
| `gpt-5.3-codex` | Coding-focused |
| `gpt-5.4-mini` | |
| `gpt-5.4` | Most capable |

### Ollama (provider: `"ollama"`)

Model names depend on what you have installed locally. Run `ollama list` to see available models, or use `list_models("ollama")` to query programmatically.

## Querying Available Models

```python
from llm_provider import list_models, list_agent_models

# LLM providers
list_models("claude")   # ["claude-haiku-4-5-20251001", "claude-sonnet-4-6", ...]
list_models("openai")   # ["gpt-4o-mini", "gpt-4o", ..., "gpt-5.4"]
list_models("ollama")   # queries http://localhost:11434/api/tags, returns installed models

# Agent backends
list_agent_models("claude-code")  # ["claude-haiku-4-5-20251001", ...]
list_agent_models("codex")        # ["gpt-5.2", "gpt-5.3-codex", "gpt-5.4-mini", "gpt-5.4"]
list_agent_models("ollama")       # queries local Ollama server
```

Ollama queries return an empty list (rather than raising) if the server is unreachable.

The raw constants are also exported for direct use: `CLAUDE_MODELS`, `OPENAI_MODELS`,
`CLAUDE_CODE_MODELS`, `CODEX_MODELS`.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `LLM` | `"claude"` | Provider to use: `"claude"`, `"openai"`, or `"ollama"` (`AI_PROVIDER` is an alias) |
| `ANTHROPIC_API_KEY` | — | API key for Claude |
| `OPENAI_API_KEY` | — | API key for OpenAI |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `qwen2.5:3b` | Ollama model name |
| `AGENT_BACKEND` | `"claude-code"` | Agent backend: `"claude-code"`, `"codex"`, or `"ollama"` |
| `CLAUDE_MODEL` | — | Model override for `ClaudeCodeAgent` |
| `CODEX_MODEL` | — | Model override for `CodexAgent` |

Keyword arguments to `get_provider()` and `get_agent()` take priority over environment variables.

## Agents

A separate agent layer wraps CLI-based coding agents (`claude`, `codex`) behind a common interface. Backends: `"claude-code"`, `"codex"`, `"ollama"` (Codex with `--oss`).

```python
from llm_provider import get_agent, AgentConfig

agent = get_agent()  # uses AGENT_BACKEND env var, defaults to "claude-code"

# Blocking run
result = agent.run("Refactor foo.py to use dataclasses.")
print(result.output)
print(result.exit_code)    # 0 on success
print(result.duration_seconds)

# Streaming
for line in agent.stream("Add type hints to bar.py."):
    print(line)
print(agent.last_result.exit_code)
```

### Agent Configuration

```python
config = AgentConfig(
    working_directory="/path/to/repo",
    model="claude-sonnet-4-6",   # or "gpt-5.4" for codex
    max_turns=10,
    timeout=120.0,
    permission_mode="acceptEdits",
)
agent = get_agent("claude-code", config=config)
result = agent.run("Fix the failing tests.", config=config)
```

### Selecting a Backend

```python
# Claude Code (claude CLI)
agent = get_agent("claude-code", model="claude-opus-4-7")

# OpenAI Codex (codex CLI)
agent = get_agent("codex", model="gpt-5.4")

# Ollama via Codex OSS (codex --oss)
agent = get_agent("ollama", model="qwen2.5:3b")
```

### Agent Models

See the [Models](#models) section and [Querying Available Models](#querying-available-models) for model IDs per backend.

## Custom Provider

Subclass `AIProvider` and implement `complete()`:

```python
from llm_provider import AIProvider

class MyProvider(AIProvider):
    def complete(self, system: str, user: str) -> str:
        # call your LLM here
        return "response text"

provider = MyProvider()
provider.complete_json("Return JSON.", "Do something.")  # inherited
provider.stream("sys", "usr")  # inherited (single-chunk fallback)
```

Override `stream()` for native streaming support.

## Resource Cleanup

Providers that hold connections (like `OllamaProvider`) can be used as context managers:

```python
with get_provider("ollama") as provider:
    result = provider.complete("system prompt", "user message")
```

Or call `provider.close()` explicitly. If neither is used, resources are cleaned up on garbage collection.

## Development

```bash
git clone https://github.com/MasonV/llm-provider.git
cd llm-provider
pip install -e ".[dev]"
pytest
```
