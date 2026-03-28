# llm-provider

Minimal pluggable LLM provider abstraction for Python. Supports Claude (Anthropic), OpenAI, and Ollama out of the box.

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

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `AI_PROVIDER` | `"claude"` | Provider to use: `"claude"`, `"openai"`, or `"ollama"` |
| `ANTHROPIC_API_KEY` | — | API key for Claude |
| `OPENAI_API_KEY` | — | API key for OpenAI |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3` | Ollama model name |

Keyword arguments to `get_provider()` take priority over environment variables.

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
