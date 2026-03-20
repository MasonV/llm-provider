# llm-provider

Minimal pluggable LLM provider abstraction for Python. Supports Claude (Anthropic) and Ollama out of the box.

## Install

```bash
# Claude only
pip install "llm-provider[claude] @ git+https://github.com/MasonV/llm-provider.git@v0.2.0"

# Ollama only
pip install "llm-provider[ollama] @ git+https://github.com/MasonV/llm-provider.git@v0.2.0"

# Both
pip install "llm-provider[all] @ git+https://github.com/MasonV/llm-provider.git@v0.2.0"
```

## Quick Start

```python
from llm_provider import get_provider

provider = get_provider()  # uses AI_PROVIDER env var, defaults to "claude"
result = provider.complete_json("Return JSON with a greeting field.", "Say hello.")
print(result)  # {"greeting": "Hello!"}
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `AI_PROVIDER` | `"claude"` | Provider to use: `"claude"` or `"ollama"` |
| `ANTHROPIC_API_KEY` | — | API key for Claude (required for Claude provider) |
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
```

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
