from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod


class AIProvider(ABC):
    @abstractmethod
    def complete(self, system: str, user: str) -> str: ...

    def complete_json(self, system: str, user: str) -> dict:
        """Call complete() and parse the result as JSON.

        Strips markdown code fences (```json ... ```) if present,
        since some models wrap JSON in fences even when asked not to.
        """
        raw = self.complete(system, user).strip()
        if raw.startswith("```"):
            # Drop opening fence line (e.g. "```json")
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw
            # Drop closing fence
            raw = raw.rsplit("```", 1)[0]
        return json.loads(raw.strip())

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

    def complete(self, system: str, user: str) -> str:
        msg = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return msg.content[0].text


class OllamaProvider(AIProvider):
    def __init__(self, base_url: str = "", model: str = "") -> None:
        import httpx

        self._base_url = base_url or os.environ.get(
            "OLLAMA_BASE_URL", "http://localhost:11434"
        )
        self._model = model or os.environ.get("OLLAMA_MODEL", "llama3")
        self._client: httpx.Client | None = httpx.Client(timeout=60.0)

    def complete(self, system: str, user: str) -> str:
        import httpx

        prompt = f"{system}\n\n{user}"
        try:
            response = self._client.post(
                f"{self._base_url}/api/generate",
                json={"model": self._model, "prompt": prompt, "stream": False},
            )
        except httpx.ConnectError:
            raise ConnectionError(
                f"Cannot reach Ollama at {self._base_url}"
            ) from None
        response.raise_for_status()
        return response.json()["response"]

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def __del__(self) -> None:
        self.close()


def get_provider(
    provider: str | None = None,
    *,
    api_key: str = "",
    ollama_base_url: str = "",
    ollama_model: str = "",
) -> AIProvider:
    """Return an AIProvider instance.

    Configuration priority (for each setting):
      1. Keyword arguments passed to this function
      2. Environment variables: AI_PROVIDER, ANTHROPIC_API_KEY,
         OLLAMA_BASE_URL, OLLAMA_MODEL
      3. Built-in defaults
    """
    name = provider or os.environ.get("AI_PROVIDER", "claude")
    if name == "claude":
        return ClaudeProvider(api_key=api_key)
    if name == "ollama":
        return OllamaProvider(base_url=ollama_base_url, model=ollama_model)
    raise ValueError(f"Unknown AI provider: {name!r}. Expected 'claude' or 'ollama'.")
