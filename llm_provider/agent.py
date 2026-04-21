"""Agent-level abstraction for CLI-based coding agents.

Wraps autonomous agent CLIs (Claude Code, OpenAI Codex) behind a common
interface so callers can swap backends without changing orchestration logic.
Each implementation shells out via subprocess — no SDK dependencies required.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

AgentCallback = Callable[[str], None]
"""Signature for optional line-by-line streaming callback."""


@dataclass(frozen=True)
class AgentResult:
    """Result of an agent execution.

    Captures the agent's output text, process exit code, timing, and any
    structured events emitted during execution (parsed from JSONL output).
    """

    output: str
    exit_code: int
    model: str = ""
    duration_seconds: float = 0.0
    backend: str = ""
    working_directory: str = ""
    raw_events: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None


@dataclass
class AgentConfig:
    """Shared configuration for all agent backends.

    Fields map to CLI flags common across backends.  Backend-specific
    translation happens inside each implementation's ``build_cmd``.
    """

    working_directory: str = ""
    model: str = ""
    max_turns: int = 0
    timeout: float = 0.0
    sandbox: str = ""
    permission_mode: str = ""
    settings_path: str = ""
    use_worktree: bool = False
    worktree_path: str = ""
    env: dict[str, str] = field(default_factory=dict)
    mcp_config_path: str = ""
    allowed_tools: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_jsonl(text: str) -> list[dict[str, Any]]:
    """Parse newline-delimited JSON, skipping non-JSON lines."""
    events: list[dict[str, Any]] = []
    for line in text.splitlines():
        parsed = _try_parse_json(line)
        if parsed is not None:
            events.append(parsed)
    return events


def _try_parse_json(line: str) -> dict[str, Any] | None:
    """Try to parse a single line as JSON dict.  Returns None on failure."""
    line = line.strip()
    if not line or not line.startswith("{"):
        return None
    try:
        obj = json.loads(line)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _extract_result_text(events: list[dict[str, Any]]) -> str:
    """Extract the final assistant text from parsed JSON events.

    Works across output formats:
    - Single-object with a ``result`` key
    - Multi-event stream with ``type: "result"`` or ``type: "assistant"``
    """
    if not events:
        return ""

    # Single-object mode (Claude --output-format json)
    if len(events) == 1 and "result" in events[0]:
        return events[0]["result"]

    parts: list[str] = []
    for ev in events:
        if ev.get("type") == "result":
            parts.append(ev.get("result", ""))
        elif ev.get("type") == "assistant" and "message" in ev:
            parts.append(ev["message"])
    return "\n".join(parts) if parts else ""


def _extract_model(events: list[dict[str, Any]]) -> str:
    """Try to find the model name in parsed JSON events."""
    for ev in events:
        if "model" in ev:
            return str(ev["model"])
    return ""


# ---------------------------------------------------------------------------
# Codex TOML helpers
# ---------------------------------------------------------------------------

def _toml_string(value: str) -> str:
    """Encode a Python string as a TOML basic string (double-quoted, escaped)."""
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _toml_string_array(items: list[str]) -> str:
    """Encode a list of strings as a TOML array literal."""
    return "[" + ",".join(_toml_string(s) for s in items) + "]"


def _toml_inline_table(mapping: dict[str, str]) -> str:
    """Encode a {str: str} mapping as a TOML inline table."""
    pairs = ",".join(f"{k}={_toml_string(v)}" for k, v in mapping.items())
    return "{" + pairs + "}"


def _codex_enabled_tools(allowed_tools: list[str]) -> dict[str, list[str] | None]:
    """Group ``mcp__server__tool`` patterns into per-server enabled_tools lists.

    Returns ``{server: [tool, ...]}`` for explicit allowlists, or
    ``{server: None}`` when a wildcard (``mcp__server__*``) is present —
    meaning "expose every tool" for that server (Codex enabled_tools doesn't
    support wildcards, so we omit the key entirely).

    Non-MCP tool names (e.g. ``Read``, ``Bash(*)``) are silently dropped:
    they are Claude built-ins and have no Codex equivalent that needs an
    allowlist.
    """
    result: dict[str, list[str] | None] = {}
    for tool in allowed_tools:
        if not tool.startswith("mcp__"):
            continue
        parts = tool.split("__", 2)
        if len(parts) != 3:
            continue
        _, server, tool_name = parts
        if "*" in tool_name:
            result[server] = None  # wildcard: allow all
            continue
        existing = result.get(server, "missing")
        if existing is None:
            continue  # already wildcard for this server
        if isinstance(existing, list):
            existing.append(tool_name)
        else:
            result[server] = [tool_name]
    return result


def _codex_mcp_overrides(mcp_config_path: str, allowed_tools: list[str]) -> list[str]:
    """Translate a Claude-style mcp config JSON file into Codex ``-c`` overrides.

    Reads ``mcp_config_path`` (a JSON file with ``{"mcpServers": {name: {...}}}``)
    and emits one ``["-c", "mcp_servers.NAME.KEY=VALUE"]`` pair per field —
    a flat, stateless alternative to writing a temp ``config.toml``.

    Returns a flat list ready to ``cmd.extend(...)`` into the codex command.
    Returns an empty list if the file is missing, malformed, or has no servers.
    """
    if not mcp_config_path:
        return []
    try:
        with open(mcp_config_path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        _log.warning("Codex MCP config read failed (%s): %s", mcp_config_path, exc)
        return []

    servers = data.get("mcpServers", {})
    if not servers:
        return []

    enabled_by_server = _codex_enabled_tools(allowed_tools)
    flags: list[str] = []
    for name, srv in servers.items():
        prefix = f"mcp_servers.{name}"
        if "command" in srv:
            flags += ["-c", f"{prefix}.command={_toml_string(srv['command'])}"]
        if "args" in srv:
            flags += ["-c", f"{prefix}.args={_toml_string_array(srv['args'])}"]
        if "cwd" in srv:
            flags += ["-c", f"{prefix}.cwd={_toml_string(srv['cwd'])}"]
        if "env" in srv:
            flags += ["-c", f"{prefix}.env={_toml_inline_table(srv['env'])}"]
        tools = enabled_by_server.get(name, "absent")
        if isinstance(tools, list) and tools:
            flags += ["-c", f"{prefix}.enabled_tools={_toml_string_array(tools)}"]
    return flags


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class AgentBackend(ABC):
    """Abstract base for CLI-based coding agent wrappers.

    Follows the same ABC pattern as :class:`AIProvider` but operates at the
    agent level — executing autonomous multi-step tasks via subprocess rather
    than making single completion API calls.
    """

    _callback: AgentCallback | None = None
    _last_result: AgentResult | None = None

    def set_callback(self, callback: AgentCallback | None) -> None:
        """Register an optional line callback for streaming output.

        Args:
            callback: Called with each output line as the agent streams.
                Pass ``None`` to clear.
        """
        self._callback = callback

    @abstractmethod
    def run(self, prompt: str, *, config: AgentConfig | None = None) -> AgentResult:
        """Execute the agent with the given prompt.  Blocks until completion."""
        ...

    @abstractmethod
    def stream(
        self, prompt: str, *, config: AgentConfig | None = None
    ) -> Iterator[str]:
        """Execute the agent, yielding output lines as they arrive.

        After the iterator is exhausted, :attr:`last_result` holds the
        :class:`AgentResult` with exit code, timing, and parsed events.
        """
        ...

    @property
    def last_result(self) -> AgentResult | None:
        """Result from the most recent :meth:`stream` call.

        Available only after the stream iterator is fully consumed.
        """
        return self._last_result

    @abstractmethod
    def executable(self) -> str:
        """Return the CLI executable name (e.g. ``"claude"``, ``"codex"``)."""
        ...

    def is_available(self) -> bool:
        """Check whether the CLI tool is installed and on ``PATH``."""
        return shutil.which(self.executable()) is not None

    def close(self) -> None:
        """Release resources held by this backend.  No-op by default."""

    def __enter__(self) -> AgentBackend:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Claude Code
# ---------------------------------------------------------------------------

class ClaudeCodeAgent(AgentBackend):
    """Wraps the ``claude`` CLI (Claude Code) for autonomous task execution.

    Uses ``claude -p <prompt> --output-format json`` in non-interactive mode.
    Supports native worktree isolation via ``--worktree``.
    """

    def __init__(
        self,
        model: str = "",
        default_config: AgentConfig | None = None,
    ) -> None:
        self._model = model or os.environ.get("CLAUDE_MODEL", "")
        self._default_config = default_config or AgentConfig()

    def executable(self) -> str:
        return "claude"

    def _resolve_config(self, config: AgentConfig | None) -> AgentConfig:
        """Return *config* if given, otherwise the constructor default."""
        return config if config is not None else self._default_config

    def build_cmd(
        self,
        prompt: str | None,
        config: AgentConfig,
        *,
        output_format: str | None = None,
    ) -> list[str]:
        """Assemble the ``claude`` CLI argument list.

        Args:
            prompt: Prompt text passed as ``-p PROMPT``.  Pass ``None`` to emit
                only the ``-p`` flag and supply the prompt on stdin instead —
                required for long prompts that exceed shell argument limits.
            config: Run configuration.
            output_format: Value for ``--output-format``.  Use ``"json"`` for
                blocking ``run()`` calls and ``"stream-json"`` for streaming.
                ``None`` omits the flag entirely (e.g. when piping to a terminal).
        """
        cmd = ["claude"]
        if config.mcp_config_path:
            cmd += ["--mcp-config", config.mcp_config_path]
        if prompt is not None:
            cmd += ["-p", prompt]
        else:
            cmd.append("-p")
        if output_format is not None:
            cmd += ["--output-format", output_format]
        model = self._model or config.model
        if model:
            cmd += ["--model", model]
        if config.max_turns > 0:
            cmd += ["--max-turns", str(config.max_turns)]
        if config.sandbox:
            cmd += ["--sandbox", config.sandbox]
        if config.permission_mode:
            cmd += ["--permission-mode", config.permission_mode]
        if config.settings_path:
            cmd += ["--settings", config.settings_path]
        for tool in config.allowed_tools:
            cmd += ["--allowedTools", tool]
        if config.use_worktree:
            cmd.append("--worktree")
        return cmd

    def run(self, prompt: str, *, config: AgentConfig | None = None) -> AgentResult:
        cfg = self._resolve_config(config)
        cmd = self.build_cmd(prompt, cfg, output_format="json")
        _log.info("ClaudeCode run: cmd=%s", cmd[:4])

        t0 = time.monotonic()
        timeout = cfg.timeout if cfg.timeout > 0 else None
        env = {**os.environ, **cfg.env} if cfg.env else None
        cwd = cfg.working_directory or None

        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=timeout, env=env, cwd=cwd,
            )
        except subprocess.TimeoutExpired as exc:
            duration = time.monotonic() - t0
            _log.warning("ClaudeCode timed out after %.1fs", duration)
            return AgentResult(
                output=exc.stdout or "",
                exit_code=-1,
                duration_seconds=duration,
                backend="claude-code",
                working_directory=cfg.working_directory,
                error=f"Timed out after {cfg.timeout}s",
            )

        duration = time.monotonic() - t0
        events = _parse_jsonl(proc.stdout)
        output_text = _extract_result_text(events) if events else proc.stdout

        result = AgentResult(
            output=output_text,
            exit_code=proc.returncode,
            model=self._model or _extract_model(events),
            duration_seconds=duration,
            backend="claude-code",
            working_directory=cfg.working_directory,
            raw_events=events,
            error=proc.stderr if proc.returncode != 0 else None,
        )
        _log.info(
            "ClaudeCode finished: exit=%d, %.1fs, %d events",
            result.exit_code, duration, len(events),
        )
        return result

    def stream(
        self, prompt: str, *, config: AgentConfig | None = None
    ) -> Iterator[str]:
        cfg = self._resolve_config(config)
        cmd = self.build_cmd(prompt, cfg, output_format="stream-json")
        _log.info("ClaudeCode stream: cmd=%s", cmd[:4])

        t0 = time.monotonic()
        env = {**os.environ, **cfg.env} if cfg.env else None
        cwd = cfg.working_directory or None

        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, env=env, cwd=cwd,
        )
        events: list[dict[str, Any]] = []
        try:
            assert proc.stdout is not None  # guaranteed by PIPE
            for line in proc.stdout:
                line = line.rstrip("\n")
                if self._callback:
                    self._callback(line)
                parsed = _try_parse_json(line)
                if parsed is not None:
                    events.append(parsed)
                yield line
        finally:
            proc.wait()
            duration = time.monotonic() - t0
            stderr = proc.stderr.read() if proc.stderr else ""
            output_text = _extract_result_text(events) if events else ""
            self._last_result = AgentResult(
                output=output_text,
                exit_code=proc.returncode,
                model=self._model or _extract_model(events),
                duration_seconds=duration,
                backend="claude-code",
                working_directory=cfg.working_directory,
                raw_events=events,
                error=stderr if proc.returncode != 0 else None,
            )
            _log.info(
                "ClaudeCode stream finished: exit=%d, %.1fs",
                proc.returncode, duration,
            )


# ---------------------------------------------------------------------------
# Codex (shared base)
# ---------------------------------------------------------------------------

class _BaseCodexAgent(AgentBackend):
    """Shared implementation for Codex-based agents.

    ``CodexAgent`` and ``OllamaCodexAgent`` inherit from this.  The only
    difference is whether ``--oss`` is passed to the ``codex`` CLI.
    """

    _oss: bool = False

    def __init__(
        self,
        model: str = "",
        default_config: AgentConfig | None = None,
    ) -> None:
        env_var = "OLLAMA_MODEL" if self._oss else "CODEX_MODEL"
        self._model = model or os.environ.get(env_var, "")
        self._default_config = default_config or AgentConfig()

    def executable(self) -> str:
        return "codex"

    @property
    def _backend_name(self) -> str:
        return "codex-oss" if self._oss else "codex"

    def _resolve_config(self, config: AgentConfig | None) -> AgentConfig:
        return config if config is not None else self._default_config

    def build_cmd(self, prompt: str | None, config: AgentConfig) -> list[str]:
        """Assemble the ``codex`` CLI argument list.

        Args:
            prompt: Prompt text passed as a positional argument to ``codex exec``.
                Pass ``None`` to omit it — Codex then reads the prompt from
                stdin, which the caller must supply via ``proc.stdin.write``.
            config: Run configuration. ``mcp_config_path`` (a Claude-style
                JSON file) is translated into Codex ``-c mcp_servers.*``
                TOML overrides; ``allowed_tools`` becomes per-server
                ``enabled_tools`` (Claude built-ins like ``Read``/``Bash`` are
                ignored — Codex provides those natively).
        """
        cmd = ["codex", "exec"]
        if self._oss:
            cmd.append("--oss")

        # MCP overrides go before the prompt so they're parsed as options.
        cmd += _codex_mcp_overrides(config.mcp_config_path, config.allowed_tools)

        # Working directory: worktree path overrides working_directory.
        # Codex accepts -C as both a flag and we still pass cwd= to Popen,
        # mirroring the ClaudeCode wrapper for parity.
        workdir = ""
        if config.use_worktree and config.worktree_path:
            workdir = config.worktree_path
        elif config.working_directory:
            workdir = config.working_directory
        if workdir:
            cmd += ["-C", workdir]

        model = self._model or config.model
        if model:
            cmd += ["-m", model]
        if config.sandbox:
            cmd += ["-s", config.sandbox]
        if config.permission_mode:
            cmd += ["-a", config.permission_mode]

        if prompt is not None:
            cmd.append(prompt)
        cmd.append("--json")
        return cmd

    def run(self, prompt: str, *, config: AgentConfig | None = None) -> AgentResult:
        cfg = self._resolve_config(config)
        cmd = self.build_cmd(prompt, cfg)
        backend = self._backend_name
        _log.info("%s run: cmd=%s", backend, cmd[:5])

        t0 = time.monotonic()
        timeout = cfg.timeout if cfg.timeout > 0 else None
        env = {**os.environ, **cfg.env} if cfg.env else None

        effective_workdir = ""
        if cfg.use_worktree and cfg.worktree_path:
            effective_workdir = cfg.worktree_path
        else:
            effective_workdir = cfg.working_directory

        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=timeout, env=env, cwd=effective_workdir or None,
            )
        except subprocess.TimeoutExpired as exc:
            duration = time.monotonic() - t0
            _log.warning("%s timed out after %.1fs", backend, duration)
            return AgentResult(
                output=exc.stdout or "",
                exit_code=-1,
                duration_seconds=duration,
                backend=backend,
                working_directory=effective_workdir,
                error=f"Timed out after {cfg.timeout}s",
            )

        duration = time.monotonic() - t0
        events = _parse_jsonl(proc.stdout)
        output_text = _extract_result_text(events) if events else proc.stdout

        result = AgentResult(
            output=output_text,
            exit_code=proc.returncode,
            model=self._model or _extract_model(events),
            duration_seconds=duration,
            backend=backend,
            working_directory=effective_workdir,
            raw_events=events,
            error=proc.stderr if proc.returncode != 0 else None,
        )
        _log.info(
            "%s finished: exit=%d, %.1fs, %d events",
            backend, result.exit_code, duration, len(events),
        )
        return result

    def stream(
        self, prompt: str, *, config: AgentConfig | None = None
    ) -> Iterator[str]:
        cfg = self._resolve_config(config)
        cmd = self.build_cmd(prompt, cfg)
        backend = self._backend_name
        _log.info("%s stream: cmd=%s", backend, cmd[:5])

        t0 = time.monotonic()
        env = {**os.environ, **cfg.env} if cfg.env else None

        effective_workdir = ""
        if cfg.use_worktree and cfg.worktree_path:
            effective_workdir = cfg.worktree_path
        else:
            effective_workdir = cfg.working_directory

        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, env=env, cwd=effective_workdir or None,
        )
        events: list[dict[str, Any]] = []
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                line = line.rstrip("\n")
                if self._callback:
                    self._callback(line)
                parsed = _try_parse_json(line)
                if parsed is not None:
                    events.append(parsed)
                yield line
        finally:
            proc.wait()
            duration = time.monotonic() - t0
            stderr = proc.stderr.read() if proc.stderr else ""
            output_text = _extract_result_text(events) if events else ""
            self._last_result = AgentResult(
                output=output_text,
                exit_code=proc.returncode,
                model=self._model or _extract_model(events),
                duration_seconds=duration,
                backend=backend,
                working_directory=effective_workdir,
                raw_events=events,
                error=stderr if proc.returncode != 0 else None,
            )
            _log.info(
                "%s stream finished: exit=%d, %.1fs",
                backend, proc.returncode, duration,
            )


class CodexAgent(_BaseCodexAgent):
    """Wraps the ``codex`` CLI for OpenAI-backed task execution.

    Uses ``codex exec <prompt> --json`` in non-interactive mode.
    Connects through the user's ChatGPT Pro membership — no per-token API
    billing.  Worktree isolation is handled by pointing ``-C`` at an
    externally-created worktree path.
    """

    _oss = False


class OllamaCodexAgent(_BaseCodexAgent):
    """Wraps ``codex --oss`` for local model execution via Ollama.

    Identical to :class:`CodexAgent` except it passes ``--oss`` to route
    inference through a locally-running Ollama instance instead of the
    OpenAI API.
    """

    _oss = True


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

CLAUDE_CODE_MODELS: list[str] = [
    "claude-haiku-4-5-20251001",
    "claude-sonnet-4-6",
    "claude-opus-4-7",
]

CODEX_MODELS: list[str] = [
    "gpt-5.2",
    "gpt-5.3-codex",
    "gpt-5.4-mini",
    "gpt-5.4",
]


def list_agent_models(backend: str, *, ollama_base_url: str = "") -> list[str]:
    """Return known model IDs for an agent backend.

    For ``"claude-code"`` and ``"codex"``, returns a static list.  For
    ``"ollama"``/``"codex-oss"``, queries the local Ollama server and returns
    installed model names; returns an empty list if unreachable.

    Args:
        backend: One of ``"claude-code"``, ``"codex"``, ``"ollama"``, or
            ``"codex-oss"``.
        ollama_base_url: Ollama server URL override.  Defaults to the
            ``OLLAMA_BASE_URL`` env var or ``http://localhost:11434``.
    """
    if backend == "claude-code":
        return list(CLAUDE_CODE_MODELS)
    if backend == "codex":
        return list(CODEX_MODELS)
    if backend in ("ollama", "codex-oss"):
        base_url = ollama_base_url or os.environ.get(
            "OLLAMA_BASE_URL", "http://localhost:11434"
        )
        import httpx
        try:
            response = httpx.get(f"{base_url}/api/tags", timeout=5.0)
            response.raise_for_status()
            data = response.json()
            return [m["name"] for m in data.get("models", [])]
        except Exception as exc:
            _log.warning(
                "list_agent_models: Ollama query failed (%s): %s", base_url, exc
            )
            return []
    raise ValueError(
        f"Unknown backend: {backend!r}. "
        "Expected 'claude-code', 'codex', or 'ollama'."
    )


def get_agent(
    backend: str | None = None,
    *,
    model: str = "",
    config: AgentConfig | None = None,
    callback: AgentCallback | None = None,
) -> AgentBackend:
    """Return an :class:`AgentBackend` instance.

    Configuration priority:
      1. Keyword arguments passed here
      2. ``AGENT_BACKEND`` environment variable
      3. Default: ``"claude-code"``

    Accepted *backend* values: ``"claude-code"``, ``"codex"``,
    ``"ollama"`` (alias ``"codex-oss"``).

    Args:
        backend: Which CLI agent to use.
        model: Model override passed to the backend constructor.
        config: Default :class:`AgentConfig` for all calls on this instance.
        callback: Optional line callback registered via
            :meth:`AgentBackend.set_callback`.
    """
    name = backend or os.environ.get("AGENT_BACKEND", "claude-code")
    _log.info("Using agent backend %r", name)

    if name == "claude-code":
        inst: AgentBackend = ClaudeCodeAgent(model=model, default_config=config)
    elif name == "codex":
        inst = CodexAgent(model=model, default_config=config)
    elif name in ("ollama", "codex-oss"):
        inst = OllamaCodexAgent(model=model, default_config=config)
    else:
        raise ValueError(
            f"Unknown agent backend: {name!r}. "
            "Expected 'claude-code', 'codex', or 'ollama'."
        )

    if callback:
        inst.set_callback(callback)
    return inst
