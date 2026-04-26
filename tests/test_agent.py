from __future__ import annotations

import json
import os
import subprocess
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from llm_provider import (
    AgentBackend,
    AgentConfig,
    AgentResult,
    ClaudeCodeAgent,
    CodexAgent,
    OllamaCodexAgent,
    get_agent,
)
from llm_provider.agent import (
    _codex_enabled_tools,
    _codex_mcp_overrides,
    _extract_model,
    _extract_result_text,
    _parse_jsonl,
    _toml_inline_table,
    _toml_string,
    _toml_string_array,
    _try_parse_json,
)


# ---------------------------------------------------------------------------
# AgentResult
# ---------------------------------------------------------------------------

class TestAgentResult:
    def test_defaults(self) -> None:
        r = AgentResult(output="done", exit_code=0)
        assert r.output == "done"
        assert r.exit_code == 0
        assert r.model == ""
        assert r.duration_seconds == 0.0
        assert r.backend == ""
        assert r.working_directory == ""
        assert r.raw_events == []
        assert r.error is None

    def test_frozen(self) -> None:
        r = AgentResult(output="ok", exit_code=0)
        with pytest.raises(AttributeError):
            r.output = "nope"  # type: ignore[misc]

    def test_full_fields(self) -> None:
        events = [{"type": "result", "result": "hi"}]
        r = AgentResult(
            output="hi",
            exit_code=0,
            model="claude-sonnet-4-6",
            duration_seconds=12.5,
            backend="claude-code",
            working_directory="/tmp/work",
            raw_events=events,
            error=None,
        )
        assert r.model == "claude-sonnet-4-6"
        assert r.duration_seconds == 12.5
        assert r.backend == "claude-code"
        assert r.working_directory == "/tmp/work"
        assert len(r.raw_events) == 1


# ---------------------------------------------------------------------------
# AgentConfig
# ---------------------------------------------------------------------------

class TestAgentConfig:
    def test_defaults(self) -> None:
        c = AgentConfig()
        assert c.working_directory == ""
        assert c.model == ""
        assert c.max_turns == 0
        assert c.timeout == 0.0
        assert c.sandbox == ""
        assert c.permission_mode == ""
        assert c.settings_path == ""
        assert c.use_worktree is False
        assert c.worktree_path == ""
        assert c.env == {}
        assert c.mcp_config_path == ""
        assert c.allowed_tools == []

    def test_custom_values(self) -> None:
        c = AgentConfig(
            working_directory="/project",
            model="o3",
            timeout=300.0,
            sandbox="workspace-write",
            use_worktree=True,
            worktree_path="/tmp/wt",
            env={"FOO": "bar"},
        )
        assert c.working_directory == "/project"
        assert c.model == "o3"
        assert c.timeout == 300.0
        assert c.sandbox == "workspace-write"
        assert c.use_worktree is True
        assert c.worktree_path == "/tmp/wt"
        assert c.env == {"FOO": "bar"}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

class TestTryParseJson:
    def test_valid_json_dict(self) -> None:
        assert _try_parse_json('{"key": "val"}') == {"key": "val"}

    def test_empty_string(self) -> None:
        assert _try_parse_json("") is None

    def test_non_json(self) -> None:
        assert _try_parse_json("hello world") is None

    def test_json_array_rejected(self) -> None:
        # Only dicts are returned
        assert _try_parse_json("[1, 2, 3]") is None

    def test_invalid_json(self) -> None:
        assert _try_parse_json("{broken") is None

    def test_whitespace_stripped(self) -> None:
        assert _try_parse_json('  {"a": 1}  ') == {"a": 1}


class TestParseJsonl:
    def test_multiple_lines(self) -> None:
        text = '{"a": 1}\nnot json\n{"b": 2}\n'
        result = _parse_jsonl(text)
        assert result == [{"a": 1}, {"b": 2}]

    def test_empty_input(self) -> None:
        assert _parse_jsonl("") == []

    def test_no_json_lines(self) -> None:
        assert _parse_jsonl("line one\nline two\n") == []


class TestExtractResultText:
    def test_single_result_object(self) -> None:
        events = [{"result": "Task completed successfully"}]
        assert _extract_result_text(events) == "Task completed successfully"

    def test_multi_event_result_type(self) -> None:
        events = [
            {"type": "start", "model": "claude-sonnet-4-6"},
            {"type": "result", "result": "Done"},
        ]
        assert _extract_result_text(events) == "Done"

    def test_assistant_message_type(self) -> None:
        events = [
            {"type": "assistant", "message": "First part"},
            {"type": "assistant", "message": "Second part"},
        ]
        assert _extract_result_text(events) == "First part\nSecond part"

    def test_empty_events(self) -> None:
        assert _extract_result_text([]) == ""

    def test_no_matching_types(self) -> None:
        events = [{"type": "tool_use", "name": "write"}]
        assert _extract_result_text(events) == ""


class TestExtractModel:
    def test_finds_model(self) -> None:
        events = [{"type": "start", "model": "o3"}, {"type": "end"}]
        assert _extract_model(events) == "o3"

    def test_no_model(self) -> None:
        events = [{"type": "end"}]
        assert _extract_model(events) == ""

    def test_empty_events(self) -> None:
        assert _extract_model([]) == ""


# ---------------------------------------------------------------------------
# ClaudeCodeAgent.build_cmd
# ---------------------------------------------------------------------------

class TestClaudeCodeBuildCmd:
    def test_minimal_with_prompt(self) -> None:
        agent = ClaudeCodeAgent()
        cmd = agent.build_cmd("do stuff", AgentConfig())
        assert cmd == ["claude", "-p", "do stuff"]

    def test_minimal_with_output_format(self) -> None:
        agent = ClaudeCodeAgent()
        cmd = agent.build_cmd("do stuff", AgentConfig(), output_format="json")
        assert cmd == ["claude", "-p", "do stuff", "--output-format", "json"]

    def test_stdin_mode_prompt_none(self) -> None:
        # prompt=None emits bare -p flag; caller writes prompt to stdin
        agent = ClaudeCodeAgent()
        cmd = agent.build_cmd(None, AgentConfig())
        assert cmd == ["claude", "-p"]
        assert "--cd" not in cmd

    def test_working_directory_not_in_cmd(self) -> None:
        # working_directory goes to cwd= kwarg on subprocess, never --cd
        agent = ClaudeCodeAgent()
        cfg = AgentConfig(working_directory="/tmp/work")
        cmd = agent.build_cmd("task", cfg)
        assert "--cd" not in cmd

    def test_model_from_constructor(self) -> None:
        agent = ClaudeCodeAgent(model="claude-opus-4-6")
        cmd = agent.build_cmd("task", AgentConfig())
        assert "--model" in cmd
        idx = cmd.index("--model")
        assert cmd[idx + 1] == "claude-opus-4-6"

    def test_model_from_config(self) -> None:
        agent = ClaudeCodeAgent()
        cfg = AgentConfig(model="claude-sonnet-4-6")
        cmd = agent.build_cmd("task", cfg)
        assert "--model" in cmd
        idx = cmd.index("--model")
        assert cmd[idx + 1] == "claude-sonnet-4-6"

    def test_constructor_model_overrides_config(self) -> None:
        agent = ClaudeCodeAgent(model="opus")
        cfg = AgentConfig(model="sonnet")
        cmd = agent.build_cmd("task", cfg)
        idx = cmd.index("--model")
        assert cmd[idx + 1] == "opus"

    def test_max_turns(self) -> None:
        agent = ClaudeCodeAgent()
        cfg = AgentConfig(max_turns=10)
        cmd = agent.build_cmd("task", cfg)
        assert "--max-turns" in cmd
        idx = cmd.index("--max-turns")
        assert cmd[idx + 1] == "10"

    def test_effort(self) -> None:
        agent = ClaudeCodeAgent()
        cfg = AgentConfig(effort="high")
        cmd = agent.build_cmd("task", cfg)
        assert "--effort" in cmd
        idx = cmd.index("--effort")
        assert cmd[idx + 1] == "high"

    def test_effort_empty_omitted(self) -> None:
        agent = ClaudeCodeAgent()
        cmd = agent.build_cmd("task", AgentConfig())
        assert "--effort" not in cmd

    def test_max_turns_zero_omitted(self) -> None:
        agent = ClaudeCodeAgent()
        cmd = agent.build_cmd("task", AgentConfig())
        assert "--max-turns" not in cmd

    def test_sandbox(self) -> None:
        agent = ClaudeCodeAgent()
        cfg = AgentConfig(sandbox="workspace-write")
        cmd = agent.build_cmd("task", cfg)
        assert "--sandbox" in cmd
        idx = cmd.index("--sandbox")
        assert cmd[idx + 1] == "workspace-write"

    def test_permission_mode(self) -> None:
        agent = ClaudeCodeAgent()
        cfg = AgentConfig(permission_mode="dontAsk")
        cmd = agent.build_cmd("task", cfg)
        assert "--permission-mode" in cmd
        idx = cmd.index("--permission-mode")
        assert cmd[idx + 1] == "dontAsk"

    def test_settings_path(self) -> None:
        agent = ClaudeCodeAgent()
        cfg = AgentConfig(settings_path="/path/to/settings.json")
        cmd = agent.build_cmd("task", cfg)
        assert "--settings" in cmd
        idx = cmd.index("--settings")
        assert cmd[idx + 1] == "/path/to/settings.json"

    def test_worktree(self) -> None:
        agent = ClaudeCodeAgent()
        cfg = AgentConfig(use_worktree=True)
        cmd = agent.build_cmd("task", cfg)
        assert "--worktree" in cmd

    def test_worktree_false_omitted(self) -> None:
        agent = ClaudeCodeAgent()
        cmd = agent.build_cmd("task", AgentConfig())
        assert "--worktree" not in cmd

    def test_mcp_config_path(self) -> None:
        agent = ClaudeCodeAgent()
        cfg = AgentConfig(mcp_config_path="/tmp/mcp.json")
        cmd = agent.build_cmd("task", cfg)
        assert "--mcp-config" in cmd
        idx = cmd.index("--mcp-config")
        assert cmd[idx + 1] == "/tmp/mcp.json"
        # mcp-config must come before -p
        assert cmd.index("--mcp-config") < cmd.index("-p")

    def test_mcp_config_omitted_when_empty(self) -> None:
        agent = ClaudeCodeAgent()
        cmd = agent.build_cmd("task", AgentConfig())
        assert "--mcp-config" not in cmd

    def test_allowed_tools_single(self) -> None:
        agent = ClaudeCodeAgent()
        cfg = AgentConfig(allowed_tools=["mcp__autodev__*"])
        cmd = agent.build_cmd("task", cfg)
        assert "--allowedTools" in cmd
        idx = cmd.index("--allowedTools")
        assert cmd[idx + 1] == "mcp__autodev__*"

    def test_allowed_tools_multiple(self) -> None:
        agent = ClaudeCodeAgent()
        cfg = AgentConfig(allowed_tools=["Read", "Write", "Bash(*)"])
        cmd = agent.build_cmd("task", cfg)
        # Each tool gets its own --allowedTools flag
        tool_pairs = [
            (cmd[i + 1], True)
            for i, tok in enumerate(cmd)
            if tok == "--allowedTools"
        ]
        tools = [cmd[i + 1] for i, tok in enumerate(cmd) if tok == "--allowedTools"]
        assert tools == ["Read", "Write", "Bash(*)"]

    def test_allowed_tools_omitted_when_empty(self) -> None:
        agent = ClaudeCodeAgent()
        cmd = agent.build_cmd("task", AgentConfig())
        assert "--allowedTools" not in cmd

    def test_output_format_none_omitted(self) -> None:
        agent = ClaudeCodeAgent()
        cmd = agent.build_cmd("task", AgentConfig(), output_format=None)
        assert "--output-format" not in cmd

    def test_output_format_stream_json(self) -> None:
        agent = ClaudeCodeAgent()
        cmd = agent.build_cmd("task", AgentConfig(), output_format="stream-json")
        assert "--output-format" in cmd
        idx = cmd.index("--output-format")
        assert cmd[idx + 1] == "stream-json"

    def test_full_config(self) -> None:
        agent = ClaudeCodeAgent(model="claude-sonnet-4-6")
        cfg = AgentConfig(
            working_directory="/project",
            mcp_config_path="/tmp/mcp.json",
            max_turns=5,
            sandbox="workspace-write",
            permission_mode="dontAsk",
            settings_path="/s.json",
            allowed_tools=["mcp__autodev__*", "Read"],
            use_worktree=True,
        )
        cmd = agent.build_cmd("build feature", cfg, output_format="stream-json")
        assert cmd[0] == "claude"
        assert "--mcp-config" in cmd
        assert "-p" in cmd
        assert "build feature" in cmd
        assert "--output-format" in cmd
        assert "--cd" not in cmd
        assert "--model" in cmd
        assert "--max-turns" in cmd
        assert "--sandbox" in cmd
        assert "--permission-mode" in cmd
        assert "--settings" in cmd
        assert "--allowedTools" in cmd
        assert "--worktree" in cmd


# ---------------------------------------------------------------------------
# Codex TOML helpers
# ---------------------------------------------------------------------------

class TestTomlString:
    def test_plain(self) -> None:
        assert _toml_string("hello") == '"hello"'

    def test_escapes_backslash(self) -> None:
        assert _toml_string("a\\b") == '"a\\\\b"'

    def test_escapes_double_quote(self) -> None:
        assert _toml_string('say "hi"') == '"say \\"hi\\""'

    def test_empty(self) -> None:
        assert _toml_string("") == '""'


class TestTomlStringArray:
    def test_single(self) -> None:
        assert _toml_string_array(["a"]) == '["a"]'

    def test_multiple(self) -> None:
        assert _toml_string_array(["a", "b", "c"]) == '["a","b","c"]'

    def test_empty(self) -> None:
        assert _toml_string_array([]) == "[]"


class TestTomlInlineTable:
    def test_single_pair(self) -> None:
        assert _toml_inline_table({"K": "v"}) == '{K="v"}'

    def test_multiple_pairs(self) -> None:
        # dict order is insertion order in 3.7+
        out = _toml_inline_table({"A": "1", "B": "2"})
        assert out == '{A="1",B="2"}'


class TestCodexEnabledTools:
    def test_wildcard_returns_none(self) -> None:
        out = _codex_enabled_tools(["mcp__autodev__*"])
        assert out == {"autodev": None}

    def test_specific_tools(self) -> None:
        out = _codex_enabled_tools(
            ["mcp__autodev__get_task", "mcp__autodev__update_status"]
        )
        assert out == {"autodev": ["get_task", "update_status"]}

    def test_wildcard_and_specific_for_same_server(self) -> None:
        # wildcard wins — once a server is wildcarded, specifics are ignored
        out = _codex_enabled_tools(
            ["mcp__autodev__*", "mcp__autodev__get_task"]
        )
        assert out == {"autodev": None}

    def test_specific_then_wildcard_promotes_to_wildcard(self) -> None:
        out = _codex_enabled_tools(
            ["mcp__autodev__get_task", "mcp__autodev__*"]
        )
        assert out == {"autodev": None}

    def test_multiple_servers(self) -> None:
        out = _codex_enabled_tools(
            ["mcp__autodev__*", "mcp__github__create_issue"]
        )
        assert out == {"autodev": None, "github": ["create_issue"]}

    def test_non_mcp_tools_dropped(self) -> None:
        # Claude built-ins are not relevant for codex enabled_tools
        assert _codex_enabled_tools(["Read", "Write", "Bash(*)"]) == {}

    def test_malformed_mcp_name_dropped(self) -> None:
        # mcp__ prefix but no server__tool split
        assert _codex_enabled_tools(["mcp__justone"]) == {}

    def test_empty(self) -> None:
        assert _codex_enabled_tools([]) == {}


class TestCodexMcpOverrides:
    def _write_config(self, data: dict) -> str:
        f = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False,
        )
        json.dump(data, f)
        f.close()
        return f.name

    def test_empty_path_returns_empty(self) -> None:
        assert _codex_mcp_overrides("", []) == []

    def test_missing_file_returns_empty(self) -> None:
        assert _codex_mcp_overrides("/nonexistent/path.json", []) == []

    def test_basic_command_args(self) -> None:
        path = self._write_config({
            "mcpServers": {
                "autodev": {
                    "command": "python3",
                    "args": ["-m", "autodev_mcp"],
                }
            }
        })
        flags = _codex_mcp_overrides(path, [])
        assert "-c" in flags
        assert 'mcp_servers.autodev.command="python3"' in flags
        assert 'mcp_servers.autodev.args=["-m","autodev_mcp"]' in flags

    def test_cwd_emitted(self) -> None:
        path = self._write_config({
            "mcpServers": {
                "autodev": {"command": "x", "cwd": "/repo"}
            }
        })
        flags = _codex_mcp_overrides(path, [])
        assert 'mcp_servers.autodev.cwd="/repo"' in flags

    def test_env_emitted_as_inline_table(self) -> None:
        path = self._write_config({
            "mcpServers": {
                "autodev": {"command": "x", "env": {"KEY": "value"}}
            }
        })
        flags = _codex_mcp_overrides(path, [])
        assert 'mcp_servers.autodev.env={KEY="value"}' in flags

    def test_wildcard_tool_skips_enabled_tools(self) -> None:
        path = self._write_config({
            "mcpServers": {"autodev": {"command": "x"}}
        })
        flags = _codex_mcp_overrides(path, ["mcp__autodev__*"])
        # No enabled_tools when wildcard — codex defaults to all enabled
        assert not any("enabled_tools" in f for f in flags)

    def test_specific_tools_emit_enabled_tools(self) -> None:
        path = self._write_config({
            "mcpServers": {"autodev": {"command": "x"}}
        })
        flags = _codex_mcp_overrides(
            path, ["mcp__autodev__get_task", "mcp__autodev__update_status"]
        )
        assert any(
            'mcp_servers.autodev.enabled_tools=["get_task","update_status"]' == f
            for f in flags
        )

    def test_no_servers_returns_empty(self) -> None:
        path = self._write_config({"mcpServers": {}})
        assert _codex_mcp_overrides(path, []) == []

    def test_malformed_json_returns_empty(self) -> None:
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        f.write("not json {{{")
        f.close()
        assert _codex_mcp_overrides(f.name, []) == []


# ---------------------------------------------------------------------------
# CodexAgent.build_cmd
# ---------------------------------------------------------------------------

class TestCodexBuildCmd:
    def test_minimal(self) -> None:
        agent = CodexAgent()
        cmd = agent.build_cmd("do stuff", AgentConfig())
        # `codex exec` always emits sandbox + approval overrides — see
        # _BaseCodexAgent.build_cmd for the rationale (MCP elicitation
        # auto-cancels otherwise).
        assert cmd == [
            "codex", "exec",
            "-s", "danger-full-access",
            "-c", 'approval_policy="never"',
            "do stuff", "--json",
        ]

    def test_stdin_mode_prompt_none(self) -> None:
        # prompt=None omits positional arg; codex reads from stdin
        agent = CodexAgent()
        cmd = agent.build_cmd(None, AgentConfig())
        assert cmd == [
            "codex", "exec",
            "-s", "danger-full-access",
            "-c", 'approval_policy="never"',
            "--json",
        ]

    def test_no_oss_flag(self) -> None:
        agent = CodexAgent()
        cmd = agent.build_cmd("task", AgentConfig())
        assert "--oss" not in cmd

    def test_working_directory(self) -> None:
        agent = CodexAgent()
        cfg = AgentConfig(working_directory="/project")
        cmd = agent.build_cmd("task", cfg)
        assert "-C" in cmd
        idx = cmd.index("-C")
        assert cmd[idx + 1] == "/project"

    def test_model(self) -> None:
        agent = CodexAgent(model="o3")
        cmd = agent.build_cmd("task", AgentConfig())
        assert "-m" in cmd
        idx = cmd.index("-m")
        assert cmd[idx + 1] == "o3"

    def test_sandbox(self) -> None:
        agent = CodexAgent()
        cfg = AgentConfig(sandbox="read-only")
        cmd = agent.build_cmd("task", cfg)
        assert "-s" in cmd
        idx = cmd.index("-s")
        assert cmd[idx + 1] == "read-only"

    def test_effort_emits_toml_override(self) -> None:
        agent = CodexAgent()
        cfg = AgentConfig(effort="high")
        cmd = agent.build_cmd("task", cfg)
        # Effort lands as a -c TOML override
        assert 'model_reasoning_effort="high"' in cmd
        idx = cmd.index('model_reasoning_effort="high"')
        assert cmd[idx - 1] == "-c"

    def test_effort_empty_omitted(self) -> None:
        agent = CodexAgent()
        cmd = agent.build_cmd("task", AgentConfig())
        assert not any("model_reasoning_effort" in t for t in cmd)

    def test_permission_mode_not_emitted(self) -> None:
        # `codex exec` has no -a flag — was a copy-paste from the Claude
        # wrapper. permission_mode is intentionally ignored for codex.
        agent = CodexAgent()
        cfg = AgentConfig(permission_mode="never")
        cmd = agent.build_cmd("task", cfg)
        assert "-a" not in cmd

    def test_worktree_uses_worktree_path(self) -> None:
        agent = CodexAgent()
        cfg = AgentConfig(
            working_directory="/project",
            use_worktree=True,
            worktree_path="/tmp/worktree",
        )
        cmd = agent.build_cmd("task", cfg)
        assert "-C" in cmd
        idx = cmd.index("-C")
        # Worktree path overrides working_directory
        assert cmd[idx + 1] == "/tmp/worktree"

    def test_worktree_without_path_uses_workdir(self) -> None:
        agent = CodexAgent()
        cfg = AgentConfig(
            working_directory="/project",
            use_worktree=True,
        )
        cmd = agent.build_cmd("task", cfg)
        idx = cmd.index("-C")
        assert cmd[idx + 1] == "/project"

    def test_mcp_config_emits_overrides_before_prompt(self) -> None:
        # Write a minimal MCP config and verify -c overrides land before
        # the prompt positional and before --json.
        f = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False,
        )
        json.dump({
            "mcpServers": {
                "autodev": {"command": "python3", "args": ["-m", "autodev_mcp"]}
            }
        }, f)
        f.close()

        agent = CodexAgent()
        cfg = AgentConfig(mcp_config_path=f.name)
        cmd = agent.build_cmd("task", cfg)
        assert "-c" in cmd
        # MCP overrides come before the positional prompt
        last_c_idx = max(i for i, t in enumerate(cmd) if t == "-c")
        assert cmd.index("task") > last_c_idx

    def test_mcp_with_allowed_tools_specific(self) -> None:
        f = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False,
        )
        json.dump({"mcpServers": {"autodev": {"command": "x"}}}, f)
        f.close()

        agent = CodexAgent()
        cfg = AgentConfig(
            mcp_config_path=f.name,
            allowed_tools=["mcp__autodev__get_task"],
        )
        cmd = agent.build_cmd("task", cfg)
        assert any(
            "enabled_tools" in t and "get_task" in t for t in cmd
        )

    def test_claude_builtin_tools_silently_dropped(self) -> None:
        # Tools like Read/Write are Claude-only; codex command should not
        # contain them in any form.
        f = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False,
        )
        json.dump({"mcpServers": {"autodev": {"command": "x"}}}, f)
        f.close()

        agent = CodexAgent()
        cfg = AgentConfig(
            mcp_config_path=f.name,
            allowed_tools=["Read", "Write", "Bash(*)"],
        )
        cmd = agent.build_cmd("task", cfg)
        joined = " ".join(cmd)
        assert "Read" not in joined
        assert "Write" not in joined
        assert "Bash" not in joined


# ---------------------------------------------------------------------------
# OllamaCodexAgent.build_cmd
# ---------------------------------------------------------------------------

class TestOllamaCodexBuildCmd:
    def test_oss_flag_present(self) -> None:
        agent = OllamaCodexAgent()
        cmd = agent.build_cmd("do stuff", AgentConfig())
        assert "--oss" in cmd
        assert cmd == [
            "codex", "exec", "--oss",
            "-s", "danger-full-access",
            "-c", 'approval_policy="never"',
            "do stuff", "--json",
        ]

    def test_model(self) -> None:
        agent = OllamaCodexAgent(model="qwen2.5:3b")
        cmd = agent.build_cmd("task", AgentConfig())
        assert "-m" in cmd
        idx = cmd.index("-m")
        assert cmd[idx + 1] == "qwen2.5:3b"

    def test_working_directory(self) -> None:
        agent = OllamaCodexAgent()
        cfg = AgentConfig(working_directory="/project")
        cmd = agent.build_cmd("task", cfg)
        assert "-C" in cmd
        idx = cmd.index("-C")
        assert cmd[idx + 1] == "/project"

    def test_stdin_mode_prompt_none(self) -> None:
        agent = OllamaCodexAgent()
        cmd = agent.build_cmd(None, AgentConfig())
        assert cmd == [
            "codex", "exec", "--oss",
            "-s", "danger-full-access",
            "-c", 'approval_policy="never"',
            "--json",
        ]


# ---------------------------------------------------------------------------
# ClaudeCodeAgent.run (mocked subprocess)
# ---------------------------------------------------------------------------

class TestClaudeCodeRun:
    @patch("llm_provider.agent.subprocess.run")
    def test_success(self, mock_run: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout='{"result": "Task done", "model": "claude-sonnet-4-6"}\n',
            stderr="",
        )
        agent = ClaudeCodeAgent()
        result = agent.run("fix bug")
        assert result.exit_code == 0
        assert result.output == "Task done"
        assert result.backend == "claude-code"
        assert result.error is None
        assert len(result.raw_events) == 1

    @patch("llm_provider.agent.subprocess.run")
    def test_failure(self, mock_run: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=1,
            stdout="", stderr="Error: something broke",
        )
        agent = ClaudeCodeAgent()
        result = agent.run("bad task")
        assert result.exit_code == 1
        assert result.error == "Error: something broke"

    @patch("llm_provider.agent.subprocess.run")
    def test_timeout(self, mock_run: MagicMock) -> None:
        exc = subprocess.TimeoutExpired(cmd=["claude"], timeout=30)
        exc.stdout = "partial output"
        mock_run.side_effect = exc
        agent = ClaudeCodeAgent()
        cfg = AgentConfig(timeout=30)
        result = agent.run("slow task", config=cfg)
        assert result.exit_code == -1
        assert "Timed out" in (result.error or "")
        assert result.output == "partial output"

    @patch("llm_provider.agent.subprocess.run")
    def test_working_directory_in_result(self, mock_run: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr="",
        )
        agent = ClaudeCodeAgent()
        cfg = AgentConfig(working_directory="/my/project")
        result = agent.run("task", config=cfg)
        assert result.working_directory == "/my/project"

    @patch("llm_provider.agent.subprocess.run")
    def test_model_from_constructor(self, mock_run: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr="",
        )
        agent = ClaudeCodeAgent(model="opus")
        result = agent.run("task")
        assert result.model == "opus"

    @patch("llm_provider.agent.subprocess.run")
    def test_model_extracted_from_events(self, mock_run: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0,
            stdout='{"model": "claude-haiku-4-5-20251001", "result": "ok"}\n',
            stderr="",
        )
        agent = ClaudeCodeAgent()
        result = agent.run("task")
        assert result.model == "claude-haiku-4-5-20251001"

    @patch("llm_provider.agent.subprocess.run")
    def test_non_json_stdout(self, mock_run: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0,
            stdout="plain text output\n",
            stderr="",
        )
        agent = ClaudeCodeAgent()
        result = agent.run("task")
        assert result.output == "plain text output\n"
        assert result.raw_events == []

    @patch("llm_provider.agent.subprocess.run")
    def test_default_config_used(self, mock_run: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr="",
        )
        default_cfg = AgentConfig(working_directory="/default")
        agent = ClaudeCodeAgent(default_config=default_cfg)
        agent.run("task")
        # working_directory goes to cwd= kwarg, not --cd flag
        call_args = mock_run.call_args[0][0]
        assert "--cd" not in call_args
        assert mock_run.call_args[1].get("cwd") == "/default"

    @patch("llm_provider.agent.subprocess.run")
    def test_per_call_config_overrides_default(self, mock_run: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr="",
        )
        default_cfg = AgentConfig(working_directory="/default")
        agent = ClaudeCodeAgent(default_config=default_cfg)
        override_cfg = AgentConfig(working_directory="/override")
        agent.run("task", config=override_cfg)
        call_args = mock_run.call_args[0][0]
        assert "--cd" not in call_args
        assert mock_run.call_args[1].get("cwd") == "/override"

    @patch("llm_provider.agent.subprocess.run")
    def test_env_passed_to_subprocess(self, mock_run: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr="",
        )
        agent = ClaudeCodeAgent()
        cfg = AgentConfig(env={"CUSTOM_VAR": "value"})
        agent.run("task", config=cfg)
        env_arg = mock_run.call_args[1].get("env")
        assert env_arg is not None
        assert env_arg["CUSTOM_VAR"] == "value"


# ---------------------------------------------------------------------------
# ClaudeCodeAgent.stream (mocked subprocess)
# ---------------------------------------------------------------------------

class TestClaudeCodeStream:
    @patch("llm_provider.agent.subprocess.Popen")
    def test_yields_lines(self, mock_popen: MagicMock) -> None:
        lines = ['{"type": "start"}\n', '{"result": "done"}\n']
        mock_proc = MagicMock()
        mock_proc.stdout = iter(lines)
        mock_proc.stderr = MagicMock()
        mock_proc.stderr.read.return_value = ""
        mock_proc.returncode = 0
        mock_popen.return_value = mock_proc

        agent = ClaudeCodeAgent()
        collected = list(agent.stream("task"))
        assert len(collected) == 2
        assert collected[0] == '{"type": "start"}'
        assert collected[1] == '{"result": "done"}'

    @patch("llm_provider.agent.subprocess.Popen")
    def test_last_result_populated(self, mock_popen: MagicMock) -> None:
        lines = ['{"result": "all good", "model": "sonnet"}\n']
        mock_proc = MagicMock()
        mock_proc.stdout = iter(lines)
        mock_proc.stderr = MagicMock()
        mock_proc.stderr.read.return_value = ""
        mock_proc.returncode = 0
        mock_popen.return_value = mock_proc

        agent = ClaudeCodeAgent()
        # Must exhaust the iterator
        for _ in agent.stream("task"):
            pass
        result = agent.last_result
        assert result is not None
        assert result.exit_code == 0
        assert result.output == "all good"
        assert result.backend == "claude-code"

    @patch("llm_provider.agent.subprocess.Popen")
    def test_callback_invoked(self, mock_popen: MagicMock) -> None:
        lines = ["line one\n", "line two\n"]
        mock_proc = MagicMock()
        mock_proc.stdout = iter(lines)
        mock_proc.stderr = MagicMock()
        mock_proc.stderr.read.return_value = ""
        mock_proc.returncode = 0
        mock_popen.return_value = mock_proc

        collected: list[str] = []
        agent = ClaudeCodeAgent()
        agent.set_callback(lambda line: collected.append(line))
        for _ in agent.stream("task"):
            pass
        assert collected == ["line one", "line two"]


# ---------------------------------------------------------------------------
# CodexAgent.run (mocked subprocess)
# ---------------------------------------------------------------------------

class TestCodexRun:
    @patch("llm_provider.agent.subprocess.run")
    def test_success(self, mock_run: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0,
            stdout='{"result": "Done by codex"}\n',
            stderr="",
        )
        agent = CodexAgent()
        result = agent.run("fix bug")
        assert result.exit_code == 0
        assert result.output == "Done by codex"
        assert result.backend == "codex"

    @patch("llm_provider.agent.subprocess.run")
    def test_timeout(self, mock_run: MagicMock) -> None:
        exc = subprocess.TimeoutExpired(cmd=["codex"], timeout=60)
        exc.stdout = ""
        mock_run.side_effect = exc
        agent = CodexAgent()
        cfg = AgentConfig(timeout=60)
        result = agent.run("task", config=cfg)
        assert result.exit_code == -1
        assert result.backend == "codex"


# ---------------------------------------------------------------------------
# OllamaCodexAgent.run (mocked subprocess)
# ---------------------------------------------------------------------------

class TestOllamaCodexRun:
    @patch("llm_provider.agent.subprocess.run")
    def test_success(self, mock_run: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0,
            stdout='{"result": "Local model done"}\n',
            stderr="",
        )
        agent = OllamaCodexAgent()
        result = agent.run("analyze code")
        assert result.exit_code == 0
        assert result.output == "Local model done"
        assert result.backend == "codex-oss"

    @patch("llm_provider.agent.subprocess.run")
    def test_oss_in_command(self, mock_run: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr="",
        )
        agent = OllamaCodexAgent()
        agent.run("task")
        call_args = mock_run.call_args[0][0]
        assert "--oss" in call_args


# ---------------------------------------------------------------------------
# is_available
# ---------------------------------------------------------------------------

class TestIsAvailable:
    @patch("shutil.which", return_value="/usr/bin/claude")
    def test_claude_available(self, _mock: MagicMock) -> None:
        assert ClaudeCodeAgent().is_available() is True

    @patch("shutil.which", return_value=None)
    def test_claude_not_available(self, _mock: MagicMock) -> None:
        assert ClaudeCodeAgent().is_available() is False

    @patch("shutil.which", return_value="/usr/bin/codex")
    def test_codex_available(self, _mock: MagicMock) -> None:
        assert CodexAgent().is_available() is True

    @patch("shutil.which", return_value="/usr/bin/codex")
    def test_ollama_codex_available(self, _mock: MagicMock) -> None:
        assert OllamaCodexAgent().is_available() is True


# ---------------------------------------------------------------------------
# executable
# ---------------------------------------------------------------------------

class TestExecutable:
    def test_claude(self) -> None:
        assert ClaudeCodeAgent().executable() == "claude"

    def test_codex(self) -> None:
        assert CodexAgent().executable() == "codex"

    def test_ollama_codex(self) -> None:
        assert OllamaCodexAgent().executable() == "codex"


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------

class TestContextManager:
    def test_claude_code_context_manager(self) -> None:
        with ClaudeCodeAgent() as agent:
            assert isinstance(agent, AgentBackend)

    def test_codex_context_manager(self) -> None:
        with CodexAgent() as agent:
            assert isinstance(agent, AgentBackend)


# ---------------------------------------------------------------------------
# get_agent factory
# ---------------------------------------------------------------------------

class TestGetAgent:
    def test_claude_code(self) -> None:
        agent = get_agent("claude-code")
        assert isinstance(agent, ClaudeCodeAgent)

    def test_codex(self) -> None:
        agent = get_agent("codex")
        assert isinstance(agent, CodexAgent)

    def test_ollama(self) -> None:
        agent = get_agent("ollama")
        assert isinstance(agent, OllamaCodexAgent)

    def test_codex_oss_alias(self) -> None:
        agent = get_agent("codex-oss")
        assert isinstance(agent, OllamaCodexAgent)

    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown agent backend"):
            get_agent("invalid-backend")

    @patch.dict(os.environ, {"AGENT_BACKEND": "codex"})
    def test_env_var_fallback(self) -> None:
        agent = get_agent()
        assert isinstance(agent, CodexAgent)

    @patch.dict(os.environ, {"AGENT_BACKEND": "ollama"})
    def test_env_var_ollama(self) -> None:
        agent = get_agent()
        assert isinstance(agent, OllamaCodexAgent)

    def test_default_is_claude_code(self) -> None:
        # Clear AGENT_BACKEND if set
        env = os.environ.copy()
        env.pop("AGENT_BACKEND", None)
        with patch.dict(os.environ, env, clear=True):
            agent = get_agent()
            assert isinstance(agent, ClaudeCodeAgent)

    def test_model_passed_to_agent(self) -> None:
        agent = get_agent("claude-code", model="opus")
        assert isinstance(agent, ClaudeCodeAgent)
        assert agent._model == "opus"

    def test_config_passed_to_agent(self) -> None:
        cfg = AgentConfig(working_directory="/work")
        agent = get_agent("codex", config=cfg)
        assert isinstance(agent, CodexAgent)
        assert agent._default_config.working_directory == "/work"

    def test_callback_registered(self) -> None:
        lines: list[str] = []
        agent = get_agent("claude-code", callback=lambda l: lines.append(l))
        assert agent._callback is not None


# ---------------------------------------------------------------------------
# Model from environment variable
# ---------------------------------------------------------------------------

class TestModelEnvVar:
    @patch.dict(os.environ, {"CLAUDE_MODEL": "claude-opus-4-6"})
    def test_claude_model_from_env(self) -> None:
        agent = ClaudeCodeAgent()
        assert agent._model == "claude-opus-4-6"

    @patch.dict(os.environ, {"CODEX_MODEL": "o3"})
    def test_codex_model_from_env(self) -> None:
        agent = CodexAgent()
        assert agent._model == "o3"

    @patch.dict(os.environ, {"OLLAMA_MODEL": "qwen2.5:3b"})
    def test_ollama_model_from_env(self) -> None:
        agent = OllamaCodexAgent()
        assert agent._model == "qwen2.5:3b"

    def test_constructor_overrides_env(self) -> None:
        with patch.dict(os.environ, {"CLAUDE_MODEL": "from-env"}):
            agent = ClaudeCodeAgent(model="from-arg")
            assert agent._model == "from-arg"


# ---------------------------------------------------------------------------
# Backend name
# ---------------------------------------------------------------------------

class TestBackendName:
    def test_codex_backend_name(self) -> None:
        assert CodexAgent()._backend_name == "codex"

    def test_ollama_backend_name(self) -> None:
        assert OllamaCodexAgent()._backend_name == "codex-oss"


# ---------------------------------------------------------------------------
# list_agent_models
# ---------------------------------------------------------------------------

from llm_provider.agent import CLAUDE_CODE_MODELS, CODEX_MODELS, list_agent_models


class TestListAgentModels:
    def test_claude_code_returns_list(self) -> None:
        models = list_agent_models("claude-code")
        assert isinstance(models, list)
        assert "claude-sonnet-4-6" in models
        assert "claude-opus-4-7" in models

    def test_claude_code_returns_copy(self) -> None:
        models = list_agent_models("claude-code")
        models.clear()
        assert len(CLAUDE_CODE_MODELS) > 0

    def test_codex_returns_list(self) -> None:
        models = list_agent_models("codex")
        assert isinstance(models, list)
        assert "gpt-5.4" in models
        assert "gpt-5.3-codex" in models

    def test_codex_returns_copy(self) -> None:
        models = list_agent_models("codex")
        models.clear()
        assert len(CODEX_MODELS) > 0

    def test_ollama_returns_names_on_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import httpx

        fake_response = MagicMock()
        fake_response.json.return_value = {
            "models": [{"name": "qwen2.5:3b"}, {"name": "llama3:8b"}]
        }
        monkeypatch.setattr(httpx, "get", lambda *a, **kw: fake_response)
        assert list_agent_models("ollama") == ["qwen2.5:3b", "llama3:8b"]

    def test_codex_oss_alias(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import httpx

        fake_response = MagicMock()
        fake_response.json.return_value = {"models": [{"name": "phi4"}]}
        monkeypatch.setattr(httpx, "get", lambda *a, **kw: fake_response)
        assert list_agent_models("codex-oss") == ["phi4"]

    def test_ollama_returns_empty_on_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import httpx

        monkeypatch.setattr(httpx, "get", lambda *a, **kw: (_ for _ in ()).throw(httpx.ConnectError("x")))
        assert list_agent_models("ollama") == []

    def test_unknown_backend_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown backend"):
            list_agent_models("nope")
