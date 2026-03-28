"""Tests for helix.agent."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from helix.agent import AgentBackend, ClaudeBackend, GeminiBackend, SessionFinished, SessionStarted, TextOutput


class TestAgentEvents:
    def test_session_started_fields(self) -> None:
        event = SessionStarted(session_id="abc-123")
        assert event.session_id == "abc-123"

    def test_text_output_fields(self) -> None:
        event = TextOutput(text="hello world")
        assert event.text == "hello world"

    def test_session_finished_fields(self) -> None:
        event = SessionFinished(turns=42, cost_usd=0.05, error=False)
        assert event.turns == 42
        assert event.cost_usd == pytest.approx(0.05)
        assert event.error is False

    def test_session_finished_error(self) -> None:
        event = SessionFinished(turns=1, cost_usd=None, error=True)
        assert event.error is True
        assert event.cost_usd is None

    def test_events_are_frozen(self) -> None:
        event = SessionStarted(session_id="x")
        with pytest.raises(AttributeError):
            event.session_id = "y"  # type: ignore[misc]


class TestAgentBackendProtocol:
    def test_claude_backend_satisfies_protocol(self) -> None:
        backend = ClaudeBackend()
        assert isinstance(backend, AgentBackend)

    def test_custom_backend_satisfies_protocol(self) -> None:
        class MyBackend:
            async def run(self, prompt, system_prompt, cwd, model, max_turns):  # type: ignore[override]
                yield SessionStarted(session_id="x")

        assert isinstance(MyBackend(), AgentBackend)

    def test_object_without_run_does_not_satisfy_protocol(self) -> None:
        class NotABackend:
            pass

        assert not isinstance(NotABackend(), AgentBackend)


class TestClaudeBackendStream:
    @pytest.mark.asyncio
    async def test_full_stream_session_started_text_finished(self, tmp_path: Path) -> None:
        """Test ClaudeBackend._stream by injecting a real-looking mock SDK module."""

        class SystemMessage:
            def __init__(self, subtype: str, data: dict[str, str]) -> None:
                self.subtype = subtype
                self.data = data

        class TextBlock:
            def __init__(self, text: str) -> None:
                self.text = text

        class AssistantMessage:
            def __init__(self, content: list[TextBlock]) -> None:
                self.content = content

        class ResultMessage:
            def __init__(self, num_turns: int, total_cost_usd: float, is_error: bool) -> None:
                self.num_turns = num_turns
                self.total_cost_usd = total_cost_usd
                self.is_error = is_error

        class ClaudeAgentOptions:
            def __init__(self, **kwargs: object) -> None:
                pass

        async def fake_query(prompt: str, options: object) -> AsyncIterator[object]:
            yield SystemMessage("init", {"session_id": "sid-stream"})
            yield AssistantMessage([TextBlock("thinking..."), TextBlock("")])
            yield ResultMessage(7, 0.05, False)

        mock_sdk = MagicMock()
        mock_sdk.SystemMessage = SystemMessage
        mock_sdk.AssistantMessage = AssistantMessage
        mock_sdk.TextBlock = TextBlock
        mock_sdk.ResultMessage = ResultMessage
        mock_sdk.ClaudeAgentOptions = ClaudeAgentOptions
        mock_sdk.query = fake_query

        with patch.dict(sys.modules, {"claude_agent_sdk": mock_sdk}):
            backend = ClaudeBackend()
            stream = await backend.run(
                prompt="test",
                system_prompt="sys",
                cwd=tmp_path,
                model="claude-opus-4-6",
                max_turns=10,
            )
            events = [e async for e in stream]

        assert isinstance(events[0], SessionStarted)
        assert events[0].session_id == "sid-stream"
        assert isinstance(events[1], TextOutput)
        assert events[1].text == "thinking..."
        assert isinstance(events[2], SessionFinished)
        assert events[2].turns == 7
        assert events[2].cost_usd == pytest.approx(0.05)
        assert events[2].error is False

    @pytest.mark.asyncio
    async def test_stream_skips_non_init_system_messages(self, tmp_path: Path) -> None:
        class SystemMessage:
            def __init__(self, subtype: str, data: dict[str, str]) -> None:
                self.subtype = subtype
                self.data = data

        class ClaudeAgentOptions:
            def __init__(self, **kwargs: object) -> None:
                pass

        async def fake_query(prompt: str, options: object) -> AsyncIterator[object]:
            yield SystemMessage("other", {"session_id": "x"})

        mock_sdk = MagicMock()
        mock_sdk.SystemMessage = SystemMessage
        mock_sdk.AssistantMessage = type("AssistantMessage", (), {})
        mock_sdk.TextBlock = type("TextBlock", (), {})
        mock_sdk.ResultMessage = type("ResultMessage", (), {})
        mock_sdk.ClaudeAgentOptions = ClaudeAgentOptions
        mock_sdk.query = fake_query

        with patch.dict(sys.modules, {"claude_agent_sdk": mock_sdk}):
            backend = ClaudeBackend()
            stream = await backend.run(
                prompt="test",
                system_prompt="sys",
                cwd=tmp_path,
                model="claude-opus-4-6",
                max_turns=10,
            )
            events = [e async for e in stream]

        assert events == []


class TestClaudeBackendImportError:
    @pytest.mark.asyncio
    async def test_helpful_error_when_sdk_not_installed(self, tmp_path: Path) -> None:
        with patch.dict(sys.modules, {"claude_agent_sdk": None}):
            backend = ClaudeBackend()
            with pytest.raises(ImportError, match="pip install 'helices\\[claude\\]'"):
                await backend.run(
                    prompt="test",
                    system_prompt="sys",
                    cwd=tmp_path,
                    model="claude-opus-4-6",
                    max_turns=10,
                )

    @pytest.mark.asyncio
    async def test_error_mentions_claude_code(self, tmp_path: Path) -> None:
        with patch.dict(sys.modules, {"claude_agent_sdk": None}):
            backend = ClaudeBackend()
            with pytest.raises(ImportError, match="Claude Code CLI"):
                await backend.run(
                    prompt="test",
                    system_prompt="sys",
                    cwd=tmp_path,
                    model="claude-opus-4-6",
                    max_turns=10,
                )


class TestClaudeBackend:
    @pytest.mark.asyncio
    async def test_yields_session_started(self, tmp_path: Path) -> None:

        mock_system = MagicMock()
        mock_system.subtype = "init"
        mock_system.data = {"session_id": "sid-999"}

        mock_result = MagicMock()
        mock_result.num_turns = 10
        mock_result.total_cost_usd = 0.12
        mock_result.is_error = False

        async def fake_query(**kwargs):  # type: ignore[no-untyped-def]
            yield mock_system
            yield mock_result

        with patch("helix.agent.ClaudeBackend.run") as mock_run:

            async def side_effect(*args, **kwargs):  # type: ignore[no-untyped-def]
                async def _stream():  # type: ignore[no-untyped-def]
                    yield SessionStarted(session_id="sid-999")
                    yield SessionFinished(turns=10, cost_usd=0.12, error=False)

                return _stream()

            mock_run.side_effect = side_effect
            backend = ClaudeBackend()
            stream = await backend.run(
                prompt="test",
                system_prompt="sys",
                cwd=tmp_path,
                model="claude-opus-4-6",
                max_turns=10,
            )
            events = [e async for e in stream]

        assert isinstance(events[0], SessionStarted)
        assert events[0].session_id == "sid-999"
        assert isinstance(events[1], SessionFinished)
        assert events[1].turns == 10


class TestGeminiBackendProtocol:
    def test_gemini_backend_satisfies_protocol(self) -> None:
        assert isinstance(GeminiBackend(), AgentBackend)


class TestGeminiBackendMissingCLI:
    @pytest.mark.asyncio
    async def test_raises_when_gemini_not_on_path(self, tmp_path: Path) -> None:
        with patch("helix.agent.shutil.which", return_value=None):
            backend = GeminiBackend()
            with pytest.raises(FileNotFoundError, match="npm install -g @google/gemini-cli"):
                await backend.run(
                    prompt="test",
                    system_prompt="",
                    cwd=tmp_path,
                    model="gemini-2.5-pro",
                    max_turns=10,
                )


def _make_gemini_proc(*lines: bytes) -> MagicMock:
    """Build a mock subprocess whose stdout async-iterates over ``lines``."""

    async def _aiter() -> AsyncIterator[bytes]:
        for line in lines:
            yield line

    mock_proc = MagicMock()
    mock_proc.stdout = _aiter()
    mock_proc.wait = AsyncMock()
    return mock_proc


class TestGeminiBackend:
    @pytest.mark.asyncio
    async def test_yields_events_from_stream_json(self, tmp_path: Path) -> None:
        mock_proc = _make_gemini_proc(
            b'{"type":"init","session_id":"gem-123","model":"gemini-2.5-pro"}\n',
            b'{"type":"message","role":"assistant","content":"Hello world","delta":true}\n',
            b'{"type":"result","status":"success","stats":{"tool_calls":3}}\n',
        )

        with (
            patch("helix.agent.shutil.which", return_value="/usr/bin/gemini"),
            patch("helix.agent.asyncio.create_subprocess_exec", return_value=mock_proc),
        ):
            backend = GeminiBackend()
            stream = await backend.run(
                prompt="test",
                system_prompt="be helpful",
                cwd=tmp_path,
                model="gemini-2.5-pro",
                max_turns=10,
            )
            events = [e async for e in stream]

        assert isinstance(events[0], SessionStarted)
        assert events[0].session_id == "gem-123"
        assert isinstance(events[1], TextOutput)
        assert events[1].text == "Hello world"
        assert isinstance(events[2], SessionFinished)
        assert events[2].turns == 3
        assert events[2].cost_usd is None
        assert events[2].error is False

    @pytest.mark.asyncio
    async def test_error_status_sets_error_flag(self, tmp_path: Path) -> None:
        mock_proc = _make_gemini_proc(
            b'{"type":"init","session_id":"gem-err","model":"gemini-2.5-pro"}\n',
            b'{"type":"result","status":"error","stats":{"tool_calls":0}}\n',
        )

        with (
            patch("helix.agent.shutil.which", return_value="/usr/bin/gemini"),
            patch("helix.agent.asyncio.create_subprocess_exec", return_value=mock_proc),
        ):
            backend = GeminiBackend()
            stream = await backend.run(
                prompt="test",
                system_prompt="",
                cwd=tmp_path,
                model="gemini-2.5-pro",
                max_turns=10,
            )
            events = [e async for e in stream]

        finished = next(e for e in events if isinstance(e, SessionFinished))
        assert finished.error is True

    @pytest.mark.asyncio
    async def test_system_prompt_prepended_to_prompt(self, tmp_path: Path) -> None:
        captured_cmd: list[str] = []

        async def fake_subprocess(*args: str, **kwargs: object) -> MagicMock:
            captured_cmd.extend(args)
            return _make_gemini_proc()

        with (
            patch("helix.agent.shutil.which", return_value="/usr/bin/gemini"),
            patch("helix.agent.asyncio.create_subprocess_exec", side_effect=fake_subprocess),
        ):
            backend = GeminiBackend()
            stream = await backend.run(
                prompt="do the thing",
                system_prompt="be concise",
                cwd=tmp_path,
                model="gemini-2.5-pro",
                max_turns=10,
            )
            [e async for e in stream]

        prompt_idx = captured_cmd.index("--prompt") + 1
        assert "be concise" in captured_cmd[prompt_idx]
        assert "do the thing" in captured_cmd[prompt_idx]

    @pytest.mark.asyncio
    async def test_blank_lines_skipped(self, tmp_path: Path) -> None:
        mock_proc = _make_gemini_proc(
            b"\n",
            b"   \n",
            b'{"type":"result","status":"success","stats":{"tool_calls":0}}\n',
        )
        with (
            patch("helix.agent.shutil.which", return_value="/usr/bin/gemini"),
            patch("helix.agent.asyncio.create_subprocess_exec", return_value=mock_proc),
        ):
            backend = GeminiBackend()
            stream = await backend.run("test", "", tmp_path, "gemini-2.5-pro", 10)
            events = [e async for e in stream]

        assert all(not isinstance(e, SessionStarted) for e in events)
        assert any(isinstance(e, SessionFinished) for e in events)

    @pytest.mark.asyncio
    async def test_invalid_json_lines_skipped(self, tmp_path: Path) -> None:
        mock_proc = _make_gemini_proc(
            b"not valid json\n",
            b'{"type":"result","status":"success","stats":{"tool_calls":0}}\n',
        )
        with (
            patch("helix.agent.shutil.which", return_value="/usr/bin/gemini"),
            patch("helix.agent.asyncio.create_subprocess_exec", return_value=mock_proc),
        ):
            backend = GeminiBackend()
            stream = await backend.run("test", "", tmp_path, "gemini-2.5-pro", 10)
            events = [e async for e in stream]

        assert len(events) == 1
        assert isinstance(events[0], SessionFinished)
