"""Tests for helix.agent."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from helix.agent import AgentBackend, ClaudeBackend, SessionFinished, SessionStarted, TextOutput


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
        with pytest.raises(Exception):
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


class TestClaudeBackend:
    @pytest.mark.asyncio
    async def test_yields_session_started(self, tmp_path: Path) -> None:
        from unittest.mock import AsyncMock

        mock_system = MagicMock()
        mock_system.subtype = "init"
        mock_system.data = {"session_id": "sid-999"}

        mock_result = MagicMock()
        mock_result.num_turns = 10
        mock_result.total_cost_usd = 0.12
        mock_result.is_error = False

        async def fake_query(**kwargs):  # type: ignore[no-untyped-def]
            from claude_agent_sdk import AssistantMessage, ResultMessage, SystemMessage
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
