"""Agent backend abstraction.

Defines the ``AgentBackend`` protocol so helix is not tied to any specific
agent SDK. The default backend uses ``claude-agent-sdk``, but any implementation
that satisfies ``AgentBackend`` can be dropped in.

To use a custom backend, pass it to ``HelixRunner``::

    runner = HelixRunner(root, tag, max_turns, backend=MyBackend())
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Protocol, runtime_checkable


@dataclass(frozen=True)
class SessionStarted:
    """Emitted once when the agent session is initialised."""

    session_id: str


@dataclass(frozen=True)
class TextOutput:
    """A text chunk from the agent."""

    text: str


@dataclass(frozen=True)
class SessionFinished:
    """Emitted once when the agent session ends."""

    turns: int
    cost_usd: float | None
    error: bool


AgentEvent = SessionStarted | TextOutput | SessionFinished


@runtime_checkable
class AgentBackend(Protocol):
    """Protocol for agent execution backends.

    Any object implementing this protocol can be used as the agent that drives
    a helix research session.
    """

    async def run(
        self,
        prompt: str,
        system_prompt: str,
        cwd: Path,
        model: str,
        max_turns: int,
    ) -> AsyncIterator[AgentEvent]:
        """Run the agent and yield events until the session ends.

        Parameters
        ----------
        prompt : str
            Initial user prompt (the research session kickoff message).
        system_prompt : str
            System-level instructions for the agent.
        cwd : Path
            Working directory for the agent (the helix root).
        model : str
            Model identifier (interpretation is backend-specific).
        max_turns : int
            Maximum number of agent turns before stopping.

        Yields
        ------
        AgentEvent
            ``SessionStarted``, ``TextOutput``, or ``SessionFinished`` events.
        """
        ...  # pragma: no cover


class ClaudeBackend:
    """Default backend powered by ``claude-agent-sdk``.

    Uses Claude Code as the agent subprocess with tool access to Read, Write,
    Edit, Bash, Glob, and Grep.
    """

    async def run(
        self,
        prompt: str,
        system_prompt: str,
        cwd: Path,
        model: str,
        max_turns: int,
    ) -> AsyncIterator[AgentEvent]:
        """Run a Claude agent session and yield normalised events.

        Parameters
        ----------
        prompt : str
            Initial user prompt.
        system_prompt : str
            System-level instructions.
        cwd : Path
            Working directory (helix root).
        model : str
            Claude model ID (e.g. ``"claude-opus-4-6"``).
        max_turns : int
            Maximum agent turns.

        Yields
        ------
        AgentEvent
            Normalised events from the claude-agent-sdk stream.
        """
        try:
            from claude_agent_sdk import (  # noqa: PLC0415
                AssistantMessage,
                ClaudeAgentOptions,
                ResultMessage,
                SystemMessage,
                TextBlock,
                query,
            )
        except ImportError as exc:
            raise ImportError(
                "claude-agent-sdk is not installed. "
                "Install it with: pip install 'helices[claude]'\n"
                "Note: Claude Code CLI must also be installed on your system. "
                "See https://claude.ai/download"
            ) from exc

        options = ClaudeAgentOptions(
            cwd=str(cwd),
            allowed_tools=["Read", "Write", "Edit", "Bash", "Glob", "Grep"],
            permission_mode="bypassPermissions",
            max_turns=max_turns,
            model=model,
            system_prompt=system_prompt,
            setting_sources=[],
        )

        async def _stream() -> AsyncIterator[AgentEvent]:
            async for message in query(prompt=prompt, options=options):
                if isinstance(message, SystemMessage) and message.subtype == "init":
                    yield SessionStarted(session_id=message.data.get("session_id", ""))

                elif isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock) and block.text:
                            yield TextOutput(text=block.text)

                elif isinstance(message, ResultMessage):
                    yield SessionFinished(
                        turns=message.num_turns,
                        cost_usd=message.total_cost_usd,
                        error=message.is_error,
                    )

        return _stream()
