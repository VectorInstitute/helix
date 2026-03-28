"""Agent backend abstraction.

Defines the ``AgentBackend`` protocol so helix is not tied to any specific
agent SDK. The default backend uses ``claude-agent-sdk``, but any implementation
that satisfies ``AgentBackend`` can be dropped in.

Built-in backends
-----------------
``ClaudeBackend``
    Spawns Claude Code as a subprocess via ``claude-agent-sdk``.
    Requires: ``pip install 'helices[claude]'`` and Claude Code CLI.

``GeminiBackend``
    Spawns Gemini CLI as a subprocess, parsing its ``stream-json`` output.
    Requires: ``npm install -g @google/gemini-cli`` (no extra Python package).

To use a custom backend, pass it to ``HelixRunner``::

    runner = HelixRunner(root, tag, max_turns, backend=MyBackend())
"""

from __future__ import annotations

import asyncio
import json
import shutil
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


class GeminiBackend:
    """Backend powered by the Gemini CLI (``@google/gemini-cli``).

    Spawns ``gemini`` as a subprocess in headless mode with ``--output-format
    stream-json``, then parses its newline-delimited JSON messages into the
    normalised ``AgentEvent`` stream.

    Requirements
    ------------
    Install the Gemini CLI via npm::

        npm install -g @google/gemini-cli

    No additional Python package is required. The CLI must be authenticated
    (run ``gemini`` interactively once, or set ``GEMINI_API_KEY``).

    Notes
    -----
    ``system_prompt`` is prepended to ``prompt`` as Gemini CLI has no
    dedicated system-prompt flag. ``max_turns`` is not directly enforced by
    the CLI; the session runs until the model stops producing tool calls.
    ``cost_usd`` is always ``None`` as the CLI does not report cost.
    """

    async def run(
        self,
        prompt: str,
        system_prompt: str,
        cwd: Path,
        model: str,
        max_turns: int,
    ) -> AsyncIterator[AgentEvent]:
        """Run a Gemini CLI session and yield normalised events.

        Parameters
        ----------
        prompt : str
            Initial user prompt.
        system_prompt : str
            System-level instructions (prepended to the prompt).
        cwd : Path
            Working directory (helix root).
        model : str
            Gemini model ID (e.g. ``"gemini-2.5-pro"``).
        max_turns : int
            Passed for interface compatibility; not enforced by the CLI.

        Yields
        ------
        AgentEvent
            Normalised events parsed from the gemini stream-json output.

        Raises
        ------
        FileNotFoundError
            If the ``gemini`` CLI binary is not found on ``PATH``.
        """
        if shutil.which("gemini") is None:
            raise FileNotFoundError(
                "gemini CLI not found. "
                "Install it with: npm install -g @google/gemini-cli\n"
                "Then authenticate by running `gemini` interactively once, "
                "or set the GEMINI_API_KEY environment variable."
            )

        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

        cmd = [
            "gemini",
            "--prompt",
            full_prompt,
            "--yolo",
            "--output-format",
            "stream-json",
            "--model",
            model,
        ]

        async def _stream() -> AsyncIterator[AgentEvent]:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )

            assert proc.stdout is not None
            tool_calls = 0

            async for raw_line in proc.stdout:
                line = raw_line.decode().strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue

                msg_type = msg.get("type")

                if msg_type == "init":
                    yield SessionStarted(session_id=msg.get("session_id", ""))

                elif msg_type == "message" and msg.get("role") == "assistant":
                    content = msg.get("content", "")
                    if content:
                        yield TextOutput(text=content)

                elif msg_type == "result":
                    stats = msg.get("stats", {})
                    tool_calls = stats.get("tool_calls", 0)
                    yield SessionFinished(
                        turns=tool_calls,
                        cost_usd=None,
                        error=msg.get("status") != "success",
                    )

            await proc.wait()

        return _stream()
