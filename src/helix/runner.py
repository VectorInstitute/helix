"""Research session runner — orchestrates the Claude agent loop for any helix."""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import os
import signal
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from types import FrameType

import anyio
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    CLIConnectionError,
    CLINotFoundError,
    ResultMessage,
    SystemMessage,
    TextBlock,
    query,
)
from rich.console import Console
from rich.rule import Rule

from .config import HelixConfig, OptimizeDirection
from .display import session_summary_panel, startup_panel
from .git import GitError, current_branch, detect_main_branch, run as git, show_file
from .results import append_experiments, best_kept, read_main_stats, read_results


console = Console()


def find_helix_root(start: Path | None = None) -> Path:
    """Walk up the directory tree to find the directory containing ``helix.yaml``.

    Parameters
    ----------
    start : Path, optional
        Directory to begin searching from. Defaults to ``Path.cwd()``.

    Returns
    -------
    Path
        Directory containing ``helix.yaml``.

    Raises
    ------
    FileNotFoundError
        If no ``helix.yaml`` is found in the tree.
    """
    path = start or Path.cwd()
    for directory in [path, *path.parents]:
        if (directory / "helix.yaml").exists():
            return directory
    raise FileNotFoundError("No helix.yaml found in the current directory or any parent.")


class HelixRunner:
    """Orchestrates a single autonomous research session for a helix.

    A session:

    1. Creates (or resumes) a ``helix/<tag>`` branch.
    2. Runs the Claude Opus agent, which reads ``program.md`` and experiments in a loop.
    3. After the agent finishes (or is interrupted), merges improvements to the main branch.
    """

    def __init__(self, helix_root: Path, tag: str, max_turns: int) -> None:
        self.root = helix_root
        self.config = HelixConfig.load(helix_root / "helix.yaml")
        self.tag = tag
        self.branch = f"helix/{tag}"
        self.max_turns = max_turns
        self._main_branch = detect_main_branch(helix_root)
        self._log_path = helix_root / "run.log"
        self._pid_path = helix_root / "run.pid"
        self._results_path = helix_root / "results.tsv"
        self._experiments_path = helix_root / "experiments.tsv"

    def _kill_experiment(self) -> None:
        """Send SIGTERM to any experiment process tracked in run.pid."""
        if not self._pid_path.exists():
            return
        with contextlib.suppress(ValueError, OSError):
            pid = int(self._pid_path.read_text().strip())
            os.kill(pid, signal.SIGTERM)
        with contextlib.suppress(OSError):
            self._pid_path.unlink()

    def _sigterm_handler(self, _signum: int, _frame: FrameType | None) -> None:
        self._kill_experiment()
        sys.exit(0)

    def _preflight(self) -> None:
        """Validate git state and create or resume the session branch."""
        cb = current_branch(self.root)
        if cb == self._main_branch:
            try:
                git("checkout", "-b", self.branch, cwd=self.root)
                console.print(f"[green]✓[/green] Created branch [bold]{self.branch}[/bold]")
            except GitError:
                console.print(
                    f"[red]✗[/red] Branch [bold]{self.branch}[/bold] already exists.\n"
                    "    Use a unique --tag or delete the branch first."
                )
                sys.exit(1)
        elif cb == self.branch:
            console.print(f"[dim]Resuming on existing branch {self.branch}[/dim]")
        else:
            console.print(
                f"[red]✗[/red] Must run from [bold]{self._main_branch}[/bold] or "
                f"[bold]{self.branch}[/bold]. Currently on: [bold]{cb}[/bold]"
            )
            sys.exit(1)

    def _build_prompt(self, main_stats: dict[str, float | None]) -> str:
        """Construct the agent kickoff prompt from helix.yaml and current stats.

        Parameters
        ----------
        main_stats : dict[str, float or None]
            Baseline and best values from the main branch experiments.

        Returns
        -------
        str
            Full prompt string passed to the Claude agent.
        """
        cfg = self.config
        baseline = main_stats.get("baseline")
        best = main_stats.get("best")
        baseline_str = f"{baseline:.4g}" if baseline is not None else "not yet established"
        best_str = f"{best:.4g}" if best is not None else "not yet established"

        primary_name = cfg.metrics.primary.name
        primary_dir = "higher" if cfg.metrics.primary.optimize == OptimizeDirection.maximize else "lower"
        eval_cmd = cfg.metrics.evaluate.command
        grep_hint = cfg.metrics.evaluate.patterns.grep_hint()

        editable = ", ".join(f"`{f}`" for f in cfg.scope.editable)
        readonly = ", ".join(f"`{f}`" for f in cfg.scope.readonly)
        hardware = os.environ.get("HELIX_HARDWARE", "unknown")

        results_cols = f"commit\t{primary_name}"
        if cfg.metrics.quality_guard:
            results_cols += f"\t{cfg.metrics.quality_guard.name}"
        results_cols += "\tstatus\tdescription"

        qg_instructions = ""
        if cfg.metrics.quality_guard:
            qg = cfg.metrics.quality_guard
            qg_dir = "lower" if qg.optimize == OptimizeDirection.minimize else "higher"
            qg_instructions = (
                f"\n## Quality guard: {qg.name} (must not degrade)\n"
                f"`{qg.name}` must stay {qg_dir} within {qg.max_degradation * 100:.1f}% of the baseline. "
                "If it degrades, discard the experiment regardless of primary metric gains.\n"
            )

        return f"""You are running an autonomous research session tagged `{self.tag}`.

## Helix: {cfg.name}
Domain: {cfg.domain}
{cfg.description}

## Environment
Hardware: {hardware}

## Your task
Read `program.md` for full domain-specific instructions. Operational rules:
- You are on branch `{self.branch}`.
- You may modify: {editable}
- Do NOT modify: {readonly}
- Run each experiment: `{eval_cmd} > run.log 2>&1 & echo $! > run.pid; wait $!; rm -f run.pid`
- Check results: `grep "{grep_hint}" run.log`
- Log to `results.tsv` (tab-separated, header: {results_cols}).
- Kept experiments: git commit stays. Discarded: `git reset --hard HEAD~1`.
- Do NOT commit results.tsv — leave it untracked.

## Primary metric: {primary_name} ({primary_dir} is better)
{qg_instructions}
## Targets to beat
- Main branch baseline: **{baseline_str} {primary_name}**
- Main branch best: **{best_str} {primary_name}**

## Start
Read `program.md` first, then scope files, then run the unmodified baseline.
NEVER stop or ask for confirmation. Run until interrupted."""

    async def _monitor_log(self) -> None:
        """Stream run.log to the console, handling file truncation between runs."""
        pos = 0
        run_announced = False
        while True:
            try:
                if self._log_path.exists():
                    size = self._log_path.stat().st_size
                    if size < pos:
                        pos = 0
                        run_announced = False
                    if size > pos:
                        with self._log_path.open("rb") as fh:
                            fh.seek(pos)
                            chunk = fh.read(size - pos)
                        for raw_line in chunk.splitlines():
                            line = raw_line.decode("utf-8", errors="replace").rstrip()
                            if not line:
                                continue
                            if "Starting evaluation" in line and not run_announced:
                                run_announced = True
                                ts = datetime.now().strftime("%H:%M:%S")
                                console.print(f"[dim]{ts}[/dim]  [cyan]▶ running…[/cyan]")
                            console.print(f"  [dim]{line}[/dim]", highlight=False)
                        pos = size
            except OSError:
                pos = 0
                run_announced = False
            await asyncio.sleep(0.15)

    async def _run_agent(self, main_stats: dict[str, float | None]) -> None:
        """Launch the Claude Opus agent and stream filtered output until done.

        Parameters
        ----------
        main_stats : dict[str, float or None]
            Baseline and best values used to build the kickoff prompt.
        """
        prompt = self._build_prompt(main_stats)
        keywords = self.config.interesting_keywords()

        console.print(startup_panel(self.tag, self.max_turns, main_stats, self.config))

        with contextlib.suppress(OSError):
            self._log_path.unlink()

        monitor_task = asyncio.create_task(self._monitor_log())

        options = ClaudeAgentOptions(
            cwd=str(self.root),
            allowed_tools=["Read", "Write", "Edit", "Bash", "Glob", "Grep"],
            permission_mode="bypassPermissions",
            max_turns=self.max_turns,
            model="claude-opus-4-6",
            system_prompt=(
                f"You are an expert researcher autonomously investigating: {self.config.description}. "
                "Be methodical and scientific. Write clean, minimal code. "
                "Never ask for permission or confirmation — act autonomously."
            ),
            setting_sources=[],
        )

        try:
            async for message in query(prompt=prompt, options=options):
                if isinstance(message, SystemMessage) and message.subtype == "init":
                    sid = message.data.get("session_id", "")
                    console.print(f"[dim]session: {sid}[/dim]\n")

                elif isinstance(message, AssistantMessage):
                    for block in message.content:
                        if not isinstance(block, TextBlock):
                            continue
                        lines = block.text.splitlines()
                        interesting = [ln for ln in lines if any(kw in ln.lower() for kw in keywords)]
                        if interesting:
                            ts = datetime.now().strftime("%H:%M:%S")
                            for ln in interesting:
                                console.print(f"[dim]{ts}[/dim]  {ln}")

                elif isinstance(message, ResultMessage):
                    status = "error" if message.is_error else "ok"
                    cost = f"  cost: ${message.total_cost_usd:.4f}" if message.total_cost_usd else ""
                    console.print(f"\n[dim]agent finished — turns: {message.num_turns}  status: {status}{cost}[/dim]")
        finally:
            monitor_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await monitor_task

    def _commit_to_main(self, rows: list[dict[str, str]]) -> None:
        """Append session experiments to main and merge improved files if applicable.

        Parameters
        ----------
        rows : list[dict[str, str]]
            Parsed rows from results.tsv.
        """
        if not rows:
            return

        session_best, session_desc = best_kept(rows, self.config)
        main_stats = read_main_stats(self._main_branch, self.root, self.config)
        main_best = main_stats.get("best")

        direction = self.config.metrics.primary.optimize
        improved = session_best is not None and (
            main_best is None
            or (direction == OptimizeDirection.maximize and session_best > main_best)
            or (direction == OptimizeDirection.minimize and session_best < main_best)
        )

        original_branch = current_branch(self.root)
        try:
            git("checkout", self._main_branch, cwd=self.root)
            append_experiments(self.tag, rows, self._experiments_path, self.config)
            files_to_add = ["experiments.tsv"]

            if improved and session_best is not None and session_desc is not None:
                for rel_path in self.config.scope.editable:
                    try:
                        content = show_file(self.branch, rel_path, self.root)
                        local = self.root / rel_path
                        if content != local.read_text():
                            local.write_text(content)
                            subprocess.run(
                                ["pre-commit", "run", "--files", rel_path],
                                cwd=self.root,
                                capture_output=True,
                                check=False,
                            )
                            files_to_add.append(rel_path)
                    except (GitError, OSError):
                        pass
                primary_name = self.config.metrics.primary.name
                msg = f"helix/{self.tag}: {session_desc} ({primary_name}: {session_best:.4g})"
            else:
                msg = f"helix/{self.tag}: {len(rows)} experiment(s), no improvement"

            git("add", *files_to_add, cwd=self.root)
            git("commit", "-m", msg, cwd=self.root)
            console.print(f"[green]✓[/green] Committed to {self._main_branch}: [italic]{msg}[/italic]")

        except Exception as exc:
            console.print(f"[red]✗[/red] Failed to commit to {self._main_branch}: {exc}")
        finally:
            git("checkout", original_branch, cwd=self.root, check=False)

    def _post_session(self, main_stats: dict[str, float | None]) -> None:
        """Print session summary and commit improvements to main.

        Parameters
        ----------
        main_stats : dict[str, float or None]
            Stats captured before the session started (used for delta display).
        """
        rows = read_results(self._results_path)
        console.print()
        console.print(session_summary_panel(rows, main_stats, self.config))
        self._commit_to_main(rows)

    def run(self) -> None:
        """Run a complete research session: preflight → agent → post-session commit."""
        atexit.register(self._kill_experiment)
        signal.signal(signal.SIGTERM, self._sigterm_handler)

        console.print(Rule(f"[bold cyan]helix[/bold cyan] · {self.config.name}"))

        self._preflight()
        main_stats = read_main_stats(self._main_branch, self.root, self.config)

        try:
            anyio.run(self._run_agent, main_stats)
        except CLINotFoundError:
            console.print("[red]✗[/red] Claude Code CLI not found. Run: pip install claude-agent-sdk")
            sys.exit(1)
        except CLIConnectionError as exc:
            console.print(f"[red]✗[/red] Connection error: {exc}")
            sys.exit(1)
        except KeyboardInterrupt:
            console.print("\n[yellow]⚠[/yellow] Interrupted.")
            self._kill_experiment()

        self._post_session(main_stats)
