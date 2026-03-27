"""Rich UI panels for helix research sessions."""

from __future__ import annotations

import os
from datetime import datetime

from rich.panel import Panel
from rich.table import Table

from .config import HelixConfig, OptimizeDirection
from .results import best_kept


def startup_panel(tag: str, max_turns: int, main_stats: dict[str, float | None], config: HelixConfig) -> Panel:
    """Build the Rich panel displayed at session start.

    Args:
        tag: Session tag (e.g. ``"mar27"``).
        max_turns: Maximum agent turns configured for this session.
        main_stats: Dict with ``"baseline"`` and ``"best"`` from the main branch.
        config: Helix configuration.

    Returns:
        A ``rich.panel.Panel`` ready to print.
    """
    hardware = os.environ.get("HELIX_HARDWARE", "unknown")
    date_str = datetime.now().strftime("%Y-%m-%d  %H:%M")
    primary_name = config.metrics.primary.name
    baseline = main_stats.get("baseline")
    best = main_stats.get("best")

    t = Table(show_header=False, box=None, padding=(0, 2))
    t.add_column(style="dim", min_width=18)
    t.add_column()

    t.add_row("helix", f"[cyan]{config.name}[/cyan]")
    t.add_row("domain", f"[dim]{config.domain}[/dim]")
    t.add_row("tag", f"[yellow]{tag}[/yellow]")
    t.add_row("branch", f"[dim]helix/{tag}[/dim]")
    t.add_row("date", f"[dim]{date_str}[/dim]")
    t.add_row("", "")
    t.add_row("hardware", f"[dim]{hardware}[/dim]")
    t.add_row("agent model", "claude-opus-4-6")
    t.add_row("time budget", f"{config.metrics.evaluate.timeout_seconds // 60} min / run")
    t.add_row("max turns", str(max_turns))
    t.add_row("", "")

    fmt = _fmt_metric(primary_name)
    t.add_row("main baseline", fmt(baseline))
    t.add_row("main best", fmt(best))

    return Panel(t, title="[bold cyan]helix[/bold cyan]", expand=False)


def session_summary_panel(
    rows: list[dict[str, str]],
    main_stats: dict[str, float | None],
    config: HelixConfig,
) -> Panel:
    """Build the Rich panel displayed at session end.

    Args:
        rows: Parsed rows from results.tsv.
        main_stats: Dict with ``"baseline"`` and ``"best"`` from the main branch.
        config: Helix configuration.

    Returns:
        A ``rich.panel.Panel`` ready to print.
    """
    primary_name = config.metrics.primary.name
    main_baseline = main_stats.get("baseline")
    main_best = main_stats.get("best")
    session_best, _ = best_kept(rows, config)

    kept = [r for r in rows if r.get("status") == "keep"]
    discarded = [r for r in rows if r.get("status") == "discard"]
    crashed = [r for r in rows if r.get("status") == "crash"]

    direction = config.metrics.primary.optimize
    improved = session_best is not None and (
        main_best is None
        or (direction == OptimizeDirection.maximize and session_best > main_best)
        or (direction == OptimizeDirection.minimize and session_best < main_best)
    )

    t = Table(show_header=False, box=None, padding=(0, 2))
    t.add_column(style="dim")
    t.add_column(style="bold")

    t.add_row("experiments", str(len(rows)))
    t.add_row("  kept", f"[green]{len(kept)}[/green]")
    t.add_row("  discarded", f"[yellow]{len(discarded)}[/yellow]")
    t.add_row("  crashed", f"[red]{len(crashed)}[/red]")
    t.add_row("", "")

    fmt = _fmt_metric(primary_name)
    t.add_row("main baseline", fmt(main_baseline))
    t.add_row("main best", fmt(main_best))
    t.add_row("session best", f"[cyan]{session_best:.4g} {primary_name}[/cyan]" if session_best is not None else "[dim]—[/dim]")
    t.add_row("", "")

    if improved and session_best is not None:
        delta = session_best - (main_best or 0.0)
        sign = "+" if direction == OptimizeDirection.maximize else ""
        t.add_row("outcome", f"[green]IMPROVED {sign}{delta:.4g} {primary_name} → committing to main[/green]")
    else:
        t.add_row("outcome", "[yellow]no improvement — recording experiments only[/yellow]")

    return Panel(t, title="[bold]session summary[/bold]", expand=False)


def _fmt_metric(name: str):  # type: ignore[no-untyped-def]
    """Return a formatting function for a metric value display."""
    def fmt(value: float | None) -> str:
        if value is None:
            return "[dim]—[/dim]"
        return f"{value:.4g} {name}"
    return fmt
