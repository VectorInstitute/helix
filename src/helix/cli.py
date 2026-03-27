"""CLI entrypoints: ``helix run`` and ``helix status``."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table

from .config import HelixConfig
from .results import read_main_stats, read_results
from .git import detect_main_branch
from .runner import HelixRunner, find_helix_root


console = Console()


def _today_tag() -> str:
    """Return today's date as a short lowercase tag, e.g. ``"mar27"``."""
    return datetime.now().strftime("%b%d").lower()


def cmd_run(args: argparse.Namespace) -> None:
    """Start (or resume) an autonomous research session."""
    root = Path(args.helix_root) if args.helix_root else find_helix_root()
    HelixRunner(helix_root=root, tag=args.tag, max_turns=args.max_turns).run()


def cmd_status(args: argparse.Namespace) -> None:
    """Print current best results and recent experiments from this helix."""
    root = Path(args.helix_root) if args.helix_root else find_helix_root()
    config = HelixConfig.load(root / "helix.yaml")
    main_branch = detect_main_branch(root)
    main_stats = read_main_stats(main_branch, root, config)

    primary_name = config.metrics.primary.name
    baseline = main_stats.get("baseline")
    best = main_stats.get("best")

    console.print(f"\n[bold cyan]{config.name}[/bold cyan]  [dim]{config.domain}[/dim]")
    console.print(f"  baseline : {baseline:.4g} {primary_name}" if baseline is not None else "  baseline : —")
    console.print(f"  best     : [green]{best:.4g} {primary_name}[/green]" if best is not None else "  best     : —")

    results_path = root / "results.tsv"
    if results_path.exists():
        rows = read_results(results_path)
        if rows:
            console.print(f"\n[dim]results.tsv ({len(rows)} experiments this session)[/dim]")
            t = Table(show_header=True, header_style="dim")
            t.add_column("commit", style="dim", width=9)
            t.add_column(primary_name)
            if config.metrics.quality_guard:
                t.add_column(config.metrics.quality_guard.name)
            t.add_column("status")
            t.add_column("description")
            for r in rows[-20:]:
                status = r.get("status", "")
                color = {"keep": "green", "discard": "yellow", "crash": "red"}.get(status, "")
                qg_val = r.get(config.metrics.quality_guard.name, "") if config.metrics.quality_guard else None
                row_data = [r.get("commit", ""), r.get(primary_name, "")]
                if qg_val is not None:
                    row_data.append(qg_val)
                row_data += [f"[{color}]{status}[/{color}]" if color else status, r.get("description", "")]
                t.add_row(*row_data)
            console.print(t)
    console.print()


def _build_parser() -> argparse.ArgumentParser:
    today = _today_tag()
    parser = argparse.ArgumentParser(
        prog="helix",
        description="Helix — autonomous research loop runner",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_p = subparsers.add_parser("run", help="Start an autonomous research session")
    run_p.add_argument(
        "--tag",
        default=today,
        help=f"Session tag used as branch suffix, e.g. 'mar27' → helix/mar27 (default: {today})",
    )
    run_p.add_argument("--max-turns", type=int, default=200, help="Maximum agent turns (default: 200)")
    run_p.add_argument("--helix-root", default=None, help="Path to helix root (default: auto-detect from helix.yaml)")
    run_p.set_defaults(func=cmd_run)

    status_p = subparsers.add_parser("status", help="Show current best results and recent experiments")
    status_p.add_argument("--helix-root", default=None, help="Path to helix root (default: auto-detect)")
    status_p.set_defaults(func=cmd_status)

    return parser


def main() -> None:
    """Entry point for the ``helix`` CLI command."""
    parser = _build_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    except FileNotFoundError as exc:
        console.print(f"[red]✗[/red] {exc}")
        sys.exit(1)
