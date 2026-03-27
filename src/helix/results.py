"""Reading and writing results.tsv and experiments.tsv."""

from __future__ import annotations

import subprocess
from pathlib import Path

from .config import HelixConfig, OptimizeDirection


def _tsv_header(config: HelixConfig, include_session: bool = False) -> str:
    """Return the TSV header row for results or experiments files.

    Column names for the metric columns are taken from ``config`` so that
    every helix uses its own meaningful names (e.g. ``tokens_per_sec`` or
    ``accuracy``) rather than generic placeholders.
    """
    primary_col = config.metrics.primary.name
    qg_col = config.metrics.quality_guard.name if config.metrics.quality_guard else "quality_guard"
    base = f"commit\t{primary_col}\t{qg_col}\tstatus\tdescription"
    return f"session\t{base}" if include_session else base


def _parse_tsv(path: Path) -> list[dict[str, str]]:
    """Parse a TSV file into a list of row dicts (empty list if file missing)."""
    if not path.exists():
        return []
    lines = path.read_text().splitlines()
    if len(lines) < 2:
        return []
    headers = lines[0].split("\t")
    rows: list[dict[str, str]] = []
    for line in lines[1:]:
        if not line.strip():
            continue
        parts = line.split("\t")
        parts += [""] * max(0, len(headers) - len(parts))
        rows.append(dict(zip(headers, parts[: len(headers)])))
    return rows


def read_results(path: Path) -> list[dict[str, str]]:
    """Parse results.tsv and return a list of row dicts."""
    return _parse_tsv(path)


def best_kept(rows: list[dict[str, str]], config: HelixConfig) -> tuple[float | None, str | None]:
    """Return ``(primary_metric_value, description)`` of the best kept experiment.

    Args:
        rows: Parsed rows from results.tsv or experiments.tsv.
        config: Helix configuration (used for metric name and optimize direction).

    Returns:
        A tuple of ``(value, description)``.  Both elements are ``None`` if no
        kept experiments exist.
    """
    kept = [r for r in rows if r.get("status") == "keep"]
    if not kept:
        return None, None

    primary_col = config.metrics.primary.name

    def get_val(r: dict[str, str]) -> float:
        try:
            return float(r.get(primary_col) or 0)
        except ValueError:
            return 0.0

    best = (
        max(kept, key=get_val)
        if config.metrics.primary.optimize == OptimizeDirection.maximize
        else min(kept, key=get_val)
    )
    return get_val(best), best.get("description", "")


def read_main_stats(main_branch: str, cwd: Path, config: HelixConfig) -> dict[str, float | None]:
    """Read baseline and best metric values from experiments.tsv on the main branch.

    Args:
        main_branch: Name of the main branch (e.g. ``"main"`` or ``"master"``).
        cwd: Repository root.
        config: Helix configuration.

    Returns:
        Dict with keys ``"baseline"`` and ``"best"``, each a float or ``None``.
    """
    try:
        result = subprocess.run(
            ["git", "-C", str(cwd), "show", f"{main_branch}:experiments.tsv"],
            capture_output=True,
            text=True,
            check=True,
        )
        raw = result.stdout
    except subprocess.CalledProcessError:
        return {"baseline": None, "best": None}

    rows = _parse_tsv_string(raw)
    kept = [r for r in rows if r.get("status") == "keep"]
    primary_col = config.metrics.primary.name

    values: list[float] = []
    for r in kept:
        val = r.get(primary_col)
        if val:
            try:
                values.append(float(val))
            except ValueError:
                pass

    if not values:
        return {"baseline": None, "best": None}

    if config.metrics.primary.optimize == OptimizeDirection.maximize:
        return {"baseline": values[0], "best": max(values)}
    return {"baseline": values[0], "best": min(values)}


def _parse_tsv_string(content: str) -> list[dict[str, str]]:
    """Parse TSV content from a string (same logic as _parse_tsv but from str)."""
    lines = content.splitlines()
    if len(lines) < 2:
        return []
    headers = lines[0].split("\t")
    rows: list[dict[str, str]] = []
    for line in lines[1:]:
        if not line.strip():
            continue
        parts = line.split("\t")
        parts += [""] * max(0, len(headers) - len(parts))
        rows.append(dict(zip(headers, parts[: len(headers)])))
    return rows


def append_experiments(tag: str, rows: list[dict[str, str]], path: Path, config: HelixConfig) -> None:
    """Append session rows to experiments.tsv, creating it with a header if needed.

    Args:
        tag: Session tag (e.g. ``"mar27"``).
        rows: Rows from results.tsv to append.
        path: Path to experiments.tsv.
        config: Helix configuration (used for metric column names).
    """
    primary_col = config.metrics.primary.name
    qg_col = config.metrics.quality_guard.name if config.metrics.quality_guard else "quality_guard"

    new_lines = "\n".join(
        f"{tag}\t{r.get('commit', '')}\t{r.get(primary_col, '')}\t"
        f"{r.get(qg_col, '')}\t{r.get('status', '')}\t{r.get('description', '')}"
        for r in rows
    )
    if not path.exists():
        path.write_text(_tsv_header(config, include_session=True) + "\n" + new_lines + "\n")
    else:
        existing = path.read_text()
        if not existing.endswith("\n"):
            existing += "\n"
        path.write_text(existing + new_lines + "\n")
