"""Reading and writing results.tsv and experiments.tsv."""

from __future__ import annotations

import contextlib
import subprocess
from pathlib import Path

from .config import HelixConfig, OptimizeDirection


def _tsv_header(config: HelixConfig, include_session: bool = False) -> str:
    """Return the TSV header row, using actual metric names as column headers.

    Parameters
    ----------
    config : HelixConfig
        Helix configuration (metric names are used as column headers).
    include_session : bool, optional
        If True, prepend a ``session`` column (for experiments.tsv).

    Returns
    -------
    str
        Tab-separated header string.
    """
    primary_col = config.metrics.primary.name
    qg_col = config.metrics.quality_guard.name if config.metrics.quality_guard else "quality_guard"
    base = f"commit\t{primary_col}\t{qg_col}\tstatus\tdescription"
    return f"session\t{base}" if include_session else base


def _parse_tsv(path: Path) -> list[dict[str, str]]:
    """Parse a TSV file into a list of row dicts.

    Parameters
    ----------
    path : Path
        Path to a tab-separated file.

    Returns
    -------
    list[dict[str, str]]
        One dict per data row; empty list if the file is missing or has no rows.
    """
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


def _parse_tsv_string(content: str) -> list[dict[str, str]]:
    """Parse TSV content from a string.

    Parameters
    ----------
    content : str
        Raw TSV text.

    Returns
    -------
    list[dict[str, str]]
        One dict per data row.
    """
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


def read_results(path: Path) -> list[dict[str, str]]:
    """Parse results.tsv and return a list of row dicts.

    Parameters
    ----------
    path : Path
        Path to ``results.tsv``.

    Returns
    -------
    list[dict[str, str]]
        Parsed rows; empty list if file does not exist.
    """
    return _parse_tsv(path)


def best_kept(rows: list[dict[str, str]], config: HelixConfig) -> tuple[float | None, str | None]:
    """Return the primary metric value and description of the best kept experiment.

    Parameters
    ----------
    rows : list[dict[str, str]]
        Parsed rows from results.tsv or experiments.tsv.
    config : HelixConfig
        Used to determine metric name and optimization direction.

    Returns
    -------
    tuple[float or None, str or None]
        ``(value, description)`` of the best kept experiment, or ``(None, None)``
        if no kept experiments exist.
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

    Parameters
    ----------
    main_branch : str
        Name of the main branch (e.g. ``"main"`` or ``"master"``).
    cwd : Path
        Repository root directory.
    config : HelixConfig
        Used to determine the metric column name and optimization direction.

    Returns
    -------
    dict[str, float or None]
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
            with contextlib.suppress(ValueError):
                values.append(float(val))

    if not values:
        return {"baseline": None, "best": None}

    if config.metrics.primary.optimize == OptimizeDirection.maximize:
        return {"baseline": values[0], "best": max(values)}
    return {"baseline": values[0], "best": min(values)}


def append_experiments(tag: str, rows: list[dict[str, str]], path: Path, config: HelixConfig) -> None:
    """Append session rows to experiments.tsv, creating it with a header if needed.

    Parameters
    ----------
    tag : str
        Session tag (e.g. ``"mar27"``).
    rows : list[dict[str, str]]
        Rows from results.tsv to append.
    path : Path
        Path to ``experiments.tsv``.
    config : HelixConfig
        Used to determine metric column names.
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
