"""Scaffold a new helix from the built-in template.

The public entry points are ``scaffold()`` and ``run_uv_lock()``.
The CLI calls them via ``helix init``.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from .config import HelixConfig
from .results import _tsv_header
from .templates import TEMPLATE


def _render(content: str, substitutions: dict[str, str]) -> str:
    """Replace ``{{key}}`` placeholders in *content* with values from *substitutions*.

    Parameters
    ----------
    content : str
        Template string containing ``{{key}}`` markers.
    substitutions : dict[str, str]
        Mapping from placeholder key to replacement value.

    Returns
    -------
    str
        Rendered string with all known placeholders replaced.
    """
    result = content
    for key, value in substitutions.items():
        result = result.replace("{{" + key + "}}", value)
    return result


def scaffold(
    name: str,
    output_dir: Path,
    domain: str = "General",
    description: str = "Autonomous research loop.",
) -> Path:
    """Create or update a helix directory from the built-in template.

    This function is non-destructive: it creates the target directory if it
    does not exist, but silently skips any file that is already present. This
    makes it safe to run against an existing repository — only the missing
    helix files will be added.

    Parameters
    ----------
    name : str
        Helix name — used as the directory name and filled into template files.
    output_dir : Path
        Parent directory where the helix directory lives (or will be created).
    domain : str, optional
        Research domain written into ``helix.yaml``. Defaults to ``"General"``.
    description : str, optional
        One-line description written into template files.
        Defaults to ``"Autonomous research loop."``.

    Returns
    -------
    Path
        Path to the helix directory.
    """
    target = output_dir / name
    target.mkdir(parents=True, exist_ok=True)

    # When name is ".", derive the project name from the resolved directory.
    resolved_name = target.resolve().name if name == "." else name
    substitutions = {"name": resolved_name, "domain": domain, "description": description}
    for filename, content in TEMPLATE.items():
        dest = target / filename
        if dest.exists():
            continue
        dest.write_text(_render(content, substitutions))

    # experiments.tsv requires the rendered helix.yaml to derive column headers.
    exp_path = target / "experiments.tsv"
    helix_yaml = target / "helix.yaml"
    if not exp_path.exists() and helix_yaml.exists():
        config = HelixConfig.load(helix_yaml)
        exp_path.write_text(_tsv_header(config, include_session=True) + "\n")

    return target


def run_uv_lock(target: Path) -> bool:
    """Run ``uv lock`` inside *target* to generate a locked dependency file.

    Parameters
    ----------
    target : Path
        Directory containing ``pyproject.toml``.

    Returns
    -------
    bool
        ``True`` if ``uv lock`` succeeded, ``False`` if it failed or if
        ``uv`` is not installed.
    """
    try:
        result = subprocess.run(
            ["uv", "lock"],
            cwd=target,
            capture_output=True,
            check=False,
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False
