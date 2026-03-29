"""Scaffold a new helix from the built-in template.

The public entry point is ``scaffold()``. The CLI calls it via ``helix init``.
"""

from __future__ import annotations

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
    """Create a new helix directory from the built-in template.

    Parameters
    ----------
    name : str
        Helix name — used as the directory name and filled into ``helix.yaml``.
    output_dir : Path
        Parent directory where the new helix directory will be created.
    domain : str, optional
        Research domain written into ``helix.yaml`` (e.g. ``"AI/ML"``).
        Defaults to ``"General"``.
    description : str, optional
        One-line description written into ``helix.yaml`` and ``program.md``.
        Defaults to ``"Autonomous research loop."``.

    Returns
    -------
    Path
        Path to the created helix directory.

    Raises
    ------
    FileExistsError
        If a directory named *name* already exists inside *output_dir*.
    """
    target = output_dir / name
    if target.exists():
        raise FileExistsError(f"Directory already exists: {target}")

    target.mkdir(parents=True)

    substitutions = {"name": name, "domain": domain, "description": description}
    for filename, content in TEMPLATE.items():
        (target / filename).write_text(_render(content, substitutions))

    # Write experiments.tsv with the correct header derived from the rendered helix.yaml.
    config = HelixConfig.load(target / "helix.yaml")
    (target / "experiments.tsv").write_text(_tsv_header(config, include_session=True) + "\n")

    return target
