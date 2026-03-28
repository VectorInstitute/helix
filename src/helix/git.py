"""Git operations for helix research sessions."""

from __future__ import annotations

import subprocess
from pathlib import Path


class GitError(Exception):
    """Raised when a git command fails unexpectedly."""


def run(*args: str, cwd: Path, check: bool = True) -> str:
    """Run a git command in ``cwd`` and return stdout.

    Parameters
    ----------
    *args : str
        Arguments passed to git (e.g. ``"checkout"``, ``"-b"``, ``"branch"``).
    cwd : Path
        Repository root directory.
    check : bool, optional
        If True (default), raise ``GitError`` on non-zero exit.

    Returns
    -------
    str
        Stripped stdout from the git command.

    Raises
    ------
    GitError
        When ``check=True`` and the command exits non-zero.
    """
    result = subprocess.run(
        ["git", "-C", str(cwd), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if check and result.returncode != 0:
        raise GitError(f"git {' '.join(args)} failed:\n{result.stderr.strip()}")
    return result.stdout.strip()


def current_branch(cwd: Path) -> str:
    """Return the name of the currently checked-out branch.

    Parameters
    ----------
    cwd : Path
        Repository root directory.

    Returns
    -------
    str
        Current branch name.
    """
    return run("rev-parse", "--abbrev-ref", "HEAD", cwd=cwd)


def show_file(ref: str, rel_path: str, cwd: Path) -> str:
    """Return the content of a file at a specific git ref.

    Parameters
    ----------
    ref : str
        Git ref (branch name, tag, or commit hash).
    rel_path : str
        File path relative to the repository root.
    cwd : Path
        Repository root directory.

    Returns
    -------
    str
        File content as a string.

    Raises
    ------
    GitError
        If the ref or path does not exist.
    """
    return run("show", f"{ref}:{rel_path}", cwd=cwd)


def branch_exists(name: str, cwd: Path) -> bool:
    """Return True if a local branch with ``name`` exists.

    Parameters
    ----------
    name : str
        Branch name to check.
    cwd : Path
        Repository root directory.

    Returns
    -------
    bool
        True if the branch exists locally.
    """
    result = subprocess.run(
        ["git", "-C", str(cwd), "rev-parse", "--verify", name],
        check=False,
        capture_output=True,
    )
    return result.returncode == 0


def detect_main_branch(cwd: Path) -> str:
    """Return ``'main'`` if it exists, otherwise ``'master'``.

    Parameters
    ----------
    cwd : Path
        Repository root directory.

    Returns
    -------
    str
        Name of the primary branch.
    """
    return "main" if branch_exists("main", cwd) else "master"
