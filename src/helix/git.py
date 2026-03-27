"""Git operations for helix research sessions."""

from __future__ import annotations

import subprocess
from pathlib import Path


class GitError(Exception):
    """Raised when a git command fails unexpectedly."""


def run(*args: str, cwd: Path, check: bool = True) -> str:
    """Run a git command in `cwd` and return stdout.

    Args:
        *args: Arguments passed to git (e.g. ``"checkout"``, ``"-b"``, ``"branch"``).
        cwd: Repository root to run the command in.
        check: If True, raise ``GitError`` on non-zero exit.

    Returns:
        Stripped stdout string.

    Raises:
        GitError: When ``check=True`` and the command exits non-zero.
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
    """Return the name of the currently checked-out branch."""
    return run("rev-parse", "--abbrev-ref", "HEAD", cwd=cwd)


def short_hash(cwd: Path) -> str:
    """Return the 7-character short hash of HEAD."""
    return run("rev-parse", "--short", "HEAD", cwd=cwd)


def show_file(ref: str, rel_path: str, cwd: Path) -> str:
    """Return the content of a file at a specific git ref.

    Args:
        ref: Git ref (branch name, tag, commit hash).
        rel_path: File path relative to the repository root.
        cwd: Repository root.

    Returns:
        File content as a string.

    Raises:
        GitError: If the ref or path does not exist.
    """
    return run("show", f"{ref}:{rel_path}", cwd=cwd)


def branch_exists(name: str, cwd: Path) -> bool:
    """Return True if a local branch with `name` exists."""
    result = subprocess.run(
        ["git", "-C", str(cwd), "rev-parse", "--verify", name],
        check=False,
        capture_output=True,
    )
    return result.returncode == 0


def detect_main_branch(cwd: Path) -> str:
    """Return 'main' if it exists, otherwise 'master'.

    Args:
        cwd: Repository root.

    Returns:
        Name of the primary branch.
    """
    return "main" if branch_exists("main", cwd) else "master"
