"""Tests for helix.git."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from helix.git import GitError, branch_exists, current_branch, detect_main_branch, run, show_file


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    """Return a path to a fresh git repo with one commit on 'main'."""
    subprocess.run(["git", "init", "-b", "main", str(tmp_path)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "test@test.com"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "Test"], check=True, capture_output=True)
    (tmp_path / "readme.txt").write_text("hello")
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-m", "init"], check=True, capture_output=True)
    return tmp_path


class TestRun:
    def test_returns_stdout(self, repo: Path) -> None:
        result = run("rev-parse", "--abbrev-ref", "HEAD", cwd=repo)
        assert result == "main"

    def test_raises_git_error_on_failure(self, repo: Path) -> None:
        with pytest.raises(GitError):
            run("checkout", "nonexistent-branch", cwd=repo)

    def test_check_false_does_not_raise(self, repo: Path) -> None:
        result = run("checkout", "nonexistent-branch", cwd=repo, check=False)
        assert result == ""


class TestCurrentBranch:
    def test_returns_branch_name(self, repo: Path) -> None:
        assert current_branch(repo) == "main"

    def test_after_branch_creation(self, repo: Path) -> None:
        subprocess.run(["git", "-C", str(repo), "checkout", "-b", "feature"], check=True, capture_output=True)
        assert current_branch(repo) == "feature"


class TestBranchExists:
    def test_existing_branch(self, repo: Path) -> None:
        assert branch_exists("main", repo) is True

    def test_nonexistent_branch(self, repo: Path) -> None:
        assert branch_exists("does-not-exist", repo) is False

    def test_after_creation(self, repo: Path) -> None:
        subprocess.run(["git", "-C", str(repo), "branch", "new-branch"], check=True, capture_output=True)
        assert branch_exists("new-branch", repo) is True


class TestDetectMainBranch:
    def test_detects_main(self, repo: Path) -> None:
        assert detect_main_branch(repo) == "main"

    def test_detects_master(self, tmp_path: Path) -> None:
        subprocess.run(["git", "init", "-b", "master", str(tmp_path)], check=True, capture_output=True)
        subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "t@t.com"], check=True, capture_output=True)
        subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "T"], check=True, capture_output=True)
        (tmp_path / "f.txt").write_text("x")
        subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True, capture_output=True)
        subprocess.run(["git", "-C", str(tmp_path), "commit", "-m", "init"], check=True, capture_output=True)
        assert detect_main_branch(tmp_path) == "master"


class TestShowFile:
    def test_shows_file_at_head(self, repo: Path) -> None:
        content = show_file("HEAD", "readme.txt", repo)
        assert content == "hello"

    def test_raises_on_missing_file(self, repo: Path) -> None:
        with pytest.raises(GitError):
            show_file("HEAD", "does_not_exist.txt", repo)

    def test_shows_file_at_branch(self, repo: Path) -> None:
        subprocess.run(["git", "-C", str(repo), "checkout", "-b", "test-branch"], check=True, capture_output=True)
        (repo / "readme.txt").write_text("modified")
        subprocess.run(["git", "-C", str(repo), "add", "."], check=True, capture_output=True)
        subprocess.run(["git", "-C", str(repo), "commit", "-m", "update"], check=True, capture_output=True)

        assert show_file("test-branch", "readme.txt", repo) == "modified"
        assert show_file("main", "readme.txt", repo) == "hello"
