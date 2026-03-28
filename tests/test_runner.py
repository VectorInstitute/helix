"""Tests for helix.runner."""

from __future__ import annotations

import contextlib
import signal
from pathlib import Path
from typing import AsyncIterator
from unittest.mock import MagicMock, patch

import pytest
import yaml

from helix.runner import HelixRunner, find_helix_root


def _make_helix_yaml(tmp_path: Path, optimize: str = "maximize", with_qg: bool = False) -> Path:
    data: dict = {
        "name": "test-helix",
        "version": "1.0.0",
        "domain": "AI/ML",
        "description": "Test helix.",
        "scope": {"editable": ["solver.py"], "readonly": ["evaluate.py", "helix.yaml"]},
        "metrics": {
            "primary": {"name": "score", "optimize": optimize},
            "evaluate": {
                "command": "python evaluate.py",
                "timeout_seconds": 60,
                "output_format": "pattern",
                "patterns": {"primary": r"^score:\s+([\d.]+)"},
            },
        },
    }
    if with_qg:
        data["metrics"]["quality_guard"] = {"name": "loss", "optimize": "minimize", "max_degradation": 0.01}
    (tmp_path / "helix.yaml").write_text(yaml.dump(data))
    return tmp_path


def _make_runner(tmp_path: Path, optimize: str = "maximize", with_qg: bool = False) -> HelixRunner:
    _make_helix_yaml(tmp_path, optimize, with_qg)
    with patch("helix.runner.detect_main_branch", return_value="main"):
        return HelixRunner(
            helix_root=tmp_path,
            tag="mar27",
            max_turns=50,
            backend=MagicMock(),
        )


class TestFindHelixRoot:
    def test_finds_in_given_dir(self, tmp_path: Path) -> None:
        (tmp_path / "helix.yaml").touch()
        assert find_helix_root(tmp_path) == tmp_path

    def test_finds_in_parent_dir(self, tmp_path: Path) -> None:
        (tmp_path / "helix.yaml").touch()
        sub = tmp_path / "sub"
        sub.mkdir()
        assert find_helix_root(sub) == tmp_path

    def test_finds_in_grandparent(self, tmp_path: Path) -> None:
        (tmp_path / "helix.yaml").touch()
        deep = tmp_path / "a" / "b"
        deep.mkdir(parents=True)
        assert find_helix_root(deep) == tmp_path

    def test_raises_when_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="helix.yaml"):
            find_helix_root(tmp_path)


class TestHelixRunnerInit:
    def test_sets_root(self, tmp_path: Path) -> None:
        runner = _make_runner(tmp_path)
        assert runner.root == tmp_path

    def test_sets_tag_and_branch(self, tmp_path: Path) -> None:
        runner = _make_runner(tmp_path)
        assert runner.tag == "mar27"
        assert runner.branch == "helix/mar27"

    def test_sets_max_turns(self, tmp_path: Path) -> None:
        assert _make_runner(tmp_path).max_turns == 50

    def test_default_backend_is_claude(self, tmp_path: Path) -> None:
        _make_helix_yaml(tmp_path)
        from helix.agent import ClaudeBackend

        with patch("helix.runner.detect_main_branch", return_value="main"):
            runner = HelixRunner(helix_root=tmp_path, tag="t", max_turns=10)
        assert isinstance(runner.backend, ClaudeBackend)

    def test_custom_backend_stored(self, tmp_path: Path) -> None:
        backend = MagicMock()
        _make_helix_yaml(tmp_path)
        with patch("helix.runner.detect_main_branch", return_value="main"):
            runner = HelixRunner(helix_root=tmp_path, tag="t", max_turns=10, backend=backend)
        assert runner.backend is backend


class TestBuildPrompt:
    def test_contains_session_tag(self, tmp_path: Path) -> None:
        prompt = _make_runner(tmp_path)._build_prompt({"baseline": None, "best": None})
        assert "mar27" in prompt

    def test_contains_session_branch(self, tmp_path: Path) -> None:
        prompt = _make_runner(tmp_path)._build_prompt({})
        assert "helix/mar27" in prompt

    def test_contains_editable_file(self, tmp_path: Path) -> None:
        prompt = _make_runner(tmp_path)._build_prompt({})
        assert "solver.py" in prompt

    def test_contains_metric_name(self, tmp_path: Path) -> None:
        prompt = _make_runner(tmp_path)._build_prompt({})
        assert "score" in prompt

    def test_numeric_baseline_shown(self, tmp_path: Path) -> None:
        prompt = _make_runner(tmp_path)._build_prompt({"baseline": 42.0, "best": 55.0})
        assert "42" in prompt
        assert "55" in prompt

    def test_none_baseline_says_not_established(self, tmp_path: Path) -> None:
        prompt = _make_runner(tmp_path)._build_prompt({"baseline": None, "best": None})
        assert "not yet established" in prompt

    def test_maximize_says_higher(self, tmp_path: Path) -> None:
        prompt = _make_runner(tmp_path, optimize="maximize")._build_prompt({})
        assert "higher" in prompt

    def test_minimize_says_lower(self, tmp_path: Path) -> None:
        prompt = _make_runner(tmp_path, optimize="minimize")._build_prompt({})
        assert "lower" in prompt

    def test_quality_guard_instructions_included(self, tmp_path: Path) -> None:
        prompt = _make_runner(tmp_path, with_qg=True)._build_prompt({})
        assert "loss" in prompt
        assert "Quality guard" in prompt

    def test_no_quality_guard_when_not_configured(self, tmp_path: Path) -> None:
        prompt = _make_runner(tmp_path, with_qg=False)._build_prompt({})
        assert "Quality guard" not in prompt


class TestSystemPrompt:
    def test_contains_description(self, tmp_path: Path) -> None:
        sp = _make_runner(tmp_path)._system_prompt()
        assert "Test helix." in sp

    def test_is_non_empty_string(self, tmp_path: Path) -> None:
        sp = _make_runner(tmp_path)._system_prompt()
        assert isinstance(sp, str) and len(sp) > 10


class TestKillExperiment:
    def test_no_pid_file_is_noop(self, tmp_path: Path) -> None:
        _make_runner(tmp_path)._kill_experiment()  # should not raise

    def test_kills_pid_from_file(self, tmp_path: Path) -> None:
        runner = _make_runner(tmp_path)
        runner._pid_path.write_text("99999\n")
        with patch("helix.runner.os.kill") as mock_kill:
            runner._kill_experiment()
        mock_kill.assert_called_once_with(99999, signal.SIGTERM)

    def test_pid_file_removed_after_kill(self, tmp_path: Path) -> None:
        runner = _make_runner(tmp_path)
        runner._pid_path.write_text("99999\n")
        with patch("helix.runner.os.kill"):
            runner._kill_experiment()
        assert not runner._pid_path.exists()

    def test_oserror_on_kill_is_suppressed(self, tmp_path: Path) -> None:
        runner = _make_runner(tmp_path)
        runner._pid_path.write_text("99999\n")
        with patch("helix.runner.os.kill", side_effect=OSError("no process")):
            runner._kill_experiment()  # should not raise
        assert not runner._pid_path.exists()

    def test_invalid_pid_content_is_ignored(self, tmp_path: Path) -> None:
        runner = _make_runner(tmp_path)
        runner._pid_path.write_text("not-a-number\n")
        runner._kill_experiment()  # ValueError suppressed
        assert not runner._pid_path.exists()


class TestPreflight:
    def test_on_main_branch_creates_session_branch(self, tmp_path: Path) -> None:
        runner = _make_runner(tmp_path)
        with (
            patch("helix.runner.current_branch", return_value="main"),
            patch("helix.runner.git") as mock_git,
        ):
            runner._preflight()
        mock_git.assert_any_call("checkout", "-b", "helix/mar27", cwd=tmp_path)

    def test_on_session_branch_resumes(self, tmp_path: Path) -> None:
        runner = _make_runner(tmp_path)
        with (
            patch("helix.runner.current_branch", return_value="helix/mar27"),
            patch("helix.runner.git") as mock_git,
        ):
            runner._preflight()
        mock_git.assert_not_called()

    def test_wrong_branch_exits_1(self, tmp_path: Path) -> None:
        runner = _make_runner(tmp_path)
        with (
            patch("helix.runner.current_branch", return_value="feature/other"),
            patch("helix.runner.git"),
        ):
            with pytest.raises(SystemExit) as exc_info:
                runner._preflight()
            assert exc_info.value.code == 1

    def test_branch_already_exists_exits_1(self, tmp_path: Path) -> None:
        from helix.git import GitError

        runner = _make_runner(tmp_path)
        with (
            patch("helix.runner.current_branch", return_value="main"),
            patch("helix.runner.git", side_effect=GitError("branch exists")),
        ):
            with pytest.raises(SystemExit) as exc_info:
                runner._preflight()
            assert exc_info.value.code == 1


class TestSigtermHandler:
    def test_calls_kill_experiment_and_exits(self, tmp_path: Path) -> None:
        runner = _make_runner(tmp_path)
        with patch.object(runner, "_kill_experiment") as mock_kill:
            with pytest.raises(SystemExit) as exc_info:
                runner._sigterm_handler(15, None)
            mock_kill.assert_called_once()
            assert exc_info.value.code == 0


class TestPostSession:
    def test_calls_commit_to_main(self, tmp_path: Path) -> None:
        runner = _make_runner(tmp_path)
        with (
            patch.object(runner, "_commit_to_main") as mock_commit,
            patch("helix.runner.read_results", return_value=[]),
            patch("helix.runner.session_summary_panel", return_value=MagicMock()),
        ):
            runner._post_session({"baseline": None, "best": None})
        mock_commit.assert_called_once()

    def test_passes_results_to_commit(self, tmp_path: Path) -> None:
        runner = _make_runner(tmp_path)
        rows = [{"commit": "abc", "score": "100", "status": "keep", "description": "x"}]
        with (
            patch.object(runner, "_commit_to_main") as mock_commit,
            patch("helix.runner.read_results", return_value=rows),
            patch("helix.runner.session_summary_panel", return_value=MagicMock()),
        ):
            runner._post_session({})
        mock_commit.assert_called_once_with(rows)


def _mock_config() -> MagicMock:
    cfg = MagicMock()
    cfg.name = "test-helix"
    cfg.domain = "AI/ML"
    cfg.description = "Test."
    cfg.interesting_keywords.return_value = ["score"]
    cfg.agent.model = "claude-opus-4-6"
    cfg.metrics.primary.name = "score"
    cfg.metrics.quality_guard = None
    cfg.metrics.evaluate.timeout_seconds = 60
    return cfg


class TestRunAgentAsync:
    @pytest.mark.asyncio
    async def test_yields_session_events(self, tmp_path: Path) -> None:
        from helix.agent import SessionFinished, SessionStarted

        runner = _make_runner(tmp_path)
        runner.config = _mock_config()

        async def _fake_stream() -> AsyncIterator[object]:
            yield SessionStarted(session_id="sid-test")
            yield SessionFinished(turns=3, cost_usd=0.01, error=False)

        async def _fake_run(**kwargs: object) -> AsyncIterator[object]:
            return _fake_stream()

        runner.backend.run = _fake_run

        with patch("helix.runner.startup_panel", return_value=MagicMock()):
            await runner._run_agent({"baseline": None, "best": None})

    @pytest.mark.asyncio
    async def test_interesting_text_printed(self, tmp_path: Path) -> None:
        from helix.agent import SessionFinished, TextOutput

        runner = _make_runner(tmp_path)
        runner.config = _mock_config()

        async def _fake_stream() -> AsyncIterator[object]:
            yield TextOutput(text="score: 100.0\nother line")
            yield SessionFinished(turns=1, cost_usd=None, error=False)

        async def _fake_run(**kwargs: object) -> AsyncIterator[object]:
            return _fake_stream()

        runner.backend.run = _fake_run

        with patch("helix.runner.startup_panel", return_value=MagicMock()):
            await runner._run_agent({})


class TestHelixRunnerRun:
    def test_run_calls_preflight_and_post_session(self, tmp_path: Path) -> None:
        runner = _make_runner(tmp_path)

        with (
            patch.object(runner, "_preflight"),
            patch.object(runner, "_post_session") as mock_post,
            patch("helix.runner.read_main_stats", return_value={"baseline": None, "best": None}),
            patch("helix.runner.anyio.run"),
        ):
            runner.run()

        mock_post.assert_called_once()

    def test_keyboard_interrupt_calls_kill(self, tmp_path: Path) -> None:
        runner = _make_runner(tmp_path)

        with (
            patch.object(runner, "_preflight"),
            patch.object(runner, "_post_session"),
            patch.object(runner, "_kill_experiment") as mock_kill,
            patch("helix.runner.read_main_stats", return_value={}),
            patch("helix.runner.anyio.run", side_effect=KeyboardInterrupt),
        ):
            runner.run()

        mock_kill.assert_called()


class TestMonitorLog:
    @pytest.mark.asyncio
    async def test_cancelled_cleanly(self, tmp_path: Path) -> None:
        import asyncio

        runner = _make_runner(tmp_path)
        task = asyncio.create_task(runner._monitor_log())
        await asyncio.sleep(0)  # allow task to start
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    @pytest.mark.asyncio
    async def test_reads_log_file_when_present(self, tmp_path: Path) -> None:
        import asyncio

        runner = _make_runner(tmp_path)
        runner._log_path.write_bytes(b"Starting evaluation\nsome output\n")

        task = asyncio.create_task(runner._monitor_log())
        await asyncio.sleep(0.05)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


class TestCommitToMain:
    def test_empty_rows_returns_early(self, tmp_path: Path) -> None:
        runner = _make_runner(tmp_path)
        with patch("helix.runner.git") as mock_git:
            runner._commit_to_main([])
        mock_git.assert_not_called()

    def test_no_improvement_commits_experiments_only(self, tmp_path: Path) -> None:
        runner = _make_runner(tmp_path)
        rows = [{"commit": "abc", "score": "80.0", "status": "discard", "description": "bad"}]

        with (
            patch("helix.runner.current_branch", return_value="helix/mar27"),
            patch("helix.runner.git") as mock_git,
            patch("helix.runner.read_main_stats", return_value={"baseline": 100.0, "best": 100.0}),
            patch("helix.runner.append_experiments"),
        ):
            runner._commit_to_main(rows)

        commit_call = next(c for c in mock_git.call_args_list if c.args[0] == "commit")
        assert "no improvement" in commit_call.args[2]

    def test_improvement_commits_editable_files(self, tmp_path: Path) -> None:
        runner = _make_runner(tmp_path)
        rows = [{"commit": "abc", "score": "200.0", "status": "keep", "description": "batch"}]
        (tmp_path / "solver.py").write_text("# improved")

        with (
            patch("helix.runner.current_branch", return_value="helix/mar27"),
            patch("helix.runner.git"),
            patch("helix.runner.read_main_stats", return_value={"baseline": 100.0, "best": 100.0}),
            patch("helix.runner.append_experiments"),
            patch("helix.runner.show_file", return_value="# new content"),
        ):
            runner._commit_to_main(rows)

    def test_show_file_error_is_silenced(self, tmp_path: Path) -> None:
        from helix.git import GitError

        runner = _make_runner(tmp_path)
        rows = [{"commit": "abc", "score": "200.0", "status": "keep", "description": "batch"}]

        with (
            patch("helix.runner.current_branch", return_value="helix/mar27"),
            patch("helix.runner.git"),
            patch("helix.runner.read_main_stats", return_value={"baseline": 100.0, "best": 100.0}),
            patch("helix.runner.append_experiments"),
            patch("helix.runner.show_file", side_effect=GitError("no file")),
        ):
            runner._commit_to_main(rows)  # should not raise

    def test_git_failure_handled_gracefully(self, tmp_path: Path) -> None:
        runner = _make_runner(tmp_path)
        rows = [{"commit": "abc", "score": "80.0", "status": "discard", "description": "bad"}]

        with (
            patch("helix.runner.current_branch", return_value="helix/mar27"),
            patch("helix.runner.git", side_effect=[None, Exception("git error"), None]),
            patch("helix.runner.read_main_stats", return_value={}),
            patch("helix.runner.append_experiments"),
        ):
            runner._commit_to_main(rows)  # outer except handles it, should not raise
