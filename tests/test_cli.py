"""Tests for helix.cli."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from helix.cli import _build_parser, _today_tag, cmd_init, cmd_run, cmd_status, main


class TestTodayTag:
    def test_returns_lowercase(self) -> None:
        assert _today_tag().islower()

    def test_matches_month_day_pattern(self) -> None:
        assert re.match(r"^[a-z]{3}\d{1,2}$", _today_tag())


class TestBuildParser:
    def test_returns_argument_parser(self) -> None:
        assert isinstance(_build_parser(), argparse.ArgumentParser)

    def test_init_name_positional(self) -> None:
        args = _build_parser().parse_args(["init", "myproject"])
        assert args.name == "myproject"

    def test_init_default_template(self) -> None:
        args = _build_parser().parse_args(["init", "myproject"])
        assert args.template == "generic"

    def test_init_custom_template(self) -> None:
        args = _build_parser().parse_args(["init", "x", "--template", "ai-inference"])
        assert args.template == "ai-inference"

    def test_init_default_domain(self) -> None:
        args = _build_parser().parse_args(["init", "x"])
        assert args.domain == "General"

    def test_run_default_max_turns(self) -> None:
        args = _build_parser().parse_args(["run"])
        assert args.max_turns == 200

    def test_run_custom_tag(self) -> None:
        args = _build_parser().parse_args(["run", "--tag", "exp1"])
        assert args.tag == "exp1"

    def test_run_custom_max_turns(self) -> None:
        args = _build_parser().parse_args(["run", "--max-turns", "50"])
        assert args.max_turns == 50

    def test_run_helix_root_none_by_default(self) -> None:
        args = _build_parser().parse_args(["run"])
        assert args.helix_root is None

    def test_status_command(self) -> None:
        args = _build_parser().parse_args(["status"])
        assert args.command == "status"

    def test_status_helix_root(self) -> None:
        args = _build_parser().parse_args(["status", "--helix-root", "/tmp/h"])
        assert args.helix_root == "/tmp/h"


class TestCmdInit:
    def test_success_calls_scaffold(self, tmp_path: Path) -> None:
        with patch("helix.cli.scaffold", return_value=tmp_path / "myhelix") as mock_scaffold:
            args = argparse.Namespace(
                name="myhelix",
                template="generic",
                domain="General",
                description="Test.",
                output_dir=str(tmp_path),
            )
            cmd_init(args)
        mock_scaffold.assert_called_once()

    def test_value_error_exits_1(self) -> None:
        with patch("helix.cli.scaffold", side_effect=ValueError("Unknown template")):
            args = argparse.Namespace(name="x", template="bad", domain="G", description="D", output_dir=None)
            with pytest.raises(SystemExit) as exc_info:
                cmd_init(args)
            assert exc_info.value.code == 1

    def test_file_exists_error_exits_1(self) -> None:
        with patch("helix.cli.scaffold", side_effect=FileExistsError("already exists")):
            args = argparse.Namespace(name="x", template="generic", domain="G", description="D", output_dir=None)
            with pytest.raises(SystemExit) as exc_info:
                cmd_init(args)
            assert exc_info.value.code == 1

    def test_none_output_dir_uses_cwd(self, tmp_path: Path) -> None:
        captured: dict[str, Path] = {}

        def fake_scaffold(name: str, output_dir: Path, **kw: object) -> Path:
            captured["output_dir"] = output_dir
            return output_dir / name

        with patch("helix.cli.scaffold", side_effect=fake_scaffold), patch("helix.cli.Path.cwd", return_value=tmp_path):
            args = argparse.Namespace(name="p", template="generic", domain="G", description="D", output_dir=None)
            cmd_init(args)
        assert captured["output_dir"] == tmp_path


class TestCmdRun:
    def test_calls_runner_run(self, tmp_path: Path) -> None:
        mock_runner = MagicMock()
        with (
            patch("helix.cli.find_helix_root", return_value=tmp_path),
            patch("helix.cli.HelixRunner", return_value=mock_runner),
        ):
            args = argparse.Namespace(tag="mar27", max_turns=50, helix_root=None)
            cmd_run(args)
        mock_runner.run.assert_called_once()

    def test_helix_root_arg_passed_directly(self, tmp_path: Path) -> None:
        mock_runner = MagicMock()
        with patch("helix.cli.HelixRunner", return_value=mock_runner):
            args = argparse.Namespace(tag="t", max_turns=10, helix_root=str(tmp_path))
            cmd_run(args)
        call_kwargs = mock_runner.run.call_args
        assert call_kwargs is not None


class TestCmdStatus:
    def _make_config(self) -> MagicMock:
        config = MagicMock()
        config.name = "test"
        config.domain = "Test"
        config.metrics.primary.name = "score"
        config.metrics.quality_guard = None
        return config

    def test_no_results_file_does_not_raise(self, tmp_path: Path) -> None:
        with (
            patch("helix.cli.find_helix_root", return_value=tmp_path),
            patch("helix.cli.HelixConfig.load", return_value=self._make_config()),
            patch("helix.cli.detect_main_branch", return_value="main"),
            patch("helix.cli.read_main_stats", return_value={"baseline": None, "best": None}),
        ):
            cmd_status(argparse.Namespace(helix_root=None))

    def test_with_results_file_and_rows(self, tmp_path: Path) -> None:
        (tmp_path / "results.tsv").write_text("commit\tscore\tstatus\tdescription\n")
        with (
            patch("helix.cli.HelixConfig.load", return_value=self._make_config()),
            patch("helix.cli.detect_main_branch", return_value="main"),
            patch("helix.cli.read_main_stats", return_value={"baseline": 100.0, "best": 150.0}),
            patch(
                "helix.cli.read_results",
                return_value=[{"commit": "abc", "score": "150.0", "status": "keep", "description": "batch"}],
            ),
        ):
            cmd_status(argparse.Namespace(helix_root=str(tmp_path)))

    def test_with_quality_guard_column(self, tmp_path: Path) -> None:
        (tmp_path / "results.tsv").write_text("commit\tscore\tloss\tstatus\tdescription\n")
        config = self._make_config()
        config.metrics.quality_guard = MagicMock()
        config.metrics.quality_guard.name = "loss"
        with (
            patch("helix.cli.HelixConfig.load", return_value=config),
            patch("helix.cli.detect_main_branch", return_value="main"),
            patch("helix.cli.read_main_stats", return_value={"baseline": 100.0, "best": 120.0}),
            patch(
                "helix.cli.read_results",
                return_value=[
                    {"commit": "abc", "score": "120.0", "loss": "0.5", "status": "keep", "description": "test"}
                ],
            ),
        ):
            cmd_status(argparse.Namespace(helix_root=str(tmp_path)))

    def test_helix_root_arg_used_directly(self, tmp_path: Path) -> None:
        with (
            patch("helix.cli.HelixConfig.load", return_value=self._make_config()),
            patch("helix.cli.detect_main_branch", return_value="main"),
            patch("helix.cli.read_main_stats", return_value={"baseline": None, "best": None}),
        ):
            cmd_status(argparse.Namespace(helix_root=str(tmp_path)))


class TestMain:
    def test_file_not_found_exits_1(self) -> None:
        mock_args = MagicMock()
        mock_args.func.side_effect = FileNotFoundError("no helix.yaml")
        mock_parser = MagicMock()
        mock_parser.parse_args.return_value = mock_args

        with patch("helix.cli._build_parser", return_value=mock_parser):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    def test_success_calls_func(self) -> None:
        mock_args = MagicMock()
        mock_parser = MagicMock()
        mock_parser.parse_args.return_value = mock_args

        with patch("helix.cli._build_parser", return_value=mock_parser):
            main()
        mock_args.func.assert_called_once_with(mock_args)
