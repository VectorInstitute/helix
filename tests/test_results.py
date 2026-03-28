"""Tests for helix.results."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import yaml

from helix.config import HelixConfig
from helix.results import (
    append_experiments,
    best_kept,
    read_main_stats,
    read_results,
)


def _make_config(tmp_path: Path, optimize: str = "maximize", with_qg: bool = True) -> HelixConfig:
    data: dict = {
        "name": "test",
        "domain": "test",
        "description": "test",
        "scope": {"editable": ["f.py"], "readonly": []},
        "metrics": {
            "primary": {"name": "score", "optimize": optimize},
            "evaluate": {
                "command": "python f.py",
                "timeout_seconds": 60,
                "output_format": "pattern",
                "patterns": {"primary": r"^score:\s+([\d.]+)"},
            },
        },
    }
    if with_qg:
        data["metrics"]["quality_guard"] = {"name": "loss", "optimize": "minimize", "max_degradation": 0.01}
    path = tmp_path / "helix.yaml"
    path.write_text(yaml.dump(data))
    return HelixConfig.load(path)


def _write_results(path: Path, rows: list[str], header: str = "commit\tscore\tloss\tstatus\tdescription") -> None:
    path.write_text(header + "\n" + "\n".join(rows) + "\n")


class TestReadResults:
    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        assert read_results(tmp_path / "results.tsv") == []

    def test_parses_rows(self, tmp_path: Path) -> None:
        path = tmp_path / "results.tsv"
        _write_results(path, ["abc1234\t100.0\t0.5\tkeep\tbaseline"])
        rows = read_results(path)
        assert len(rows) == 1
        assert rows[0]["commit"] == "abc1234"
        assert rows[0]["score"] == "100.0"
        assert rows[0]["status"] == "keep"

    def test_skips_blank_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "results.tsv"
        _write_results(path, ["abc1234\t100.0\t0.5\tkeep\tbaseline", "", "def5678\t110.0\t0.5\tkeep\tbatch"])
        rows = read_results(path)
        assert len(rows) == 2

    def test_header_only_returns_empty(self, tmp_path: Path) -> None:
        path = tmp_path / "results.tsv"
        path.write_text("commit\tscore\tloss\tstatus\tdescription\n")
        assert read_results(path) == []


class TestBestKept:
    def test_empty_rows(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path)
        assert best_kept([], cfg) == (None, None)

    def test_no_kept_rows(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path)
        rows = [{"commit": "a", "score": "100.0", "loss": "0.5", "status": "discard", "description": "x"}]
        assert best_kept(rows, cfg) == (None, None)

    def test_single_kept_row(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path)
        rows = [{"commit": "a", "score": "100.0", "loss": "0.5", "status": "keep", "description": "baseline"}]
        val, desc = best_kept(rows, cfg)
        assert val == pytest.approx(100.0)
        assert desc == "baseline"

    def test_maximize_returns_highest(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path, optimize="maximize")
        rows = [
            {"commit": "a", "score": "100.0", "loss": "0.5", "status": "keep", "description": "a"},
            {"commit": "b", "score": "200.0", "loss": "0.5", "status": "keep", "description": "b"},
            {"commit": "c", "score": "150.0", "loss": "0.5", "status": "keep", "description": "c"},
        ]
        val, desc = best_kept(rows, cfg)
        assert val == pytest.approx(200.0)
        assert desc == "b"

    def test_minimize_returns_lowest(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path, optimize="minimize")
        rows = [
            {"commit": "a", "score": "0.9", "loss": "0.5", "status": "keep", "description": "a"},
            {"commit": "b", "score": "0.3", "loss": "0.5", "status": "keep", "description": "b"},
            {"commit": "c", "score": "0.6", "loss": "0.5", "status": "keep", "description": "c"},
        ]
        val, desc = best_kept(rows, cfg)
        assert val == pytest.approx(0.3)
        assert desc == "b"

    def test_ignores_non_kept_rows(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path, optimize="maximize")
        rows = [
            {"commit": "a", "score": "100.0", "loss": "0.5", "status": "keep", "description": "a"},
            {"commit": "b", "score": "999.0", "loss": "0.5", "status": "discard", "description": "b"},
            {"commit": "c", "score": "888.0", "loss": "0.5", "status": "crash", "description": "c"},
        ]
        val, _ = best_kept(rows, cfg)
        assert val == pytest.approx(100.0)


class TestAppendExperiments:
    def test_creates_file_with_header(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path)
        rows = [{"commit": "abc", "score": "95.0", "loss": "0.5", "status": "keep", "description": "test"}]
        path = tmp_path / "experiments.tsv"

        append_experiments("mar27", rows, path, cfg)

        content = path.read_text()
        assert content.startswith("session\tcommit\tscore\tloss\tstatus\tdescription")
        assert "mar27\tabc\t95.0\t0.5\tkeep\ttest" in content

    def test_appends_to_existing(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path)
        path = tmp_path / "experiments.tsv"
        rows1 = [{"commit": "a", "score": "90.0", "loss": "0.5", "status": "keep", "description": "first"}]
        rows2 = [{"commit": "b", "score": "95.0", "loss": "0.5", "status": "keep", "description": "second"}]

        append_experiments("s1", rows1, path, cfg)
        append_experiments("s2", rows2, path, cfg)

        lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
        assert len(lines) == 3  # header + 2 rows
        assert "s1\ta" in lines[1]
        assert "s2\tb" in lines[2]

    def test_column_names_match_config(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path)
        path = tmp_path / "experiments.tsv"
        append_experiments("t", [], path, cfg)

        header = path.read_text().splitlines()[0]
        assert "score" in header
        assert "loss" in header

    def test_no_quality_guard_column(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path, with_qg=False)
        path = tmp_path / "experiments.tsv"
        append_experiments("t", [], path, cfg)

        header = path.read_text().splitlines()[0]
        assert "quality_guard" in header
        assert "loss" not in header


class TestReadMainStats:
    @pytest.fixture()
    def git_repo(self, tmp_path: Path) -> Path:
        subprocess.run(["git", "init", "-b", "main", str(tmp_path)], check=True, capture_output=True)
        subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "t@t.com"], check=True, capture_output=True)
        subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "T"], check=True, capture_output=True)
        return tmp_path

    def _commit_experiments(self, repo: Path, content: str) -> None:
        (repo / "experiments.tsv").write_text(content)
        subprocess.run(["git", "-C", str(repo), "add", "."], check=True, capture_output=True)
        subprocess.run(["git", "-C", str(repo), "commit", "-m", "add experiments"], check=True, capture_output=True)

    def test_no_experiments_file(self, tmp_path: Path, git_repo: Path) -> None:
        cfg = _make_config(tmp_path)
        stats = read_main_stats("main", git_repo, cfg)
        assert stats == {"baseline": None, "best": None}

    def test_reads_baseline_and_best(self, tmp_path: Path, git_repo: Path) -> None:
        cfg = _make_config(tmp_path)
        content = (
            "session\tcommit\tscore\tloss\tstatus\tdescription\n"
            "s1\ta\t100.0\t0.5\tkeep\tbaseline\n"
            "s1\tb\t150.0\t0.5\tkeep\tbatch\n"
            "s1\tc\t80.0\t0.5\tdiscard\tfailed\n"
        )
        self._commit_experiments(git_repo, content)

        stats = read_main_stats("main", git_repo, cfg)
        assert stats["baseline"] == pytest.approx(100.0)
        assert stats["best"] == pytest.approx(150.0)

    def test_minimize_best_is_lowest(self, tmp_path: Path, git_repo: Path) -> None:
        cfg = _make_config(tmp_path, optimize="minimize")
        content = (
            "session\tcommit\tscore\tloss\tstatus\tdescription\n"
            "s1\ta\t0.9\t0.5\tkeep\tbaseline\n"
            "s1\tb\t0.3\t0.5\tkeep\tbetter\n"
        )
        self._commit_experiments(git_repo, content)

        stats = read_main_stats("main", git_repo, cfg)
        assert stats["baseline"] == pytest.approx(0.9)
        assert stats["best"] == pytest.approx(0.3)

    def test_no_kept_rows_returns_none(self, tmp_path: Path, git_repo: Path) -> None:
        cfg = _make_config(tmp_path)
        content = "session\tcommit\tscore\tloss\tstatus\tdescription\ns1\ta\t100.0\t0.5\tdiscard\tfailed\n"
        self._commit_experiments(git_repo, content)

        stats = read_main_stats("main", git_repo, cfg)
        assert stats == {"baseline": None, "best": None}


class TestParseTsvStringEdgeCases:
    def test_single_line_returns_empty(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path)
        subprocess.run(["git", "init", "-b", "main", str(tmp_path)], check=True, capture_output=True)
        subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "t@t.com"], check=True, capture_output=True)
        subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "T"], check=True, capture_output=True)
        (tmp_path / "experiments.tsv").write_text("just-one-line")
        subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True, capture_output=True)
        subprocess.run(["git", "-C", str(tmp_path), "commit", "-m", "add"], check=True, capture_output=True)
        stats = read_main_stats("main", tmp_path, cfg)
        assert stats == {"baseline": None, "best": None}

    def test_blank_data_lines_skipped(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path)
        subprocess.run(["git", "init", "-b", "main", str(tmp_path)], check=True, capture_output=True)
        subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "t@t.com"], check=True, capture_output=True)
        subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "T"], check=True, capture_output=True)
        content = "session\tcommit\tscore\tloss\tstatus\tdescription\n\n\ns1\ta\t100.0\t0.5\tkeep\ttest\n"
        (tmp_path / "experiments.tsv").write_text(content)
        subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True, capture_output=True)
        subprocess.run(["git", "-C", str(tmp_path), "commit", "-m", "add"], check=True, capture_output=True)
        stats = read_main_stats("main", tmp_path, cfg)
        assert stats["baseline"] == pytest.approx(100.0)


class TestBestKeptInvalidScore:
    def test_invalid_score_string_treated_as_zero(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path, optimize="maximize")
        rows = [
            {"commit": "a", "score": "not-a-float", "loss": "0.5", "status": "keep", "description": "x"},
            {"commit": "b", "score": "50.0", "loss": "0.5", "status": "keep", "description": "y"},
        ]
        val, desc = best_kept(rows, cfg)
        assert val == pytest.approx(50.0)
        assert desc == "y"


class TestAppendExperimentsNoTrailingNewline:
    def test_appends_even_without_trailing_newline(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path)
        path = tmp_path / "experiments.tsv"
        # Write existing content without trailing newline
        path.write_text("session\tcommit\tscore\tloss\tstatus\tdescription\ns1\ta\t90.0\t0.5\tkeep\tfirst")
        rows = [{"commit": "b", "score": "95.0", "loss": "0.5", "status": "keep", "description": "second"}]
        append_experiments("s2", rows, path, cfg)
        lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
        assert len(lines) == 3
        assert "s2\tb" in lines[2]
