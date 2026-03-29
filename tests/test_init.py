"""Unit tests for helix.init: scaffold() and _render()."""

from __future__ import annotations

from pathlib import Path

import pytest

from helix.config import HelixConfig
from helix.init import _render, scaffold


class TestRender:
    """Tests for the _render() template substitution helper."""

    def test_single_placeholder(self) -> None:
        assert _render("hello {{name}}", {"name": "world"}) == "hello world"

    def test_multiple_placeholders(self) -> None:
        result = _render("{{a}} + {{b}} = {{c}}", {"a": "1", "b": "2", "c": "3"})
        assert result == "1 + 2 = 3"

    def test_repeated_placeholder(self) -> None:
        result = _render("{{x}} and {{x}}", {"x": "foo"})
        assert result == "foo and foo"

    def test_unknown_placeholder_left_intact(self) -> None:
        result = _render("hello {{unknown}}", {"name": "world"})
        assert result == "hello {{unknown}}"

    def test_empty_substitutions(self) -> None:
        result = _render("no placeholders here", {})
        assert result == "no placeholders here"

    def test_empty_content(self) -> None:
        assert _render("", {"name": "x"}) == ""


class TestScaffold:
    """Tests for scaffold()."""

    def test_creates_directory(self, tmp_path: Path) -> None:
        target = scaffold("myhelix", tmp_path)
        assert target.is_dir()
        assert target.name == "myhelix"

    def test_creates_all_files(self, tmp_path: Path) -> None:
        target = scaffold("myhelix", tmp_path)
        assert (target / "README.md").exists()
        assert (target / "helix.yaml").exists()
        assert (target / "program.md").exists()
        assert (target / "solver.py").exists()
        assert (target / "evaluate.py").exists()
        assert (target / "experiments.tsv").exists()

    def test_name_substituted_in_helix_yaml(self, tmp_path: Path) -> None:
        target = scaffold("coolproject", tmp_path)
        content = (target / "helix.yaml").read_text()
        assert "coolproject" in content
        assert "{{name}}" not in content

    def test_domain_substituted_in_helix_yaml(self, tmp_path: Path) -> None:
        target = scaffold("proj", tmp_path, domain="AI/ML")
        content = (target / "helix.yaml").read_text()
        assert "AI/ML" in content
        assert "{{domain}}" not in content

    def test_description_substituted_in_program_md(self, tmp_path: Path) -> None:
        desc = "Optimize inference throughput on WikiText-2."
        target = scaffold("proj", tmp_path, description=desc)
        content = (target / "program.md").read_text()
        assert desc in content
        assert "{{description}}" not in content

    def test_experiments_tsv_has_correct_header(self, tmp_path: Path) -> None:
        target = scaffold("myhelix", tmp_path)
        header = (target / "experiments.tsv").read_text().strip()
        assert "score" in header
        assert "status" in header
        assert "description" in header

    def test_helix_yaml_is_valid_config(self, tmp_path: Path) -> None:
        target = scaffold("proj", tmp_path)
        config = HelixConfig.load(target / "helix.yaml")
        assert config.name == "proj"
        assert config.metrics.primary.name == "score"

    def test_returns_path_to_created_dir(self, tmp_path: Path) -> None:
        result = scaffold("x", tmp_path)
        assert isinstance(result, Path)
        assert result == tmp_path / "x"

    def test_raises_file_exists_error_if_dir_exists(self, tmp_path: Path) -> None:
        (tmp_path / "x").mkdir()
        with pytest.raises(FileExistsError, match="already exists"):
            scaffold("x", tmp_path)

    def test_default_domain_is_general(self, tmp_path: Path) -> None:
        target = scaffold("proj", tmp_path)
        config = HelixConfig.load(target / "helix.yaml")
        assert config.domain == "General"

    def test_custom_domain_propagates_to_config(self, tmp_path: Path) -> None:
        target = scaffold("proj", tmp_path, domain="Robotics")
        config = HelixConfig.load(target / "helix.yaml")
        assert config.domain == "Robotics"

    def test_readme_name_substituted(self, tmp_path: Path) -> None:
        target = scaffold("myproject", tmp_path)
        content = (target / "README.md").read_text()
        assert "myproject" in content
        assert "{{name}}" not in content

    def test_readme_description_substituted(self, tmp_path: Path) -> None:
        desc = "Solve the travelling salesman problem faster."
        target = scaffold("tsp", tmp_path, description=desc)
        content = (target / "README.md").read_text()
        assert desc in content
        assert "{{description}}" not in content

    def test_readme_no_unrendered_placeholders(self, tmp_path: Path) -> None:
        target = scaffold("proj", tmp_path, domain="Robotics", description="Optimize grasping.")
        content = (target / "README.md").read_text()
        assert "{{" not in content

    def test_readme_links_to_helix_framework(self, tmp_path: Path) -> None:
        target = scaffold("proj", tmp_path)
        content = (target / "README.md").read_text()
        assert "github.com/VectorInstitute/helix" in content
