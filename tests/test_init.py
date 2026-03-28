"""Unit tests for helix.init: scaffold(), _render(), list_templates()."""

from __future__ import annotations

from pathlib import Path

import pytest

from helix.config import HelixConfig
from helix.init import _render, list_templates, scaffold


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


class TestListTemplates:
    """Tests for list_templates()."""

    def test_returns_sorted_list(self) -> None:
        templates = list_templates()
        assert templates == sorted(templates)

    def test_contains_expected_templates(self) -> None:
        templates = list_templates()
        assert "generic" in templates
        assert "ai-inference" in templates

    def test_returns_list(self) -> None:
        assert isinstance(list_templates(), list)


class TestScaffold:
    """Tests for scaffold()."""

    def test_generic_template_creates_directory(self, tmp_path: Path) -> None:
        target = scaffold("myhelix", tmp_path)
        assert target.is_dir()
        assert target.name == "myhelix"

    def test_generic_template_creates_all_files(self, tmp_path: Path) -> None:
        target = scaffold("myhelix", tmp_path)
        assert (target / "README.md").exists()
        assert (target / "helix.yaml").exists()
        assert (target / "program.md").exists()
        assert (target / "solver.py").exists()
        assert (target / "evaluate.py").exists()
        assert (target / "experiments.tsv").exists()

    def test_ai_inference_template_creates_files(self, tmp_path: Path) -> None:
        target = scaffold("myinfer", tmp_path, template="ai-inference")
        assert (target / "README.md").exists()
        assert (target / "helix.yaml").exists()
        assert (target / "program.md").exists()
        assert (target / "experiments.tsv").exists()

    def test_ai_inference_template_no_solver(self, tmp_path: Path) -> None:
        target = scaffold("myinfer", tmp_path, template="ai-inference")
        assert not (target / "solver.py").exists()

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
        # generic template uses "score" as primary metric
        assert "score" in header
        assert "status" in header
        assert "description" in header

    def test_experiments_tsv_ai_inference_header(self, tmp_path: Path) -> None:
        target = scaffold("myinfer", tmp_path, template="ai-inference")
        header = (target / "experiments.tsv").read_text().strip()
        assert "tokens_per_sec" in header
        assert "bpb" in header

    def test_helix_yaml_is_valid_config(self, tmp_path: Path) -> None:
        target = scaffold("proj", tmp_path)
        config = HelixConfig.load(target / "helix.yaml")
        assert config.name == "proj"
        assert config.metrics.primary.name == "score"

    def test_returns_path_to_created_dir(self, tmp_path: Path) -> None:
        result = scaffold("x", tmp_path)
        assert isinstance(result, Path)
        assert result == tmp_path / "x"

    def test_raises_value_error_on_unknown_template(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Unknown template"):
            scaffold("x", tmp_path, template="nonexistent")

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

    def test_ai_inference_readme_mentions_primary_metric(self, tmp_path: Path) -> None:
        target = scaffold("myinfer", tmp_path, template="ai-inference")
        content = (target / "README.md").read_text()
        assert "tokens_per_sec" in content

    def test_ai_inference_readme_mentions_quality_guard(self, tmp_path: Path) -> None:
        target = scaffold("myinfer", tmp_path, template="ai-inference")
        content = (target / "README.md").read_text()
        assert "bpb" in content

    def test_ai_inference_readme_no_unrendered_placeholders(self, tmp_path: Path) -> None:
        target = scaffold("myinfer", tmp_path, template="ai-inference", description="Fast inference.")
        content = (target / "README.md").read_text()
        assert "{{" not in content
