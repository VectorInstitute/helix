"""Unit tests for helix.init: scaffold(), _render(), and run_uv_lock()."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from helix.config import HelixConfig
from helix.init import _render, run_uv_lock, scaffold


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

    def test_creates_helix_files(self, tmp_path: Path) -> None:
        target = scaffold("myhelix", tmp_path)
        assert (target / "helix.yaml").exists()
        assert (target / "program.md").exists()
        assert (target / "README.md").exists()
        assert (target / "experiments.tsv").exists()

    def test_creates_reproducibility_files(self, tmp_path: Path) -> None:
        target = scaffold("myhelix", tmp_path)
        assert (target / "pyproject.toml").exists()
        assert (target / ".python-version").exists()

    def test_does_not_create_solver_py(self, tmp_path: Path) -> None:
        target = scaffold("myhelix", tmp_path)
        assert not (target / "solver.py").exists()

    def test_does_not_create_evaluate_py(self, tmp_path: Path) -> None:
        target = scaffold("myhelix", tmp_path)
        assert not (target / "evaluate.py").exists()

    def test_name_substituted_in_helix_yaml(self, tmp_path: Path) -> None:
        target = scaffold("coolproject", tmp_path)
        content = (target / "helix.yaml").read_text()
        assert "coolproject" in content
        assert "{{name}}" not in content

    def test_name_substituted_in_pyproject_toml(self, tmp_path: Path) -> None:
        target = scaffold("coolproject", tmp_path)
        content = (target / "pyproject.toml").read_text()
        assert 'name = "coolproject"' in content
        assert "{{name}}" not in content

    def test_description_substituted_in_pyproject_toml(self, tmp_path: Path) -> None:
        desc = "Optimize training throughput."
        target = scaffold("proj", tmp_path, description=desc)
        content = (target / "pyproject.toml").read_text()
        assert desc in content
        assert "{{description}}" not in content

    def test_domain_substituted_in_helix_yaml(self, tmp_path: Path) -> None:
        target = scaffold("proj", tmp_path, domain="AI/ML")
        content = (target / "helix.yaml").read_text()
        assert "AI/ML" in content
        assert "{{domain}}" not in content

    def test_description_substituted_in_program_md(self, tmp_path: Path) -> None:
        desc = "Optimize inference throughput."
        target = scaffold("proj", tmp_path, description=desc)
        content = (target / "program.md").read_text()
        assert desc in content
        assert "{{description}}" not in content

    def test_python_version_file_contents(self, tmp_path: Path) -> None:
        target = scaffold("proj", tmp_path)
        content = (target / ".python-version").read_text().strip()
        assert content == "3.12"

    def test_pyproject_requires_python(self, tmp_path: Path) -> None:
        target = scaffold("proj", tmp_path)
        content = (target / "pyproject.toml").read_text()
        assert "requires-python" in content
        assert "3.12" in content

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

    def test_returns_path_to_directory(self, tmp_path: Path) -> None:
        result = scaffold("x", tmp_path)
        assert isinstance(result, Path)
        assert result == tmp_path / "x"

    def test_default_domain_is_general(self, tmp_path: Path) -> None:
        target = scaffold("proj", tmp_path)
        config = HelixConfig.load(target / "helix.yaml")
        assert config.domain == "General"

    def test_custom_domain_propagates_to_config(self, tmp_path: Path) -> None:
        target = scaffold("proj", tmp_path, domain="Robotics")
        config = HelixConfig.load(target / "helix.yaml")
        assert config.domain == "Robotics"

    def test_readme_no_unrendered_placeholders(self, tmp_path: Path) -> None:
        target = scaffold("proj", tmp_path, domain="Robotics", description="Optimize grasping.")
        content = (target / "README.md").read_text()
        assert "{{" not in content

    def test_readme_links_to_helix_framework(self, tmp_path: Path) -> None:
        target = scaffold("proj", tmp_path)
        content = (target / "README.md").read_text()
        assert "github.com/VectorInstitute/helix" in content


class TestScaffoldNonDestructive:
    """scaffold() must not overwrite files that already exist."""

    def test_existing_directory_is_not_an_error(self, tmp_path: Path) -> None:
        (tmp_path / "myhelix").mkdir()
        target = scaffold("myhelix", tmp_path)
        assert target.is_dir()

    def test_existing_pyproject_toml_is_not_overwritten(self, tmp_path: Path) -> None:
        (tmp_path / "myhelix").mkdir()
        existing = tmp_path / "myhelix" / "pyproject.toml"
        existing.write_text("[project]\nname = 'custom'\n")

        scaffold("myhelix", tmp_path)

        assert existing.read_text() == "[project]\nname = 'custom'\n"

    def test_existing_python_version_is_not_overwritten(self, tmp_path: Path) -> None:
        (tmp_path / "myhelix").mkdir()
        pv = tmp_path / "myhelix" / ".python-version"
        pv.write_text("3.11\n")

        scaffold("myhelix", tmp_path)

        assert pv.read_text() == "3.11\n"

    def test_missing_helix_files_are_created_alongside_existing_ones(self, tmp_path: Path) -> None:
        (tmp_path / "proj").mkdir()
        (tmp_path / "proj" / "pyproject.toml").write_text("[project]\nname = 'existing'\n")

        target = scaffold("proj", tmp_path)

        # helix-specific files were created
        assert (target / "helix.yaml").exists()
        assert (target / "program.md").exists()
        # existing file was preserved
        assert (target / "pyproject.toml").read_text() == "[project]\nname = 'existing'\n"

    def test_dot_name_scaffolds_into_current_directory(self, tmp_path: Path) -> None:
        # `helix init .` should scaffold into the given output_dir itself.
        target = scaffold(".", tmp_path)
        assert target.resolve() == tmp_path.resolve()
        assert (tmp_path / "helix.yaml").exists()

    def test_dot_name_uses_directory_name_in_templates(self, tmp_path: Path) -> None:
        # The directory name, not ".", should appear in helix.yaml.
        scaffold(".", tmp_path)
        content = (tmp_path / "helix.yaml").read_text()
        assert f"name: {tmp_path.name}" in content
        assert "name: ." not in content

    def test_existing_experiments_tsv_is_not_overwritten(self, tmp_path: Path) -> None:
        (tmp_path / "proj").mkdir()
        exp = tmp_path / "proj" / "experiments.tsv"
        exp.write_text("session\tcommit\tscore\tstatus\tdescription\nmar01\tabc\t0.9\tkeep\tfirst\n")

        scaffold("proj", tmp_path)

        assert "mar01" in exp.read_text()


class TestRunUvLock:
    """Tests for run_uv_lock()."""

    def test_returns_true_on_success(self, tmp_path: Path) -> None:
        mock_result = MagicMock()
        mock_result.returncode = 0
        with patch("helix.init.subprocess.run", return_value=mock_result) as mock_run:
            assert run_uv_lock(tmp_path) is True
        mock_run.assert_called_once_with(["uv", "lock"], cwd=tmp_path, capture_output=True, check=False)

    def test_returns_false_on_nonzero_exit(self, tmp_path: Path) -> None:
        mock_result = MagicMock()
        mock_result.returncode = 1
        with patch("helix.init.subprocess.run", return_value=mock_result):
            assert run_uv_lock(tmp_path) is False

    def test_returns_false_when_uv_not_installed(self, tmp_path: Path) -> None:
        with patch("helix.init.subprocess.run", side_effect=FileNotFoundError):
            assert run_uv_lock(tmp_path) is False

    def test_passes_target_as_cwd(self, tmp_path: Path) -> None:
        mock_result = MagicMock()
        mock_result.returncode = 0
        with patch("helix.init.subprocess.run", return_value=mock_result) as mock_run:
            run_uv_lock(tmp_path)
        assert mock_run.call_args.kwargs["cwd"] == tmp_path
