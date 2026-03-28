"""Tests for helix.config."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from helix.config import HelixConfig, OptimizeDirection, OutputPatterns


def _minimal_config_dict() -> dict:
    return {
        "name": "test-helix",
        "domain": "AI/ML",
        "description": "A test helix.",
        "scope": {"editable": ["solver.py"], "readonly": ["eval.py"]},
        "metrics": {
            "primary": {"name": "accuracy", "optimize": "maximize"},
            "evaluate": {
                "command": "python eval.py",
                "timeout_seconds": 60,
                "output_format": "pattern",
                "patterns": {"primary": r"^accuracy:\s+([\d.]+)"},
            },
        },
    }


class TestHelixConfigLoad:
    def test_load_minimal(self, tmp_path: Path) -> None:
        data = _minimal_config_dict()
        path = tmp_path / "helix.yaml"
        path.write_text(yaml.dump(data))

        cfg = HelixConfig.load(path)

        assert cfg.name == "test-helix"
        assert cfg.domain == "AI/ML"
        assert cfg.metrics.primary.name == "accuracy"
        assert cfg.metrics.primary.optimize == OptimizeDirection.maximize

    def test_load_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            HelixConfig.load(tmp_path / "nonexistent.yaml")

    def test_defaults_applied(self, tmp_path: Path) -> None:
        data = _minimal_config_dict()
        path = tmp_path / "helix.yaml"
        path.write_text(yaml.dump(data))

        cfg = HelixConfig.load(path)

        assert cfg.version == "1.0.0"
        assert cfg.author is None
        assert cfg.agent.model == "claude-opus-4-6"
        assert cfg.agent.max_turns == 200
        assert cfg.requirements.python == ">=3.10"
        assert cfg.requirements.gpu is None

    def test_quality_guard_optional(self, tmp_path: Path) -> None:
        data = _minimal_config_dict()
        path = tmp_path / "helix.yaml"
        path.write_text(yaml.dump(data))

        cfg = HelixConfig.load(path)

        assert cfg.metrics.quality_guard is None

    def test_quality_guard_loaded(self, tmp_path: Path) -> None:
        data = _minimal_config_dict()
        data["metrics"]["quality_guard"] = {
            "name": "loss",
            "optimize": "minimize",
            "max_degradation": 0.05,
        }
        path = tmp_path / "helix.yaml"
        path.write_text(yaml.dump(data))

        cfg = HelixConfig.load(path)

        assert cfg.metrics.quality_guard is not None
        assert cfg.metrics.quality_guard.name == "loss"
        assert cfg.metrics.quality_guard.optimize == OptimizeDirection.minimize
        assert cfg.metrics.quality_guard.max_degradation == pytest.approx(0.05)

    def test_agent_model_configurable(self, tmp_path: Path) -> None:
        data = _minimal_config_dict()
        data["agent"] = {"model": "claude-sonnet-4-6", "max_turns": 50}
        path = tmp_path / "helix.yaml"
        path.write_text(yaml.dump(data))

        cfg = HelixConfig.load(path)

        assert cfg.agent.model == "claude-sonnet-4-6"
        assert cfg.agent.max_turns == 50

    def test_invalid_optimize_direction(self, tmp_path: Path) -> None:
        data = _minimal_config_dict()
        data["metrics"]["primary"]["optimize"] = "sideways"
        path = tmp_path / "helix.yaml"
        path.write_text(yaml.dump(data))

        with pytest.raises(Exception):
            HelixConfig.load(path)

    def test_scope_editable_and_readonly(self, tmp_path: Path) -> None:
        data = _minimal_config_dict()
        path = tmp_path / "helix.yaml"
        path.write_text(yaml.dump(data))

        cfg = HelixConfig.load(path)

        assert cfg.scope.editable == ["solver.py"]
        assert cfg.scope.readonly == ["eval.py"]

    def test_real_inference_opt_helix(self) -> None:
        """The reference helix must load without errors."""
        helix_yaml = Path(__file__).parent.parent / "examples" / "inference-opt" / "helix.yaml"
        if not helix_yaml.exists():
            pytest.skip("examples/inference-opt/helix.yaml not found")

        cfg = HelixConfig.load(helix_yaml)

        assert cfg.name == "inference-opt"
        assert cfg.metrics.primary.name == "tokens_per_sec"
        assert cfg.metrics.primary.optimize == OptimizeDirection.maximize
        assert cfg.metrics.quality_guard is not None
        assert cfg.metrics.quality_guard.name == "bpb"


class TestInterestingKeywords:
    def test_includes_metric_names(self, tmp_path: Path) -> None:
        data = _minimal_config_dict()
        data["metrics"]["quality_guard"] = {
            "name": "loss",
            "optimize": "minimize",
            "max_degradation": 0.01,
        }
        path = tmp_path / "helix.yaml"
        path.write_text(yaml.dump(data))
        cfg = HelixConfig.load(path)

        keywords = cfg.interesting_keywords()

        assert "accuracy" in keywords
        assert "loss" in keywords
        assert "keep" in keywords
        assert "discard" in keywords

    def test_no_quality_guard(self, tmp_path: Path) -> None:
        data = _minimal_config_dict()
        path = tmp_path / "helix.yaml"
        path.write_text(yaml.dump(data))
        cfg = HelixConfig.load(path)

        keywords = cfg.interesting_keywords()

        assert "accuracy" in keywords
        assert "loss" not in keywords


class TestOutputPatternsGrepHint:
    def test_simple_prefix(self) -> None:
        patterns = OutputPatterns(primary=r"^tokens_per_sec:\s+([\d.]+)")
        assert patterns.grep_hint() == "tokens_per_sec:"

    def test_anchored_pattern(self) -> None:
        patterns = OutputPatterns(primary=r"^accuracy:\s+([\d.]+)")
        assert patterns.grep_hint() == "accuracy:"

    def test_without_anchor(self) -> None:
        patterns = OutputPatterns(primary=r"score:\s+([\d.]+)")
        assert patterns.grep_hint().startswith("score:")
