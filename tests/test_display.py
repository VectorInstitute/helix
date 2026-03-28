"""Tests for helix.display."""

from __future__ import annotations

from unittest.mock import MagicMock

from rich.panel import Panel

from helix.config import OptimizeDirection
from helix.display import _fmt_metric, session_summary_panel, startup_panel


def _make_config(
    direction: OptimizeDirection = OptimizeDirection.maximize,
    with_qg: bool = False,
) -> MagicMock:
    config = MagicMock()
    config.name = "test-helix"
    config.domain = "AI/ML"
    config.description = "Test helix."
    config.metrics.primary.name = "score"
    config.metrics.primary.optimize = direction
    config.metrics.evaluate.timeout_seconds = 300
    config.metrics.quality_guard = None
    if with_qg:
        config.metrics.quality_guard = MagicMock()
        config.metrics.quality_guard.name = "loss"
    return config


class TestFmtMetric:
    def test_none_returns_dim_dash(self) -> None:
        fmt = _fmt_metric("score")
        assert "—" in fmt(None)

    def test_float_includes_metric_name(self) -> None:
        fmt = _fmt_metric("score")
        result = fmt(100.0)
        assert "score" in result
        assert "100" in result

    def test_four_sig_figs(self) -> None:
        fmt = _fmt_metric("tps")
        result = fmt(1234.5678)
        assert "1235" in result


class TestStartupPanel:
    def test_returns_panel_instance(self) -> None:
        config = _make_config()
        panel = startup_panel("mar27", 100, {"baseline": None, "best": None}, config)
        assert isinstance(panel, Panel)

    def test_none_stats_does_not_raise(self) -> None:
        config = _make_config()
        panel = startup_panel("mar27", 200, {"baseline": None, "best": None}, config)
        assert isinstance(panel, Panel)

    def test_with_numeric_stats(self) -> None:
        config = _make_config()
        panel = startup_panel("exp1", 50, {"baseline": 50.0, "best": 75.0}, config)
        assert isinstance(panel, Panel)

    def test_hardware_env_var_used(self, monkeypatch: MagicMock) -> None:
        monkeypatch.setenv("HELIX_HARDWARE", "A100")
        config = _make_config()
        panel = startup_panel("mar27", 10, {}, config)
        assert isinstance(panel, Panel)


class TestSessionSummaryPanel:
    def test_returns_panel_instance(self) -> None:
        config = _make_config()
        rows = [
            {"status": "keep", "score": "100.0", "description": "baseline"},
            {"status": "discard", "score": "90.0", "description": "failed"},
            {"status": "crash", "score": "", "description": "oom"},
        ]
        panel = session_summary_panel(rows, {"baseline": 80.0, "best": 100.0}, config)
        assert isinstance(panel, Panel)

    def test_empty_rows(self) -> None:
        config = _make_config()
        panel = session_summary_panel([], {"baseline": None, "best": None}, config)
        assert isinstance(panel, Panel)

    def test_improved_maximize(self) -> None:
        config = _make_config(direction=OptimizeDirection.maximize)
        rows = [{"status": "keep", "score": "200.0", "description": "batch"}]
        panel = session_summary_panel(rows, {"baseline": 100.0, "best": 100.0}, config)
        assert isinstance(panel, Panel)

    def test_improved_minimize(self) -> None:
        config = _make_config(direction=OptimizeDirection.minimize)
        rows = [{"status": "keep", "score": "0.1", "description": "better"}]
        panel = session_summary_panel(rows, {"baseline": 0.9, "best": 0.9}, config)
        assert isinstance(panel, Panel)

    def test_no_improvement(self) -> None:
        config = _make_config()
        rows = [{"status": "discard", "score": "50.0", "description": "bad"}]
        panel = session_summary_panel(rows, {"baseline": 100.0, "best": 100.0}, config)
        assert isinstance(panel, Panel)

    def test_session_best_none_when_no_kept_rows(self) -> None:
        config = _make_config()
        rows = [{"status": "discard", "score": "80.0", "description": "bad"}]
        panel = session_summary_panel(rows, {"baseline": None, "best": None}, config)
        assert isinstance(panel, Panel)

    def test_main_best_none_still_improved(self) -> None:
        config = _make_config(direction=OptimizeDirection.maximize)
        rows = [{"status": "keep", "score": "100.0", "description": "first"}]
        panel = session_summary_panel(rows, {"baseline": None, "best": None}, config)
        assert isinstance(panel, Panel)
