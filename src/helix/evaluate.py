"""Metric extraction from experiment command output."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .config import HelixConfig, OptimizeDirection


@dataclass(frozen=True)
class ExperimentResult:
    """Parsed metrics from a single experiment run."""

    primary: float
    quality_guard: float | None


class MetricParser:
    """Extracts metric values from experiment stdout using regex patterns.

    Patterns are defined in ``helix.yaml`` under ``metrics.evaluate.patterns``.
    Each pattern must have exactly one capture group that matches the numeric value.
    """

    def __init__(self, config: HelixConfig) -> None:
        self._primary_re = re.compile(config.metrics.evaluate.patterns.primary, re.MULTILINE)
        qg_pattern = config.metrics.evaluate.patterns.quality_guard
        self._qg_re = re.compile(qg_pattern, re.MULTILINE) if qg_pattern else None

    def parse(self, output: str) -> ExperimentResult | None:
        """Parse experiment stdout and return an ``ExperimentResult``.

        Args:
            output: Full stdout from the experiment command.

        Returns:
            Parsed result, or ``None`` if the primary metric was not found
            (indicating a crash or incomplete run).
        """
        match = self._primary_re.search(output)
        if not match:
            return None
        primary = float(match.group(1))

        quality_guard: float | None = None
        if self._qg_re:
            qg_match = self._qg_re.search(output)
            if qg_match:
                quality_guard = float(qg_match.group(1))

        return ExperimentResult(primary=primary, quality_guard=quality_guard)


def is_improvement(result: ExperimentResult, baseline: ExperimentResult | None, config: HelixConfig) -> bool:
    """Return True if ``result`` beats ``baseline`` without violating the quality guard.

    Args:
        result: Metrics from the current experiment.
        baseline: Best known metrics so far (None if no baseline exists yet).
        config: Helix configuration (optimize direction, quality guard tolerance).

    Returns:
        True if the result is an improvement and the quality guard is satisfied.
    """
    if baseline is None:
        return True

    direction = config.metrics.primary.optimize
    primary_ok = (
        result.primary > baseline.primary
        if direction == OptimizeDirection.maximize
        else result.primary < baseline.primary
    )
    if not primary_ok:
        return False

    qg_cfg = config.metrics.quality_guard
    if qg_cfg is None or result.quality_guard is None or baseline.quality_guard is None:
        return True

    tol = qg_cfg.max_degradation
    if qg_cfg.optimize == OptimizeDirection.minimize:
        return result.quality_guard <= baseline.quality_guard * (1 + tol)
    return result.quality_guard >= baseline.quality_guard * (1 - tol)
