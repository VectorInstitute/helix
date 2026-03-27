"""Helix configuration schema — loads and validates helix.yaml."""

from __future__ import annotations

import re
from enum import Enum
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class OptimizeDirection(str, Enum):
    """Direction for metric optimization."""

    maximize = "maximize"
    minimize = "minimize"


class PrimaryMetric(BaseModel):
    """Primary optimization metric."""

    name: str = Field(description="Metric name, used as the column header in results.tsv")
    optimize: OptimizeDirection = Field(description="Whether higher or lower values are better")


class QualityGuard(BaseModel):
    """Secondary metric that must not degrade past a tolerance threshold."""

    name: str = Field(description="Metric name, used as the column header in results.tsv")
    optimize: OptimizeDirection = Field(description="Whether higher or lower values are better")
    max_degradation: float = Field(
        default=0.01,
        ge=0.0,
        description="Maximum allowed relative degradation (0.01 = 1%)",
    )


class OutputPatterns(BaseModel):
    """Regex patterns to extract metric values from experiment stdout.

    Each pattern must contain exactly one capture group ``(...)`` that
    matches the numeric value.
    """

    primary: str = Field(description="Regex pattern with one capture group for the primary metric")
    quality_guard: str | None = Field(default=None, description="Regex pattern for the quality guard metric")

    def grep_hint(self) -> str:
        """Return a simple grep-compatible prefix derived from the primary pattern.

        Strips the capture group so the agent can use it directly in grep
        to confirm a metric appeared in the log.

        Returns
        -------
        str
            Literal prefix suitable for use in ``grep "..."  run.log``.
        """
        prefix = re.split(r"[\(\[\\\+\*\?]", self.primary.lstrip("^"))[0]
        return prefix.rstrip("\\s").rstrip()


class EvaluateConfig(BaseModel):
    """How to run a single experiment and extract its results."""

    command: str = Field(description="Shell command that runs the experiment and prints metrics to stdout")
    timeout_seconds: int = Field(default=300, gt=0, description="Wall-clock budget per experiment in seconds")
    output_format: Literal["pattern"] = "pattern"
    patterns: OutputPatterns


class MetricsConfig(BaseModel):
    """Full metric specification for a helix."""

    primary: PrimaryMetric
    quality_guard: QualityGuard | None = None
    evaluate: EvaluateConfig


class ScopeConfig(BaseModel):
    """Defines which files the agent may modify versus only read."""

    editable: list[str] = Field(
        default_factory=list,
        description="File paths or globs the agent is allowed to modify",
    )
    readonly: list[str] = Field(
        default_factory=list,
        description="File paths the agent should read but must not modify",
    )


class RequirementsConfig(BaseModel):
    """Runtime requirements (informational, shown in startup panel)."""

    python: str = ">=3.10"
    gpu: str | None = None


class HelixConfig(BaseModel):
    """Full helix.yaml configuration.

    Load with ``HelixConfig.load(path)``; the constructor is for tests only.
    """

    name: str
    version: str = "1.0.0"
    domain: str
    description: str
    author: str | None = None
    scope: ScopeConfig
    metrics: MetricsConfig
    requirements: RequirementsConfig = Field(default_factory=RequirementsConfig)

    @classmethod
    def load(cls, path: Path) -> HelixConfig:
        """Load and validate a helix.yaml file.

        Parameters
        ----------
        path : Path
            Absolute path to a ``helix.yaml`` file.

        Returns
        -------
        HelixConfig
            Validated configuration object.

        Raises
        ------
        FileNotFoundError
            If the path does not exist.
        pydantic.ValidationError
            If the YAML content does not match the schema.
        """
        with path.open() as fh:
            data = yaml.safe_load(fh)
        return cls.model_validate(data)

    def interesting_keywords(self) -> set[str]:
        """Return keywords worth surfacing when filtering agent log output.

        Returns
        -------
        set[str]
            Set of lowercase keywords (metric names plus standard status words).
        """
        base: set[str] = {
            "keep",
            "discard",
            "crash",
            "baseline",
            "experiment",
            "improvement",
            "error",
            "failed",
            "✓",
            "✗",
            "→",
            "running",
            "result",
        }
        base.add(self.metrics.primary.name.lower())
        if self.metrics.quality_guard:
            base.add(self.metrics.quality_guard.name.lower())
        return base
