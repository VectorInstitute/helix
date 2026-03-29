"""Tests for helix.hardware."""

from __future__ import annotations

import contextlib
import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from helix.hardware import detect_hardware
from helix.runner import HelixRunner


def _nvidia_result(names: list[str]) -> MagicMock:
    return MagicMock(returncode=0, stdout="\n".join(names) + "\n")


def _sysctl_result(name: str) -> MagicMock:
    return MagicMock(returncode=0, stdout=f"{name}\n")


class TestDetectHardwareNvidia:
    def test_single_gpu(self) -> None:
        with patch("subprocess.run", return_value=_nvidia_result(["NVIDIA H100 80GB HBM3"])):
            assert detect_hardware() == "NVIDIA H100 80GB HBM3"

    def test_multiple_identical_gpus_deduplicated(self) -> None:
        with patch("subprocess.run", return_value=_nvidia_result(["NVIDIA A100"] * 3)):
            assert detect_hardware() == "NVIDIA A100"

    def test_multiple_different_gpus(self) -> None:
        with patch("subprocess.run", return_value=_nvidia_result(["NVIDIA A100", "NVIDIA H100"])):
            assert detect_hardware() == "NVIDIA A100, NVIDIA H100"

    def test_not_found_falls_through(self) -> None:
        with (
            patch("subprocess.run", side_effect=FileNotFoundError),
            patch("platform.system", return_value="Linux"),
            patch("platform.processor", return_value="x86_64"),
        ):
            assert detect_hardware() == "x86_64"

    def test_nonzero_exit_falls_through(self) -> None:
        with (
            patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "nvidia-smi")),
            patch("platform.system", return_value="Linux"),
            patch("platform.processor", return_value="x86_64"),
        ):
            assert detect_hardware() == "x86_64"

    def test_timeout_falls_through(self) -> None:
        with (
            patch("subprocess.run", side_effect=subprocess.TimeoutExpired("nvidia-smi", 5)),
            patch("platform.system", return_value="Linux"),
            patch("platform.processor", return_value="x86_64"),
        ):
            assert detect_hardware() == "x86_64"

    def test_empty_output_falls_through(self) -> None:
        with (
            patch("subprocess.run", return_value=MagicMock(returncode=0, stdout="\n  \n")),
            patch("platform.system", return_value="Linux"),
            patch("platform.processor", return_value="x86_64"),
        ):
            assert detect_hardware() == "x86_64"


class TestDetectHardwareMacOS:
    def test_apple_silicon_via_sysctl(self) -> None:
        call_count = 0

        def _side_effect(cmd, **kwargs):
            nonlocal call_count
            call_count += 1
            if "nvidia-smi" in cmd:
                raise FileNotFoundError
            if "sysctl" in cmd:
                return _sysctl_result("Apple M4 Pro")
            raise FileNotFoundError

        with (
            patch("subprocess.run", side_effect=_side_effect),
            patch("platform.system", return_value="Darwin"),
        ):
            assert detect_hardware() == "Apple M4 Pro"

    def test_sysctl_not_found_falls_through(self) -> None:
        with (
            patch("subprocess.run", side_effect=FileNotFoundError),
            patch("platform.system", return_value="Darwin"),
            patch("platform.processor", return_value="arm"),
        ):
            assert detect_hardware() == "arm"

    def test_sysctl_empty_output_falls_through(self) -> None:
        def _side_effect(cmd, **kwargs):
            if "nvidia-smi" in cmd:
                raise FileNotFoundError
            return MagicMock(returncode=0, stdout="  \n")

        with (
            patch("subprocess.run", side_effect=_side_effect),
            patch("platform.system", return_value="Darwin"),
            patch("platform.processor", return_value="arm"),
        ):
            assert detect_hardware() == "arm"

    def test_non_darwin_skips_sysctl(self) -> None:
        commands_called: list[list[str]] = []

        def _track(cmd, **kwargs):
            commands_called.append(list(cmd))
            raise FileNotFoundError

        with (
            patch("subprocess.run", side_effect=_track),
            patch("platform.system", return_value="Linux"),
            patch("platform.processor", return_value="x86_64"),
        ):
            detect_hardware()

        assert not any("sysctl" in cmd for cmd in commands_called)


class TestDetectHardwareFallback:
    def test_processor_returned(self) -> None:
        with (
            patch("subprocess.run", side_effect=FileNotFoundError),
            patch("platform.system", return_value="Linux"),
            patch("platform.processor", return_value="Intel(R) Core(TM) i9"),
        ):
            assert detect_hardware() == "Intel(R) Core(TM) i9"

    def test_machine_returned_when_processor_empty(self) -> None:
        with (
            patch("subprocess.run", side_effect=FileNotFoundError),
            patch("platform.system", return_value="Linux"),
            patch("platform.processor", return_value=""),
            patch("platform.machine", return_value="aarch64"),
        ):
            assert detect_hardware() == "aarch64"

    def test_unknown_when_all_empty(self) -> None:
        with (
            patch("subprocess.run", side_effect=FileNotFoundError),
            patch("platform.system", return_value="Linux"),
            patch("platform.processor", return_value=""),
            patch("platform.machine", return_value=""),
        ):
            assert detect_hardware() == "unknown"


class TestRunnerHardwareEnvVar:
    def _make_runner(self, tmp_path: Path) -> HelixRunner:
        with (
            patch("helix.runner.HelixConfig.load"),
            patch("helix.runner.detect_main_branch", return_value="main"),
        ):
            return HelixRunner(tmp_path, "test", max_turns=1)

    def _run_with_patches(self, runner: HelixRunner, **extra_patches: object) -> dict:
        """Run runner.run() with all infrastructure patched out. Returns entered mocks."""
        patches = {
            "helix.runner.HelixRunner._preflight": None,
            "helix.runner.read_main_stats": dict,
            "helix.runner.anyio.run": None,
            "helix.runner.HelixRunner._post_session": None,
            "helix.runner.console.print": None,
            "helix.runner.atexit.register": None,
            "helix.runner.signal.signal": None,
            **extra_patches,
        }
        mocks: dict = {}
        with contextlib.ExitStack() as stack:
            for target, retval in patches.items():
                kw = {} if retval is None or retval is dict else {"return_value": retval}
                if retval is dict:
                    kw = {"return_value": {}}
                mocks[target] = stack.enter_context(patch(target, **kw))
            runner.run()
        return mocks

    def test_env_var_not_overwritten(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Explicit HELIX_HARDWARE always wins — detect_hardware is never called."""
        monkeypatch.setenv("HELIX_HARDWARE", "My Custom GPU")
        runner = self._make_runner(tmp_path)
        mocks = self._run_with_patches(runner, **{"helix.runner.detect_hardware": None})
        mocks["helix.runner.detect_hardware"].assert_not_called()
        assert os.environ["HELIX_HARDWARE"] == "My Custom GPU"

    def test_env_var_set_when_missing(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """When env var absent, detect_hardware result is written to HELIX_HARDWARE."""
        monkeypatch.delenv("HELIX_HARDWARE", raising=False)
        runner = self._make_runner(tmp_path)
        with patch("helix.runner.detect_hardware", return_value="Apple M4 Pro"):
            self._run_with_patches(runner)
        assert os.environ["HELIX_HARDWARE"] == "Apple M4 Pro"
