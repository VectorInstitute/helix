"""Tests for helix.hardware."""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from helix.hardware import detect_hardware


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
        with patch("subprocess.run", side_effect=FileNotFoundError):
            with patch("platform.system", return_value="Linux"):
                with patch("platform.processor", return_value="x86_64"):
                    assert detect_hardware() == "x86_64"

    def test_nonzero_exit_falls_through(self) -> None:
        with patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "nvidia-smi")):
            with patch("platform.system", return_value="Linux"):
                with patch("platform.processor", return_value="x86_64"):
                    assert detect_hardware() == "x86_64"

    def test_timeout_falls_through(self) -> None:
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("nvidia-smi", 5)):
            with patch("platform.system", return_value="Linux"):
                with patch("platform.processor", return_value="x86_64"):
                    assert detect_hardware() == "x86_64"

    def test_empty_output_falls_through(self) -> None:
        with patch("subprocess.run", return_value=MagicMock(returncode=0, stdout="\n  \n")):
            with patch("platform.system", return_value="Linux"):
                with patch("platform.processor", return_value="x86_64"):
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

        with patch("subprocess.run", side_effect=_side_effect):
            with patch("platform.system", return_value="Darwin"):
                assert detect_hardware() == "Apple M4 Pro"

    def test_sysctl_not_found_falls_through(self) -> None:
        with patch("subprocess.run", side_effect=FileNotFoundError):
            with patch("platform.system", return_value="Darwin"):
                with patch("platform.processor", return_value="arm"):
                    assert detect_hardware() == "arm"

    def test_sysctl_empty_output_falls_through(self) -> None:
        def _side_effect(cmd, **kwargs):
            if "nvidia-smi" in cmd:
                raise FileNotFoundError
            return MagicMock(returncode=0, stdout="  \n")

        with patch("subprocess.run", side_effect=_side_effect):
            with patch("platform.system", return_value="Darwin"):
                with patch("platform.processor", return_value="arm"):
                    assert detect_hardware() == "arm"

    def test_non_darwin_skips_sysctl(self) -> None:
        commands_called: list[list[str]] = []

        def _track(cmd, **kwargs):
            commands_called.append(list(cmd))
            raise FileNotFoundError

        with patch("subprocess.run", side_effect=_track):
            with patch("platform.system", return_value="Linux"):
                with patch("platform.processor", return_value="x86_64"):
                    detect_hardware()

        assert not any("sysctl" in cmd for cmd in commands_called)


class TestDetectHardwareFallback:
    def test_processor_returned(self) -> None:
        with patch("subprocess.run", side_effect=FileNotFoundError):
            with patch("platform.system", return_value="Linux"):
                with patch("platform.processor", return_value="Intel(R) Core(TM) i9"):
                    assert detect_hardware() == "Intel(R) Core(TM) i9"

    def test_machine_returned_when_processor_empty(self) -> None:
        with patch("subprocess.run", side_effect=FileNotFoundError):
            with patch("platform.system", return_value="Linux"):
                with patch("platform.processor", return_value=""):
                    with patch("platform.machine", return_value="aarch64"):
                        assert detect_hardware() == "aarch64"

    def test_unknown_when_all_empty(self) -> None:
        with patch("subprocess.run", side_effect=FileNotFoundError):
            with patch("platform.system", return_value="Linux"):
                with patch("platform.processor", return_value=""):
                    with patch("platform.machine", return_value=""):
                        assert detect_hardware() == "unknown"


class TestRunnerHardwareEnvVar:
    def test_env_var_not_overwritten(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Explicit HELIX_HARDWARE always wins — detect_hardware is never called."""
        monkeypatch.setenv("HELIX_HARDWARE", "My Custom GPU")
        with patch("helix.runner.detect_hardware") as mock_detect:
            import os
            if "HELIX_HARDWARE" not in os.environ:
                os.environ["HELIX_HARDWARE"] = mock_detect()
            assert os.environ["HELIX_HARDWARE"] == "My Custom GPU"
            mock_detect.assert_not_called()

    def test_env_var_set_when_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When env var absent, detect_hardware result is written to HELIX_HARDWARE."""
        monkeypatch.delenv("HELIX_HARDWARE", raising=False)
        with patch("helix.runner.detect_hardware", return_value="Apple M4 Pro"):
            import os
            if "HELIX_HARDWARE" not in os.environ:
                from helix.runner import detect_hardware as _dh
                os.environ["HELIX_HARDWARE"] = _dh()
            assert os.environ["HELIX_HARDWARE"] == "Apple M4 Pro"
