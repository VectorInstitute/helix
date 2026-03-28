"""Hardware auto-detection for helix session display and agent prompts."""

from __future__ import annotations

import platform
import subprocess


def detect_hardware() -> str:
    """Return a human-readable hardware description.

    Detection order:

    1. NVIDIA GPU via ``nvidia-smi`` (works on Linux and Windows).
    2. Apple Silicon / macOS CPU via ``sysctl``.
    3. ``platform.processor()`` → ``platform.machine()`` → ``"unknown"``.

    Returns
    -------
    str
        Hardware description, e.g. ``"NVIDIA H100 80GB HBM3"`` or ``"Apple M4 Pro"``.
    """
    # 1. NVIDIA GPU
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        names = [n.strip() for n in result.stdout.strip().splitlines() if n.strip()]
        if names:
            return ", ".join(dict.fromkeys(names))  # deduplicate, preserve order
    except Exception:  # noqa: BLE001
        pass

    # 2. Apple Silicon / macOS CPU
    if platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            )
            name = result.stdout.strip()
            if name:
                return name
        except Exception:  # noqa: BLE001
            pass

    # 3. Generic CPU fallback
    return platform.processor() or platform.machine() or "unknown"
