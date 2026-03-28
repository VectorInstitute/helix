"""Shared pytest configuration and fixtures."""

import pytest


# Configure pytest-asyncio to auto mode so async tests work without extra markers.
def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "asyncio: mark test as async")
