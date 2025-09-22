"""Global pytest configuration for anyio testing."""

import pytest

pytest_plugins = ("anyio",)


@pytest.fixture
def anyio_backend() -> str:
    """Configure anyio to test with trio backend only."""
    return "trio"
