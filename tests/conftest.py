"""Global pytest configuration for anyio testing."""

import pytest

pytest_plugins = ("anyio",)


@pytest.fixture(params=["asyncio", "trio"])
def anyio_backend(request):
    """Configure anyio to test with both asyncio and trio backends."""
    return request.param
