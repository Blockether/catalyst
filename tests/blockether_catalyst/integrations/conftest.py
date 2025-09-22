"""Global configuration and fixtures for integration tests.

ALL INTEGRATION TESTS USE REAL LLM - NO MOCKS!
"""

import os

import httpx
import pytest
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAILike

# Global LLM Configuration - Defaults to local proxy (no API key needed)
LLM_BASE_URL = os.getenv("LLM_BASE_URL", os.getenv("OPENAI_BASE_URL", "http://localhost:3005/v1"))
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")

# Determine API key based on whether we're using local proxy or not
if "localhost" in LLM_BASE_URL or "127.0.0.1" in LLM_BASE_URL:
    # Local proxy doesn't need real API key - use dummy key format
    LLM_API_KEY = os.getenv("LLM_API_KEY", os.getenv("OPENAI_API_KEY", "sk-proj-local-proxy-key-not-needed"))
else:
    # Real OpenAI or external service - require actual API key
    LLM_API_KEY = os.getenv("LLM_API_KEY", os.getenv("OPENAI_API_KEY"))
    if not LLM_API_KEY:
        raise ValueError(
            f"API key required for non-local LLM service at {LLM_BASE_URL}. "
            "Set OPENAI_API_KEY or LLM_API_KEY environment variable."
        )

# Always set OPENAI_API_KEY for Agno compatibility (even for local proxy)
if not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = LLM_API_KEY

# Also set OPENAI_BASE_URL for consistency
if not os.getenv("OPENAI_BASE_URL"):
    os.environ["OPENAI_BASE_URL"] = LLM_BASE_URL


def check_llm_server() -> None:
    """Check if LLM server is available. FAIL IMMEDIATELY if not."""
    try:
        with httpx.Client() as client:
            response = client.get(f"{LLM_BASE_URL}/models", timeout=2.0)
            assert response.status_code == 200, (
                f"\n❌ LLM SERVER NOT RESPONDING at {LLM_BASE_URL}\n"
                f"Status: {response.status_code}\n"
                "ALL INTEGRATION TESTS REQUIRE REAL LLM SERVER!\n"
                "NO MOCKS ALLOWED!"
            )
    except Exception as e:
        pytest.fail(
            f"\n❌ CANNOT CONNECT TO LLM SERVER at {LLM_BASE_URL}\n"
            f"Error: {e}\n"
            "ALL INTEGRATION TESTS REQUIRE REAL LLM SERVER!\n"
            "NO MOCKS ALLOWED!"
        )


@pytest.fixture(autouse=True, scope="session")
def require_llm_server():
    """Fixture that ensures LLM server is available for ALL integration tests."""
    check_llm_server()


@pytest.fixture(scope="session")
def test_llm():
    """Global REAL LLM instance for all integration tests."""
    return OpenAILike(
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        id=LLM_MODEL,
        temperature=0.1,  # Low temperature for consistent test results
    )


@pytest.fixture(scope="function")
def test_db():
    """Fresh in-memory database for each test."""
    return SqliteDb()


@pytest.fixture(scope="session")
def shared_db():
    """Shared database for tests that need persistence."""
    return SqliteDb()


# Markers for different test types
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "real_llm: marks tests that make REAL LLM calls (no mocks)")
    config.addinivalue_line("markers", "slow: marks tests that take longer due to LLM calls")
    config.addinivalue_line(
        "markers",
        "require_llm_server: marks tests that require LLM server to be running",
    )


# Automatically mark all integration tests
def pytest_collection_modifyitems(config, items):
    """Automatically add markers to integration tests."""
    for item in items:
        # All tests in integrations folder use real LLM
        if "integrations" in str(item.fspath):
            item.add_marker(pytest.mark.real_llm)
