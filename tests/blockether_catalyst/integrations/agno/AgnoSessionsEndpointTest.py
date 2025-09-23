"""Tests for the AgnoOsASGIModule sessions endpoint."""

from typing import Any, Dict
from unittest.mock import MagicMock, Mock, patch

import pytest
from fastapi import APIRouter
from fastapi.testclient import TestClient

from blockether_catalyst.asgi.ASGICoreApplication import ASGICoreApplication
from blockether_catalyst.integrations.agno.AgnoOsASGIModule import (
    AgnoOsASGIModule,
    AssistantConfig,
    ChatConfig,
)


class TestAgnoSessionsEndpoint:
    """Test suite for AgnoOsASGIModule sessions endpoint."""

    @pytest.fixture
    def mock_runner(self):
        """Create a mock runner with database."""
        runner = MagicMock()
        runner.db = MagicMock()
        return runner

    @pytest.fixture
    def mock_runner_no_db(self):
        """Create a mock runner without database."""
        runner = MagicMock()
        runner.db = None
        return runner

    @pytest.fixture
    def mock_workflow(self):
        """Create a mock workflow."""
        workflow = MagicMock()
        workflow.name = "Test Workflow"
        workflow.db = MagicMock()
        return workflow

    @pytest.fixture
    def agno_module_with_db(self, mock_runner, mock_workflow):
        """Create AgnoOsASGIModule with database."""
        chat_config = ChatConfig(
            assistant=AssistantConfig(name="Test Assistant", runner=mock_runner),
            base_url="http://localhost:8000",
        )

        module = AgnoOsASGIModule(
            title="Test Module",
            description="Test Description",
            chat=chat_config,
            workflows=[mock_workflow],
        )
        return module

    @pytest.fixture
    def agno_module_no_db(self, mock_runner_no_db, mock_workflow):
        """Create AgnoOsASGIModule without database."""
        chat_config = ChatConfig(
            assistant=AssistantConfig(name="Test Assistant", runner=mock_runner_no_db),
            base_url="http://localhost:8000",
        )

        module = AgnoOsASGIModule(
            title="Test Module",
            description="Test Description",
            chat=chat_config,
            workflows=[mock_workflow],
        )
        return module

    @pytest.fixture
    def test_app_with_db(self, agno_module_with_db):
        """Create test app with database."""
        from fastapi import FastAPI

        app = FastAPI(title="Test App", description="Test", version="1.0.0")
        router = APIRouter()
        agno_module_with_db.mount(app, router)
        app.include_router(router, prefix="/os")  # Add the /os prefix
        return app

    @pytest.fixture
    def test_app_no_db(self, agno_module_no_db):
        """Create test app without database."""
        from fastapi import FastAPI

        app = FastAPI(title="Test App", description="Test", version="1.0.0")
        router = APIRouter()
        agno_module_no_db.mount(app, router)
        app.include_router(router, prefix="/os")  # Add the /os prefix
        return app

    @pytest.fixture
    def test_client_with_db(self, test_app_with_db):
        """Create test client with database."""
        return TestClient(test_app_with_db)

    @pytest.fixture
    def test_client_no_db(self, test_app_no_db):
        """Create test client without database."""
        return TestClient(test_app_no_db)

    def skip_test_sessions_endpoint_with_query_params(self, test_client_no_db):
        """Test /sessions endpoint with query parameters."""
        response = test_client_no_db.get(
            "/os/sessions",
            params={
                "type": "workflow",  # type is required, cannot be empty
                "component_id": "",
                "limit": 20,
                "page": 1,
                "sort_by": "created_at",
                "sort_order": "desc",
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert data["sessions"] == []
        assert data["total"] == 0
        assert data["page"] == 1
        assert data["limit"] == 20
        assert data["total_pages"] == 0

    def skip_test_sessions_endpoint_with_custom_pagination(self, test_client_no_db):
        """Test /sessions endpoint with custom pagination parameters."""
        response = test_client_no_db.get("/os/sessions", params={"type": "workflow", "limit": 50, "page": 2})

        assert response.status_code == 200
        data = response.json()

        assert data["sessions"] == []
        assert data["total"] == 0
        assert data["page"] == 2
        assert data["limit"] == 50
        assert data["total_pages"] == 0

    def test_session_runs_endpoint_no_database(self, test_client_no_db):
        """Test /sessions/{session_id}/runs endpoint without database."""
        response = test_client_no_db.get("/os/sessions/test-session-123/runs?type=workflow")

        assert response.status_code == 200
        data = response.json()
        assert data == []

    def test_executor_runs_with_empty_body(self, test_client_no_db):
        """Test /executor/runs endpoint with empty body."""
        response = test_client_no_db.post("/os/executor/runs", data={})

        assert response.status_code == 200
        data = response.json()

        # Should return error for missing message
        assert data["status"] == "error"
        assert "Missing required field: 'message'" in data["error"]

    def test_executor_runs_with_invalid_json(self, test_client_no_db):
        """Test /executor/runs endpoint with invalid form data."""
        response = test_client_no_db.post(
            "/os/executor/runs",
            data={"message": ""},  # Empty message
        )

        assert response.status_code == 200
        data = response.json()

        # Should return error for empty message
        assert data["status"] == "error"
        assert "Missing required field: 'message'" in data["error"]

    def test_executor_runs_with_valid_input(self, test_client_no_db, mock_runner_no_db):
        """Test /executor/runs endpoint with valid input."""
        # Mock the runner to have a run method that returns WorkflowRunOutput-like object
        from types import SimpleNamespace

        mock_result = SimpleNamespace(content="Test response", session_id="test-123", run_id="run-456")
        mock_runner_no_db.run = MagicMock(return_value=mock_result)

        response = test_client_no_db.post(
            "/os/executor/runs",
            data={"message": "test query", "session_id": "test-123"},
        )

        assert response.status_code == 200
        data = response.json()

        # Should execute successfully
        assert "error" not in data or data.get("status") != "error"
        assert "run_id" in data
        assert "session_id" in data
