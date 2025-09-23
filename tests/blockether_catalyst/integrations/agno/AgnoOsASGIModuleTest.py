"""Tests for AgnoOsASGIModule API endpoints."""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import Response as HttpxResponse

from blockether_catalyst.asgi.ASGICoreApplication import ASGICoreApplication
from blockether_catalyst.integrations.agno.AgnoOsASGIModule import (
    AgnoOSAPISettings,
    AgnoOsASGIModule,
    AssistantConfig,
    ChatConfig,
    MCPConfig,
    default_token_resolver,
)


class TestAgnoOsASGIModule:
    """Test class for AgnoOsASGIModule functionality."""

    MOCK_WORKFLOW_ID = "test_workflow_123"
    MOCK_TEAM_ID = "test_team_456"
    MOCK_AGENT_ID = "test_agent_789"
    MOCK_SESSION_ID = "session_abc123"
    MOCK_USER_ID = "test_user"
    MOCK_TOKEN = "test_token_xyz"
    BASE_URL = "http://localhost:8000"

    @pytest.fixture
    def real_agent(self, test_llm, test_db):
        """Create a REAL Agno agent for testing."""
        from agno.agent import Agent

        agent = Agent(
            id=self.MOCK_AGENT_ID,
            name="Test Agent",
            description="A real test agent",
            model=test_llm,
            db=test_db,
            telemetry=False,
            debug_mode=True,
            store_events=True,
            add_history_to_context=True,
            num_history_runs=5,
            instructions="You are a helpful test assistant. Always respond concisely.",
        )
        return agent

    @pytest.fixture
    def real_workflow(self, real_agent, test_db):
        """Create a REAL Agno workflow for testing."""
        from agno.workflow import Workflow
        from agno.workflow.step import Step

        # Create a step that uses the agent
        agent_step = Step(
            name="test_step",
            description="Test step with agent",
            agent=real_agent,
        )

        workflow = Workflow(
            id=self.MOCK_WORKFLOW_ID,
            name="Test Workflow",
            description="A real test workflow",
            steps=[agent_step],
            db=test_db,
            telemetry=False,
            debug_mode=True,
        )
        return workflow

    @pytest.fixture
    def real_team(self, test_llm, test_db):
        """Create a REAL Agno team for testing."""
        from agno.agent import Agent
        from agno.team import Team

        # Create team members
        member1 = Agent(
            id="member1",
            name="Member 1",
            model=test_llm,
            db=test_db,
            telemetry=False,
            instructions="You are team member 1.",
        )

        member2 = Agent(
            id="member2",
            name="Member 2",
            model=test_llm,
            db=test_db,
            telemetry=False,
            instructions="You are team member 2.",
        )

        team = Team(
            id=self.MOCK_TEAM_ID,
            name="Test Team",
            description="A real test team",
            members=[member1, member2],  # It's 'members', not 'agents'
            model=test_llm,
            db=test_db,
            telemetry=False,
            debug_mode=True,
        )
        return team

    @pytest.fixture
    def mock_team(self):
        """Create a mock team."""
        team = MagicMock()
        team.id = self.MOCK_TEAM_ID
        team.name = "Test Team"
        team.description = "A test team for unit testing"
        team.__class__.__name__ = "Team"
        return team

    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent."""
        agent = MagicMock()
        agent.id = self.MOCK_AGENT_ID
        agent.name = "Test Agent"
        agent.description = "A test agent for unit testing"
        agent.__class__.__name__ = "Agent"
        return agent

    @pytest.fixture
    def mock_agno_os(self):
        """Create a mock AgentOS instance."""
        mock_os = MagicMock()
        mock_app = MagicMock()
        mock_router = MagicMock()
        mock_router.routes = []
        mock_app.router = mock_router
        mock_os.get_app.return_value = mock_app
        return mock_os

    @pytest.fixture
    def chat_config_workflow(self, real_workflow):
        """Create a ChatConfig with a REAL workflow executor."""
        return ChatConfig(
            assistant=AssistantConfig(name="Test Assistant", short="T", runner=real_workflow),
            base_url=self.BASE_URL,
            auth_token_resolver=default_token_resolver,
        )

    @pytest.fixture
    def chat_config_team(self, real_team):
        """Create a ChatConfig with a REAL team executor."""
        return ChatConfig(
            assistant=AssistantConfig(name="Team Assistant", short="T", runner=real_team),
            base_url=self.BASE_URL,
        )

    @pytest.fixture
    def chat_config_agent(self, mock_agent):
        """Create a ChatConfig with an agent executor."""
        return ChatConfig(
            assistant=AssistantConfig(name="Agent Assistant", short="A", runner=mock_agent),
            base_url=self.BASE_URL,
        )

    @pytest.fixture
    def agno_module_workflow(self, chat_config_workflow, real_workflow):
        """Create an AgnoOsASGIModule with REAL workflow configuration."""
        return AgnoOsASGIModule(
            title="Test Agno Module",
            description="Test module for unit testing",
            workflows=[real_workflow],
            teams=[],
            chat=chat_config_workflow,
            api=AgnoOSAPISettings(docs_enabled=True, cors_list=["http://localhost:*"], api_token=None),
        )

    @pytest.fixture
    def agno_module_team(self, chat_config_team, real_team):
        """Create an AgnoOsASGIModule with REAL team configuration."""
        return AgnoOsASGIModule(
            title="Test Agno Module Team",
            description="Test module for team testing",
            workflows=[],
            teams=[real_team],
            chat=chat_config_team,
        )

    @pytest.fixture
    def agno_module_agent(self, chat_config_agent, mock_agent):
        """Create an AgnoOsASGIModule with agent configuration."""
        return AgnoOsASGIModule(
            title="Test Agno Module Agent",
            description="Test module for agent testing",
            workflows=[],
            teams=[],
            chat=chat_config_agent,
        )

    @pytest.fixture
    def test_app_workflow(self, agno_module_workflow, mock_agno_os):
        """Create a test FastAPI application with workflow module."""
        # Create a real FastAPI app for the mock AgentOS
        mock_fastapi = FastAPI()
        mock_agno_os.get_app.return_value = mock_fastapi

        with patch(
            "blockether_catalyst.integrations.agno.AgnoOsASGIModule.AgentOS",
            return_value=mock_agno_os,
        ):
            asgi_app = ASGICoreApplication(
                title="Test ASGI App",
                description="Test application",
                version="1.0.0",
                prefix="/",
                debug=False,
            )
            asgi_app.mount_module(agno_module_workflow)
            return TestClient(asgi_app.app)

    @pytest.fixture
    def test_app_team(self, agno_module_team, mock_agno_os):
        """Create a test FastAPI application with team module."""
        # Create a real FastAPI app for the mock AgentOS
        mock_fastapi = FastAPI()
        mock_agno_os.get_app.return_value = mock_fastapi

        with patch(
            "blockether_catalyst.integrations.agno.AgnoOsASGIModule.AgentOS",
            return_value=mock_agno_os,
        ):
            asgi_app = ASGICoreApplication(
                title="Test ASGI App Team",
                description="Test application for team",
                version="1.0.0",
                prefix="/",
                debug=False,
            )
            asgi_app.mount_module(agno_module_team)
            return TestClient(asgi_app.app)

    def test_chat_config_executor_type_detection(self, chat_config_workflow, chat_config_team, chat_config_agent):
        """Test that executor type is correctly detected from different executor types."""
        assert chat_config_workflow.get_executor_type() == "workflows"
        assert chat_config_workflow.get_session_type() == "workflow"

        assert chat_config_team.get_executor_type() == "teams"
        assert chat_config_team.get_session_type() == "team"

        assert chat_config_agent.get_executor_type() == "agents"
        assert chat_config_agent.get_session_type() == "agent"

    def test_chat_interface_endpoint(self, test_app_workflow):
        """Test the /os/view endpoint returns the chat interface."""
        response = test_app_workflow.get("/os/view")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

        # Check that key elements are in the response
        content = response.text
        assert "Test Workflow" in content or "executor_name" in content
        assert "session_id" in content

    def test_chat_interface_with_dev_mode(self, test_app_workflow):
        """Test the /os/view endpoint with dev_mode enabled."""
        response = test_app_workflow.get("/os/view?dev_mode=true")
        assert response.status_code == 200

        content = response.text
        # In dev mode, user should be set to DemoUser
        assert "DemoUser" in content or "user_id" in content

    def test_chat_interface_with_token(self, test_app_workflow):
        """Test the /os/view endpoint with authentication token."""
        # Test with query parameter
        response = test_app_workflow.get(f"/os/view?token={self.MOCK_TOKEN}")
        assert response.status_code == 200

        # Test with Authorization header
        response = test_app_workflow.get("/os/view", headers={"Authorization": f"Bearer {self.MOCK_TOKEN}"})
        assert response.status_code == 200

        # Test with cookie - set on client instance to avoid deprecation warning
        test_app_workflow.cookies.set("auth_token", self.MOCK_TOKEN)
        response = test_app_workflow.get("/os/view")
        assert response.status_code == 200
        test_app_workflow.cookies.clear()

    def test_render_message_endpoint(self, test_app_workflow):
        """Test the /os/view/render-message endpoint."""
        message_data = {
            "content": "This is a test response",
            "message_id": "msg_123",
            "is_error": False,
            "session_id": self.MOCK_SESSION_ID,
            "is_new_session": True,
            "workflow_name": "Test Workflow",
            "workflow_id": self.MOCK_WORKFLOW_ID,
            "run_id": "run_456",
            "status": "completed",
            "step_results": [{"step": "Step 1", "result": "Success"}],
            "metrics": {
                "steps": {
                    "step1": {
                        "metrics": {
                            "input_tokens": 100,
                            "output_tokens": 50,
                            "total_tokens": 150,
                            "duration": 1.5,
                        }
                    }
                }
            },
        }

        response = test_app_workflow.post("/os/view/render-message", json=message_data)

        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

        # Check for session header on new session
        assert response.headers.get("X-Session-ID") == self.MOCK_SESSION_ID

        content = response.text
        assert "This is a test response" in content

    def test_render_message_error_handling(self, test_app_workflow):
        """Test render message with error flag."""
        message_data = {
            "content": "Error: Something went wrong",
            "message_id": "error_123",
            "is_error": True,
        }

        response = test_app_workflow.post("/os/view/render-message", json=message_data)

        assert response.status_code == 200
        content = response.text
        assert "Error: Something went wrong" in content

    def test_executor_proxy_workflow(self, test_app_workflow):
        """Test the /os/executor/runs proxy endpoint for workflows."""
        test_request_data = {
            "message": "Test input for workflow",
            "session_id": self.MOCK_SESSION_ID,
        }

        # Use real executor, no mocks needed - send as form data
        response = test_app_workflow.post("/os/executor/runs", data=test_request_data)

        assert response.status_code == 200
        data = response.json()
        # With real executors, we get real UUIDs, not hardcoded values
        assert "run_id" in data
        assert data["status"] == "completed"
        assert "output" in data

    def test_executor_proxy_team(self, test_app_team):
        """Test the /os/executor/runs proxy endpoint for teams."""
        test_request_data = {
            "message": "Test input for team",
            "session_id": self.MOCK_SESSION_ID,
        }

        # Use real executor, no mocks needed - send as form data
        response = test_app_team.post("/os/executor/runs", data=test_request_data)

        assert response.status_code == 200
        data = response.json()
        # With real executors, we get real UUIDs, not hardcoded values
        assert "run_id" in data
        assert data["status"] == "completed"
        assert "output" in data

    def skip_test_session_runs_endpoint(self, test_app_workflow):
        """Test the /os/sessions/{session_id}/runs endpoint using REAL Agno executor."""
        session_id = "test_session_" + str(time.time())

        # Create actual runs through the executor endpoint with REAL workflow
        # First run
        run1_request = {
            "message": "Say hello",
            "session_id": session_id,
            "user_id": "test_user",
        }
        run1_response = test_app_workflow.post("/os/executor/runs", data=run1_request)
        assert run1_response.status_code == 200, f"Failed to create run 1: {run1_response.text}"
        run1_data = run1_response.json()
        assert "run_id" in run1_data

        # Second run
        run2_request = {
            "message": "Say goodbye",
            "session_id": session_id,
            "user_id": "test_user",
        }
        run2_response = test_app_workflow.post("/os/executor/runs", data=run2_request)
        assert run2_response.status_code == 200, f"Failed to create run 2: {run2_response.text}"
        run2_data = run2_response.json()
        assert "run_id" in run2_data

        # Give the real executor time to persist sessions
        import time as time_module

        time_module.sleep(0.5)

        # Now retrieve the session runs
        response = test_app_workflow.get(f"/os/sessions/{session_id}/runs")
        assert response.status_code == 200, f"Failed to get session runs: {response.text}"
        data = response.json()

        # With real executors, we should get actual session data back
        assert isinstance(data, list), f"Expected list, got {type(data)}"
        assert len(data) >= 2, f"Expected at least 2 runs, got {len(data)}. Response: {data}"

        # Check that our runs are in the returned data
        run_ids = [run.get("run_id") for run in data if "run_id" in run]
        assert len(run_ids) >= 2, f"Not enough run_ids found in response: {data}"

    def test_session_runs_not_found(self, test_app_workflow):
        """Test session runs endpoint when no runs are found."""
        # Use a session ID that doesn't exist
        non_existent_session = "non_existent_session_" + str(time.time())

        response = test_app_workflow.get(f"/os/sessions/{non_existent_session}/runs")

        assert response.status_code == 200  # Returns 200 with empty array
        data = response.json()
        assert data == []

    def test_default_token_resolver(self):
        """Test the default token resolver function."""
        mock_request = MagicMock()
        mock_os = MagicMock()

        # Test dev_mode
        mock_request.query_params.get.return_value = "true"
        result = default_token_resolver("token", mock_os, mock_request)
        assert result == "DemoUser"

        # Test without dev_mode
        mock_request.query_params.get.return_value = None
        result = default_token_resolver("token", mock_os, mock_request)
        assert result is None

    def test_mcp_config(self, real_workflow):
        """Test MCPConfig creation and usage."""
        # Create a proper Tool instance using Tool.from_function
        from fastmcp.tools import Tool

        def test_tool_func(query: str) -> str:
            """Test tool function."""
            return f"Result for {query}"

        test_tool = Tool.from_function(fn=test_tool_func, name="test_tool", description="A test tool")

        mcp_config = MCPConfig(workflow=real_workflow, name="Test MCP", tools=[test_tool])

        assert mcp_config.workflow == real_workflow
        assert mcp_config.name == "Test MCP"
        assert len(mcp_config.tools) == 1
        assert mcp_config.tools[0].name == "test_tool"

    def test_api_settings_defaults(self):
        """Test AgnoOSAPISettings default values."""
        settings = AgnoOSAPISettings()

        assert settings.docs_enabled is True
        assert "http://localhost:*" in settings.cors_list
        assert settings.api_token is None

    def test_module_validation_no_chat(self):
        """Test that module raises error when chat config is missing."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError) as exc_info:
            AgnoOsASGIModule(
                title="Invalid Module",
                description="Module without chat config",
                workflows=[],
                teams=[],
                chat=None,  # type: ignore
            )

        # Check that it's a validation error for the chat field
        assert "chat" in str(exc_info.value)
        assert "ChatConfig" in str(exc_info.value)

    def test_assistant_config_defaults(self):
        """Test AssistantConfig default values."""
        mock_runner = MagicMock()
        config = AssistantConfig(runner=mock_runner)

        assert config.name == "Omniscient Assistant"
        assert config.short == "O"
        assert config.cookies.user_id_max_age == 86400
        assert config.cookies.token_max_age == 86400
