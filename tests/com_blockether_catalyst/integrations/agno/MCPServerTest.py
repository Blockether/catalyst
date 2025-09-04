"""Tests for MCPServer integration."""

import json
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from agno.run.response import RunResponse
from agno.workflow import Workflow
from httpx import AsyncClient

from com_blockether_catalyst.integrations.agno.WorkflowASGIModule import (
    MCPServer,
    MessageRequest,
    SessionInfo,
    WorkflowApiASGIModule,
)
from com_blockether_catalyst.integrations.agno.WorkflowTypes import (
    AgnoWorkflowAPIModule,
    MCPConfig,
    WorkflowConfig,
)


class TestMCPServer:
    """Test cases for MCPServer."""

    def test_mcp_server_initialization(self) -> None:
        """Test MCP server initializes correctly."""
        mock_workflow = MagicMock(spec=Workflow)
        server = MCPServer(
            server_name="test-server",
            workflow=mock_workflow,
        )

        # Check basic initialization
        assert server.workflow == mock_workflow
        assert server.memory_mode == "ephemeral"
        assert server.mcp is not None
        assert server.mcp.name == "test-server"

    def test_mcp_server_with_custom_tools(self) -> None:
        """Test MCP server with custom tools."""
        mock_workflow = MagicMock(spec=Workflow)

        # Define a custom tool
        def custom_tool(message: str) -> str:
            return f"Custom: {message}"

        custom_tools = {"custom_tool": custom_tool}

        server = MCPServer(
            server_name="test-server",
            workflow=mock_workflow,
            custom_tools=custom_tools,  # type: ignore[arg-type]
        )

        # Check that server initializes with custom tools
        assert server._custom_tools == custom_tools

    def test_mcp_server_get_asgi_app(self) -> None:
        """Test that MCP server returns an ASGI app."""
        mock_workflow = MagicMock(spec=Workflow)
        server = MCPServer(
            server_name="test-server",
            workflow=mock_workflow,
        )

        # Get the ASGI app
        app = server.get_asgi_app()

        # Check that it returns something (the actual Starlette app)
        assert app is not None

    def test_mcp_server_memory_mode(self) -> None:
        """Test MCP server memory mode."""
        mock_workflow = MagicMock(spec=Workflow)
        server = MCPServer(
            server_name="test-server",
            workflow=mock_workflow,
        )

        # Check that memory mode is always ephemeral
        assert server.memory_mode == "ephemeral"

    @pytest.mark.asyncio
    async def test_new_session_tool(self) -> None:
        """Test that ephemeral mode doesn't have new_session tool."""
        mock_workflow = MagicMock(spec=Workflow)
        server = MCPServer(
            server_name="test-server",
            workflow=mock_workflow,  # Always ephemeral now
        )

        # Get the registered tools
        tools = server.mcp._tool_manager.list_tools()
        tool_names = [t.name for t in tools]

        # In ephemeral mode (only mode now), new_session should not exist
        assert "new_session" not in tool_names
        assert "send_message" in tool_names

    @pytest.mark.asyncio
    async def test_send_message_tool(self) -> None:
        """Test the send_message MCP tool."""
        mock_workflow = MagicMock(spec=Workflow)

        # Mock the workflow's deep_copy method to return itself
        mock_workflow.deep_copy.return_value = mock_workflow
        mock_workflow.workflow_id = "test-workflow"

        # Mock the workflow's arun method
        mock_response = RunResponse(content="Test response", session_id="test-session")
        mock_workflow.arun = AsyncMock(return_value=mock_response)

        server = MCPServer(
            server_name="test-server",
            workflow=mock_workflow,
        )

        # Get the tool and call it
        tools = server.mcp._tool_manager.list_tools()
        send_message_tool = next(t for t in tools if t.name == "send_message")

        result = await send_message_tool.fn(MessageRequest(message="Test message", session_id="test-session"))

        assert result == "Test response"

    @pytest.mark.asyncio
    async def test_playground_api_with_mcp_server(self) -> None:
        """Test WorkflowApiASGIModule with MCP server enabled."""

        # Mock workflow callback
        async def mock_workflow(**kwargs: Any) -> RunResponse:
            return RunResponse(content="Test response", session_id="test-session")

        # Create API with MCP server enabled
        api = WorkflowApiASGIModule(
            workflow=WorkflowConfig(alias="TestWorkflow", run_callback=mock_workflow),
            api=AgnoWorkflowAPIModule(name="Test Playground", mcp=MCPConfig(server_name="test-mcp-server")),
        )

        # Verify MCP server was created and mounted
        assert api.mcp_server_instance is not None
        assert isinstance(api.mcp_server_instance, MCPServer)

        # Simulate mounting by calling setup_routes
        from fastapi import APIRouter

        router = APIRouter()
        api.setup_routes(router)

        # Check that MCP mounting info is available after setup
        mcp_info = api.get_mcp_mount_info()
        assert mcp_info is not None
        assert mcp_info[1] == "/mcp"

    @pytest.mark.asyncio
    async def test_playground_api_with_custom_formatter(self) -> None:
        """Test WorkflowApiASGIModule with custom MCP formatter."""

        # Mock workflow callback
        async def mock_workflow(**kwargs: Any) -> RunResponse:
            return RunResponse(content={"data": "test"}, session_id="test-session")

        # Custom tool with simple signature
        def custom_tool(message: str) -> str:
            return f"FORMATTED: {message}"

        # Create API with custom tool
        api = WorkflowApiASGIModule(
            workflow=WorkflowConfig(alias="TestWorkflow", run_callback=mock_workflow),
            api=AgnoWorkflowAPIModule(
                name="Test Playground",
                mcp=MCPConfig(
                    server_name="test-mcp-server",
                    custom_tools={"format_test": custom_tool},  # Add as custom tool
                ),
            ),
        )

        # Verify MCP server was created with custom tools
        assert api.mcp_server_instance is not None
        # Check that the MCP server was initialized with custom tools
        assert api.mcp_server_instance._custom_tools is not None

    def test_playground_api_without_mcp_server(self) -> None:
        """Test WorkflowApiASGIModule without MCP server."""

        # Mock workflow callback
        async def mock_workflow(**kwargs: Any) -> RunResponse:
            return RunResponse(content="Test response", session_id="test-session")

        # Create API without MCP server
        api = WorkflowApiASGIModule(
            workflow=WorkflowConfig(alias="TestWorkflow", run_callback=mock_workflow),
            api=AgnoWorkflowAPIModule(
                name="Test Playground",
                mcp=None,  # No MCP configuration - MCP disabled when None
            ),
        )

        # Verify MCP server was not created
        assert api.mcp_server_instance is None

    def test_api_with_mcp_memory_mode(self) -> None:
        """Test WorkflowApiASGIModule creates MCPServer in ephemeral mode."""

        # Mock workflow callback
        async def mock_workflow(**kwargs: Any) -> RunResponse:
            return RunResponse(content="Test", session_id="test")

        # Create API with MCP server
        api = WorkflowApiASGIModule(
            workflow=WorkflowConfig(alias="TestWorkflow", run_callback=mock_workflow),
            api=AgnoWorkflowAPIModule(name="Test API", mcp=MCPConfig()),
        )

        # Verify MCP server was created with ephemeral memory mode (always)
        assert api.mcp_server_instance is not None
        assert api.mcp_server_instance.memory_mode == "ephemeral"

    def test_ephemeral_mode_tools(self) -> None:
        """Test that ephemeral mode only has send_message tool."""
        mock_workflow = MagicMock(spec=Workflow)

        server = MCPServer(
            server_name="test-server",
            workflow=mock_workflow,
        )

        # Get all tools
        tools = server.mcp._tool_manager.list_tools()
        tool_names = [t.name for t in tools]

        # In ephemeral mode, only send_message should exist
        assert "send_message" in tool_names
        assert "new_session" not in tool_names
        assert "clear_session" not in tool_names
        assert "list_sessions" not in tool_names
