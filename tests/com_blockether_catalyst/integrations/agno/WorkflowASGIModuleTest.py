"""Integration tests for WorkflowASGIModule."""

from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest
from agno.run.response import RunResponse

from com_blockether_catalyst.integrations.agno import (
    AgnoWorkflowAPIModule,
    WorkflowApiASGIModule,
    WorkflowConfig,
)


@pytest.mark.integration
class TestWorkflowASGIModule:
    """Test the Agno workflow API."""

    def test_api_initialization(self) -> None:
        """Test that the API can be initialized with proper settings."""

        # Mock the workflow entrypoint
        async def mock_workflow_on_run_callback(
            workflow: Any, request_context: Any, message: Any, **kwargs: Any
        ) -> RunResponse:
            return RunResponse(
                content="Test response",
                workflow_id="test-workflow",
                run_id="test-run-id",
            )

        # Create the API instance
        api = WorkflowApiASGIModule(
            workflow=WorkflowConfig(
                description="Test workflow",
                run_callback=mock_workflow_on_run_callback,
            ),
            api=AgnoWorkflowAPIModule(
                name="Test Playground",
                app_id="test-app",
            ),
        )

        # Verify API is initialized
        assert api.workflow_instance is not None
        assert api.workflow_configuration is not None
        assert api.prefix == "/v1"

    def test_workflow_settings_propagation(self) -> None:
        """Test that workflow settings are properly propagated."""

        # Mock the workflow entrypoint
        async def mock_workflow_on_run_callback(
            workflow: Any, request_context: Any, message: Any, **kwargs: Any
        ) -> RunResponse:
            return RunResponse(
                content="Test response",
                workflow_id="test-workflow",
                run_id="test-run-id",
            )

        # Create custom settings
        workflow_config = WorkflowConfig(
            description="Custom description",
            debug_mode=True,
            monitoring=True,
            telemetry=True,
            app_id="custom-app",
        )

        # Update workflow config with run_callback
        workflow_config.run_callback = mock_workflow_on_run_callback

        # Create the API instance
        api = WorkflowApiASGIModule(
            workflow=workflow_config,
        )

        # Verify workflow was created with settings
        assert api.workflow_instance is not None
        # The workflow class should have the description from settings
        assert hasattr(api.workflow_instance.__class__, "description")

    def test_api_settings_propagation(self) -> None:
        """Test that API settings are properly propagated."""

        # Mock the workflow entrypoint
        async def mock_workflow_on_run_callback(
            workflow: Any, request_context: Any, message: Any, **kwargs: Any
        ) -> RunResponse:
            return RunResponse(
                content="Test response",
                workflow_id="test-workflow",
                run_id="test-run-id",
            )

        # Create custom API settings
        api_config = AgnoWorkflowAPIModule(
            name="Custom Playground",
            app_id="custom-playground-app",
            description="Custom playground description",
            monitoring=True,
            prefix="/api/v2",
        )

        # Create the API instance
        api = WorkflowApiASGIModule(
            workflow=WorkflowConfig(
                run_callback=mock_workflow_on_run_callback,
            ),
            api=api_config,
        )

        # Verify API settings
        assert api.prefix == "/api/v2"
        # Note: CORS should be configured at ASGICore level, not module level
        # So we no longer check for CORS middleware here

    def test_route_interception(self) -> None:
        """Test that the workflow run route is properly intercepted."""

        # Mock the workflow entrypoint
        async def mock_workflow_on_run_callback(
            workflow: Any, request_context: Any, message: Any, **kwargs: Any
        ) -> RunResponse:
            return RunResponse(
                content="Test response",
                workflow_id="test-workflow",
                run_id="test-run-id",
            )

        # Create the API instance
        api = WorkflowApiASGIModule(
            workflow=WorkflowConfig(
                run_callback=mock_workflow_on_run_callback,
            ),
        )

        # Check that the intercepted route exists (including in mounted apps)
        routes = api.get_all_routes()
        assert any("/workflow/run" in route for route in routes)

    def test_api_configuration(self) -> None:
        """Test that API configuration is properly stored."""

        # Mock the workflow entrypoint
        async def mock_workflow_on_run_callback(
            workflow: Any, request_context: Any, message: Any, **kwargs: Any
        ) -> RunResponse:
            return RunResponse(
                content="Test response",
                workflow_id="test-workflow",
                run_id="test-run-id",
            )

        # Create the API instance with custom configuration
        api = WorkflowApiASGIModule(
            workflow=WorkflowConfig(
                run_callback=mock_workflow_on_run_callback,
            ),
            api=AgnoWorkflowAPIModule(
                name="Custom API",
                app_id="custom-app",
                description="Custom Description",
                monitoring=True,
                prefix="/api",
            ),
        )

        # Verify API configuration is stored
        assert api.api_configuration.name == "Custom API"
        assert api.api_configuration.app_id == "custom-app"
        assert api.api_configuration.description == "Custom Description"
        assert api.api_configuration.monitoring is True
        assert api.prefix == "/api"

    def test_custom_endpoints_registration(self) -> None:
        """Test that custom endpoints can be registered."""
        from fastapi import APIRouter

        # Mock the workflow entrypoint
        async def mock_workflow_on_run_callback(
            workflow: Any, request_context: Any, message: Any, **kwargs: Any
        ) -> RunResponse:
            return RunResponse(
                content="Test response",
                workflow_id="test-workflow",
                run_id="test-run-id",
            )

        # Track if custom endpoint registration was called
        custom_endpoint_called = False

        def register_custom_endpoints(router: Any, workflow: Any) -> None:
            nonlocal custom_endpoint_called
            custom_endpoint_called = True

            @router.get("/custom/endpoint")
            async def custom_endpoint() -> Dict[str, str]:
                return {"message": "Custom endpoint"}

        # Create the API instance with custom endpoint registration
        api = WorkflowApiASGIModule(
            workflow=WorkflowConfig(
                run_callback=mock_workflow_on_run_callback,
            ),
            api=AgnoWorkflowAPIModule(
                custom_endpoints=register_custom_endpoints,
            ),
        )

        # Setup routes by simulating the mount process
        router = APIRouter(prefix="/v1")
        api.setup_routes(router)

        # Verify custom endpoint registration was called
        assert custom_endpoint_called

        # Note: Custom endpoints are registered directly on the router,
        # so we can't check them in get_all_routes. The fact that
        # custom_endpoint_called is True verifies it was registered.
