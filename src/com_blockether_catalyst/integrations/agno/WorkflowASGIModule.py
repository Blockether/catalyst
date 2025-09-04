"""API for Agno workflows."""

import json
import logging
from typing import Any, Callable, Coroutine, Dict, List, Optional, Type, Union, cast
from uuid import uuid4

from agno.app.playground.schemas import WorkflowRunRequest
from agno.run.response import RunResponse
from agno.workflow import Workflow
from fastapi import APIRouter, HTTPException, Request
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.session import ServerSession
from pydantic import BaseModel, Field

from com_blockether_catalyst.asgi.ASGICoreModule import ASGICoreModule

from .MainWorkflow import MainWorkflow
from .WorkflowTypes import (
    AgnoWorkflowAPIModule,
    MCPToolDefinition,
    RequestContextModel,
    WorkflowConfig,
)

logger = logging.getLogger(__name__)


class SessionInfo(BaseModel):
    """Information about a workflow session."""

    session_id: str = Field(description="Unique session identifier")
    message_count: int = Field(default=0, description="Number of messages in session")
    created_at: str = Field(description="Session creation timestamp")


class MessageRequest(BaseModel):
    """Simplified message request structure."""

    message: str = Field(description="The message to send to the workflow")
    session_id: Optional[str] = Field(default=None, description="Session ID for continuing conversation")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Optional context data to pass to the workflow")


async def setup_workflow_instance(
    workflow: Workflow, session_id: str | None = None, user_id: str | None = None
) -> Workflow:
    new_workflow_instance = workflow.deep_copy(update={"workflow_id": workflow.workflow_id})

    if session_id is not None:
        new_workflow_instance.session_id = session_id

    if user_id is not None:
        new_workflow_instance.user_id = user_id

    # Set mode, debug, workflow_id, session_id, initialize memory
    new_workflow_instance.set_storage_mode()
    new_workflow_instance.set_debug()
    new_workflow_instance.set_monitoring()
    new_workflow_instance.set_workflow_id()
    new_workflow_instance.set_session_id()
    new_workflow_instance.initialize_memory()

    # Ensure memory is not None (for type checker)
    if new_workflow_instance.memory is None:
        raise ValueError("memory is not initialized")

    if not new_workflow_instance.storage:
        raise ValueError("storage is not initialized")

    return new_workflow_instance


async def create_workflow_run(workflow: Workflow, body: WorkflowRunRequest) -> RunResponse:
    if body.session_id is not None:
        logger.debug(f"Continuing session: {body.session_id}")
    else:
        logger.debug("Creating new session: ")

    workflow_instance = await setup_workflow_instance(workflow, session_id=body.session_id, user_id=body.user_id)

    # Return based on the response type
    try:
        result = await workflow_instance.arun(**body.input)
        return cast(RunResponse, result)
    except Exception as e:
        # Handle unexpected runtime errors
        raise HTTPException(status_code=500, detail=f"Error running workflow: {str(e)}")


class MCPServer:
    """MCP server wrapper for Agno workflows.

    Provides a Model Context Protocol interface to interact with Agno workflows,
    enabling LLM tools to send messages through a standardized protocol.

    Memory mode:
    - 'ephemeral': Uses InMemoryStorage for temporary sessions (default)
    """

    def __init__(
        self,
        server_name: str,
        workflow: Workflow,
        custom_tools: Optional[Dict[str, Union[Callable, MCPToolDefinition]]] = None,
    ):
        """Initialize the MCP server.

        Args:
            server_name: Name for the MCP server
            workflow: The Agno workflow instance to interact with
            custom_tools: Optional custom tools to register
        """
        self.workflow = workflow
        self.memory_mode = "ephemeral"  # Always ephemeral for simplicity

        # Store custom tools for stateless handler access
        self._custom_tools = custom_tools or {}

        # Create MCP server with stateless HTTP for ASGI integration
        self.mcp = FastMCP(name=server_name, stateless_http=True)

        # Register default MCP tools
        self._register_default_tools()

        # Register custom tools if provided
        if custom_tools:
            self._register_custom_tools(custom_tools)

    async def _setup_ephemeral_workflow(self) -> Workflow:
        """Setup workflow with ephemeral memory for a session.

        Args:
            session_id: Session identifier
            user_id: Optional user identifier

        Returns:
            Workflow instance with ephemeral memory
        """
        from agno.storage.in_memory import InMemoryStorage

        # Create a copy of the workflow
        workflow_instance = self.workflow.deep_copy(update={"workflow_id": self.workflow.workflow_id})

        # Set the ephemeral storage
        workflow_instance.storage = InMemoryStorage(mode="workflow")

        return workflow_instance

    async def _create_workflow_run(
        self, session_id: str, input_data: Dict[str, Any], user_id: Optional[str] = None
    ) -> RunResponse:
        """Create a workflow run using the workflow instance.

        Args:
            session_id: Session identifier
            input_data: Input data for the workflow
            user_id: Optional user identifier

        Returns:
            RunResponse from the workflow
        """
        if self.memory_mode == "ephemeral":
            # Use ephemeral memory for this session
            workflow_instance = await self._setup_ephemeral_workflow()
        else:
            # Use inherited memory from the workflow
            workflow_instance = await setup_workflow_instance(self.workflow, session_id=session_id, user_id=user_id)

        # Create and execute workflow run request
        request = WorkflowRunRequest(input=input_data, session_id=session_id, user_id=user_id)

        # Run the workflow directly
        if request.input is None:
            request.input = {}

        if "context" not in request.input:
            request.input["context"] = {
                "session_id": session_id,
                "user_id": user_id,
            }

        request.input["context"]["is_ephemeral"] = self.memory_mode == "ephemeral"

        result = await workflow_instance.arun(**request.input)
        return cast(RunResponse, result)

    def _register_default_tools(self) -> None:
        """Register default MCP tools for workflow interaction."""
        from datetime import datetime, timezone

        if not self.memory_mode == "ephemeral":

            @self.mcp.tool(description="Start a new conversation session with the workflow")
            async def new_session() -> SessionInfo:
                """Start a new workflow session.

                Creates a fresh session for conversation with the workflow.
                Returns session information including the session ID.
                """
                session_id = str(uuid4())
                session_info = SessionInfo(
                    session_id=session_id,
                    message_count=0,
                    created_at=datetime.now(timezone.utc).isoformat(),
                )
                return session_info

        @self.mcp.tool(description="Send a message to the workflow")
        async def send_message(request: MessageRequest) -> str:
            """Send a message to the workflow within a session context.

            Args:
                request: Message request containing the message and optional session_id

            Returns:
                Formatted response from the workflow
            """
            # Use provided session_id or create new one
            session_id = request.session_id
            if not session_id:
                session_info = await new_session()
                session_id = session_info.session_id

            # Create workflow run
            input_data: Dict[str, Any] = {
                "message": request.message,
                "context": request.context or {},
            }

            run_response = await self._create_workflow_run(session_id, input_data)

            # Format and return response
            return str(run_response.content)

    def _register_custom_tools(self, custom_tools: Dict[str, Union[Callable, MCPToolDefinition]]) -> None:
        """Register custom MCP tools.

        Args:
            custom_tools: Dictionary mapping tool names to callables or MCPToolDefinition objects
        """
        for name, tool_def in custom_tools.items():
            if isinstance(tool_def, MCPToolDefinition):
                # Extract metadata from MCPToolDefinition
                self.mcp.tool(
                    name=name,
                    description=tool_def.description,
                    title=tool_def.title,
                )(tool_def.function)
            else:
                # Simple callable - just use the name
                self.mcp.tool(name=name)(tool_def)

    def get_asgi_app(self) -> Any:
        """Get the ASGI app for mounting in FastAPI.

        Returns:
            ASGI application that can be mounted in FastAPI
        """
        return self.mcp.sse_app()


class WorkflowApiASGIModule(ASGICoreModule):
    """Minimal API for Agno workflows.

    Provides REST API endpoints for interacting with a single Agno workflow.
    Features:
    - REST API endpoints for workflow execution
    - Session management
    - Request context injection
    - Optional MCP server integration

    Example:
        ```python
        from com_blockether_catalyst.asgi import ASGICoreApplication

        # Create the API module
        api = WorkflowApiASGIModule(
            workflow=WorkflowConfig(
                run_callback=my_workflow_function,
                description="My workflow"
            ),
            api=AgnoWorkflowAPIModule(
                name="My API"
            )
        )

        # Create the application and mount the module
        app = ASGICoreApplication()
        app.mount_module(api)
        ```
    """

    # Add Pydantic fields for this module - using private attributes
    mcp_server_instance: Optional[MCPServer] = Field(default=None, exclude=True, description="MCP server instance")
    api_configuration: AgnoWorkflowAPIModule = Field(
        default_factory=AgnoWorkflowAPIModule,
        exclude=True,
        description="API configuration",
    )
    workflow_configuration: Optional[WorkflowConfig] = Field(
        default=None, exclude=True, description="Workflow configuration"
    )
    workflow_instance: Optional[Workflow] = Field(default=None, exclude=True, description="Workflow instance")
    workflow_sessions: Dict[str, Workflow] = Field(
        default_factory=dict, exclude=True, description="Active workflow sessions"
    )

    def __init__(
        self,
        workflow: WorkflowConfig,
        api: Optional[AgnoWorkflowAPIModule] = None,
        **kwargs: Any,
    ):
        """
        Initialize the API with a more idiomatic interface.

        Args:
            workflow: Workflow configuration including run_callback
            api: API configuration including middlewares and custom endpoints
            **kwargs: Additional arguments passed to parent
        """
        # Use provided settings or defaults
        api = api or AgnoWorkflowAPIModule()

        # Get the template directory for this module
        # Use resolve() to get absolute path for library compatibility
        from pathlib import Path

        module_dir = Path(__file__).resolve().parent
        template_dir = module_dir / "templates"

        # Prepare initialization data
        init_data = {
            "prefix": api.prefix,
            "title": api.name,
            "description": api.description,
            "template_dirs": [template_dir] if template_dir.exists() else [],
            "htmx_enabled": True,
            "api_configuration": api,
            "workflow_configuration": workflow,
            **kwargs,
        }

        # Initialize base class with all fields
        super().__init__(**init_data)

        # Initialize private attributes for MCP mounting
        self._mcp_app: Optional[Any] = None
        self._mcp_mount_path: Optional[str] = None

    def model_post_init(self, __context: Any) -> None:
        """Post-initialization to set up workflow and MCP server."""
        # Call parent post-init
        super().model_post_init(__context)

        # Validate workflow configuration exists
        if not self.workflow_configuration:
            raise ValueError("workflow_configuration is required")

        # Validate workflow has run_callback
        if not self.workflow_configuration.run_callback:
            raise ValueError("workflow.run_callback is required")

        # Create and configure the workflow instance
        self.workflow_instance = self._create_workflow(
            self.workflow_configuration.run_callback,
            self.workflow_configuration.model_dump(),
        )

        # Initialize MCP server if mcp config is provided
        if self.api_configuration.mcp:
            self._initialize_mcp_server(self.api_configuration)

    def _create_workflow(self, run_callback: Callable, workflow_settings: Dict[str, Any]) -> Workflow:
        """
        Create a workflow instance with injected run method.

        Args:
            run_callback: The run function to inject
            workflow_settings: Settings for workflow initialization

        Returns:
            Configured workflow instance
        """
        # Create workflow instance with workflow_id
        workflow_init_params = workflow_settings.copy()
        # workflow_id is always "mainworkflow" internally
        workflow_init_params["workflow_id"] = "mainworkflow"
        workflow_init_params["name"] = "mainworkflow"

        # Extract description, telemetry, and alias from settings and remove from init_params
        description = workflow_init_params.pop("description", "Main workflow")
        telemetry = workflow_init_params.pop("telemetry", False)
        # Remove alias as it's not used by MainWorkflow
        workflow_init_params.pop("alias", None)

        # Remove run_callback from init_params as it's not a Workflow parameter
        workflow_init_params.pop("run_callback", None)

        # Create MainWorkflow instance with injected callback
        workflow = MainWorkflow(
            run_callback=run_callback,
            description=description,
            telemetry=telemetry,
            **workflow_init_params,
        )

        # Set storage mode to workflow
        if workflow.storage:
            workflow.storage.mode = "workflow"

        return workflow

    def setup_routes(self, router: APIRouter) -> None:
        """Set up minimal workflow routes.

        Args:
            router: The APIRouter instance to add routes to
        """
        # If MCP server is configured, mount its ASGI app
        if self.mcp_server_instance:
            # Get the MCP ASGI app
            mcp_app = self.mcp_server_instance.get_asgi_app()

            # Mount the MCP app at /mcp endpoint
            # Note: We need to use the parent app's mount method, not the router
            # This will be handled by registering a mount callback
            self._mcp_app = mcp_app
            self._mcp_mount_path = "/mcp"

        # Add workflow info endpoint - single workflow, no ID needed
        @router.get("/workflow")
        async def get_workflow() -> Dict[str, Any]:
            """Get workflow information."""
            if not self.workflow_instance:
                raise HTTPException(status_code=500, detail="Workflow not initialized")
            return {
                "workflow_id": self.workflow_instance.workflow_id,
                "name": (
                    self.workflow_configuration.alias if self.workflow_configuration else "Workflow"
                ),  # Use alias if provided
                "description": self.workflow_instance.description,
                "storage": (
                    self.workflow_instance.storage.__class__.__name__ if self.workflow_instance.storage else None
                ),
            }

        # Add workflow run endpoint - single workflow, no ID needed
        @router.post("/workflow/run", response_model=None)
        async def handle_workflow_run(request: Request) -> RunResponse:
            """Execute workflow with request context injection."""
            # Internal workflow_id is always "mainworkflow", no need to check

            # Parse request body
            body = await request.body()
            try:
                data = json.loads(body)
            except json.JSONDecodeError as e:
                raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(e)}")

            # Extract input and session info
            input_data = data.get("input", {})
            session_id = data.get("session_id")
            user_id = data.get("user_id")

            # Check for stream parameter and reject it
            if data.get("stream", False):
                raise HTTPException(
                    status_code=400,
                    detail="Streaming is disabled. This API only supports non-streaming responses.",
                )

            # Add request context
            request_context = RequestContextModel(
                headers=dict(request.headers),
                url=str(request.url),
                method=request.method,
                client_ip=request.client.host if request.client else None,
                user_agent=request.headers.get("user-agent"),
                path=request.url.path,
                query_params=dict(request.query_params),
                correlation_id=request.headers.get("x-correlation-id"),
            )
            input_data["request_context"] = request_context.model_dump()

            # Create workflow run request
            workflow_run_request = WorkflowRunRequest(input=input_data, session_id=session_id, user_id=user_id)

            # Execute workflow using the module-level function
            if not self.workflow_instance:
                raise HTTPException(status_code=500, detail="Workflow not initialized")
            result = await create_workflow_run(self.workflow_instance, workflow_run_request)
            return result

        # Add custom endpoints if provided
        if self.api_configuration.custom_endpoints and self.workflow_instance:
            self.api_configuration.custom_endpoints(router, self.workflow_instance)

        # Add chat interface routes if templates are configured
        if self.templates:
            from datetime import datetime

            from fastapi.responses import HTMLResponse

            @router.get("/chat", response_class=HTMLResponse)
            async def chat_interface(request: Request) -> HTMLResponse:
                """Render the chat interface."""
                # Generate a unique session ID
                session_id = f"session_{uuid4().hex[:8]}"

                # Get the full path prefix including app prefix
                # request.url.path is like /calc/api/chat, we want /calc/api
                full_prefix = (
                    str(request.url.path).rsplit("/chat", 1)[0] if "/chat" in str(request.url.path) else self.prefix
                )

                return cast(
                    HTMLResponse,
                    self.render_template(
                        "chat.html",
                        {
                            "request": request,
                            "workflow_name": (
                                self.workflow_configuration.alias
                                if self.workflow_configuration
                                else "Workflow Assistant"
                            ),
                            "workflow_description": (
                                self.workflow_configuration.description
                                if self.workflow_configuration
                                else "Chat with the workflow"
                            ),
                            "session_id": session_id,
                            "chat_api_prefix": full_prefix,  # Use different name to avoid override
                        },
                        request=request,
                    ),
                )

            @router.post("/chat/send", response_class=HTMLResponse)
            async def send_message(request: Request) -> HTMLResponse:
                """Handle chat message and return ONLY the workflow response.

                The user message and loading indicator are added by JavaScript before this request.
                """
                form_data = await request.form()
                message = cast(str, form_data.get("message", "")).strip()
                session_id = cast(str, form_data.get("session_id", f"session_{uuid4().hex[:8]}"))

                if not message:
                    return HTMLResponse(content="", status_code=204)

                # Get message ID from header (set by JavaScript)
                message_id = request.headers.get("X-Message-ID", f"msg_{uuid4().hex[:8]}")

                # Process the message through the workflow
                try:
                    # Create request context for the workflow
                    request_context = RequestContextModel(
                        headers=dict(request.headers),
                        url=str(request.url),
                        method=request.method,
                        client_ip=request.client.host if request.client else None,
                        user_agent=request.headers.get("user-agent"),
                        path=request.url.path,
                        query_params=dict(request.query_params),
                        correlation_id=request.headers.get("x-correlation-id"),
                    )

                    # Create workflow input with request context
                    workflow_input = {
                        "message": message,
                        "session_id": session_id,
                        "timestamp": datetime.utcnow().isoformat(),
                        "request_context": request_context.model_dump(),
                    }

                    # Create and execute workflow run
                    workflow_run_request = WorkflowRunRequest(
                        input=workflow_input,
                        session_id=session_id,
                    )

                    # Execute workflow
                    if not self.workflow_instance:
                        raise HTTPException(status_code=500, detail="Workflow not initialized")
                    result = await create_workflow_run(self.workflow_instance, workflow_run_request)

                    # Extract response content
                    response_content = result.content if hasattr(result, "content") else str(result)

                except Exception as e:
                    logger.exception("Error processing chat message")
                    response_content = f"Error: {str(e)}"

                    # Return error response with error flag
                    workflow_message_html = self.render_partial(
                        "workflow_message.html",
                        {
                            "response": response_content,
                            "message_id": message_id,
                            "is_error": True,
                        },
                        request=request,
                    )
                    return cast(HTMLResponse, workflow_message_html)

                # Return ONLY workflow response partial which will replace the loading indicator
                workflow_message_html = self.render_partial(
                    "workflow_message.html",
                    {
                        "response": response_content,
                        "message_id": message_id,
                        "is_error": False,
                    },
                    request=request,
                )

                # Return only the workflow response (it will replace the loading indicator)
                return cast(HTMLResponse, workflow_message_html)

    def _initialize_mcp_server(self, settings: AgnoWorkflowAPIModule) -> None:
        """Initialize the MCP server.

        Args:
            settings: API settings containing MCP configuration
        """
        if not settings.mcp:
            # Use defaults if no MCP config provided
            from .WorkflowTypes import MCPConfig

            settings.mcp = MCPConfig()

        # Create MCP server instance with workflow
        if not self.workflow_instance:
            raise ValueError("workflow_instance is not initialized")

        self.mcp_server_instance = MCPServer(
            server_name=settings.mcp.server_name,
            workflow=self.workflow_instance,
            custom_tools=settings.mcp.custom_tools,
        )

        logger.info(f"MCP server initialized with name '{settings.mcp.server_name}'")

    @property
    def workflow(self) -> Optional[Workflow]:
        """
        Get the workflow instance.

        Returns:
            The workflow instance
        """
        return self.workflow_instance

    def get_mcp_mount_info(self) -> Optional[tuple[Any, str]]:
        """
        Get MCP app and mount path if configured.

        Returns:
            Tuple of (mcp_app, mount_path) if MCP is configured, None otherwise
        """
        if self._mcp_app and self._mcp_mount_path:
            return (self._mcp_app, self._mcp_mount_path)
        return None

    def get_all_routes(self) -> List[str]:
        """
        Get all registered routes for this module.

        Returns:
            List of route paths
        """
        routes = [
            f"{self.prefix}/workflow",
            f"{self.prefix}/workflow/run",
        ]

        # Add chat routes if templates are configured
        if self.templates:
            routes.extend(
                [
                    f"{self.prefix}/chat",
                    f"{self.prefix}/chat/send",
                ]
            )

        # Add MCP routes if configured
        if self._mcp_mount_path:
            routes.append(f"{self.prefix}{self._mcp_mount_path}")

        return routes
