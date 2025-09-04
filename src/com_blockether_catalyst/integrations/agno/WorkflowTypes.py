"""Pydantic models for Agno workflow configuration."""

from typing import (
    Any,
    Callable,
    Coroutine,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Type,
    TypeVar,
    Union,
)

from agno.run.response import RunResponse, RunResponseEvent
from agno.storage.base import Storage
from agno.workflow import Workflow
from fastapi import APIRouter, FastAPI
from pydantic import BaseModel, ConfigDict, Field

# Type variable for message content
T = TypeVar("T")

# Type alias for the workflow run callback callable that returns RunResponse (no streaming)
# Takes workflow instance and **kwargs containing message and request_context
OnRunCallable = Callable[..., Coroutine[Any, Any, RunResponse]]


class RequestContextModel(BaseModel):
    """HTTP request context for workflow execution.

    Captures essential HTTP request information that gets automatically
    injected into workflow inputs. This enables workflows to access
    request metadata for logging, routing, or processing decisions.

    Allows arbitrary additional fields to be passed through.

    Core attributes:
        headers: Complete HTTP request headers
        url: Full request URL including protocol and query string
        method: HTTP method (GET, POST, PUT, DELETE, etc.)
        client_ip: Client's IP address if available
        user_agent: Browser/client identification string
        path: URL path without domain or query string
        query_params: Parsed query string parameters
        correlation_id: Tracking ID for distributed tracing
    """

    model_config = ConfigDict(extra="allow")  # Allow arbitrary additional fields

    headers: Dict[str, str] = Field(description="HTTP request headers")
    url: str = Field(description="Full request URL")
    method: str = Field(description="HTTP method (GET, POST, etc.)")
    client_ip: Optional[str] = Field(default=None, description="Client IP address")
    user_agent: Optional[str] = Field(default=None, description="User-Agent header value")
    path: str = Field(description="Request path")
    query_params: Dict[str, str] = Field(default_factory=dict, description="Query parameters")
    correlation_id: Optional[str] = Field(default=None, description="Correlation ID from headers")


class WorkflowInputModel(BaseModel):
    """Base model for Agno workflow inputs.

    Provides a standard structure for workflow inputs with a required
    message field. Can be extended with additional fields for specific
    workflow requirements.

    The model allows extra fields (extra="allow") to support flexible
    workflow designs without strict schema enforcement.

    Attributes:
        message: Primary input message/query for the workflow to process
    """

    model_config = ConfigDict(extra="allow")  # Allow additional fields

    message: str = Field(description="The main message/query to process")
    # Additional fields can be added by extending this model


class WorkflowInputWithContextModel(WorkflowInputModel):
    """Workflow input model with HTTP request context.

    Combines the base workflow input with automatic request context
    injection. Use this when your workflow needs access to HTTP
    request metadata like headers, client IP, or correlation IDs.

    Attributes:
        request_context: Automatically populated HTTP request information
        message: Inherited from WorkflowInputModel
    """

    request_context: RequestContextModel = Field(description="Automatically injected request context (required)")


class WorkflowRunRequestModel(BaseModel):
    """Model for workflow run request body."""

    input: Dict[str, Any] = Field(description="Input data for the workflow")
    session_: Optional[str] = Field(default=None, description="User identifier")
    user_id: Optional[str] = Field(default=None, description="User identifier")


class WorkflowConfig(BaseModel):
    """Workflow configuration.

    Encapsulates all workflow-specific settings including the run callback
    function, storage configuration, and operational flags.
    """

    model_config = ConfigDict(
        extra="allow",  # Allow additional workflow-specific settings
        arbitrary_types_allowed=True,  # Allow Storage and other Agno types
    )

    run_callback: Optional[OnRunCallable] = Field(default=None, description="The workflow run callback function")
    alias: Optional[str] = Field(default=None, description="Display name/alias for the workflow (shown in UI)")
    description: Optional[str] = Field(default="", description="Workflow description")
    storage: Optional[Storage] = Field(default=None, description="Storage configuration")
    debug_mode: bool = Field(default=False, description="Enable debug mode")
    monitoring: bool = Field(default=False, description="Enable monitoring")
    telemetry: bool = Field(default=False, description="Enable telemetry")
    app_id: Optional[str] = Field(default=None, description="Application identifier")


class CORSConfig(BaseModel):
    """Cross-Origin Resource Sharing (CORS) configuration.

    Controls browser security policies for cross-origin requests
    to the workflow API. Essential for web-based integrations.

    Attributes:
        allow_origins: List of allowed origin domains
        allow_credentials: Whether to include cookies/auth headers
        allow_methods: Allowed HTTP methods
        allow_headers: Allowed request headers
        max_age: Cache duration for preflight requests
        expose_headers: Additional headers visible to browser
    """

    allow_origins: List[str] = Field(default=["*"], description="Allowed origins")
    allow_credentials: bool = Field(default=True, description="Allow credentials")
    allow_methods: List[str] = Field(default=["*"], description="Allowed HTTP methods")
    allow_headers: List[str] = Field(default=["*"], description="Allowed headers")
    max_age: int = Field(default=3600, description="Max age for preflight requests in seconds")
    expose_headers: List[str] = Field(default=[], description="Headers to expose to the browser")


class MCPToolDefinition(BaseModel):
    """Definition for a custom MCP tool.

    Allows users to specify complete metadata for their custom tools
    including description, title, and the callable function.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    function: Callable = Field(description="The tool's callable function")
    description: Optional[str] = Field(default=None, description="Brief description of what the tool does")
    title: Optional[str] = Field(default=None, description="Human-readable title for the tool")


class MCPConfig(BaseModel):
    """MCP (Model Context Protocol) configuration.

    Groups all MCP-related settings together for better organization.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    server_name: str = Field(default="agno-workflow-mcp", description="MCP server name")
    memory_mode: str = Field(
        default="ephemeral",
        description="Memory mode: 'ephemeral' for in-memory sessions or 'inherited' to use workflow's memory",
    )
    custom_tools: Optional[Dict[str, Union[Callable, MCPToolDefinition]]] = Field(
        default=None,
        description="Optional custom MCP tools to register. Can be callables or MCPToolDefinition objects",
    )


class AgnoWorkflowAPIModule(BaseModel):
    """Configuration for the Agno Workflow API Module.

    Organizes workflow API settings including module configuration,
    custom endpoints, and optional MCP server settings.

    Note: CORS and other middleware should be configured at the ASGICore level,
    not in individual modules.

    Attributes:
        name: Display name for the API module
        app_id: Unique application identifier
        description: API module description
        monitoring: Enable performance monitoring
        prefix: API route prefix (e.g., '/v1')
        custom_endpoints: Function to register custom API endpoints
        mcp: Optional MCP server configuration (if None, MCP is disabled)
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(default="Agno API", description="API module name")
    app_id: str = Field(default="agno-api", description="Application identifier")
    description: str = Field(default="Agno API", description="API module description")
    monitoring: bool = Field(default=False, description="Enable monitoring")
    prefix: str = Field(default="/v1", description="API route prefix")
    custom_endpoints: Optional[Callable[[APIRouter, Workflow], None]] = Field(
        default=None, description="Function to register custom API endpoints"
    )
    mcp: Optional[MCPConfig] = Field(default=None, description="MCP server configuration (None = disabled)")
