"""Agno workflow integration for Catalyst.

This module provides integration with Agno workflows, including:
- WorkflowApiASGIModule: ASGI module for Agno workflows
- MCPServer: Model Context Protocol server for LLM integration
- MainWorkflow: Base workflow implementation with request context extraction
- Various configuration models for API and workflow settings

Example:
    ```python
    from com_blockether_catalyst.integrations.agno import (
        WorkflowApiASGIModule,
        WorkflowConfig,
        AgnoWorkflowAPIModule,
        MCPConfig
    )

    api = WorkflowApiASGIModule(
        workflow=WorkflowConfig(
            run_callback=my_workflow_function,
            description="My workflow description"
        ),
        api=AgnoWorkflowAPIModule(
            name="My API",
            mcp=MCPConfig()  # MCP enabled when config provided
        )
    )
    ```
"""

from .MainWorkflow import MainWorkflow
from .WorkflowASGIModule import MCPServer, WorkflowApiASGIModule
from .WorkflowTypes import WorkflowInputWithContextModel  # Deprecated alias
from .WorkflowTypes import (
    AgnoWorkflowAPIModule,
    MCPConfig,
    MCPToolDefinition,
    OnRunCallable,
    RequestContextModel,
    WorkflowConfig,
    WorkflowInputModel,
    WorkflowRunRequestModel,
)

__all__ = [
    "MainWorkflow",
    "MCPServer",
    "WorkflowApiASGIModule",
    "WorkflowConfig",
    "AgnoWorkflowAPIModule",
    "OnRunCallable",
    "MCPConfig",
    "MCPToolDefinition",
    "RequestContextModel",
    "WorkflowInputModel",
    "WorkflowInputWithContextModel",
    "WorkflowInputWithContextModel",  # Deprecated, use WorkflowInputWithContextModel
    "WorkflowRunRequestModel",
]
