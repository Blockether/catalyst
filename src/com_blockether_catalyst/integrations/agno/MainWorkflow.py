"""Main workflow implementation for Agno integration."""

from typing import Any, Callable, Coroutine, Dict, Optional

import anyio
from agno.run.response import RunResponse
from agno.workflow import Workflow

from .WorkflowTypes import OnRunCallable, RequestContextModel


class MainWorkflow(Workflow):
    """
    Workflow implementation with request context extraction and validation.

    This workflow:
    - Rejects streaming requests
    - Extracts and validates request_context
    - Extracts message field
    - Delegates to an injected run_callback function
    """

    def __init__(
        self,
        run_callback: OnRunCallable,
        description: str | None = "Main workflow",
        telemetry: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Initialize MainWorkflow with injected entrypoint.

        Args:
            run_callback: The async function to call for workflow execution
            description: Workflow description
            telemetry: Whether to enable telemetry
            **kwargs: Additional workflow configuration parameters
        """
        super().__init__(description=description, telemetry=telemetry, **kwargs)
        self._run_callback = run_callback

    async def arun(self, **kwargs: Any) -> RunResponse:  # type: ignore[override]
        """
        Asynchronous run method with request context extraction and validation.

        This method:
        1. Validates that streaming is disabled
        2. Extracts and validates the request_context parameter
        3. Extracts the required message field
        4. Delegates to the injected run_callback

        Args:
            **kwargs: Workflow input parameters including:
                - request_context: Required HTTP request metadata
                - message: Required message field
                - stream: Optional streaming flag (must be False or absent)

        Returns:
            RunResponse from the workflow execution

        Raises:
            ValueError: If streaming is enabled, request_context is missing,
                       or message field is missing
        """
        # Check for stream parameter and reject it
        if kwargs.get("stream", False):
            raise ValueError("Streaming is disabled. This API only supports non-streaming responses.")

        # Extract request context (required)
        if "request_context" not in kwargs:
            raise ValueError("request_context is required but not found in input")

        # Handle both dict and RequestContextModel instances
        request_context_data = kwargs["request_context"]
        if isinstance(request_context_data, RequestContextModel):
            request_context = request_context_data
        else:
            request_context = RequestContextModel(**request_context_data)

        # Extract message
        message = kwargs.get("message")
        if message is None:
            raise ValueError("message field is required but not found in input")

        # Remove processed fields from kwargs
        remaining_kwargs = kwargs.copy()
        remaining_kwargs.pop("message", None)
        remaining_kwargs.pop("request_context", None)
        remaining_kwargs.pop("stream", None)  # Remove stream if present

        # Call run_callback and get RunResponse directly
        return await self._run_callback(
            self,
            message=message,
            request_context=request_context,
            **remaining_kwargs,
        )

    def deep_copy(self, update: Optional[Dict[str, Any]] = None) -> "MainWorkflow":  # type: ignore
        """Create a deep copy of the workflow with optional updates.

        Args:
            update: Optional dictionary of attributes to update in the copy

        Returns:
            A new MainWorkflow instance with the same configuration
        """
        # Get current workflow settings
        copy_params: Dict[str, Any] = {
            "workflow_id": self.workflow_id,
            "name": self.name,
            "description": self.description,
            "telemetry": self.telemetry,
        }

        # Apply any updates
        if update:
            copy_params.update(update)

        # Create new instance with run_callback
        new_workflow = MainWorkflow(run_callback=self._run_callback, **copy_params)

        # Copy session and user IDs if they exist
        if hasattr(self, "session_id"):
            new_workflow.session_id = self.session_id
        if hasattr(self, "user_id"):
            new_workflow.user_id = self.user_id

        # Copy storage if it exists
        if hasattr(self, "storage") and self.storage:
            new_workflow.storage = self.storage

        return new_workflow
