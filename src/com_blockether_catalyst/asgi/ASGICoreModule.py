"""
ASGI Core Module - Base class for modular ASGI components
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, ConfigDict, Field

from .ASGITypes import (
    HTMXConfig,
    HTMXRequest,
    HTMXResponse,
    StaticMount,
    TailwindConfig,
)


class ASGICoreModule(BaseModel, ABC):
    """Base class for ASGI modules that can be mounted to ASGICoreApplication."""

    # Pydantic fields with proper typing and defaults
    prefix: str = Field(default="", description="URL prefix for this module (e.g., '/knowledge')")
    title: Optional[str] = Field(default=None, description="Module title")
    description: Optional[str] = Field(default=None, description="Module description")
    template_dirs: List[Path] = Field(default_factory=list, description="Template directories for this module")
    static_mounts: List[StaticMount] = Field(default_factory=list, description="Static file mounts for this module")
    htmx_enabled: bool = Field(default=False, description="Whether HTMX is enabled for this module")
    htmx_config: HTMXConfig = Field(default_factory=HTMXConfig, description="HTMX configuration")
    tailwind_config: TailwindConfig = Field(default_factory=TailwindConfig, description="Tailwind CSS configuration")

    # Non-serialized fields (excluded from Pydantic serialization)
    templates: Optional[Jinja2Templates] = Field(default=None, exclude=True)

    # Pydantic configuration
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        """Post-initialization to set up computed fields and templates."""
        # Set title if not provided
        if self.title is None:
            self.title = self.__class__.__name__

        # Set description if not provided
        if self.description is None:
            self.description = f"{self.title} Module"

        # Setup templates if configured
        self._setup_templates()

    def _setup_templates(self) -> None:
        """Set up template engine if configured."""
        if self.template_dirs:
            # Use the first template directory as the main one
            self.templates = Jinja2Templates(directory=str(self.template_dirs[0]))

    @abstractmethod
    def setup_routes(self, router: APIRouter) -> None:
        """Set up module-specific routes.

        This method must be implemented by subclasses to define their routes.
        The router provided will already have the module prefix applied.

        Args:
            router: The APIRouter instance to add routes to

        Example:
            @router.get("/")
            async def index():
                return {"message": "Hello from module"}
        """
        pass

    # HTMX Helper Methods

    def get_htmx_request(self, request: Request) -> HTMXRequest:
        """Extract HTMX information from request headers.

        Args:
            request: FastAPI request object

        Returns:
            HTMXRequest object with HTMX headers parsed
        """
        headers = {k.lower(): v for k, v in request.headers.items()}
        return HTMXRequest.from_headers(headers)

    def htmx_response(
        self,
        content: str,
        *,
        push_url: Optional[str] = None,
        redirect: Optional[str] = None,
        refresh: bool = False,
        replace_url: Optional[str] = None,
        retarget: Optional[str] = None,
        reswap: Optional[str] = None,
        trigger: Optional[str] = None,
        trigger_after_settle: Optional[str] = None,
        trigger_after_swap: Optional[str] = None,
        **kwargs: Any,
    ) -> HTMLResponse:
        """Create an HTML response with HTMX headers.

        Args:
            content: HTML content to return
            push_url: URL to push to browser history
            redirect: URL to redirect to
            refresh: Whether to refresh the page
            replace_url: URL to replace in browser history
            retarget: CSS selector to retarget the response
            reswap: How to swap the response
            trigger: Client-side events to trigger
            trigger_after_settle: Events to trigger after settle
            trigger_after_swap: Events to trigger after swap
            **kwargs: Additional HTMLResponse parameters

        Returns:
            HTMLResponse with HTMX headers set
        """
        htmx_resp = HTMXResponse(
            push_url=push_url,
            redirect=redirect,
            refresh=refresh,
            replace_url=replace_url,
            retarget=retarget,
            reswap=reswap,
            trigger=trigger,
            trigger_after_settle=trigger_after_settle,
            trigger_after_swap=trigger_after_swap,
        )

        headers = htmx_resp.to_headers()
        if "headers" in kwargs:
            kwargs["headers"].update(headers)
        else:
            kwargs["headers"] = headers

        return HTMLResponse(content=content, **kwargs)

    def render_template(
        self,
        name: str,
        context: Dict[str, Any],
        request: Optional[Request] = None,
        htmx_headers: Optional[HTMXResponse] = None,
    ) -> HTMLResponse:
        """Render a template with optional HTMX headers.

        Args:
            name: Template name
            context: Template context dictionary
            request: Optional request object (required for some template features)
            htmx_headers: Optional HTMX response headers

        Returns:
            HTMLResponse with rendered template
        """
        if not self.templates:
            raise RuntimeError("Templates not configured for this module")

        # Add request to context if provided
        if request:
            context["request"] = request

        # Add HTMX config to context if enabled
        if self.htmx_enabled:
            context["htmx_config"] = self.htmx_config
            context["htmx_cdn_url"] = self.htmx_config.cdn_url

        # Add Tailwind config to context
        if self.tailwind_config.cdn_enabled:
            context["tailwind_cdn_url"] = self.tailwind_config.cdn_url

        # Add module info to context
        context["module_prefix"] = self.prefix
        context["module_title"] = self.title

        # Render template
        html = self.templates.TemplateResponse(name, context)

        # Add HTMX headers if provided
        if htmx_headers:
            headers = htmx_headers.to_headers()
            html.headers.update(headers)

        return html

    def render_partial(
        self,
        name: str,
        context: Dict[str, Any],
        request: Optional[Request] = None,
        **htmx_options: Any,
    ) -> HTMLResponse:
        """Render a partial template (for HTMX requests).

        This is a convenience method for rendering partial templates
        that are typically returned for HTMX requests.

        Args:
            name: Partial template name (from partials directory)
            context: Template context
            request: Optional request object
            **htmx_options: HTMX response options (push_url, trigger, etc.)

        Returns:
            HTMLResponse with partial content and HTMX headers
        """
        # Prefix with partials directory if not already
        if not name.startswith("partials/"):
            name = f"partials/{name}"

        # Create HTMX response if options provided
        htmx_resp = HTMXResponse(**htmx_options) if htmx_options else None

        return self.render_template(name, context, request, htmx_resp)

    def render_html(self, html_content: str) -> HTMLResponse:
        """Render raw HTML content as a response.

        This is useful when you want to return HTML without using templates.

        Args:
            html_content: Raw HTML string

        Returns:
            HTMLResponse with the HTML content
        """
        return HTMLResponse(content=html_content)
