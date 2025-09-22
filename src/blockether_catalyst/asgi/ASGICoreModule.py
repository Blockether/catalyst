"""
ASGI Core Module - Base class for modular ASGI components
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from jinja2 import ChoiceLoader, Environment, FileSystemLoader
from pydantic import BaseModel, ConfigDict, Field


class HTMXRequest(BaseModel):
    """HTMX request information extracted from headers."""

    is_htmx: bool = Field(default=False)
    target: Optional[str] = None  # HX-Target header
    trigger: Optional[str] = None  # HX-Trigger header
    trigger_name: Optional[str] = None  # HX-Trigger-Name header
    current_url: Optional[str] = None  # HX-Current-URL header
    prompt: Optional[str] = None  # HX-Prompt header
    boosted: bool = Field(default=False)  # HX-Boosted header

    @classmethod
    def from_headers(cls, headers: Dict[str, str]) -> "HTMXRequest":
        """Create HTMXRequest from request headers."""
        return cls(
            is_htmx=headers.get("hx-request", "").lower() == "true",
            target=headers.get("hx-target"),
            trigger=headers.get("hx-trigger"),
            trigger_name=headers.get("hx-trigger-name"),
            current_url=headers.get("hx-current-url"),
            prompt=headers.get("hx-prompt"),
            boosted=headers.get("hx-boosted", "").lower() == "true",
        )


class HTMXResponse(BaseModel):
    """HTMX response configuration."""

    push_url: Optional[str] = None  # HX-Push-Url header
    redirect: Optional[str] = None  # HX-Redirect header
    refresh: bool = Field(default=False)  # HX-Refresh header
    replace_url: Optional[str] = None  # HX-Replace-Url header
    retarget: Optional[str] = None  # HX-Retarget header
    reswap: Optional[str] = None  # HX-Reswap header
    trigger: Optional[str] = None  # HX-Trigger header (client-side events)
    trigger_after_settle: Optional[str] = None  # HX-Trigger-After-Settle
    trigger_after_swap: Optional[str] = None  # HX-Trigger-After-Swap

    def to_headers(self) -> Dict[str, str]:
        """Convert to HTTP headers."""
        headers = {}
        if self.push_url is not None:
            headers["HX-Push-Url"] = self.push_url
        if self.redirect:
            headers["HX-Redirect"] = self.redirect
        if self.refresh:
            headers["HX-Refresh"] = "true"
        if self.replace_url is not None:
            headers["HX-Replace-Url"] = self.replace_url
        if self.retarget:
            headers["HX-Retarget"] = self.retarget
        if self.reswap:
            headers["HX-Reswap"] = self.reswap
        if self.trigger:
            headers["HX-Trigger"] = self.trigger
        if self.trigger_after_settle:
            headers["HX-Trigger-After-Settle"] = self.trigger_after_settle
        if self.trigger_after_swap:
            headers["HX-Trigger-After-Swap"] = self.trigger_after_swap
        return headers


class StaticMount(BaseModel):
    """Configuration for static file serving."""

    mount: str  # URL path like "/static" or "/assets"
    directory: Path  # Directory containing static files
    name: Optional[str] = None  # Optional mount name
    html: bool = False  # Whether to serve HTML files


class HTMXConfig(BaseModel):
    """HTMX-specific configuration for our custom integration."""

    # CDN configuration
    cdn_enabled: bool = Field(default=True, description="Whether to load HTMX from CDN")
    cdn_url: str = Field(default="https://cdn.jsdelivr.net/npm/htmx.org@2.0.7/dist/htmx.min.js")

    # Default behaviors
    default_swap: Literal[
        "innerHTML",
        "outerHTML",
        "beforebegin",
        "afterbegin",
        "beforeend",
        "afterend",
        "delete",
        "none",
    ] = Field(default="innerHTML")
    default_trigger: str = Field(default="click")

    # Response headers we'll set
    push_url: bool = Field(default=True)  # Whether to push URL to browser history
    retarget: bool = Field(default=False)  # Whether to retarget responses
    reswap: bool = Field(default=False)  # Whether to change swap behavior

    # Common trigger patterns (for reference/documentation)
    trigger_patterns: Dict[str, str] = Field(
        default_factory=lambda: {
            "search": "keyup changed delay:500ms",
            "filter": "change",
            "lazy": "revealed",
            "poll": "every 2s",
        }
    )

    # Extension URLs (if needed)
    extensions: List[str] = Field(default_factory=list)


class TailwindConfig(BaseModel):
    """Tailwind CSS configuration."""

    cdn_enabled: bool = Field(default=True, description="Whether to load Tailwind CSS from CDN")
    cdn_url: str = Field(default="https://cdn.jsdelivr.net/npm/@tailwindcss/browser@4")
    custom_css: List[str] = Field(default_factory=list)


class ASGIModuleExtensionsConfig(BaseModel):
    """Configuration for ASGI module extensions."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    tailwind: Optional[TailwindConfig] = Field(default=None, description="Tailwind CSS configuration")
    htmx: Optional[HTMXConfig] = Field(default=None, description="HTMX configuration")


class ASGIModuleConfig(BaseModel):
    """Base class for ASGI modules that can be mounted to ASGICoreApplication."""

    # Pydantic fields with proper typing and defaults
    prefix: str = Field(default="", description="URL prefix for this module (e.g., '/knowledge')")
    title: str = Field(description="Module title")
    description: str = Field(description="Module description")

    statics: List[StaticMount] = Field(default_factory=list, description="Static file mounts for this module")
    templates: Optional[Jinja2Templates] = Field(default=None, description="Jinja2 templates for this module")
    template_dirs: Optional[Union[Path, List[Path]]] = Field(
        default=None, description="Template directories for this module"
    )
    extensions: Optional[ASGIModuleExtensionsConfig] = Field(
        default=None,
        description="Optional extensions configuration",
    )
    # Pydantic configuration
    model_config = ConfigDict(arbitrary_types_allowed=True)


class ASGICoreModule(ASGIModuleConfig, ABC):
    """Base class for ASGI modules that can be mounted to ASGICoreApplication."""

    def _setup_template_inheritance(self) -> None:
        """Set up template inheritance with ChoiceLoader.

        This method configures a template loader that searches in multiple directories:
        1. Module-specific templates (if provided)
        2. Parent ASGI templates as fallback
        """
        # Convert single path to list for uniform handling
        if isinstance(self.template_dirs, Path):
            template_paths = [self.template_dirs]
        else:
            template_paths = self.template_dirs if self.template_dirs else []

        # Add parent ASGI templates as fallback
        parent_template_path = Path(__file__).parent / "templates"
        if parent_template_path.exists() and parent_template_path not in template_paths:
            template_paths.append(parent_template_path)

        # Create loaders for each path
        loaders = [FileSystemLoader(path) for path in template_paths if path and Path(path).exists()]

        if loaders:
            # Create ChoiceLoader for template inheritance
            template_loader = ChoiceLoader(loaders) if len(loaders) > 1 else loaders[0]

            # Create Jinja2 environment with security enabled
            template_env = Environment(loader=template_loader, autoescape=True)

            # Initialize templates
            self.templates = Jinja2Templates(env=template_env)

    def model_post_init(self, __context: Any) -> None:
        """Post-initialization to set up computed fields and templates."""
        # Set title if not provided
        if self.title is None:
            self.title = self.__class__.__name__

        # Set description if not provided
        if self.description is None:
            self.description = f"{self.title} Module"

        # Set up templates with inheritance if template_dirs is specified
        if self.template_dirs and not self.templates:
            self._setup_template_inheritance()

    @abstractmethod
    def mount(self, app: FastAPI, router: APIRouter) -> None:
        """Mount the ASGI application to the main FastAPI app.

        This method is called to integrate the module's routes and functionality
        into the main application.

        """
        pass

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

        if self.extensions:
            if self.extensions.htmx:
                context["htmx_config"] = self.extensions.htmx

            if self.extensions.tailwind:
                context["tailwind_config"] = self.extensions.tailwind

        # Add request to context (required for TemplateResponse)
        if request:
            context["request"] = request

        # Add module info to context
        context["module_prefix"] = self.prefix
        context["module_title"] = self.title

        # Render template with new API (request as first parameter)
        if request:
            html = self.templates.TemplateResponse(request, name, context)
        else:
            # Fallback to old API if no request provided (will show deprecation warning)
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
