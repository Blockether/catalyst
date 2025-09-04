from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Type

from fastapi import APIRouter
from pydantic import BaseModel, Field


class StaticMount(BaseModel):
    """Configuration for static file serving."""

    url: str  # URL path like "/static" or "/assets"
    directory: Path  # Directory containing static files
    name: Optional[str] = None  # Optional mount name
    html: bool = False  # Whether to serve HTML files


class HTMXConfig(BaseModel):
    """HTMX-specific configuration for our custom integration."""

    # CDN configuration
    cdn_url: str = Field(default="https://unpkg.com/htmx.org@1.9.10")
    cdn_enabled: bool = Field(default=True)

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

    cdn_enabled: bool = Field(default=True)
    cdn_url: str = Field(default="https://cdn.tailwindcss.com")
    custom_css: List[str] = Field(default_factory=list)  # Additional CSS files


class CORSConfig(BaseModel):
    """CORS middleware configuration."""

    model_config = {"arbitrary_types_allowed": True}

    allow_origins: List[str] = Field(default=["*"])
    allow_credentials: bool = Field(default=True)
    allow_methods: List[str] = Field(default=["*"])
    allow_headers: List[str] = Field(default=["*"])
    max_age: int = Field(default=3600)
    expose_headers: List[str] = Field(default=[])


class MiddlewareConfig(BaseModel):
    """Middleware configuration."""

    model_config = {"arbitrary_types_allowed": True}

    middleware_class: Type[Any]
    options: Dict[str, Any] = Field(default_factory=dict)


class RouteConfig(BaseModel):
    """Route configuration."""

    model_config = {"arbitrary_types_allowed": True}

    router: Optional[APIRouter] = None
    prefix: str = Field(default="")
    tags: List[str] = Field(default_factory=list)
    dependencies: Optional[List[Any]] = None


class ASGIConfig(BaseModel):
    """ASGI application configuration."""

    model_config = {"arbitrary_types_allowed": True}

    # Core Settings
    title: str = Field(default="Catalyst API")
    description: str = Field(default="Enterprise LLM Toolkit API")
    version: str = Field(default="1.0.0")

    # URL Configuration
    prefix: str = Field(default="")  # Root application prefix (e.g., "/api/v1")

    # Static Assets
    static_mounts: List[StaticMount] = Field(default_factory=list)
    public_dir: Optional[Path] = None  # Main public directory

    # Template Configuration
    template_dirs: List[Path] = Field(default_factory=list)
    template_engine: Literal["jinja2", "string", "none"] = Field(default="none")

    # HTMX Configuration
    htmx_enabled: bool = Field(default=False)
    htmx_config: HTMXConfig = Field(default_factory=HTMXConfig)

    # Tailwind CSS Configuration
    tailwind_config: TailwindConfig = Field(default_factory=TailwindConfig)

    # API Documentation
    debug: bool = Field(default=False)
    docs_url: Optional[str] = Field(default="/docs")
    redoc_url: Optional[str] = Field(default="/redoc")
    openapi_url: Optional[str] = Field(default="/openapi.json")
    openapi_prefix: str = Field(default="")

    # CORS and Middleware
    cors_config: Optional[CORSConfig] = None
    middleware: List[MiddlewareConfig] = Field(default_factory=list)
    gzip_enabled: bool = Field(default=True)
    gzip_minimum_size: int = Field(default=1000)  # Minimum response size to compress

    # Routes
    routes: List[RouteConfig] = Field(default_factory=list)

    # Server Settings
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    reload: bool = Field(default=False)

    # Security
    trusted_hosts: Optional[List[str]] = None
    max_request_size: int = Field(default=100 * 1024 * 1024)  # 100MB default

    # Custom Extensions
    extensions: Dict[str, Any] = Field(default_factory=dict)

    def get_prefixed_url(self, url: str, include_module: bool = True) -> str:
        """Get URL with prefix applied.

        Args:
            url: The URL path
            include_module: Whether to include the module prefix (for compatibility)

        Returns:
            Full prefixed URL path
        """
        if not url:
            return self.prefix or "/"
        if not self.prefix or not include_module:
            return url
        # Ensure single slash between prefix and URL
        prefix = self.prefix.rstrip("/")
        url = url.lstrip("/")
        return f"{prefix}/{url}" if url else prefix


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
