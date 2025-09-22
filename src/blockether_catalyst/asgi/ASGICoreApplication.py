"""
ASGI Core Application - Root application manager with module support
"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncGenerator, Dict, List, Optional, cast

from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from .ASGICoreModule import ASGICoreModule, ASGIModuleConfig, StaticMount

logger = logging.getLogger(__name__)


class CORSConfig(BaseModel):
    """CORS middleware configuration."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    allow_origins: List[str] = Field(default=["*"])
    allow_credentials: bool = Field(default=True)
    allow_methods: List[str] = Field(default=["*"])
    allow_headers: List[str] = Field(default=["*"])
    max_age: int = Field(default=3600)
    expose_headers: List[str] = Field(default=[])


class ASGIApplicationConfig(BaseModel):
    """ASGI application configuration."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Core Settings
    title: str = Field(default="Catalyst API")
    description: str = Field(default="Enterprise LLM Toolkit API")
    version: str = Field(default="1.0.0")
    prefix: str = Field(default="")
    statics: List[StaticMount] = Field(default_factory=list, description="Static file mounts for this module")

    # Modules
    modules: List[ASGICoreModule] = Field(default_factory=list)

    # CORS
    cors: Optional[CORSConfig] = None

    # Server
    host: str = Field(default="127.0.0.1")
    port: int = Field(default=8000)

    # Compression
    gzip_enabled: bool = Field(default=True)
    gzip_minimum_size: int = Field(default=1000)

    # Logging
    debug: bool = Field(default=False)
    log_level: str = Field(default="INFO")
    log_requests: bool = Field(default=True)

    # OpenAPI/Swagger
    docs_url: Optional[str] = Field(default="/docs")
    redoc_url: Optional[str] = Field(default="/redoc")
    openapi_url: Optional[str] = Field(default="/openapi.json")

    # Security
    trusted_hosts: Optional[List[str]] = None


class ASGICoreApplication(ASGIApplicationConfig):
    """Root ASGI application that manages modules."""

    def model_post_init(self, __context: Any) -> None:
        """Post-initialization to set up computed fields and templates."""
        self._app = self._create_application()
        self._configure_middleware()

        for module in self.modules:
            self.mount_module(module)

    @asynccontextmanager
    async def _lifespan(self, app: FastAPI) -> AsyncGenerator[None, None]:
        """Manage application lifespan events.

        Args:
            app: FastAPI application instance.

        Yields:
            None during application lifetime.
        """

        logger = logging.getLogger(__name__)
        logger.info(f"Starting {self.title} application...")

        yield

        # Shutdown
        logger.info(f"Shutting down {self.title} application...")

    @property
    def app(self) -> FastAPI:
        """Get the FastAPI application instance.

        Returns:
            FastAPI application instance.
        """
        return self._app

    def _create_application(self) -> FastAPI:
        """Create the FastAPI application instance.

        Returns:
            FastAPI application instance.
        """
        return FastAPI(
            title=self.title,
            description=self.description,
            version=self.version,
            debug=self.debug,
            lifespan=self._lifespan,
        )

    def _configure_middleware(self) -> None:
        """Configure global middleware for the application."""
        # Add GZip middleware if enabled
        if self.gzip_enabled:
            self._app.add_middleware(GZipMiddleware, minimum_size=self.gzip_minimum_size)

        # Add trusted host middleware if configured
        if self.trusted_hosts:
            self._app.add_middleware(TrustedHostMiddleware, allowed_hosts=self.trusted_hosts)

        # Add CORS middleware if configured
        if self.cors:
            self._app.add_middleware(
                CORSMiddleware,
                allow_origins=self.cors.allow_origins,
                allow_credentials=self.cors.allow_credentials,
                allow_methods=self.cors.allow_methods,
                allow_headers=self.cors.allow_headers,
                max_age=self.cors.max_age,
                expose_headers=self.cors.expose_headers,
            )

    def mount_module(self, module: ASGICoreModule, prefix: Optional[str] = None) -> None:
        """Mount an ASGICoreModule to this application.

        Args:
            module: The ASGICoreModule instance to mount
            prefix: Optional prefix override (uses module's prefix if not provided)
        """

        # Use module's prefix or override
        module_prefix = prefix or module.prefix

        if not module_prefix.startswith("/"):
            raise ValueError("Module prefix must start with '/'")

        # Combine app prefix with module prefix
        full_prefix = f"{self.prefix}{module_prefix[1:]}"

        # Store module reference
        module_name = module.__class__.__name__

        print(f"Mounting module {module_name} at prefix {full_prefix}")
        router = APIRouter(tags=[module_name])

        module.mount(self._app, router)
        self._app.include_router(router, prefix=full_prefix)

        # Mount module's static files if any
        for mount in module.statics:
            if mount.directory.exists():
                mount_url = os.path.join(full_prefix, mount.mount.lstrip("/"))
                self._app.mount(
                    mount_url,
                    StaticFiles(
                        directory=os.path.join(os.getcwd(), mount.directory),
                        html=mount.html,
                    ),
                    name=f"{module_name}_{mount.mount.replace('/', '_')}",
                )

    def run(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """Run the application using uvicorn.

        Args:
            host: Host to bind to (uses config if not provided)
            port: Port to bind to (uses config if not provided)
            reload: Enable auto-reload (uses config if not provided)
            **kwargs: Additional uvicorn parameters
        """
        import uvicorn

        uvicorn.run(
            self._app,
            host=host or self.host,
            port=port or self.port,
            **kwargs,
        )
