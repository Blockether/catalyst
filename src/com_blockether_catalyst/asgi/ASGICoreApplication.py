"""
ASGI Core Application - Root application manager with module support
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncGenerator, Dict, List, Optional, cast

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.staticfiles import StaticFiles

from .ASGITypes import ASGIConfig, MiddlewareConfig

if TYPE_CHECKING:
    from .ASGICoreModule import ASGICoreModule

# Startup
import logging


class ASGICoreApplication:
    """Root ASGI application that manages _modules."""

    def __init__(self, config: Optional[ASGIConfig] = None) -> None:
        """Initialize the root ASGI application.

        Args:
            config: Optional ASGI configuration.
        """
        self.config = config or ASGIConfig()
        self.app: FastAPI = self._create_application()
        self._modules: Dict[str, "ASGICoreModule"] = {}

        # Configure global middleware and settings
        self._configure_middleware()
        self._configure_global_static()

    @asynccontextmanager
    async def _lifespan(self, app: FastAPI) -> AsyncGenerator[None, None]:
        """Manage application lifespan events.

        Args:
            app: FastAPI application instance.

        Yields:
            None during application lifetime.
        """

        logger = logging.getLogger(__name__)
        logger.info(f"Starting {self.config.title} application...")

        yield

        # Shutdown
        logger.info(f"Shutting down {self.config.title} application...")

    def _create_application(self) -> FastAPI:
        """Create the FastAPI application instance.

        Returns:
            FastAPI application instance.
        """
        # Apply prefix to documentation URLs
        docs_url = None
        redoc_url = None
        openapi_url = None

        if self.config.docs_url:
            docs_url = f"{self.config.prefix}{self.config.docs_url}" if self.config.prefix else self.config.docs_url
        if self.config.redoc_url:
            redoc_url = f"{self.config.prefix}{self.config.redoc_url}" if self.config.prefix else self.config.redoc_url
        if self.config.openapi_url:
            openapi_url = (
                f"{self.config.prefix}{self.config.openapi_url}" if self.config.prefix else self.config.openapi_url
            )

        return FastAPI(
            title=self.config.title,
            description=self.config.description,
            version=self.config.version,
            debug=self.config.debug,
            docs_url=docs_url,
            redoc_url=redoc_url,
            openapi_url=openapi_url,
            lifespan=self._lifespan,
        )

    def _configure_middleware(self) -> None:
        """Configure global middleware for the application."""
        # Add GZip middleware if enabled
        if self.config.gzip_enabled:
            self.app.add_middleware(GZipMiddleware, minimum_size=self.config.gzip_minimum_size)

        # Add trusted host middleware if configured
        if self.config.trusted_hosts:
            self.app.add_middleware(TrustedHostMiddleware, allowed_hosts=self.config.trusted_hosts)

        # Add CORS middleware if configured
        if self.config.cors_config:
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=self.config.cors_config.allow_origins,
                allow_credentials=self.config.cors_config.allow_credentials,
                allow_methods=self.config.cors_config.allow_methods,
                allow_headers=self.config.cors_config.allow_headers,
                max_age=self.config.cors_config.max_age,
                expose_headers=self.config.cors_config.expose_headers,
            )

        # Add custom middleware
        for middleware in self.config.middleware:
            self.app.add_middleware(cast(Any, middleware.middleware_class), **middleware.options)

    def _configure_global_static(self) -> None:
        """Configure global static file serving."""
        # Mount global public directory if configured
        if self.config.public_dir and self.config.public_dir.exists():
            public_url = f"{self.config.prefix}/public" if self.config.prefix else "/public"
            self.app.mount(
                public_url,
                StaticFiles(directory=str(self.config.public_dir), html=True),
                name="public",
            )

        # Mount global static directories
        for mount in self.config.static_mounts:
            if mount.directory.exists():
                mount_url = f"{self.config.prefix}{mount.url}" if self.config.prefix else mount.url
                self.app.mount(
                    mount_url,
                    StaticFiles(directory=str(mount.directory), html=mount.html),
                    name=mount.name or f"static_{mount.url.replace('/', '_')}",
                )

    def mount_module(self, module: "ASGICoreModule", prefix: Optional[str] = None) -> None:
        """Mount an ASGICoreModule to this application.

        Args:
            module: The ASGICoreModule instance to mount
            prefix: Optional prefix override (uses module's prefix if not provided)
        """
        from fastapi import APIRouter

        # Use module's prefix or override
        module_prefix = prefix or module.prefix

        # Combine app prefix with module prefix
        full_prefix = f"{self.config.prefix}{module_prefix}" if self.config.prefix else module_prefix

        # Create a router for the module
        router = APIRouter(prefix=full_prefix)

        # Let the module setup its routes on the router
        module.setup_routes(router)

        # Include the router in the main app
        self.app.include_router(router)

        # Store module reference
        module_name = module.__class__.__name__
        self._modules[module_name] = module

        # Mount module's static files if any
        if hasattr(module, "static_mounts"):
            for mount in module.static_mounts:
                if mount.directory.exists():
                    mount_url = f"{full_prefix}{mount.url}"
                    self.app.mount(
                        mount_url,
                        StaticFiles(directory=str(mount.directory), html=mount.html),
                        name=f"{module_name}_{mount.url.replace('/', '_')}",
                    )

        # Check if module has MCP app to mount
        if hasattr(module, "get_mcp_mount_info"):
            mcp_info = module.get_mcp_mount_info()
            if mcp_info:
                mcp_app, mcp_path = mcp_info
                # Mount the MCP Starlette app under the module's prefix
                mcp_mount_url = f"{full_prefix}{mcp_path}"
                self.app.mount(mcp_mount_url, mcp_app, name=f"{module_name}_mcp")

    def add_middleware(self, middleware_config: MiddlewareConfig) -> None:
        """Add middleware to the application.

        Args:
            middleware_config: Middleware configuration to add.
        """
        self.app.add_middleware(
            cast(Any, middleware_config.middleware_class),
            **middleware_config.options,
        )

    def run(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        reload: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        """Run the application using uvicorn.

        Args:
            host: Host to bind to (uses config if not provided)
            port: Port to bind to (uses config if not provided)
            reload: Enable auto-reload (uses config if not provided)
            **kwargs: Additional uvicorn parameters
        """
        uvicorn.run(
            self.app,
            host=host or self.config.host,
            port=port or self.config.port,
            reload=reload if reload is not None else self.config.reload,
            **kwargs,
        )
