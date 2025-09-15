import logging
from datetime import datetime
from pathlib import Path
from textwrap import dedent
from typing import TYPE_CHECKING, Any, Callable, List, Optional

from agno.os import AgentOS
from agno.os.settings import AgnoAPISettings
from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates
from fastmcp import FastMCP
from fastmcp.prompts import Prompt
from fastmcp.resources import Resource
from fastmcp.tools import Tool
from pydantic import BaseModel, Field

from blockether_catalyst.asgi.ASGICoreModule import (
    ASGICoreModule,
    ASGIModuleExtensionsConfig,
    HTMXConfig,
    TailwindConfig,
)

if TYPE_CHECKING:
    from agno.agent import Agent
    from agno.team import Team
    from agno.workflow import Workflow
else:
    Agent = Any
    Workflow = Any
    Team = Any

logger = logging.getLogger(__name__)


AuthTokenResolver = Callable[[str, AgentOS, Request], Optional[str]]


def default_token_resolver(token: str, os: AgentOS, request: Request) -> Optional[str]:
    """Default token resolver for development/testing.

    WARNING: This resolver is NOT secure for production!
    Only use this for development/demo purposes

    Args:
        token: The authentication token
        os: The AgentOS instance
        request: The FastAPI request object

    Returns:
        User ID if valid, None otherwise
    """
    # Check if dev_mode is explicitly requested
    if request.query_params.get("dev_mode") == "true":
        return "DemoUser"

    # In production, this should validate the token
    logger.warning("Using default token resolver without dev_mode. Implement a secure resolver for production!")
    return None


class MCPConfig(BaseModel):
    """Configuration for MCP integration."""

    model_config = {"arbitrary_types_allowed": True}

    agent: "Agent" = Field(..., description="The MCP agent to use")
    name: str = Field(description="Name of the MCP application")
    tools: List[Tool] = Field(default_factory=list, description="List of tools to register with MCP")
    resources: List[Resource] = Field(default_factory=list, description="List of resources for MCP")
    prompts: List[Prompt] = Field(default_factory=list, description="List of prompts for MCP")


class ChatConfig(BaseModel):
    chat_agent: "Agent"
    user_id_cookie_max_age: int = 86400
    token_cookie_max_age: int = 86400
    assistant_name: str = "Omniscient Assistant"
    assistant_avatar: str = "O"
    base_url: str = "http://localhost:8002"
    auth_token_resolver: AuthTokenResolver = default_token_resolver


class AgnoOsASGIModule(ASGICoreModule):
    templates: Optional[Jinja2Templates] = Jinja2Templates(directory=Path(__file__).parent / "templates")
    extensions: Optional[ASGIModuleExtensionsConfig] = ASGIModuleExtensionsConfig(
        tailwind=TailwindConfig(cdn_enabled=True), htmx=HTMXConfig(cdn_enabled=True)
    )
    prefix: str = "/os"
    version: str = "0.1.0"
    chat: ChatConfig
    agents: List["Agent"] = []
    workflows: List["Workflow"] = []
    teams: List["Team"] = []
    docs_enabled: bool = True
    mcp: Optional[MCPConfig] = None
    api_token: Optional[str] = None
    cors_list: List[str] = []

    def model_post_init(self, __context: Any) -> None:
        if not self.chat:
            raise ValueError("Chat configuration must be provided for AgnoOsASGIModule")

        return super().model_post_init(__context)

    def mount(self, app: FastAPI, router: APIRouter) -> None:
        """Mount the ASGI application to the main FastAPI app."""

        self._os = AgentOS(
            os_id=self.title.strip().replace(" ", "_"),
            agents=self.agents,
            version=self.version,
            workflows=self.workflows,
            teams=self.teams,
            settings=AgnoAPISettings(
                os_security_key=self.api_token,
                docs_enabled=self.docs_enabled,
                cors_origin_list=self.cors_list,
            ),
            enable_mcp=False,
            telemetry=False,
        )

        router_os = self._os.get_app().router
        if self.mcp:
            mcp = FastMCP(name=self.mcp.name, version=self.version)
            for tool in self.mcp.tools:
                mcp.add_tool(tool)

        if not self.chat or not self.chat.chat_agent:
            raise ValueError("Chat configuration or chat agent is not properly set")

        @router_os.get("/view", response_class=HTMLResponse, include_in_schema=False)
        async def chat_interface(request: Request) -> HTMLResponse:
            """Render the chat interface for Knowledge Extraction Workflow."""
            # Start with no session - will be created on first message
            session_id = ""

            # Try to get token from various sources
            # 1. Query parameter (useful for iframe embedding)
            token = request.query_params.get("token")

            # 2. Authorization header
            if not token:
                auth_header = request.headers.get("Authorization")
                if auth_header and auth_header.startswith("Bearer "):
                    token = auth_header[7:]

            # 3. Cookie
            if not token:
                token = request.cookies.get("auth_token")

            # Check for dev_mode first
            user_id = None
            if request.query_params.get("dev_mode") == "true":
                user_id = "DemoUser"
            elif token:
                # If we have a token, resolve the user
                user_id = self.chat.auth_token_resolver(token, self._os, request)  # type: ignore

            # Check if user was already set in cookie (from previous resolution)
            if not user_id:
                user_id = request.cookies.get("user_id")

            # Create the response using render_template which returns a TemplateResponse
            response = self.render_template(
                "chat.j2",
                {
                    "agent_name": self.chat.chat_agent.name,
                    "agent_description": self.chat.chat_agent.description,
                    "agent_id": self.chat.chat_agent.id,
                    "session_id": session_id,
                    "user_id": user_id or "",  # Pass resolved user_id to template
                    "auth_token": token or "",  # Pass token to template
                    "base_url": self.chat.base_url,  # Pass base URL to template
                    "api_prefix": self.prefix,  # /os
                    "assistant_name": self.chat.assistant_name,  # Pass configurable assistant name
                    "assistant_avatar": self.chat.assistant_avatar,  # Pass configurable avatar
                },
                request=request,
            )

            # If we resolved a user from token, set it in a cookie
            if user_id and token:
                response.set_cookie(
                    key="user_id",
                    value=user_id,
                    httponly=True,
                    samesite="lax",
                    max_age=self.chat.user_id_cookie_max_age,
                )
                response.set_cookie(
                    key="auth_token",
                    value=token,
                    httponly=True,
                    samesite="lax",
                    max_age=self.chat.token_cookie_max_age,
                )

            return response

        @router_os.post(
            "/view/render-message",
            response_class=HTMLResponse,
            include_in_schema=False,
        )
        async def render_message(request: Request) -> Response:
            """Render a workflow response message as HTML."""

            # Parse JSON body
            data = await request.json()

            message_content = data.get("content", "")
            message_id = data.get("message_id", f"msg_{datetime.now().timestamp()}")
            is_error = data.get("is_error", False)
            session_id = data.get("session_id")
            is_new_session = data.get("is_new_session", False)

            # Render the workflow response as HTML - render_partial returns a TemplateResponse
            response_obj = self.render_partial(
                "partials/workflow_message.j2",
                {
                    "response": message_content,
                    "timestamp": datetime.now().strftime("%I:%M %p"),
                    "message_id": message_id,
                    "is_error": is_error,
                    "assistant_name": self.chat.assistant_name,
                    "assistant_avatar": self.chat.assistant_avatar,
                },
                request=request,
            )

            # Add session ID header if new session
            if is_new_session and session_id:
                response_obj.headers["X-Session-ID"] = session_id

            return response_obj

        @router_os.get("/view/script.js", response_class=Response, include_in_schema=False)
        async def embed_script_js(request: Request) -> Response:
            """Serve the Catalyst Chat Embed JavaScript library with .js extension for better compatibility."""

            script_content = dedent(
                """
                /**
                * Catalyst Chat Embed - JavaScript Library
                * Embeds the Catalyst chat interface as an iframe with authentication and context support
                */
                class CatalystChatEmbed {
                    constructor(target, options = {}) {
                        this.baseUrl = options.baseUrl || 'http://localhost:8002';
                        this.endpoint = options.endpoint || '/os/view';
                        this.token = options.token || null;
                        this.targetSelector = target || '#catalyst-chat';
                        this.width = options.width || '100%';
                        this.height = options.height || '600px';
                        this.devMode = options.devMode || false;
                        this.assistantName = options.assistantName || null;
                        this.assistantAvatar = options.assistantAvatar || null;

                        // Default iframe styles
                        this.iframeStyles = {
                            border: 'none',
                            borderRadius: '8px',
                            boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
                            ...options.styles
                        };
                        this.iframe = null;
                        this.targetElement = null;

                        // Auto-initialize if target is provided
                        if (target) {
                            this.init();
                        }
                    }

                    /**
                    * Initialize and render the chat embed
                    */
                    init() {
                        this.targetElement = document.querySelector(this.targetSelector);

                        if (!this.targetElement) {
                            console.error(`Catalyst Chat Embed: Target element "${this.targetSelector}" not found`);
                            return false;
                        }
                        this.createIframe();
                        this.render();

                        return true;
                    }

                    /**
                    * Create the iframe element with proper configuration
                    */
                    createIframe() {
                        this.iframe = document.createElement('iframe');
                        this.iframe.src = this.buildIframeUrl();
                        this.iframe.width = this.width;
                        this.iframe.height = this.height;

                        // Apply styles
                        Object.assign(this.iframe.style, this.iframeStyles);

                        // Accessibility and security attributes
                        this.iframe.setAttribute('title', 'Catalyst Chat Interface');
                        this.iframe.setAttribute('allow', 'clipboard-write');
                        this.iframe.setAttribute('sandbox', 'allow-scripts allow-same-origin allow-forms allow-popups allow-modals');
                    }

                    /**
                    * Build the iframe URL with authentication and parameters
                    */
                    buildIframeUrl() {
                        const url = new URL(this.endpoint, this.baseUrl);

                        // Add authentication
                        if (this.token) {
                            url.searchParams.set('token', this.token);
                        } else if (this.devMode) {
                            url.searchParams.set('dev_mode', 'true');
                        }

                        // Add assistant customization
                        if (this.assistantName) {
                            url.searchParams.set('assistant_name', this.assistantName);
                        }
                        if (this.assistantAvatar) {
                            url.searchParams.set('assistant_avatar', this.assistantAvatar);
                        }

                        return url.toString();
                    }

                    /**
                    * Render the iframe into the target element
                    */
                    render() {
                        // Clear existing content
                        this.targetElement.innerHTML = '';

                        // Add the iframe
                        this.targetElement.appendChild(this.iframe);

                        // Add loading indicator
                        this.showLoadingIndicator();

                        // Handle iframe load
                        this.iframe.addEventListener('load', () => {
                            this.hideLoadingIndicator();
                            this.onIframeLoaded();
                        });

                        // Handle iframe errors
                        this.iframe.addEventListener('error', (error) => {
                            this.hideLoadingIndicator();
                            this.showErrorMessage('Failed to load chat interface');
                            console.error('Catalyst Chat Embed: Iframe load error', error);
                        });
                    }

                    /**
                    * Show loading indicator
                    */
                    showLoadingIndicator() {
                        const loader = document.createElement('div');
                        loader.id = 'catalyst-chat-loader';
                        loader.style.cssText = `
                            position: absolute;
                            top: 50%;
                            left: 50%;
                            transform: translate(-50%, -50%);
                            text-align: center;
                            color: #6b7280;
                            font-family: system-ui, -apple-system, sans-serif;
                            font-size: 14px;
                        `;
                        loader.innerHTML = `
                            <div style="
                                width: 32px;
                                height: 32px;
                                border: 3px solid #f3f4f6;
                                border-top: 3px solid #FFCC00;
                                border-radius: 50%;
                                animation: spin 1s linear infinite;
                                margin: 0 auto 8px;
                            "></div>
                            Loading chat...
                            <style>
                                @keyframes spin {
                                    0% { transform: rotate(0deg); }
                                    100% { transform: rotate(360deg); }
                                }
                            </style>
                        `;

                        // Make target element relative for absolute positioning
                        if (getComputedStyle(this.targetElement).position === 'static') {
                            this.targetElement.style.position = 'relative';
                        }

                        this.targetElement.appendChild(loader);
                    }

                    /**
                    * Hide loading indicator
                    */
                    hideLoadingIndicator() {
                        const loader = document.getElementById('catalyst-chat-loader');
                        if (loader) {
                            loader.remove();
                        }
                    }

                    /**
                    * Show error message
                    */
                    showErrorMessage(message) {
                        const error = document.createElement('div');
                        error.style.cssText = `
                            padding: 20px;
                            text-align: center;
                            color: #ef4444;
                            background: #fef2f2;
                            border: 1px solid #fecaca;
                            border-radius: 8px;
                            font-family: system-ui, -apple-system, sans-serif;
                            font-size: 14px;
                        `;
                        error.textContent = message;

                        this.targetElement.appendChild(error);
                    }

                    /**
                    * Called when iframe is successfully loaded
                    */
                    onIframeLoaded() {
                        console.log('Catalyst Chat Embed: Chat interface loaded successfully');

                        // Trigger custom event
                        const event = new CustomEvent('catalystChatLoaded', {
                            detail: { embed: this }
                        });
                        document.dispatchEvent(event);
                    }

                    /**
                    * Set context data that will be persisted in cookies
                    * @param {Object} contextData - The context data to store
                    */
                    setContext(contextData) {
                        try {
                            const contextString = JSON.stringify(contextData);
                            this.setCookie('context', contextString, 7); // 7 days expiry

                            console.log('Catalyst Chat Embed: Context set', contextData);
                            return true;
                        } catch (error) {
                            console.error('Catalyst Chat Embed: Failed to set context', error);
                            return false;
                        }
                    }

                    /**
                    * Get context data from cookies
                    * @returns {Object|null} The context data or null if not found
                    */
                    getContext() {
                        try {
                            const contextString = this.getCookie('context');
                            if (!contextString) return null;

                            return JSON.parse(contextString);
                        } catch (error) {
                            console.error('Catalyst Chat Embed: Failed to get context', error);
                            return null;
                        }
                    }

                    /**
                    * Clear context data
                    */
                    clearContext() {
                        this.setCookie('context', '', -1);
                        console.log('Catalyst Chat Embed: Context cleared');
                    }

                    /**
                    * Update authentication token
                    * @param {string} token - New authentication token
                    */
                    setToken(token) {
                        this.token = token;

                        // Reload iframe with new token
                        if (this.iframe) {
                            this.iframe.src = this.buildIframeUrl();
                        }

                        console.log('Catalyst Chat Embed: Token updated');
                    }

                    /**
                    * Enable or disable dev mode
                    * @param {boolean} enabled - Whether to enable dev mode
                    */
                    setDevMode(enabled) {
                        this.devMode = enabled;

                        // Reload iframe
                        if (this.iframe) {
                            this.iframe.src = this.buildIframeUrl();
                        }

                        console.log(`Catalyst Chat Embed: Dev mode ${enabled ? 'enabled' : 'disabled'}`);
                    }

                    /**
                    * Show the chat interface
                    */
                    show() {
                        if (this.targetElement) {
                            this.targetElement.style.display = 'block';
                        }
                    }

                    /**
                    * Hide the chat interface
                    */
                    hide() {
                        if (this.targetElement) {
                            this.targetElement.style.display = 'none';
                        }
                    }

                    /**
                    * Toggle the chat interface visibility
                    */
                    toggle() {
                        if (this.targetElement) {
                            const isVisible = this.targetElement.style.display !== 'none';
                            this.targetElement.style.display = isVisible ? 'none' : 'block';
                        }
                    }

                    /**
                    * Destroy the embed and clean up
                    */
                    destroy() {
                        if (this.targetElement) {
                            this.targetElement.innerHTML = '';
                        }

                        this.iframe = null;
                        this.targetElement = null;

                        console.log('Catalyst Chat Embed: Destroyed');
                    }

                    /**
                    * Reload the chat interface
                    */
                    reload() {
                        if (this.iframe) {
                            // Show loading indicator
                            this.showLoadingIndicator();
                            
                            // Create a one-time load event listener for this reload
                            const onReloadComplete = () => {
                                this.hideLoadingIndicator();
                                this.onIframeLoaded();
                                // Remove the listener after it fires
                                this.iframe.removeEventListener('load', onReloadComplete);
                            };
                            
                            // Add the load listener before changing src
                            this.iframe.addEventListener('load', onReloadComplete);
                            
                            // Force reload by appending timestamp to prevent caching
                            const url = new URL(this.buildIframeUrl());
                            url.searchParams.set('_reload', Date.now());
                            this.iframe.src = url.toString();
                        }
                    }

                    /**
                    * Utility: Set cookie
                    * @param {string} name - Cookie name
                    * @param {string} value - Cookie value
                    * @param {number} days - Expiry in days
                    */
                    setCookie(name, value, days) {
                        const expires = new Date();
                        expires.setTime(expires.getTime() + (days * 24 * 60 * 60 * 1000));

                        const cookieString = `${name}=${encodeURIComponent(value)}; expires=${expires.toUTCString()}; path=/; SameSite=Lax`;
                        document.cookie = cookieString;
                    }

                    /**
                    * Utility: Get cookie
                    * @param {string} name - Cookie name
                    * @returns {string|null} Cookie value or null
                    */
                    getCookie(name) {
                        const nameEQ = name + "=";
                        const ca = document.cookie.split(';');

                        for (let i = 0; i < ca.length; i++) {
                            let c = ca[i];
                            while (c.charAt(0) === ' ') {
                                c = c.substring(1, c.length);
                            }
                            if (c.indexOf(nameEQ) === 0) {
                                return decodeURIComponent(c.substring(nameEQ.length, c.length));
                            }
                        }

                        return null;
                    }
                }

                /**
                * Global factory function for easy initialization
                * @param {string} target - CSS selector for target element
                * @param {Object} options - Configuration options
                * @returns {CatalystChatEmbed} Embed instance
                */
                window.CatalystChat = function(target, options) {
                    const embed = new CatalystChatEmbed(target, options);
                    return embed;
                };

                // Export for module systems
                if (typeof module !== 'undefined' && module.exports) {
                    module.exports = CatalystChatEmbed;
                }
            """
            ).strip()

            return Response(
                content=script_content,
                media_type="application/javascript",
                headers={
                    "Cache-Control": "public, max-age=3600",  # Cache for 1 hour
                    "Content-Type": "application/javascript; charset=utf-8",
                },
            )

        router.routes = list(router.routes) + list(router_os.routes)
