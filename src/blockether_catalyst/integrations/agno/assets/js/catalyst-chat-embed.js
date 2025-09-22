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