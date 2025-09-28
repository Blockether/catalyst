"""
Celery configuration module using Celery's standard configuration approach.

This follows Celery best practices:
- Use module-based configuration (celeryconfig.py)
- Leverage Celery's built-in Settings object
- Keep configuration simple and declarative
"""

import os

from kombu import Queue

# Get Redis configuration from environment or use defaults
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))
REDIS_URL = f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"

# Broker settings
broker_url = REDIS_URL
broker_connection_retry_on_startup = True

# Result backend
result_backend = REDIS_URL
result_expires = 3600  # Results expire after 1 hour

# Task execution settings
task_serializer = "pickle"
accept_content = ["pickle", "json"]
result_serializer = "pickle"
timezone = "UTC"
enable_utc = True

# Worker settings
worker_prefetch_multiplier = 1
# Restart worker after 300 tasks to prevent memory leaks
worker_max_tasks_per_child = 300
worker_disable_rate_limits = True
# REQUIRED for pypdfium2 thread-safety
# DONT CHANGE to "threads" - it breaks multiprocessing
worker_pool = "prefork"

# Task routing and queues
task_queues = (
    Queue("default", routing_key="default"),
    Queue("pdf_processing", routing_key="stage0[12].#"),
    Queue("text_processing", routing_key="stage0[3-9].#|stage10.#"),
    Queue("high_priority", routing_key="high.#"),
)

task_routes = {
    "stage01.*": {"queue": "pdf_processing"},
    "stage02.*": {"queue": "pdf_processing"},
    "stage03.*": {"queue": "text_processing"},
    "stage04.*": {"queue": "text_processing"},
    "stage05.*": {"queue": "text_processing"},
    "stage06.*": {"queue": "text_processing"},
    "stage07.*": {"queue": "text_processing"},
    "stage08.*": {"queue": "text_processing"},
    "stage09.*": {"queue": "text_processing"},
    "stage10.*": {"queue": "text_processing"},
    "orchestrate.*": {"queue": "default"},
}

# Task settings
task_time_limit = 1800  # 30 minutes hard limit
task_soft_time_limit = 1500  # 25 minutes soft limit
task_acks_late = True
task_default_retry_delay = 60  # Default retry delay
task_max_retries = 3  # Default max retries

# Task execution logging
task_publish_retry = True
task_publish_retry_policy = {
    "max_retries": 3,
    "interval_start": 0,
    "interval_step": 0.2,
    "interval_max": 0.5,
}

# Redis-specific optimizations
broker_transport_options = {
    "visibility_timeout": 3600,
    "fanout_prefix": True,
    "fanout_patterns": True,
}

# Security settings
task_always_eager = False  # Don't run tasks locally in tests by default
task_eager_propagates = False  # Don't propagate errors in eager mode

# Monitoring and debugging
worker_enable_remote_control = True  # Enable remote control commands
worker_send_task_event_heartbeats = True  # Send heartbeats for long tasks
worker_task_event_heartbeat_interval = 30  # Heartbeat every 30 seconds

# Logging configuration for better visibility in Flower
worker_hijack_root_logger = False  # Don't hijack root logger
worker_log_color = True  # Colorize logs
worker_log_format = "[%(asctime)s: %(levelname)s/%(processName)s] %(message)s"
worker_task_log_format = "[%(asctime)s: %(levelname)s/%(processName)s][%(task_name)s(%(task_id)s)] %(message)s"

# Logtool configuration - for debugging and monitoring
worker_redirect_stdouts = False  # Don't redirect to avoid conflicts with Rich console
worker_redirect_stdouts_level = "INFO"  # Log level for redirected output (when enabled)

# Enable task events for better monitoring
task_track_started = True
task_send_sent_event = True
worker_send_task_events = True

# Result backend settings for better debugging
result_extended = True  # Store extended task info
result_compression = "gzip"  # Compress results to save memory
task_result_expires = 7200  # Keep results for 2 hours (was 1 hour)

# Imports - automatically import task modules
imports = ("tools.knowledge_extraction.CeleryTasks",)

# Task autodiscovery - ensure all tasks are properly registered
task_autodiscover_packages = [
    "tools.knowledge_extraction",
]

# Include all task annotations
task_annotations = {
    "*": {
        "rate_limit": "100/m",  # Default rate limit
        "time_limit": 1800,  # 30 minutes
    },
}
