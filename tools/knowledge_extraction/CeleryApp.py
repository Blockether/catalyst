"""
Main Celery application using standard configuration approach.

To run workers:
    # Default worker
    uv run celery -A tools.knowledge_extraction.CeleryApp worker --pool=prefork --loglevel=info

    # PDF processing worker
    uv run celery -A tools.knowledge_extraction.CeleryApp worker -Q pdf_processing --pool=prefork -n pdf_worker

    # Text processing worker
    uv run celery -A tools.knowledge_extraction.CeleryApp worker -Q text_processing --pool=prefork -n text_worker

To run Flower monitoring:
    uv run celery -A tools.knowledge_extraction.CeleryApp flower --port=5555
"""

import logging

from celery import Celery

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Create Celery app using standard configuration
celery_app = Celery("knowledge_extraction")

# Load configuration from module (Celery's recommended approach)
celery_app.config_from_object("tools.knowledge_extraction.CeleryConfig")

# Log configuration details
logger.info(f"Celery broker: {celery_app.conf.broker_url}")
logger.info(f"Celery backend: {celery_app.conf.result_backend}")
logger.info(f"Worker pool type: {celery_app.conf.worker_pool}")

# Export the app for Celery command line
app = celery_app

# Import tasks to register them with Celery
# This import must happen after the app is created to avoid circular imports
import tools.knowledge_extraction.CeleryTasks  # noqa: E402, F401

if __name__ == "__main__":
    print("=" * 70)
    print("PARALLEL KNOWLEDGE EXTRACTION - CELERY")
    print("=" * 70)
    print(f"Broker URL: {celery_app.conf.broker_url}")
    print(f"Backend URL: {celery_app.conf.result_backend}")
    print(f"Worker Pool: {celery_app.conf.worker_pool}")
    print(f"Task Serializer: {celery_app.conf.task_serializer}")
    print(f"Max Tasks Per Child: {celery_app.conf.worker_max_tasks_per_child}")
    print()
    print("Registered Queues:")
    for queue in celery_app.conf.task_queues:
        print(f"  - {queue.name}")
    print()
    print("To start workers:")
    print("  uv run celery -A tools.knowledge_extraction.CeleryApp worker --pool=prefork")
    print()
    print("To start Flower monitoring:")
    print("  uv run celery -A tools.knowledge_extraction.CeleryApp flower --port=5555")
    print("=" * 70)
