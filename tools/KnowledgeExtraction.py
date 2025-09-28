#!/usr/bin/env python3
"""
Knowledge Extraction - Distributed PDF Processing with Monitoring.

This script demonstrates how to use Celery for parallel knowledge extraction.

Prerequisites:
    1. Redis must be installed:
        macOS: brew install redis
        Ubuntu: apt-get install redis-server

Usage:
    # Basic usage (auto-starts everything, outputs to 'public' directory)
    uv run python3 tools/KnowledgeExtraction.py "input/*.pdf"

    # With custom output directory
    uv run python3 tools/KnowledgeExtraction.py "input/*.pdf" my_output/

    # With Flower monitoring UI
    uv run python3 tools/KnowledgeExtraction.py "input/*.pdf" --with-flower

    # With custom Redis host
    uv run python3 tools/KnowledgeExtraction.py "input/*.pdf" --redis-host 192.168.1.100
"""

import argparse
import asyncio
import glob as glob_module
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

# Add the project root to the path BEFORE importing from tools
sys.path.insert(0, str(Path(__file__).parent.parent))

# Optimization imports removed - now handled in CeleryTasks stages 11 and 12
from tools.knowledge_extraction.CeleryTasks import (
    chunk_documents_task,
    classify_chunks_task,
    extract_pdf_batch,
    process_terms_workflow,
    optimize_pdfs_task,
    optimize_images_task,
)
from blockether_catalyst.knowledge.KnowledgeTypes import KnowledgeProcessorSettings
from blockether_catalyst.knowledge.extraction.ModelSettings import (
    ExtractionModelSettings,
)
from celery.result import AsyncResult
from celery import chain

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table


# Simple logging config
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class KnowledgeExtraction:
    """Knowledge extraction using distributed parallel processing."""

    def __init__(
        self,
        input_glob: str,
        output_dir: Path,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        with_flower: bool = False,
    ):
        """
        Initialize parallel knowledge extraction.

        Args:
            input_glob: Glob pattern for input PDF files
            output_dir: Output directory for results
            redis_host: Redis server hostname
            redis_port: Redis server port
            with_flower: Whether to start Flower monitoring UI
        """
        self.input_glob = input_glob
        self.output_dir = output_dir
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.with_flower = with_flower
        # Ensure console writes to stderr to avoid creating '-' file
        self.console = Console(stderr=True)

        # Track subprocesses for cleanup
        self.redis_process: Optional[subprocess.Popen] = None
        self.worker_process: Optional[subprocess.Popen] = None
        self.flower_process: Optional[subprocess.Popen] = None

        # Set environment variables for CeleryConfig
        os.environ["REDIS_HOST"] = redis_host
        os.environ["REDIS_PORT"] = str(redis_port)

        # Show template configuration environment variables
        self._show_template_configuration()

        # Create Celery app (it will use CeleryConfig.py)
        from celery import Celery

        self.celery_app = Celery("knowledge_extraction")
        self.celery_app.config_from_object("tools.knowledge_extraction.CeleryConfig")

    def _show_template_configuration(self) -> None:
        """Show current template configuration from environment variables."""
        self.console.print("\n[bold cyan]📝 Template Configuration:[/bold cyan]")

        # Check for template directory override
        templates_path = os.getenv("KNOWLEDGE_TEMPLATES_PATH")
        if templates_path:
            self.console.print(
                f"  [green]✓[/green] Template directory: {templates_path}"
            )
        else:
            self.console.print(
                "  [dim]Template directory: src/blockether_catalyst/assets/knowledge/prompts (default)[/dim]"
            )

        # Check for individual template overrides
        template_overrides = [
            ("KNOWLEDGE_TEMPLATE_TERM_REFINEMENT", "Term refinement"),
            ("KNOWLEDGE_TEMPLATE_DOCUMENT_CHUNKING", "Document chunking"),
            ("KNOWLEDGE_TEMPLATE_CHUNK_CLASSIFICATION", "Chunk classification"),
        ]

        has_overrides = False
        for env_var, description in template_overrides:
            template_path = os.getenv(env_var)
            if template_path:
                self.console.print(f"  [green]✓[/green] {description}: {template_path}")
                has_overrides = True

        if not has_overrides and not templates_path:
            self.console.print(
                "\n[dim]Tip: You can customize templates using these environment variables:[/dim]"
            )
            self.console.print(
                "[dim]  - KNOWLEDGE_TEMPLATES_PATH: Override entire template directory[/dim]"
            )
            self.console.print(
                "[dim]  - KNOWLEDGE_TEMPLATE_TERM_REFINEMENT: Override term refinement template[/dim]"
            )
            self.console.print(
                "[dim]  - KNOWLEDGE_TEMPLATE_DOCUMENT_CHUNKING: Override document chunking template[/dim]"
            )
            self.console.print(
                "[dim]  - KNOWLEDGE_TEMPLATE_CHUNK_CLASSIFICATION: Override chunk classification template[/dim]"
            )

        self.console.print()  # Add blank line for readability

    def find_pdf_files(self) -> List[str]:
        """Find all PDF files matching the input pattern."""
        matching_files = glob_module.glob(self.input_glob, recursive=True)
        pdf_files = [f for f in matching_files if f.lower().endswith(".pdf")]
        return pdf_files

    def check_redis_running(self) -> bool:
        """Check if Redis server is running."""
        try:
            import redis

            r = redis.Redis(
                host=self.redis_host, port=self.redis_port, socket_connect_timeout=1
            )
            r.ping()
            return True
        except Exception:
            return False

    def start_redis(self) -> bool:
        """Start Redis server if not running."""
        if self.check_redis_running():
            self.console.print("[bold green]✓ Redis is already running[/bold green]")
            return True

        self.console.print("[yellow]Starting Redis server...[/yellow]")

        try:
            # Try to start Redis server
            self.redis_process = subprocess.Popen(
                ["redis-server", "--port", str(self.redis_port)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            # Wait for Redis to start
            for _ in range(10):
                if self.check_redis_running():
                    self.console.print(
                        "[bold green]✓ Redis server started successfully[/bold green]"
                    )
                    return True
                time.sleep(0.5)

            self.console.print("[bold red]❌ Redis failed to start[/bold red]")
            return False

        except FileNotFoundError:
            self.console.print("[bold red]❌ Redis is not installed![/bold red]")
            self.console.print("\n[yellow]Install Redis first:[/yellow]")
            self.console.print("  macOS: brew install redis")
            self.console.print("  Ubuntu: apt-get install redis-server")
            return False

    def start_celery_worker(self) -> bool:
        """Start Celery worker process."""
        self.console.print("[yellow]Starting Celery worker...[/yellow]")

        # Kill any existing workers to avoid duplicates
        try:
            subprocess.run(
                ["pkill", "-f", "celery.*worker"], capture_output=True, timeout=2
            )
            time.sleep(1)  # Give it time to die
        except Exception:
            pass  # Ignore if pkill fails

        try:
            # Start Celery worker with unique name to avoid duplicate warnings
            import uuid

            worker_name = f"worker_{uuid.uuid4().hex[:8]}@{os.uname().nodename}"

            cmd = [
                sys.executable,
                "-m",
                "celery",
                "-A",
                "tools.knowledge_extraction.CeleryApp",
                "worker",
                "--pool=prefork",
                "--loglevel=INFO",  # Keep at INFO to avoid library debug logs
                "--logfile=processing.log",  # Log to file
                "--concurrency=6",
                "-E",  # Send events for real-time monitoring
                "-n",
                worker_name,  # Unique name to prevent duplicate warnings
            ]

            # Start worker with real-time log streaming
            self.worker_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,  # Line buffered
                env={**os.environ, "PYTHONUNBUFFERED": "1"},  # Force unbuffered output
            )

            # Start a thread to stream worker logs
            import threading

            from rich.markup import escape

            def stream_worker_logs():
                """Stream worker logs in real-time."""
                if not self.worker_process or not self.worker_process.stdout:
                    return
                for line in iter(self.worker_process.stdout.readline, ""):
                    if line:
                        # Escape Rich markup characters to prevent parsing errors
                        line = escape(line.strip())

                        # Format and print worker logs based on content
                        if "ERROR" in line or "CRITICAL" in line:
                            self.console.print(
                                f"[red]Worker: {line}[/red]", markup=True
                            )
                        elif "WARNING" in line:
                            self.console.print(
                                f"[yellow]Worker: {line}[/yellow]", markup=True
                            )
                        elif "Stage" in line or "Orchestrator" in line:
                            self.console.print(
                                f"[cyan]Worker: {line}[/cyan]", markup=True
                            )
                        elif "DEBUG" in line:
                            # Truncate very long debug lines (like PDF metadata)
                            if len(line) > 200:
                                line = line[:200] + "..."
                            self.console.print(
                                f"[dim]Worker: {line}[/dim]", markup=True
                            )
                        else:
                            # Truncate very long lines to prevent display issues
                            if len(line) > 300:
                                line = line[:300] + "..."
                            self.console.print(
                                f"[green]Worker: {line}[/green]", markup=True
                            )

            log_thread = threading.Thread(target=stream_worker_logs, daemon=True)
            log_thread.start()

            # Wait for worker to be ready
            time.sleep(3)

            if self.worker_process.poll() is None:
                self.console.print(
                    "[bold green]✓ Celery worker started successfully[/bold green]"
                )
                return True
            else:
                self.console.print(
                    "[bold red]❌ Celery worker failed to start[/bold red]"
                )
                return False

        except Exception as e:
            self.console.print(
                f"[bold red]❌ Error starting Celery worker: {e}[/bold red]"
            )
            return False

    def start_flower(self) -> bool:
        """Start Flower monitoring UI."""
        if not self.with_flower:
            return True

        self.console.print("[yellow]Starting Flower monitoring UI...[/yellow]")

        try:
            cmd = [
                sys.executable,
                "-m",
                "celery",
                "-A",
                "tools.knowledge_extraction.CeleryApp",
                "flower",
                "--port=5555",
            ]

            self.flower_process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            time.sleep(2)

            if self.flower_process.poll() is None:
                self.console.print(
                    "[bold green]✓ Flower UI started at http://localhost:5555[/bold green]"
                )
                return True
            else:
                self.console.print(
                    "[bold yellow]⚠ Flower UI failed to start (non-critical)[/bold yellow]"
                )
                return True  # Non-critical, continue anyway

        except Exception as e:
            self.console.print(
                f"[bold yellow]⚠ Could not start Flower: {e}[/bold yellow]"
            )
            return True  # Non-critical

    def check_celery_connection(self) -> bool:
        """Check if Celery workers are available."""
        # Try to inspect active workers
        inspector = self.celery_app.control.inspect()
        active = inspector.active()

        if active is None:
            return False

        worker_count = len(active)
        self.console.print(
            f"[bold green]✓ Found {worker_count} active worker(s)[/bold green]"
        )

        # Show worker details
        table = Table(title="Active Workers")
        table.add_column("Worker", style="cyan")
        table.add_column("Active Tasks", style="yellow")

        for worker_name, tasks in active.items():
            table.add_row(worker_name, str(len(tasks)))

        self.console.print(table)
        return True

    def cleanup(self):
        """Clean up all started processes."""
        self.console.print("\n[yellow]Cleaning up...[/yellow]")

        if self.flower_process:
            try:
                self.flower_process.terminate()
                self.flower_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.flower_process.kill()
                self.flower_process.wait(timeout=2)

        if self.worker_process:
            try:
                self.worker_process.terminate()
                self.worker_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.console.print(
                    "[yellow]Worker taking longer to shutdown, forcing...[/yellow]"
                )
                self.worker_process.kill()
                self.worker_process.wait(timeout=2)

        if self.redis_process:
            try:
                self.redis_process.terminate()
                self.redis_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.redis_process.kill()
                self.redis_process.wait(timeout=2)

        self.console.print("[bold green]✓ Cleanup complete[/bold green]")

    async def run_extraction(self) -> None:
        """Run the knowledge extraction pipeline using Celery."""
        try:
            # Always auto-start services
            self.console.print("\n[bold cyan]🚀 Starting services...[/bold cyan]")

            # Start Redis
            if not self.start_redis():
                self.console.print(
                    "[bold red]Failed to start Redis. Exiting.[/bold red]"
                )
                sys.exit(1)

            # Start Celery worker
            if not self.start_celery_worker():
                self.console.print(
                    "[bold red]Failed to start Celery worker. Exiting.[/bold red]"
                )
                sys.exit(1)

            # Start Flower (optional)
            self.start_flower()

            # Give services time to fully initialize
            self.console.print("[dim]Waiting for services to initialize...[/dim]")
            await asyncio.sleep(2)

            # Find PDF files
            pdf_files = self.find_pdf_files()

            if not pdf_files:
                self.console.print(
                    f"[bold red]❌ No PDF files found matching: {self.input_glob}[/bold red]"
                )
                return

            self.console.print(
                f"\n[bold cyan]📚 Found {len(pdf_files)} PDF files:[/bold cyan]"
            )
            for i, pdf in enumerate(pdf_files[:5], 1):
                self.console.print(f"  {i}. {Path(pdf).name}")
            if len(pdf_files) > 5:
                self.console.print(f"  ... and {len(pdf_files) - 5} more")

            # Check Celery connection
            if not self.check_celery_connection():
                self.console.print(
                    "[bold red]Workers not ready after auto-start. Exiting.[/bold red]"
                )
                return

            # Create settings for extraction (use knowledge_extraction subdirectory)
            knowledge_settings = KnowledgeProcessorSettings(
                extraction_output_dir=self.output_dir / "knowledge_extraction"
            )

            # Create model settings with default configuration
            model_settings = ExtractionModelSettings.default_settings()

            self.console.print(
                "\n[bold yellow]🚀 Starting parallel extraction...[/bold yellow]"
            )
            if self.with_flower:
                self.console.print(
                    "[dim]Monitor progress at: http://localhost:5555[/dim]"
                )

            # Start the extraction pipeline
            start_time = time.time()
            extraction_timestamp = int(time.time())
            image_output_dir = f"{self.output_dir}/knowledge_extraction/images"

            workflow = chain(
                # Stage 1: Extract PDFs (using original files)
                extract_pdf_batch.si(pdf_files, image_output_dir, knowledge_settings),
                # Stage 2: Chunk documents
                chunk_documents_task.s(model_settings, knowledge_settings),
                # Stage 3: Classify chunks
                classify_chunks_task.s(model_settings, knowledge_settings),
                # Stage 4-10: Process terms
                process_terms_workflow.s(
                    model_settings,
                    knowledge_settings,
                    extraction_timestamp,
                    str(self.output_dir / "knowledge_extraction"),
                    pdf_files,
                ),
                # Stage 11: Optimize PDFs
                optimize_pdfs_task.s(),
                # Stage 12: Optimize images
                optimize_images_task.s(),
            )

            # Start the workflow
            result = workflow.apply_async()
            task_id = result.id

            self.console.print(
                f"\n[bold green]✓ Pipeline started with workflow ID: {task_id}[/bold green]"
            )

            # Monitor progress with better status display
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console,
            ) as progress:
                task = progress.add_task("Processing PDFs...", total=None)

                last_status = None
                status_count = 0
                while True:
                    workflow_result = AsyncResult(task_id)

                    status = {
                        "task_id": task_id,
                        "ready": workflow_result.ready(),
                        "successful": workflow_result.successful()
                        if workflow_result.ready()
                        else False,
                        "status": workflow_result.status,
                        "error": None,
                    }

                    # If failed, get the actual error
                    if workflow_result.ready() and workflow_result.failed():
                        try:
                            # Try to get the traceback
                            status["error"] = workflow_result.traceback
                        except Exception:
                            # Fallback to getting the exception info
                            try:
                                status["error"] = str(workflow_result.info)
                            except Exception:
                                status["error"] = "Unknown error - check worker logs"

                    # Show more detailed status
                    from rich.markup import escape

                    current_status = str(status["status"])
                    if current_status != last_status:
                        # Escape status in case it contains markup-like characters
                        safe_status = escape(current_status)
                        self.console.print(
                            f"\n[cyan]Pipeline status changed: {safe_status}[/cyan]",
                            markup=True,
                        )
                        last_status = current_status
                        status_count = 0
                    else:
                        status_count += 1
                        # Show that we're still checking after many iterations
                        if status_count > 10:
                            self.console.print(
                                f"[dim]Still in {current_status} state... checking task {task_id}[/dim]"
                            )
                            status_count = 0

                    if status["ready"]:
                        progress.update(task, description=f"Status: {current_status}")
                        break

                    # Show meta information if available
                    if "meta" in status and status["meta"]:
                        meta = status["meta"]
                        if isinstance(meta, dict):
                            if "current" in meta and "total" in meta:
                                desc = f"Stage {meta['current']}/{meta['total']}: {meta.get('status', current_status)}"
                            else:
                                desc = f"Status: {meta.get('status', current_status)}"
                            progress.update(task, description=desc)
                    else:
                        progress.update(task, description=f"Status: {current_status}")

                    await asyncio.sleep(2)

            # Calculate duration
            duration = time.time() - start_time

            # Show results
            if status.get("successful"):
                self.console.print(
                    "\n[bold green]✅ Extraction completed successfully![/bold green]"
                )
                self.console.print(f"[cyan]Duration: {duration:.2f} seconds[/cyan]")
                self.console.print(f"[cyan]Output directory: {self.output_dir}[/cyan]")

                # Get the result and persist it
                self.console.print(
                    "\n[yellow]📝 Retrieving and persisting extraction results...[/yellow]"
                )

                # Get the LinkedKnowledge result from the workflow
                workflow_result = AsyncResult(task_id)
                linked_knowledge = workflow_result.get(timeout=30)

                if linked_knowledge:
                    # Import here to avoid circular imports
                    from blockether_catalyst.knowledge.search.SearchCore import (
                        KnowledgeSearchCore,
                    )

                    # Create pickle directory if it doesn't exist
                    extraction_dir = self.output_dir / "knowledge_extraction"
                    extraction_dir.mkdir(parents=True, exist_ok=True)

                    # Create KnowledgeSearchCore and persist
                    pickle_path = extraction_dir / "knowledge_search.pkl"
                    search_core = KnowledgeSearchCore(
                        linked_knowledge=linked_knowledge,
                        pickle_path=pickle_path,
                        auto_load=False,
                    )

                    # Persist to pickle
                    search_core.persist()

                    self.console.print(
                        f"[bold green]✓ Knowledge base saved to: {pickle_path}[/bold green]"
                    )
                    self.console.print(
                        f"  - Documents: {len(linked_knowledge.documents)}"
                    )
                    self.console.print(f"  - Terms: {len(linked_knowledge.terms)}")
                    self.console.print(f"  - Chunks: {len(linked_knowledge.chunks)}")
                else:
                    self.console.print(
                        "[yellow]⚠ No extraction result to persist[/yellow]"
                    )

                # Optimization is now handled by stages 11 and 12 in the Celery pipeline
            else:
                self.console.print("\n[bold red]❌ Extraction failed![/bold red]")

                # Always try to get error details
                error_msg = status.get("error")
                if error_msg:
                    from rich.markup import escape

                    safe_error = escape(str(error_msg)[:2000])  # Show more of the error
                    self.console.print("\n[red]Error details:[/red]", markup=True)
                    self.console.print(f"[red]{safe_error}[/red]", markup=True)
                else:
                    # If no error captured, tell user to check logs
                    self.console.print("[red]No error details captured.[/red]")
                    self.console.print(
                        "[yellow]Check the worker output above for error messages.[/yellow]"
                    )
                    self.console.print(
                        "[yellow]Look for lines with ERROR or Exception.[/yellow]"
                    )

            # Show final statistics
            self._show_statistics()

        finally:
            self.cleanup()

    def _show_statistics(self) -> None:
        """Show extraction statistics."""
        # Get Celery statistics
        inspector = self.celery_app.control.inspect()
        stats = inspector.stats()

        if stats:
            self.console.print("\n[bold cyan]📊 Worker Statistics:[/bold cyan]")
            for worker, worker_stats in stats.items():
                pool = worker_stats.get("pool", {})
                self.console.print(f"\n[yellow]{worker}:[/yellow]")
                self.console.print(f"  Pool size: {pool.get('max-concurrency', 'N/A')}")
                self.console.print(
                    f"  Tasks processed: {pool.get('writes', {}).get('total', 0)}"
                )


async def main():
    """Main entry point for parallel extraction."""
    parser = argparse.ArgumentParser(
        description="Parallel knowledge extraction using distributed processing"
    )

    parser.add_argument(
        "input_glob",
        help="Glob pattern for input PDF files (e.g., 'input/*.pdf')",
    )

    parser.add_argument(
        "output_dir",
        nargs="?",
        default="public",
        help="Output directory for extraction results (default: public)",
    )

    parser.add_argument(
        "--redis-host",
        default="localhost",
        help="Redis host address (default: localhost)",
    )

    parser.add_argument(
        "--redis-port",
        type=int,
        default=6379,
        help="Redis port (default: 6379)",
    )

    parser.add_argument(
        "--with-flower",
        action="store_true",
        help="Start Flower monitoring UI at http://localhost:5555",
    )

    parser.add_argument(
        "--show-worker-logs",
        action="store_true",
        default=True,
        help="Show worker logs in console (default: True)",
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="DEBUG",
        help="Worker log level (default: DEBUG)",
    )

    args = parser.parse_args()

    # Create and run extractor
    extractor = KnowledgeExtraction(
        input_glob=args.input_glob,
        output_dir=Path(args.output_dir),
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        with_flower=args.with_flower,
    )

    # Store additional args as instance attributes (they'll be used via getattr)
    # This is intentional for runtime configuration
    extractor.show_worker_logs = args.show_worker_logs  # type: ignore[attr-defined]
    extractor.log_level = args.log_level  # type: ignore[attr-defined]

    # Set up signal handlers for cleanup
    def signal_handler(sig, frame):
        extractor.cleanup()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    await extractor.run_extraction()


if __name__ == "__main__":
    asyncio.run(main())
