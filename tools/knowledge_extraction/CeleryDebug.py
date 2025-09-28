#!/usr/bin/env python3
"""
Celery debugging utilities for knowledge extraction pipeline.

Usage:
    # Show all registered tasks
    uv run python tools/knowledge_extraction/CeleryDebug.py tasks

    # Show active tasks
    uv run python tools/knowledge_extraction/CeleryDebug.py active

    # Show worker stats
    uv run python tools/knowledge_extraction/CeleryDebug.py stats

    # Enable debug logging
    uv run python tools/knowledge_extraction/CeleryDebug.py debug

    # Purge all pending tasks
    uv run python tools/knowledge_extraction/CeleryDebug.py purge
"""

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.tree import Tree

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tools.knowledge_extraction.CeleryApp import celery_app

console = Console()


def show_registered_tasks():
    """Show all registered tasks."""
    console.print("\n[bold cyan]📋 Registered Tasks[/bold cyan]\n")

    tasks = celery_app.tasks

    # Group tasks by category
    categories = {}
    for name, task in tasks.items():
        if "." in name:
            category = name.split(".")[0]
            if category not in categories:
                categories[category] = []
            categories[category].append(name)

    # Display as tree
    tree = Tree("Tasks")
    for category in sorted(categories.keys()):
        branch = tree.add(f"[yellow]{category}[/yellow]")
        for task_name in sorted(categories[category]):
            branch.add(f"[cyan]{task_name}[/cyan]")

    console.print(tree)
    console.print(f"\n[green]Total: {len(tasks)} tasks registered[/green]")


def show_active_tasks():
    """Show currently active tasks."""
    inspector = celery_app.control.inspect()
    active = inspector.active()

    if not active:
        console.print("[yellow]No workers are currently active[/yellow]")
        return

    console.print("\n[bold cyan]🔄 Active Tasks[/bold cyan]\n")

    for worker_name, tasks in active.items():
        console.print(f"[bold yellow]{worker_name}[/bold yellow]")

        if not tasks:
            console.print("  [dim]No active tasks[/dim]")
        else:
            table = Table()
            table.add_column("Task", style="cyan")
            table.add_column("ID", style="magenta")
            table.add_column("Args", style="green")

            for task in tasks:
                args = (
                    str(task.get("args", ""))[:50] + "..."
                    if len(str(task.get("args", ""))) > 50
                    else str(task.get("args", ""))
                )
                table.add_row(task.get("name", "Unknown"), task.get("id", ""), args)

            console.print(table)


def show_worker_stats():
    """Show worker statistics."""
    inspector = celery_app.control.inspect()
    stats = inspector.stats()

    if not stats:
        console.print("[yellow]No workers found[/yellow]")
        return

    console.print("\n[bold cyan]📊 Worker Statistics[/bold cyan]\n")

    for worker_name, worker_stats in stats.items():
        console.print(f"[bold yellow]{worker_name}[/bold yellow]")

        table = Table()
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        # Extract key metrics
        pool = worker_stats.get("pool", {})
        table.add_row("Pool Type", worker_stats.get("pool", {}).get("implementation", "N/A"))
        table.add_row("Max Concurrency", str(pool.get("max-concurrency", "N/A")))
        table.add_row("Tasks Processed", str(pool.get("writes", {}).get("total", 0)))

        # Rusage info
        rusage = worker_stats.get("rusage", {})
        if rusage:
            table.add_row("User Time", f"{rusage.get('utime', 0):.2f}s")
            table.add_row("System Time", f"{rusage.get('stime', 0):.2f}s")
            table.add_row("Max RSS", f"{rusage.get('maxrss', 0) / 1024:.1f} MB")

        console.print(table)
        console.print()


def enable_debug_logging():
    """Enable debug logging for all workers."""
    control = celery_app.control
    control.broadcast("pool_restart", arguments={"loglevel": "DEBUG"})
    console.print("[green]✓ Debug logging enabled for all workers[/green]")
    console.print("[dim]Note: Workers will restart with DEBUG level[/dim]")


def purge_tasks():
    """Purge all pending tasks."""
    celery_app.control.purge()
    console.print("[green]✓ All pending tasks purged[/green]")


def show_pending_tasks():
    """Show pending tasks in queues."""
    inspector = celery_app.control.inspect()
    reserved = inspector.reserved()

    console.print("\n[bold cyan]📦 Reserved/Pending Tasks[/bold cyan]\n")

    if not reserved:
        console.print("[yellow]No reserved tasks found[/yellow]")
        return

    for worker_name, tasks in reserved.items():
        console.print(f"[bold yellow]{worker_name}[/bold yellow]")
        console.print(f"  [cyan]Reserved tasks: {len(tasks)}[/cyan]")

        if tasks and len(tasks) <= 5:  # Show details for small number of tasks
            for task in tasks:
                console.print(f"    - {task.get('name', 'Unknown')} ({task.get('id', '')[:8]}...)")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Celery debugging utilities")
    parser.add_argument(
        "command", choices=["tasks", "active", "stats", "debug", "purge", "pending"], help="Command to execute"
    )

    args = parser.parse_args()

    commands = {
        "tasks": show_registered_tasks,
        "active": show_active_tasks,
        "stats": show_worker_stats,
        "debug": enable_debug_logging,
        "purge": purge_tasks,
        "pending": show_pending_tasks,
    }

    try:
        commands[args.command]()
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        console.print("[dim]Make sure Redis and Celery workers are running[/dim]")


if __name__ == "__main__":
    main()
