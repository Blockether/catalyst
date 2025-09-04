#!/usr/bin/env python3
"""
Interactive CLI for refining knowledge extraction prompts using PromptAlignmentCore.

This tool allows users to:
1. Process a document chunk by chunk
2. Refine prompts for each chunk interactively
3. Use PromptAlignmentCore to optimize prompts
4. Save refined prompts for future use
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm, Prompt
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from com_blockether_catalyst.knowledge import KnowledgeExtractionCore
from com_blockether_catalyst.prompt.internal.PromptAlignmentTypes import PromptTemplate
from com_blockether_catalyst.prompt import PromptAlignmentCore
from com_blockether_catalyst.prompt.PromptAlignmentCore import (
    PromptConfiguration,
    AlignmentResult,
)
from com_blockether_catalyst.prompt.internal.PromptAlignmentTypes import (
    AlignmentFeedback,
    AlignmentPrinciple,
    EvaluationResult,
)

# Initialize Rich console for better terminal output
console = Console()


class MockTypedCall:
    """Mock implementation of TypedCall for demo purposes."""

    async def call(self, x: str) -> EvaluationResult:
        """Simple mock evaluation."""
        # In real usage, this would call an actual LLM
        return EvaluationResult(
            alignment_score=0.7,
            feedback="Mock evaluation feedback",
            suggested_improvements=["Add more context", "Clarify instructions"],
            reasoning="This is a mock evaluation for demonstration purposes",
        )


class MockAlignmentCall:
    """Mock implementation of alignment call for demo purposes."""

    async def call(self, x: str) -> AlignmentFeedback:
        """Simple mock alignment feedback."""
        return AlignmentFeedback(
            overall_assessment="Mock overall assessment for alignment",
            specific_issues=["Could be more specific", "Needs more examples"],
            improvement_suggestions=["Be more specific", "Add examples"],
            principles_to_apply=[
                AlignmentPrinciple(
                    principle="Clarity is key",
                    importance=0.9,
                )
            ],
            reasoning="Mock alignment reasoning",
        )


class PromptRefinementCLI:
    """Interactive CLI for refining knowledge extraction prompts."""

    def __init__(self, file_path: Path):
        """
        Initialize the CLI with a document file.

        Args:
            file_path: Path to the document to process
        """
        self.file_path = file_path
        self.console = console
        # Initialize prompt templates
        self.acronym_template = PromptTemplate(
            template="Extract all acronyms from the following text with their full forms."
        )
        self.keyword_template = PromptTemplate(
            template="Extract all important keywords and key terms from the following text."
        )
        self.refined_prompts: Dict[str, str] = {}
        self.current_chunk_index = 0

        # Note: For demo purposes, we're skipping KnowledgeProcessorSettings
        # and PDFKnowledgeExtractor initialization as they require full LLM call
        # implementations. In production, you would initialize these properly.

        # For demo purposes, use mock calls
        # In production, replace these with actual LLM calls
        self.target_model = MockTypedCall()
        self.alignment_model = MockAlignmentCall()

        # Initialize PromptAlignmentCore
        self.prompt_aligner = PromptAlignmentCore(
            target_model=self.target_model,
            alignment_model=self.alignment_model,
        )

        # Storage for refined templates
        self.refined_templates: Dict[str, Any] = {}

    async def run(self):
        """Main entry point for the CLI."""
        self.console.print(
            Panel.fit(
                "[bold cyan]Knowledge Extraction Prompt Refinement CLI[/bold cyan]\n"
                f"Processing: {self.file_path.name}",
                title="Welcome",
            )
        )

        # Extract document content
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            progress.add_task("Extracting document content...", total=None)
            extraction_result = await self._extract_document()
            progress.stop()

        if not extraction_result:
            self.console.print("[red]Failed to extract document content.[/red]")
            return

        # Process chunks
        chunks = await self._create_chunks(extraction_result)
        self.console.print(
            f"\n[green]Created {len(chunks)} chunks from document.[/green]\n"
        )

        # Main refinement loop
        while self.current_chunk_index < len(chunks):
            chunk = chunks[self.current_chunk_index]
            await self._process_chunk(chunk, self.current_chunk_index, len(chunks))

            # Ask user what to do next
            if self.current_chunk_index < len(chunks) - 1:
                action = await self._prompt_next_action()
                if action == "continue":
                    self.current_chunk_index += 1
                elif action == "process_all":
                    await self._process_remaining_chunks(
                        chunks[self.current_chunk_index + 1 :]
                    )
                    break
                elif action == "quit":
                    break
            else:
                self.console.print("\n[green]✓ All chunks processed![/green]")
                break

        # Save refined prompts
        if self.refined_templates:
            await self._save_refined_templates()

    async def _extract_document(self):
        """Extract content from the document."""
        try:
            # For demo purposes, return mock data
            # In production, use the actual extractor
            from com_blockether_catalyst.knowledge.internal.KnowledgeExtractionTypes import (
                KnowledgeExtractionResult,
                KnowledgePageDataWithRawText,
            )

            # Read file content directly for demo
            with open(self.file_path, "rb") as f:
                f.read()  # Read but don't use for demo

            # Create mock extraction result
            return KnowledgeExtractionResult(
                id="mock_id",
                filename=self.file_path.name,
                source_type="pdf",
                pages=[
                    KnowledgePageDataWithRawText(
                        page=1,
                        text="Mock extracted text from the document. This is a demo.",
                        raw_text="Mock raw text",
                        lines=1,
                    )
                ],
                raw="Mock raw content",
            )
        except Exception as e:
            self.console.print(f"[red]Error extracting document: {e}[/red]")
            return None

    async def _create_chunks(self, extraction_result):
        """Create chunks from extraction result."""
        chunks = []
        for page in extraction_result.pages:
            # Simple chunking for demo - in production, use the actual chunking strategy
            text = page.text
            chunk_size = 500  # Characters per chunk

            for i in range(0, len(text), chunk_size):
                chunk_text = text[i : i + chunk_size]
                chunks.append(
                    {
                        "text": chunk_text,
                        "page": page.page,
                        "index": len(chunks),
                    }
                )

        return chunks

    async def _process_chunk(self, chunk: Dict, index: int, total: int):
        """Process a single chunk interactively."""
        self.console.print(f"\n[bold]═══ Chunk {index + 1}/{total} ═══[/bold]")
        self.console.print(
            Panel(
                chunk["text"][:300] + "..."
                if len(chunk["text"]) > 300
                else chunk["text"]
            )
        )

        # Display current prompts
        self.console.print("\n[yellow]Current Prompt Templates:[/yellow]")

        # Show acronym prompt
        self._display_prompt_template("Acronym Validation", self.acronym_template)

        # Show keyword prompt
        self._display_prompt_template("Keyword Validation", self.keyword_template)

        # Ask if user wants to refine
        if Confirm.ask("\nWould you like to refine these prompts for this chunk?"):
            await self._refine_prompts_for_chunk(chunk)

    def _display_prompt_template(self, title: str, template):
        """Display a prompt template."""
        self.console.print(f"\n[cyan]{title}:[/cyan]")
        # Show first 200 chars of template
        preview = (
            template.template[:200] + "..."
            if len(template.template) > 200
            else template.template
        )
        self.console.print(Syntax(preview, "text", theme="monokai", line_numbers=False))

    async def _refine_prompts_for_chunk(self, chunk: Dict):
        """Refine prompts for a specific chunk."""
        self.console.print("\n[bold cyan]Starting prompt refinement...[/bold cyan]")

        # Select which prompt to refine
        prompt_type = Prompt.ask(
            "Which prompt would you like to refine?",
            choices=["acronym", "keyword", "both", "skip"],
            default="both",
        )

        if prompt_type == "skip":
            return

        if prompt_type in ["acronym", "both"]:
            await self._refine_single_prompt("acronym", chunk)

        if prompt_type in ["keyword", "both"]:
            await self._refine_single_prompt("keyword", chunk)

    async def _refine_single_prompt(self, prompt_type: str, chunk: Dict):
        """Refine a single prompt type using PromptAlignmentCore."""
        self.console.print(f"\n[cyan]Refining {prompt_type} prompt...[/cyan]")

        # Get current template
        if prompt_type == "acronym":
            current_template = self.acronym_template.template
        else:
            current_template = self.keyword_template.template

        # Get user's target behavior
        target_behavior = Prompt.ask(
            f"Describe the desired behavior for {prompt_type} extraction",
            default=f"Extract valid {prompt_type}s with high precision and recall",
        )

        # Configure alignment
        config = PromptConfiguration(
            initial_prompt=current_template,
            target_behavior=target_behavior,
            max_iterations=3,
            score_threshold=0.85,
            preserve_context=True,
        )

        # Run alignment
        with self.console.status(f"Aligning {prompt_type} prompt..."):
            result = await self.prompt_aligner.align_prompt(config)

        # Display results
        self._display_alignment_result(result)

        # Ask if user wants to keep the refined prompt
        if Confirm.ask("\nKeep this refined prompt?"):
            # Update the template
            if prompt_type == "acronym":
                self.acronym_template.template = result.aligned_prompt
                self.refined_templates["acronym"] = result.aligned_prompt
            else:
                self.keyword_template.template = result.aligned_prompt
                self.refined_templates["keyword"] = result.aligned_prompt

            self.console.print(
                f"[green]✓ {prompt_type.capitalize()} prompt updated![/green]"
            )

            # Store principles learned
            # domain = f"{prompt_type}_extraction"  # Removed unused variable
            if result.principles_applied:
                self.console.print(
                    f"\n[cyan]Stored {len(result.principles_applied)} principles for future use.[/cyan]"
                )

    def _display_alignment_result(self, result: AlignmentResult):
        """Display the alignment result."""
        table = Table(title="Alignment Result")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Iterations", str(result.iterations_used))
        table.add_row("Final Score", f"{result.final_score:.2%}")
        table.add_row("Principles Applied", str(len(result.principles_applied)))

        self.console.print(table)

        # Show evolution history
        if result.evolution_history:
            self.console.print("\n[yellow]Evolution History:[/yellow]")
            for evolution in result.evolution_history:
                self.console.print(
                    f"  Iteration {evolution.iteration}: Score {evolution.score:.2%}"
                )

    async def _prompt_next_action(self) -> str:
        """Prompt user for next action."""
        choice = Prompt.ask(
            "\nWhat would you like to do?",
            choices=["continue", "process_all", "quit"],
            default="continue",
        )
        return choice

    async def _process_remaining_chunks(self, chunks: List[Dict]):
        """Process all remaining chunks with current prompts."""
        self.console.print(
            "\n[cyan]Processing remaining chunks with refined prompts...[/cyan]"
        )

        with Progress(console=self.console) as progress:
            task = progress.add_task("Processing chunks...", total=len(chunks))

            for chunk in chunks:
                # Simulate processing
                await asyncio.sleep(0.1)  # In production, actually process the chunk
                progress.advance(task)

        self.console.print("[green]✓ All chunks processed![/green]")

    async def _save_refined_templates(self):
        """Save refined templates to file."""
        output_file = (
            self.file_path.parent / f"{self.file_path.stem}_refined_prompts.json"
        )

        if Confirm.ask(f"\nSave refined prompts to {output_file.name}?"):
            # Export templates
            templates_data = {
                "acronym": self.refined_templates.get("acronym"),
                "keyword": self.refined_templates.get("keyword"),
                "metadata": {
                    "source_file": str(self.file_path),
                    "chunks_processed": self.current_chunk_index + 1,
                },
            }

            with open(output_file, "w") as f:
                json.dump(templates_data, f, indent=2)

            self.console.print(
                f"[green]✓ Saved refined prompts to {output_file}[/green]"
            )

            # Also export principles
            principles_file = (
                self.file_path.parent / f"{self.file_path.stem}_principles.json"
            )
            principles_data = self.prompt_aligner.export_principles()

            if principles_data:
                with open(principles_file, "w") as f:
                    json.dump(principles_data, f, indent=2)
                self.console.print(
                    f"[green]✓ Saved learned principles to {principles_file}[/green]"
                )


async def main():
    """Main entry point for the CLI."""
    if len(sys.argv) < 2:
        console.print("[red]Usage: python PromptRefinementCLI.py <document_path>[/red]")
        console.print("\nExample: python PromptRefinementCLI.py document.pdf")
        sys.exit(1)

    file_path = Path(sys.argv[1])

    if not file_path.exists():
        console.print(f"[red]Error: File '{file_path}' not found.[/red]")
        sys.exit(1)

    if file_path.suffix.lower() not in [".pdf"]:
        console.print("[red]Error: Currently only PDF files are supported.[/red]")
        sys.exit(1)

    # Run the CLI
    cli = PromptRefinementCLI(file_path)

    try:
        await cli.run()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user.[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        import traceback

        if console.is_terminal:
            traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
