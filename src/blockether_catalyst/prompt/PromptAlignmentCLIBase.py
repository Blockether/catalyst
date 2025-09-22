"""
Base class for creating prompt refinement CLIs.

This provides a reusable framework for building interactive CLIs that refine
prompts using the PromptAlignmentCore system.
"""

try:
    import rich
except ImportError:
    raise ImportError(
        "The 'rich' library is required for PromptAlignmentCLIBase. Please install it via 'pip install rich/uv add rich'."
    )

import json
import re
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generic, List, Optional, TypeVar

from pydantic import BaseModel
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.syntax import Syntax
from rich.table import Table

from blockether_catalyst.prompt import PromptAlignmentCore
from blockether_catalyst.prompt.PromptAlignmentCore import (
    AlignmentResult,
    PromptConfiguration,
)

# Type variable for the response model type
TResponse = TypeVar("TResponse", bound=BaseModel)


class PromptAlignmentCLIBase(ABC, Generic[TResponse]):
    """Base class for creating prompt refinement CLIs."""

    def __init__(
        self,
        prompt_name: str,
        prompt_dir: Path = Path("prompts"),
        console: Optional[Console] = None,
        output_dir: Optional[Path] = None,
    ):
        """
        Initialize the CLI base.

        Args:
            prompt_name: Name for the prompt (used for saving)
            prompt_dir: Directory to save prompts
            console: Rich console for output
            output_dir: Directory to save outputs (responses, filled prompts, etc.)
        """
        self.prompt_name = prompt_name
        self.prompt_dir = Path(prompt_dir)
        self.console = console or Console()

        # Create output directory for responses
        self.output_dir = output_dir or Path(f"output/{prompt_name}_responses")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Will be set by subclass
        self.prompt_template: str = ""
        self.prompt_aligner: Optional[PromptAlignmentCore] = None

        # Initialize components
        self._init_components()

        # Load saved prompt or default
        self.prompt_template = self._load_prompt_template()

    @abstractmethod
    def _init_components(self):
        """Initialize LLM components and prompt aligner."""
        pass

    @abstractmethod
    async def _test_prompt(self, prompt: str) -> TResponse:
        """Test the prompt with actual data and return results.

        Args:
            prompt: The prompt to test

        Returns:
            Response model with test results
        """
        pass

    @abstractmethod
    def _display_test_results(self, results: TResponse):
        """Display the test results in a formatted way.

        Args:
            results: The test results to display
        """
        pass

    def _get_current_data(self) -> Optional[Any]:
        """
        Get the current data being processed (e.g., batch of pages, documents).
        Override this method to provide data for viewing/processing.

        Returns:
            Current data or None if not available
        """
        return None

    def _display_raw_json(self, results: BaseModel, title: str = "Raw JSON Results"):
        """
        Display results as pretty-printed JSON.

        Args:
            results: BaseModel instance to display as JSON
            title: Title for the JSON panel
        """
        from rich.syntax import Syntax

        # Convert BaseModel to JSON with proper formatting
        json_str = results.model_dump_json(indent=2)

        # Use Rich's syntax highlighting for JSON with word wrap
        syntax = Syntax(
            json_str,
            "json",
            theme="monokai",
            line_numbers=False,
            word_wrap=True,  # Enable word wrapping for long lines
        )

        # Display in a panel
        self.console.print(
            Panel(
                syntax,
                title=f"[cyan]{title}[/cyan]",
                border_style="cyan",
                expand=False,  # Don't expand to full width unnecessarily
            )
        )

    def _save_response_to_file(self, response_data: BaseModel, prefix: str = "response") -> Path:
        """
        Save response data to a JSON file without any truncation.

        Args:
            response_data: The BaseModel data to save
            prefix: Prefix for the filename

        Returns:
            Path to the saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}.json"
        filepath = self.output_dir / filename

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w") as f:
            # Write Pydantic model as JSON
            f.write(response_data.model_dump_json(indent=2))

        self.console.print(f"[green]✓ Response saved to: {filepath}[/green]")
        return filepath

    def _save_text_to_file(self, text: str, prefix: str = "text") -> Path:
        """
        Save text content to a file.

        Args:
            text: The text to save
            prefix: Prefix for the filename

        Returns:
            Path to the saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}.txt"
        filepath = self.output_dir / filename

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w") as f:
            f.write(text)

        self.console.print(f"[green]✓ Text saved to: {filepath}[/green]")
        return filepath

    def _view_saved_responses(self):
        """View saved response files in the output directory."""
        response_files = sorted(self.output_dir.glob("*.json"), reverse=True)

        if not response_files:
            self.console.print("[yellow]No saved responses found[/yellow]")
            return

        self.console.print("\n[cyan]Saved Response Files:[/cyan]")
        for idx, file in enumerate(response_files[:10], 1):
            self.console.print(f"{idx}. {file.name}")

        if len(response_files) > 10:
            self.console.print(f"[dim]... and {len(response_files) - 10} more files[/dim]")

        file_num = IntPrompt.ask("Select file to view (0 to cancel)", default=0)
        if file_num > 0 and file_num <= len(response_files):
            selected_file = response_files[file_num - 1]
            with open(selected_file, "r") as f:
                data = json.load(f)

            self.console.print(f"\n[green]Contents of {selected_file.name}:[/green]")
            # Display the full JSON without truncation
            from rich.syntax import Syntax

            json_str = json.dumps(data, indent=2, default=str)
            syntax = Syntax(json_str, "json", theme="monokai", line_numbers=True)
            self.console.print(syntax)

            # Offer to export to a separate file
            from rich.prompt import Confirm, Prompt

            if Confirm.ask("\nSave to a new file for external viewing?", default=False):
                new_filename = Prompt.ask("Enter filename", default=f"export_{selected_file.stem}.json")
                export_path = Path(new_filename)
                with open(export_path, "w") as f:
                    json.dump(data, f, indent=2, default=str)
                self.console.print(f"[green]✓ Exported to {export_path}[/green]")

    def _load_prompt_template(self) -> str:
        """Load saved prompt or use default."""
        template_file = self.prompt_dir / f"{self.prompt_name}.txt"

        with open(template_file, "r") as f:
            return f.read()

    def _save_prompt_template(self, template: str):
        """Save refined prompt."""
        self.prompt_dir.mkdir(parents=True, exist_ok=True)

        # Save current version
        template_file = self.prompt_dir / f"{self.prompt_name}.txt"
        with open(template_file, "w") as f:
            f.write(template)

        # Save timestamped version for history
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        history_file = self.prompt_dir / f"{self.prompt_name}_{timestamp}.txt"
        with open(history_file, "w") as f:
            f.write(template)

        self.console.print(f"[green]✓ Saved prompt template to {template_file}[/green]")

    def _view_prompt(self):
        """Display current prompt template."""
        from rich.text import Text

        self.console.print("\n[yellow]Current Prompt Template:[/yellow]")

        # Use Text object with word wrapping
        text_obj = Text(self.prompt_template, overflow="fold")
        self.console.print(
            Panel(
                text_obj,
                expand=False,  # Don't expand to full width unnecessarily
                border_style="yellow",
            )
        )

        # Show placeholders if any
        placeholders = self._get_prompt_placeholders()
        if placeholders:
            self.console.print("\n[cyan]Available placeholders:[/cyan]")
            for p in placeholders:
                self.console.print(f"  • {p}")

    def _get_prompt_placeholders(self) -> List[str]:
        """
        Get list of placeholders in the prompt.
        Override to provide specific placeholders.
        """

        # Find all {placeholder} patterns
        return list(set(re.findall(r"\{(\w+)\}", self.prompt_template)))

    def _validate_placeholders(self, prompt: str, values: Optional[Dict[str, Any]] = None) -> tuple[bool, List[str]]:
        """
        Validate that all placeholders in the prompt have values.

        Args:
            prompt: The prompt template to validate
            values: Dictionary of placeholder values (optional)

        Returns:
            Tuple of (is_valid, missing_placeholders)
        """
        import re

        placeholders = set(re.findall(r"\{(\w+)\}", prompt))

        if not placeholders:
            return True, []

        if values is None:
            # If no values provided, all placeholders are missing
            return False, list(placeholders)

        missing = [p for p in placeholders if p not in values]
        return len(missing) == 0, missing

    def _fill_template(self, prompt: str, values: Dict[str, Any]) -> str:
        """
        Fill prompt template with values, with validation.

        Args:
            prompt: The prompt template
            values: Dictionary of placeholder values

        Returns:
            Filled prompt string

        Raises:
            ValueError: If required placeholders are missing
        """
        is_valid, missing = self._validate_placeholders(prompt, values)

        if not is_valid:
            raise ValueError(
                f"Missing required placeholders: {', '.join(missing)}. "
                f"Available values: {', '.join(values.keys()) if values else 'none'}"
            )

        return prompt.format(**values)

    def _display_alignment_result(self, result: AlignmentResult):
        """Display alignment results with detailed information."""
        table = Table(title="Alignment Result")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Iterations Used", str(result.iterations_used))
        table.add_row("Final Score", f"{result.final_score:.2%}")
        table.add_row("Principles Applied", str(len(result.principles_applied)))

        self.console.print(table)

        # Show evolution with more details
        if result.evolution_history:
            self.console.print("\n[yellow]Score Evolution:[/yellow]")
            for evo in result.evolution_history:
                self.console.print(f"  Iteration {evo.iteration}: {evo.score:.2%}")
                if hasattr(evo, "feedback") and evo.feedback:
                    self.console.print(f"    [dim]Feedback: {evo.feedback[:100]}...[/dim]")

        # Show applied principles
        if result.principles_applied:
            self.console.print("\n[cyan]Applied Principles:[/cyan]")
            for idx, principle in enumerate(result.principles_applied[:5], 1):
                if hasattr(principle, "principle"):
                    self.console.print(f"  {idx}. {principle.principle}")
                else:
                    self.console.print(f"  {idx}. {str(principle)[:100]}...")

            if len(result.principles_applied) > 5:
                self.console.print(f"  [dim]... and {len(result.principles_applied) - 5} more[/dim]")
        else:
            self.console.print("\n[yellow]No principles were applied (prompt may already be optimal)[/yellow]")

        # Show final feedback if available (check using hasattr only)
        if hasattr(result, "final_feedback"):
            feedback = getattr(result, "final_feedback", None)
            if feedback:
                self.console.print("\n[bold]Final Assessment:[/bold]")
                self.console.print(Panel(str(feedback), border_style="green", expand=False))

    def _manual_edit(self):
        """Manually edit the prompt."""
        self.console.print("\n[cyan]Enter new prompt (Ctrl+D when done):[/cyan]")

        lines = []
        try:
            while True:
                lines.append(input())
        except EOFError:
            pass

        new_template = "\n".join(lines)

        if new_template.strip():
            self.prompt_template = new_template
            self.console.print("\n[green]✓ Prompt updated[/green]")

    def _display_text_with_syntax(
        self,
        text: str,
        title: str = "Content",
        language: str = "text",
        line_numbers: bool = True,
        offer_save: bool = True,
        save_prefix: str = "content",
    ):
        """
        Display text with syntax highlighting in a panel.

        Args:
            text: The text to display
            title: Title for the panel
            language: Language for syntax highlighting
            line_numbers: Whether to show line numbers
            offer_save: Whether to offer saving the content
            save_prefix: Prefix for saved file
        """
        syntax = Syntax(
            text,
            language,
            theme="monokai",
            line_numbers=line_numbers,
            word_wrap=True,  # Enable word wrapping for long lines
        )
        self.console.print(
            Panel(
                syntax,
                title=f"[bold]{title}[/bold]",
                border_style="blue",
                expand=False,  # Don't expand to full width unnecessarily
            )
        )

        if offer_save and Confirm.ask(f"\nSave {title.lower()} to file?", default=False):
            self._save_text_to_file(text, prefix=save_prefix)

    def _display_content_preview(self, content: str, title: str, max_chars: int = 2000) -> str:
        """
        Display a preview of content with truncation if needed.

        Args:
            content: The content to preview
            title: Title for display
            max_chars: Maximum characters to show

        Returns:
            The preview text
        """
        from rich.text import Text

        preview = content[:max_chars] if len(content) > max_chars else content
        if len(content) > max_chars:
            preview += f"\n\n[dim]... (truncated from {len(content):,} characters)[/dim]"

        # Use Text object to ensure proper word wrapping
        text_obj = Text(preview, overflow="fold")

        self.console.print(
            Panel(
                text_obj,
                title=f"[bold]{title}[/bold] - {len(content):,} characters",
                border_style="green",
                expand=False,  # Don't expand to full width
            )
        )
        return preview

    def _compare_prompts(self, prompt1: str, prompt2: str):
        """Compare two prompts side by side."""
        from rich.columns import Columns
        from rich.text import Text

        col1 = Panel(
            Text(prompt1, style="yellow", overflow="fold"),
            title="Original",
            border_style="yellow",
            expand=False,
        )
        col2 = Panel(
            Text(prompt2, style="green", overflow="fold"),
            title="Refined",
            border_style="green",
            expand=False,
        )

        self.console.print(Columns([col1, col2]))

    async def _principle_based_refinement(self):
        """PRINCIPLE-BASED REFINEMENT through critique and feedback.

        This approach learns from critique of prompts to extract principles:
        1. Evaluate current prompt against target behavior
        2. Extract principles from the critique/feedback
        3. Apply principles systematically to improve the prompt
        4. Build knowledge base of principles over time
        """
        if not self.prompt_aligner:
            self.console.print("[red]Prompt aligner not initialized[/red]")
            return

        self.console.print("\n[bold cyan]═══ PRINCIPLE-BASED REFINEMENT (Critique Method) ═══[/bold cyan]")
        self.console.print("[dim]Learning from critique to extract and apply reusable principles[/dim]\n")

        # Check current principles
        existing_principles = self.export_principles()
        if existing_principles:
            self.console.print(f"[green]✓ {len(existing_principles)} principles in knowledge base[/green]")

            # Offer to view or clear them
            if Confirm.ask("View existing principles?", default=False):
                self.console.print("\n[bold]Current Principles:[/bold]")
                for idx, p in enumerate(existing_principles[:5], 1):
                    if isinstance(p, dict):
                        principle_text = p.get("principle", str(p))
                        importance = p.get("importance", 0)
                        self.console.print(f"  {idx}. {principle_text}")
                        self.console.print(f"      [dim]Importance: {importance:.2f}[/dim]")
                    else:
                        self.console.print(f"  {idx}. {str(p)}")
                if len(existing_principles) > 5:
                    self.console.print(f"  [dim]... and {len(existing_principles) - 5} more principles[/dim]")
        else:
            self.console.print("[yellow]No principles learned yet[/yellow]")
            self.console.print("[dim]Principles will be extracted from critique during refinement[/dim]")

        # Get target behavior for the prompt
        self.console.print("\n[bold]Define Target Behavior[/bold]")
        self.console.print("[dim]Describe what the prompt should achieve[/dim]\n")

        target_behavior = Prompt.ask(
            "What should the refined prompt accomplish?",
            default="Generate accurate, detailed, and well-structured responses",
        )

        # Configure alignment
        config = PromptConfiguration(
            initial_prompt=self.prompt_template,
            target_behavior=target_behavior,
            max_iterations=IntPrompt.ask("Max refinement iterations", default=3),
            score_threshold=0.85,
            preserve_context=True,
        )

        original_prompt = self.prompt_template

        # Apply principle-based alignment
        self.console.print("\n[cyan]Refining prompt through critique and principle extraction...[/cyan]")
        self.console.print("[dim]The system will:[/dim]")
        self.console.print("[dim]1. Evaluate the prompt against target behavior[/dim]")
        self.console.print("[dim]2. Extract principles from the critique[/dim]")
        self.console.print("[dim]3. Apply principles to improve the prompt[/dim]")
        self.console.print("[dim]4. Repeat until optimal or max iterations reached[/dim]\n")

        with self.console.status("Applying principle-based refinement..."):
            result = await self.prompt_aligner.align_prompt(config)

        # Display detailed results
        self._display_alignment_result(result)

        # Show what changed
        if result.aligned_prompt != original_prompt:
            self.console.print("\n[bold cyan]═══ REFINEMENT RESULTS ═══[/bold cyan]")

            # Show comparison
            self.console.print("\n[yellow]Prompt Comparison:[/yellow]")
            self._compare_prompts(original_prompt, result.aligned_prompt)

            # Show detailed diff if requested
            if Confirm.ask("\nView detailed diff?", default=True):
                diff_text = self._show_diff(
                    original_prompt,
                    result.aligned_prompt,
                    "Original",
                    "Refined",
                    context_lines=5,
                )
                from rich.panel import Panel

                self.console.print(
                    Panel(
                        diff_text,
                        title="[bold]Principle-Based Refinement Diff[/bold]",
                        border_style="green",
                        expand=False,
                    )
                )

            # Accept changes
            if Confirm.ask("\nAccept refined prompt?", default=True):
                self.prompt_template = result.aligned_prompt
                self.console.print("[green]✓ Prompt updated with principle-based improvements[/green]")
                self.console.print(f"[dim]Final score: {result.final_score:.2%}[/dim]")

                # Show number of principles now in database
                current_principles = self.export_principles()
                self.console.print(f"[dim]Total principles in knowledge base: {len(current_principles)}[/dim]")
        else:
            self.console.print("\n[yellow]Prompt already optimal according to target behavior[/yellow]")
            self.console.print("[dim]Consider adjusting the target behavior for further refinement[/dim]")

    def _manage_principles(self):
        """View and manage learned principles."""
        if not self.prompt_aligner:
            self.console.print("[red]Prompt aligner not initialized[/red]")
            return

        principles = self.export_principles()

        if not principles:
            self.console.print("[yellow]No principles learned yet.[/yellow]")
            self.console.print("[dim]Use option 4 to learn from good examples first.[/dim]")
            return

        self.console.print(f"\n[cyan]Learned Principles ({len(principles)} total):[/cyan]")

        for idx, principle in enumerate(principles, 1):
            if isinstance(principle, dict):
                p_text = principle.get("principle", str(principle))
                p_type = principle.get("type", "unknown")
                strength = principle.get("strength", 0)

                self.console.print(f"\n{idx}. [bold]{p_text}[/bold]")
                self.console.print(f"   Type: {p_type} | Strength: {strength:.2f}")
            else:
                self.console.print(f"{idx}. {str(principle)}")

        # Offer management options
        self.console.print("\n[yellow]Principle Management Options:[/yellow]")
        self.console.print("1. Export principles to file")
        self.console.print("2. Clear all principles")
        self.console.print("3. Return to main menu")

        choice = Prompt.ask("Select option", default="3")

        if choice == "1":
            filename = Prompt.ask("Enter filename", default="principles.json")
            with open(filename, "w") as f:
                json.dump(principles, f, indent=2)
            self.console.print(f"[green]✓ Exported {len(principles)} principles to {filename}[/green]")
        elif choice == "2":
            if Confirm.ask("Clear all learned principles?", default=False):
                self.import_principles([])
                self.console.print("[yellow]All principles cleared[/yellow]")

    async def _process_all_data(self):
        """
        Process all available data in sequence.
        Override this method to implement specific batch processing logic.
        """
        self.console.print("[yellow]Process all not implemented for this CLI[/yellow]")
        self.console.print("[dim]Override _process_all_data() to implement batch processing[/dim]")

    def _get_data_batches(self) -> List[Any]:
        """
        Get all data batches for processing.
        Override this method to provide specific data batches.

        Returns:
            List of data batches to process
        """
        return []

    def _compare_prompts_with_saved(self):
        """Compare current prompt with the saved version."""
        saved_prompt_file = self.prompt_dir / f"{self.prompt_name}.txt"
        if not saved_prompt_file.exists():
            self.console.print("[yellow]No saved prompt to compare with[/yellow]")
            self.console.print("[dim]Save your prompt first using option 6[/dim]")
            return

        with open(saved_prompt_file, "r") as f:
            saved_prompt = f.read()

        if saved_prompt == self.prompt_template:
            self.console.print("\n[green]Current prompt matches the saved version[/green]")
        else:
            self.console.print("\n[cyan]Comparing Prompts:[/cyan]")
            self._compare_prompts(saved_prompt, self.prompt_template)

    def _show_diff(
        self,
        text1: str,
        text2: str,
        label1: str = "Original",
        label2: str = "Modified",
        context_lines: int = 3,
    ) -> Any:
        """
        Show a unified diff between two texts with color highlighting.

        Args:
            text1: First text to compare
            text2: Second text to compare
            label1: Label for first text
            label2: Label for second text
            context_lines: Number of context lines to show

        Returns:
            The formatted diff Text object
        """
        import difflib

        from rich.text import Text

        # Split texts into lines for comparison
        lines1 = text1.splitlines(keepends=True)
        lines2 = text2.splitlines(keepends=True)

        # Generate unified diff
        diff = difflib.unified_diff(lines1, lines2, fromfile=label1, tofile=label2, n=context_lines, lineterm="")

        # Format diff with colors
        diff_text = Text()
        diff_lines = list(diff)

        if not diff_lines:
            return Text("No differences found", style="green")

        # No size limitation - always show full diff
        for line in diff_lines:
            if line.startswith("+++"):
                diff_text.append(line + "\n", style="bold green")
            elif line.startswith("---"):
                diff_text.append(line + "\n", style="bold red")
            elif line.startswith("@@"):
                diff_text.append(line + "\n", style="bold blue")
            elif line.startswith("+"):
                diff_text.append(line + "\n", style="green")
            elif line.startswith("-"):
                diff_text.append(line + "\n", style="red")
            else:
                diff_text.append(line + "\n", style="dim")

        # No truncation - always show full diff

        return diff_text

    def _show_value_diff(
        self,
        value1: Any,
        value2: Any,
        field_name: str,
        model1_name: str = "Model 1",
        model2_name: str = "Model 2",
    ):
        """
        Show diff between two values from consensus models.

        Args:
            value1: First value
            value2: Second value
            field_name: Name of the field being compared
            model1_name: Name of first model
            model2_name: Name of second model
        """
        import json

        from rich.panel import Panel

        self.console.print(f"\n[bold cyan]Difference in field: {field_name}[/bold cyan]")

        # Convert values to comparable strings
        if isinstance(value1, (list, dict)):
            str1 = json.dumps(value1, indent=2, default=str)
            str2 = json.dumps(value2, indent=2, default=str)
        else:
            str1 = str(value1)
            str2 = str(value2)

        # Show the diff
        diff_text = self._show_diff(str1, str2, model1_name, model2_name)

        self.console.print(
            Panel(
                diff_text,
                title=f"[bold]Diff: {field_name}[/bold]",
                border_style="yellow",
                expand=False,
            )
        )

    async def run(self):
        """Main CLI loop."""
        self.console.print(
            Panel.fit(
                f"[bold cyan]PRINCIPLE-BASED Prompt Refinement[/bold cyan]\n"
                f"[dim]Using Google's Guidelines Method[/dim]\n"
                f"Prompt: {self.prompt_name}\n"
                f"Directory: {self.prompt_dir}",
                title="Principle-Based Refinement (Guidelines Only)",
            )
        )

        while True:
            self.console.print("\n[bold cyan]Options:[/bold cyan]")
            self.console.print("1. View current prompt")
            self.console.print("2. Test current prompt")
            self.console.print("3. Refine prompt")
            self.console.print("4. Manually edit prompt")
            self.console.print("5. Save prompt")
            self.console.print("6. Load saved prompt")
            self.console.print("7. Compare prompts (current vs saved)")
            self.console.print("8. Process all data")
            self.console.print("m. Manage principles")
            self.console.print("v. View saved responses")
            self.console.print("0. Exit")

            choice = Prompt.ask("Select option", default="2")

            try:
                if choice == "1":
                    self._view_prompt()
                elif choice == "2":
                    results = await self._test_prompt(self.prompt_template)
                    self._display_test_results(results)
                elif choice == "3":
                    await self._principle_based_refinement()
                elif choice == "4":
                    self._manual_edit()
                elif choice == "5":
                    self._save_prompt_template(self.prompt_template)
                elif choice == "6":
                    self.prompt_template = self._load_prompt_template()
                    self.console.print("[green]✓ Loaded saved prompt[/green]")
                elif choice == "7":
                    self._compare_prompts_with_saved()
                elif choice == "8":
                    await self._process_all_data()
                elif choice.lower() == "m":
                    self._manage_principles()
                elif choice.lower() == "v":
                    self._view_saved_responses()
                elif choice == "0":
                    break
                else:
                    self.console.print("[red]Invalid option[/red]")
            except Exception as e:
                self.console.print(f"[red]Error: {e}[/red]")
                import traceback

                if Confirm.ask("Show traceback?", default=False):
                    traceback.print_exc()

    def export_principles(self) -> List[Dict]:
        """Export learned principles."""
        if self.prompt_aligner:
            return self.prompt_aligner.export_principles()
        return []

    def import_principles(self, principles_data: List[Dict]):
        """Import principles from another source."""
        if self.prompt_aligner:
            self.prompt_aligner.import_principles(principles_data)
            self.console.print(f"[green]✓ Imported {len(principles_data)} principles[/green]")
