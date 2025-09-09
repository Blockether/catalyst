"""
Pretty printing utilities for consensus rounds with tables and similarity scores.

This module provides beautiful, informative table-based logging for consensus rounds,
showing model responses, similarity scores, and field-by-field comparisons.
"""

import json
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel

# Try importing rich library with proper error handling
try:
    from rich import box
    from rich.columns import Columns
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

from .ConsensusTypes import (
    ConsensusRound,
    DisagreementAnalysis,
    ModelResponse,
    VerbosityLevel,
)


class ConsensusPrettyPrinter:
    """Pretty printer for consensus rounds with rich formatting."""

    @staticmethod
    def print_round_summary(
        round_num: int,
        round_data: ConsensusRound,
        query: str,
        verbosity: VerbosityLevel = VerbosityLevel.NORMAL,
        max_fields_normal: int = 4,
        prompts: Optional[Dict[str, str]] = None,
    ) -> None:
        """Print a beautiful summary of a consensus round.

        Args:
            round_num: The round number
            round_data: The consensus round data
            query: The original query
            verbosity: Verbosity level for field display
            max_fields_normal: Maximum fields to show in NORMAL mode
            prompts: Optional dict mapping model IDs to their specific prompts
        """
        if not HAS_RICH:
            # Simple fallback when rich is not installed
            print(f"\n{'='*60}")
            print(f"Consensus Round {round_num}")
            print(f"{'='*60}")
            print(f"Query: {query[:100]}...")
            print(f"Consensus Achieved: {'✓' if round_data.consensus_achieved else '✗'}")
            print(f"Number of Models: {len(round_data.responses)}")
            for response in round_data.responses:
                print(f"\nModel {response.id}:")
                if hasattr(response.content, 'model_dump'):
                    content = response.content.model_dump()
                    for key, value in list(content.items())[:3]:  # Show first 3 fields
                        print(f"  {key}: {str(value)[:100]}")
                else:
                    print(f"  {str(response.content)[:200]}")
            print(f"{'='*60}\n")
            return
        
        # Print round header
        console = Console()
        console.print()
        console.print(
            Panel(
                f"[bold yellow]🔄 Consensus Round {round_num}[/bold yellow]",
                style="bold blue",
                box=box.DOUBLE,
            )
        )

        # Print the original query/prompt
        console.print(
            Panel(
                f"[bold]Query:[/bold]\n{query}",
                title="💭 Input",
                border_style="cyan",
                box=box.ROUNDED,
                padding=(1, 2),
            )
        )

        # If we have model-specific prompts, show them
        if prompts and verbosity >= VerbosityLevel.VERBOSE:
            console.print("\n[bold]Model-Specific Prompts:[/bold]")
            for model_id, prompt in prompts.items():
                # Show first 200 chars of prompt if too long
                prompt_display = prompt[:200] + "..." if len(prompt) > 200 else prompt
                console.print(
                    Panel(
                        prompt_display,
                        title=f"📨 Prompt for {model_id}",
                        border_style="dim",
                        box=box.SIMPLE,
                        padding=(0, 1),
                    )
                )

        # Calculate vote groups to determine consensus status
        vote_groups = ConsensusPrettyPrinter._get_vote_groups(round_data.responses)
        total_models = len(round_data.responses)

        # Print each model's response in its own panel
        console.print("\n[bold]Model Responses:[/bold]\n")

        for response in round_data.responses:
            model_id = response.id
            vote_hash = ConsensusPrettyPrinter._get_vote_hash(response)

            # Determine consensus status for this model
            group_size = vote_groups.get(vote_hash, 1)
            consensus_status = ConsensusPrettyPrinter._format_consensus_status(group_size, total_models)

            # Get response details
            content_dict = response.content.model_dump()
            reasoning = content_dict.pop("reasoning", None)

            # Create response details text
            response_text = ""

            # Add main fields
            field_count = 0
            for key, value in content_dict.items():
                field_count += 1

                # In NORMAL mode, limit fields shown
                if verbosity == VerbosityLevel.NORMAL and field_count > max_fields_normal:
                    remaining = len(content_dict) - max_fields_normal
                    if remaining > 0:
                        response_text += f"\n[dim]... and {remaining} more fields[/dim]"
                    break

                # Format value based on type and verbosity
                if isinstance(value, (list, dict)):
                    if verbosity == VerbosityLevel.VERBOSE:
                        # Show full content in verbose mode
                        if isinstance(value, list):
                            # Show list items with better formatting
                            value_str = f"[{len(value)} items]\n"
                            for i, item in enumerate(value[:10]):  # Show first 10 items
                                item_str = json.dumps(item, indent=2) if not isinstance(item, str) else item
                                # Truncate long items
                                if len(item_str) > 200:
                                    item_str = item_str[:197] + "..."
                                value_str += f"    [{i}]: {item_str}\n"
                            if len(value) > 10:
                                value_str += f"    ... and {len(value) - 10} more items"
                        else:
                            # Show dict with proper formatting
                            value_str = json.dumps(value, indent=2)
                            if len(value_str) > 500:
                                value_str = value_str[:497] + "..."
                    else:
                        # Normal mode - just show count
                        if isinstance(value, list):
                            value_str = f"[{len(value)} items]"
                        else:
                            value_str = f"{{{len(value)} fields}}"
                elif isinstance(value, str):
                    # In verbose mode, show more of string values
                    if verbosity == VerbosityLevel.VERBOSE:
                        value_str = f'"{value}"' if len(value) < 200 else f'"{value[:197]}..."'
                    else:
                        value_str = f'"{value}"' if len(value) < 50 else f'"{value[:47]}..."'
                else:
                    value_str = json.dumps(value)

                response_text += f"[cyan]{key}:[/cyan] {value_str}\n"

            # Add reasoning if in verbose mode (show full reasoning)
            if verbosity == VerbosityLevel.VERBOSE and reasoning:
                response_text += f"\n[dim]Reasoning:[/dim]\n{reasoning}"

            # Create panel title with metadata
            title = f"🤖 {model_id} | Vote: {vote_hash[:8]} | {consensus_status}"

            # Determine border color based on consensus status
            if "Majority" in consensus_status or "Consensus" in consensus_status:
                border_style = "green"
            elif "Minority" in consensus_status:
                border_style = "yellow"
            else:  # Outlier
                border_style = "red"

            console.print(
                Panel(
                    response_text.strip(),
                    title=title,
                    border_style=border_style,
                    box=box.ROUNDED,
                    padding=(0, 1),
                )
            )

        # Print disagreement analysis if available
        if round_data.disagreement_analysis:
            ConsensusPrettyPrinter._print_disagreement_analysis(round_data.disagreement_analysis, verbosity)

    @staticmethod
    def print_consensus_achieved(
        round_num: int,
        num_responses: int,
        duration_ms: Optional[float] = None,
    ) -> None:
        """Print a success message for consensus achievement.

        Args:
            round_num: The round number where consensus was achieved
            num_responses: Number of model responses
            duration_ms: Optional duration in milliseconds
        """
        message = "✅ [bold green]Consensus Achieved![/bold green]\n"
        message += f"Round: {round_num} | Models: {num_responses}"

        if duration_ms:
            message += f" | Duration: {duration_ms:.1f}ms"

        panel = Panel(
            message,
            title="🎯 Success",
            border_style="green",
            box=box.ROUNDED,
        )
        console = Console()
        console.print(panel)

    @staticmethod
    def print_round_comparison(
        rounds: List[ConsensusRound],
        verbosity: VerbosityLevel = VerbosityLevel.NORMAL,
    ) -> None:
        """Print a comparison table showing evolution across rounds.

        Args:
            rounds: List of consensus rounds
            verbosity: Verbosity level for display
        """
        if len(rounds) < 2:
            return

        table = Table(
            title="📊 Consensus Evolution Across Rounds",
            box=box.SIMPLE_HEAD,
            show_header=True,
            header_style="bold magenta",
        )

        # Add columns
        table.add_column("Round", style="bold", width=8)
        table.add_column("Models", style="cyan", width=8)
        table.add_column("Unique Votes", style="yellow", width=12)
        table.add_column("Agreement %", style="green", width=12)
        table.add_column("Consensus Fields", style="blue", width=30)
        table.add_column("Disputed Fields", style="red", width=30)

        for round_data in rounds:
            # Calculate metrics
            num_models = len(round_data.responses)
            unique_votes = len(ConsensusPrettyPrinter._get_unique_votes(round_data.responses))
            agreement_pct = ((num_models - unique_votes + 1) / num_models * 100) if num_models > 0 else 0

            # Get field info
            consensus_fields = []
            disputed_fields = []

            if round_data.disagreement_analysis:
                consensus_fields = round_data.disagreement_analysis.consensus_fields[:3]
                disputed_fields = list(round_data.disagreement_analysis.disagreement_fields.keys())[:3]

            table.add_row(
                str(round_data.round_number),
                str(num_models),
                str(unique_votes),
                f"{agreement_pct:.1f}%",
                ", ".join(consensus_fields) if consensus_fields else "—",
                ", ".join(disputed_fields) if disputed_fields else "—",
            )

        console = Console()
        console.print(table)

    @staticmethod
    def _calculate_similarity_scores(
        responses: List[ModelResponse],
    ) -> Dict[str, float]:
        """Calculate average similarity scores for each model response.

        Args:
            responses: List of model responses

        Returns:
            Dictionary mapping model ID to average similarity score
        """
        scores = {}

        for i, response1 in enumerate(responses):
            similarities = []
            for j, response2 in enumerate(responses):
                if i != j:
                    similarity = ConsensusPrettyPrinter._calculate_response_similarity(response1, response2)
                    similarities.append(similarity)

            if similarities:
                scores[response1.id] = sum(similarities) / len(similarities)
            else:
                scores[response1.id] = 1.0

        return scores

    @staticmethod
    def _calculate_response_similarity(response1: ModelResponse, response2: ModelResponse) -> float:
        """Calculate similarity between two responses.

        Args:
            response1: First response
            response2: Second response

        Returns:
            Similarity score between 0 and 1
        """
        from .ConsensusTypes import ComparisonStrategy

        # Get response content as dicts
        dict1 = response1.content.model_dump()
        dict2 = response2.content.model_dump()

        # Get the model class to check field metadata
        model_class = response1.content.__class__

        # Count matching fields (excluding fields with IGNORE strategy)
        matching_fields = 0
        total_fields = 0

        for field_name in dict1:
            # Check if field should be ignored based on VotingField metadata
            field_info = model_class.model_fields.get(field_name)

            # Check if field has IGNORE strategy
            if field_info and field_info.json_schema_extra:
                extra = field_info.json_schema_extra
                if isinstance(extra, dict) and "voting_comparison" in extra:
                    voting_comparison = extra["voting_comparison"]
                    if isinstance(voting_comparison, dict):
                        strategy = voting_comparison.get("strategy")
                        # Check both enum and string values
                        if (
                            (isinstance(strategy, ComparisonStrategy) and strategy == ComparisonStrategy.IGNORE)
                            or (isinstance(strategy, str) and strategy == "ignore")
                            or strategy == "IGNORE"
                        ):
                            continue  # Skip ignored fields

            total_fields += 1
            if field_name in dict2 and dict1[field_name] == dict2[field_name]:
                matching_fields += 1

        return matching_fields / total_fields if total_fields > 0 else 0.0

    @staticmethod
    def _get_vote_hash(response: ModelResponse) -> str:
        """Get the voting hash for a response.

        Args:
            response: Model response

        Returns:
            Vote hash string
        """
        vote_key = response.content.get_voting_key()
        return str(vote_key)

    @staticmethod
    def _get_unique_votes(responses: List[ModelResponse]) -> Set[str]:
        """Get unique vote hashes from responses.

        Args:
            responses: List of model responses

        Returns:
            Set of unique vote hashes
        """
        return {ConsensusPrettyPrinter._get_vote_hash(response) for response in responses}

    @staticmethod
    def _get_vote_groups(responses: List[ModelResponse]) -> Dict[str, int]:
        """Get vote groups showing how many models voted for each option.

        Args:
            responses: List of model responses

        Returns:
            Dictionary mapping vote hash to count of models with that vote
        """
        vote_counts = {}
        for response in responses:
            vote_hash = ConsensusPrettyPrinter._get_vote_hash(response)
            vote_counts[vote_hash] = vote_counts.get(vote_hash, 0) + 1
        return vote_counts

    @staticmethod
    def _format_consensus_status(group_size: int, total_models: int) -> str:
        """Format the consensus status for a model based on its vote group.

        Args:
            group_size: Number of models with the same vote
            total_models: Total number of models

        Returns:
            Formatted consensus status string
        """
        if group_size == total_models:
            # All models agree
            return f"[green]✓ Consensus ({group_size}/{total_models})[/green]"
        elif group_size > total_models / 2:
            # This model is in the majority
            return f"[green]✓ Majority ({group_size}/{total_models})[/green]"
        elif group_size > 1:
            # This model has some agreement but not majority
            return f"[yellow]⚠ Minority ({group_size}/{total_models})[/yellow]"
        else:
            # This model is alone
            return "[red]✘ Outlier[/red]"

    @staticmethod
    def _format_similarity(score: float) -> str:
        """Format a similarity score with color coding.

        Args:
            score: Similarity score between 0 and 1

        Returns:
            Formatted string with color
        """
        percentage = score * 100
        if percentage >= 80:
            return f"[green]{percentage:.1f}%[/green]"
        elif percentage >= 60:
            return f"[yellow]{percentage:.1f}%[/yellow]"
        else:
            return f"[red]{percentage:.1f}%[/red]"

    @staticmethod
    def _get_response_summary(
        response: ModelResponse,
        verbosity: VerbosityLevel,
        max_fields: int = 4,
    ) -> str:
        """Get a summary of the response based on verbosity.

        Args:
            response: Model response
            verbosity: Verbosity level
            max_fields: Maximum fields to show in NORMAL mode

        Returns:
            Formatted response summary
        """
        content_dict = response.content.model_dump()

        # Remove reasoning field as it's usually too long
        content_dict.pop("reasoning", None)

        if verbosity == VerbosityLevel.VERBOSE:
            # Show all fields
            return ConsensusPrettyPrinter._format_dict_compact(content_dict)
        else:
            # Show only first few fields
            limited_dict = dict(list(content_dict.items())[:max_fields])
            if len(content_dict) > max_fields:
                limited_dict["..."] = f"({len(content_dict) - max_fields} more fields)"
            return ConsensusPrettyPrinter._format_dict_compact(limited_dict)

    @staticmethod
    def _format_dict_compact(d: Dict[str, Any]) -> str:
        """Format a dictionary in a compact way for display.

        Args:
            d: Dictionary to format

        Returns:
            Compact string representation
        """
        items = []
        for key, value in d.items():
            if isinstance(value, (list, dict)):
                if isinstance(value, list):
                    value_str = f"[{len(value)} items]"
                else:
                    value_str = f"{{{len(value)} fields}}"
            elif isinstance(value, str) and len(value) > 30:
                value_str = f'"{value[:27]}..."'
            else:
                value_str = json.dumps(value)
            items.append(f"{key}: {value_str}")

        return "\n".join(items)

    @staticmethod
    def _print_disagreement_analysis(
        analysis: DisagreementAnalysis,
        verbosity: VerbosityLevel,
    ) -> None:
        """Print disagreement analysis in a nice format.

        Args:
            analysis: Disagreement analysis
            verbosity: Verbosity level
        """
        if not analysis.disagreement_fields:
            return

        console = Console()

        # Create a table for better visibility
        table = Table(
            title="⚠️  Field Disagreements",
            box=box.DOUBLE_EDGE,
            show_header=True,
            header_style="bold yellow",
            border_style="yellow",
            title_style="bold red",
            expand=True,
        )

        # Add columns
        table.add_column("Field", style="bold cyan", width=20)
        table.add_column("Different Values", style="bold magenta", width=15, justify="center")
        table.add_column("Value Examples", style="white", overflow="fold")

        max_fields = 10 if verbosity == VerbosityLevel.VERBOSE else 5

        for field, values in list(analysis.disagreement_fields.items())[:max_fields]:
            unique_values = list(set(str(v) for v in values))
            num_different = len(unique_values)

            # Format value examples
            if num_different <= 3:
                # Show all unique values if there are 3 or fewer
                value_examples = []
                for v_str in unique_values:
                    if len(v_str) > 50:
                        # For long values, show on separate lines with better truncation
                        value_examples.append(f"• {v_str[:47]}...")
                    else:
                        value_examples.append(f"• {v_str}")
                value_display = "\n".join(value_examples)
            else:
                # Show first 2 examples if there are many different values
                examples = []
                for v_str in unique_values[:2]:
                    if len(v_str) > 40:
                        examples.append(f"• {v_str[:37]}...")
                    else:
                        examples.append(f"• {v_str}")
                examples.append(f"[dim]... and {num_different - 2} more variations[/dim]")
                value_display = "\n".join(examples)

            # Add row to table with colored indicators
            if num_different == 2:
                num_display = f"[yellow]{num_different}[/yellow]"
            elif num_different == 3:
                num_display = f"[orange3]{num_different}[/orange3]"
            else:
                num_display = f"[red]{num_different}[/red]"

            table.add_row(field, num_display, value_display)

        if len(analysis.disagreement_fields) > max_fields:
            remaining = len(analysis.disagreement_fields) - max_fields
            table.add_row(
                "[dim]...[/dim]",
                "[dim]...[/dim]",
                f"[dim italic]({remaining} more fields with disagreements)[/dim italic]",
            )

        # Print the table
        console.print()
        console.print(table)

        # Add a summary panel if there are many disagreements
        if len(analysis.disagreement_fields) > 3:
            summary_text = (
                f"[bold yellow]⚠️  High Disagreement Detected[/bold yellow]\n"
                f"[white]Total fields with disagreements: [bold red]{len(analysis.disagreement_fields)}[/bold red][/white]\n"
            )

            if verbosity == VerbosityLevel.NORMAL:
                summary_text += "[dim]Use verbose mode to see more details[/dim]"

            console.print(
                Panel(
                    summary_text,
                    border_style="red",
                    box=box.HEAVY,
                    padding=(0, 1),
                )
            )
