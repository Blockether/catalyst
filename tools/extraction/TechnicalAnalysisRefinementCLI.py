#!/usr/bin/env python3
"""
Technical Analysis Prompt Refinement CLI.

Phase 1 of document analysis - focuses on extracting:
- Document sections with types (concept, procedure, requirement, example, data, narrative)
- Terms and acronyms
- Summaries and key points

All types are defined in this file for simplicity.
Prompts are loaded from tools/extraction/prompts/technical_analysis.txt
"""

import sys
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import anyio
from pydantic import BaseModel
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.table import Table

from com_blockether_catalyst.consensus import ConsensusCore, TypedCallBaseForConsensus
from com_blockether_catalyst.consensus.internal.Consensus import Consensus
from com_blockether_catalyst.consensus.internal.ConsensusTypes import (
    ConsensusSettings,
    VerbosityLevel,
)
from com_blockether_catalyst.consensus.internal.VotingComparison import (
    BaseModelWithReasoning,
    ComparisonStrategy,
    VotingField,
)
from com_blockether_catalyst.knowledge.internal import PDFKnowledgeExtractor
from com_blockether_catalyst.knowledge.internal.KnowledgeExtractionTypes import (
    KnowledgePageData,
)
from com_blockether_catalyst.knowledge.internal.PDKnowledgeExtractorTypes import PDFKnowledgeProcessorSettings, PDFPageCropOffset
from com_blockether_catalyst.prompt import PromptAlignmentCore
from com_blockether_catalyst.prompt.PromptAlignmentCLIBase import PromptAlignmentCLIBase
from com_blockether_catalyst.prompt.internal.PromptAlignmentTypes import (
    AlignmentFeedback,
    EvaluationResult,
)
from com_blockether_catalyst.utils.instructor.InstructorLLMCall import InstructorLLMCall

console = Console()


# ============================================================================
# TECHNICAL ANALYSIS TYPES
# ============================================================================

class SectionType(str, Enum):
    """Document section types - simplified for LLM recognition."""
    CONCEPT = "concept"            # Definitions, terminology, what something is
    PROCEDURE = "procedure"        # How to do something, steps, instructions
    REQUIREMENT = "requirement"    # Rules, policies, constraints, must/must not
    EXAMPLE = "example"            # Sample usage, case studies, scenarios
    DATA = "data"                  # Tables, metrics, statistics, lists
    NARRATIVE = "narrative"        # General descriptive text, explanations


class Section(BaseModelWithReasoning):
    """A section the LLM found in the batch."""
    title: str = VotingField(description="Section title", comparison=ComparisonStrategy.EXACT)
    start_page: int = VotingField(description="Starting page number", comparison=ComparisonStrategy.EXACT)
    end_page: int = VotingField(description="Ending page number", comparison=ComparisonStrategy.EXACT)
    level: int = VotingField(description="Heading level (1=main, 2=sub, etc.)", comparison=ComparisonStrategy.EXACT)
    text: str = VotingField(description="Full text of section", comparison=ComparisonStrategy.EXACT)
    summary: str = VotingField(description="Brief summary of section", comparison=ComparisonStrategy.EXACT)
    type: SectionType = VotingField(
        description="Type: concept, procedure, requirement, example, data, or narrative",
        comparison=ComparisonStrategy.EXACT
    )

class TermType(str, Enum):
    """Type of term identified."""
    ACRONYM = "acronym"
    KEYWORD = "keyword"


class TermCandidate(BaseModelWithReasoning):
    """A term/acronym candidate the LLM found."""
    term: str = VotingField(description="The keyword or acronym text", comparison=ComparisonStrategy.EXACT)
    type: TermType = VotingField(description="Type: 'acronym' or 'keyword'", comparison=ComparisonStrategy.EXACT)
    full_form: Optional[str] = VotingField(default=None, description="Full form if acronym", comparison=ComparisonStrategy.EXACT)
    definition: Optional[str] = VotingField(comparison=ComparisonStrategy.EXACT,
                                            default=None, description="Brief definition of a term")
    page_found: int = VotingField(comparison=ComparisonStrategy.EXACT, description="Page where found")


class Concept(BaseModelWithReasoning):
    """A key concept or idea the LLM found."""
    concept: str = VotingField(
        description="The concept text", comparison=ComparisonStrategy.EXACT,
        min_length=150
    )
    context: Optional[str] = VotingField(
        default=None, description="Context or explanation", comparison=ComparisonStrategy.SEMANTIC,
        min_length=20, threshold=0.7
    )
    page_found: int = VotingField(
        description="Page where found", comparison=ComparisonStrategy.EXACT
    )


class TechnicalBatchResponse(TypedCallBaseForConsensus):
    """What LLM returns for technical analysis - simple lists!"""

    sections: List[Section] = VotingField(
        default_factory=list,
        comparison=ComparisonStrategy.SEQUENCE_ORDERED_DERIVED,
        description="Sections found in this batch"
    )

    terms: List[TermCandidate] = VotingField(
        default_factory=list,
        comparison=ComparisonStrategy.SEQUENCE_UNORDERED_ALIKE,
        description="Terms and acronyms found"
    )

    summary: str = VotingField(
        min_length=150,
        threshold=0.8,
        comparison=ComparisonStrategy.SEMANTIC,
        description="Brief paragraph summarizing what this batch covers"
    )

    concepts: List[Concept] = VotingField(
        default_factory=list,
        comparison=ComparisonStrategy.SEQUENCE_UNORDERED_ALIKE,
        description="Key concepts and ideas extracted from this batch"
    )


# ============================================================================
# CLI IMPLEMENTATION
# ============================================================================

# 1. We have technical analysis of the content (sections, concepts, summary, terms etc.)
#.   Remove decorational elements like headers, footers, watermarks
# 2. Now we need to go over the sections and create a table of content with explanation of each section and keywords to section + keywords for the full document.. (heuristics)
# 3. We need to build a higher overview of the document - author, name, purpose, audience,
# concepts, like summary of the book (detailed.. On this page there this one, and on the other one there is somethng else..) (heuristics + LLM)
# 4. For each section generate the questions and answers then put them in the vector store..
     # Answer: terms, reasoning, text, references: page, section title, document i author (LLM)

class TechnicalAnalysisRefinementCLI(PromptAlignmentCLIBase):
    """CLI for refining technical analysis prompts."""

    def __init__(
        self,
        pdf_path: Path,
        batch_size: int = 2,
        api_base_url: str = "http://localhost:3005/v1",
        api_key: str = "test-key"
    ):
        """
        Initialize the technical analysis CLI.

        Args:
            pdf_path: Path to PDF for testing
            batch_size: Pages per batch
            api_base_url: Base URL for the Instructor API
            api_key: API key for the Instructor API
        """
        self.pdf_path = pdf_path
        self.batch_size = batch_size
        self.api_base_url = api_base_url
        self.api_key = api_key
        self.pages: List[KnowledgePageData] = []
        self.current_batch_index = 0

        # Initialize PDF extractor
        self.pdf_extractor = PDFKnowledgeExtractor(settings=PDFKnowledgeProcessorSettings(
            pdf_page_crop_offset=PDFPageCropOffset(top=48, bottom=48),
        ))

        super().__init__(
            prompt_name="technical_analysis",
            prompt_dir=Path("tools/extraction/prompts"),
            output_dir=Path("output/technical_analysis_responses"),
            console=console
        )

    def _init_components(self):
        """Initialize LLM components and prompt aligner."""
        # Create model for technical analysis
        self.tech_model: InstructorLLMCall[TechnicalBatchResponse] = InstructorLLMCall(
            response_model=TechnicalBatchResponse,
            model="gpt-4o",
            temperature=0.3,
            base_url=self.api_base_url,
            api_key=self.api_key,
        )

        self.technical_consensus: Consensus[TechnicalBatchResponse] = ConsensusCore[TechnicalBatchResponse].consensus(
            models=[
                ConsensusCore[TechnicalBatchResponse].model(
                    id="tech_analyzer",
                    executor=self.tech_model,
                    perspective="Extract sections, terms, and summaries from document.",
                    weight_multiplier=1.0,
                ),
                ConsensusCore[TechnicalBatchResponse].model(
                    id="tech_analyzer_2",
                    executor=self.tech_model,
                    perspective="Extract sections, terms, and summaries from document. Avoid classification without sufficient evidence.",
                    weight_multiplier=1.0,
                ),
            ],
            judge=self.tech_model,
            settings=ConsensusSettings(max_rounds=3, threshold=0.9, verbosity=VerbosityLevel.VERBOSE),
        )

        # Create alignment models
        eval_model = InstructorLLMCall(
            response_model=EvaluationResult,
            model="gpt-4o",
            temperature=0.3,
            base_url=self.api_base_url,
            api_key=self.api_key,
        )

        feedback_model = InstructorLLMCall(
            response_model=AlignmentFeedback,
            model="gpt-4o",
            temperature=0.3,
            base_url=self.api_base_url,
            api_key=self.api_key,
        )

        # Create consensus for alignment
        target_consensus = ConsensusCore.consensus(
            models=[
                ConsensusCore.model(
                    id="evaluator",
                    executor=eval_model,
                    perspective="Evaluate extraction quality and completeness.",
                    weight_multiplier=1.0,
                ),

                ConsensusCore.model(
                    id="evaluator1",
                    executor=eval_model,
                    perspective="Evaluate extraction quality and completeness. Take into account all of the rules and remember to be concise and strict.",
                    weight_multiplier=1.0,
                ),
            ],
            judge=eval_model,
            settings=ConsensusSettings(max_rounds=2, threshold=0.9, verbosity=VerbosityLevel.VERBOSE),
        )

        alignment_consensus = ConsensusCore.consensus(
            models=[
                ConsensusCore.model(
                    id="aligner",
                    executor=feedback_model,
                    perspective="Provide feedback for improving extraction prompts.",
                    weight_multiplier=1.0,
                ),
            ],
            judge=feedback_model,
            settings=ConsensusSettings(max_rounds=3, threshold=0.85, verbosity=VerbosityLevel.VERBOSE),
        )

        # Initialize prompt aligner
        self.prompt_aligner = PromptAlignmentCore(
            target_consensus=target_consensus,
            alignment_consensus=alignment_consensus,
        )

    def _get_default_prompt(self) -> str:
        """Get the default prompt template from external file."""
        prompt_file = self.prompt_dir / f"{self.prompt_name}.txt"
        if prompt_file.exists():
            with open(prompt_file, "r") as f:
                return f.read()
        raise ValueError(f"Prompt file not found: {prompt_file}")

    def _get_prompt_placeholders(self) -> List[str]:
        """Get list of placeholders in the prompt."""
        return [
            "{document_name}",
            "{batch_number}",
            "{total_batches}",
            "{pages_text}",
            "{previous_sections}",
            "{author}",
        ]

    async def _extract_document(self):
        """Extract PDF document."""
        with self.console.status("Extracting document..."):
            result = self.pdf_extractor.extract(self.pdf_path)
            if result:
                self.pages = list(result.pages)
                self.author: str | None = result.metadata.author
                self.console.print(f"[green]✓ Extracted {len(self.pages)} pages[/green]")

    def _get_current_batch(self) -> List[KnowledgePageData]:
        """Get current batch of pages."""
        if not self.pages:
            return []

        start_idx = self.current_batch_index * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.pages))
        return self.pages[start_idx:end_idx]

    def _fill_prompt(self, batch_pages: List[KnowledgePageData]) -> str:
        """Fill the prompt template with batch data."""
        if not batch_pages:
            return ""

        # Format pages text
        pages_text = "\n\n---PAGE BREAK---\n\n".join(
            f"Page {p.page}:{"\n\n".join(map(lambda t: t.to_ascii_table(), p.tables)) if p.tables else ''}\n{p.text}" for p in batch_pages
        )

        # Calculate totals
        total_batches = (len(self.pages) + self.batch_size - 1) // self.batch_size

        # Fill template
        prompt = self.prompt_template
        prompt = prompt.replace("{document_name}", self.pdf_path.name)
        prompt = prompt.replace("{batch_number}", str(self.current_batch_index + 1))
        prompt = prompt.replace("{total_batches}", str(total_batches))
        prompt = prompt.replace("{pages_text}", pages_text)
        # Default author, can be extracted from PDF metadata later
        prompt = prompt.replace("{author}", self.author or "Unknown")

        return prompt

    def _save_full_response(self, response_data: Dict[str, Any], batch_info: Dict[str, Any]) -> Path:
        """Save the full response to a JSON file without any truncation."""
        full_data = {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "batch_info": batch_info,
            "response": response_data
        }

        # Use base class method with custom prefix
        prefix = f"batch_{batch_info.get('batch_num', 0)}_pages_{batch_info.get('pages', 'unknown')}"
        return self._save_response_to_file(full_data, prefix=prefix)

    async def _test_prompt(self, prompt: str) -> Dict[str, Any]:
        """Test the current prompt on current batch."""
        batch_pages = self._get_current_batch()
        if not batch_pages:
            return {"error": "No pages to test"}

        # Fill and run prompt
        filled_prompt = self._fill_prompt(batch_pages)

        result = await self.technical_consensus.call(filled_prompt)

        if result and result.final_response:
            response = result.final_response

            # Create the response dictionary
            batch_info = {
                "pages": f"{batch_pages[0].page}-{batch_pages[-1].page}",
                "batch_num": self.current_batch_index + 1,
            }

            # Get the full response data
            full_response_data = response.model_dump()

            # Save the full response to file (no truncation)
            saved_file = self._save_full_response(full_response_data, batch_info)

            # Store consensus result for diff viewing
            self.last_consensus_result = result

            return {
                "sections": response.sections,
                "terms": response.terms,
                "summary": response.summary,
                "concepts": response.concepts,
                "batch_info": batch_info,
                "final_response": full_response_data,
                "saved_to": str(saved_file),
                "consensus_result": result
            }

        return {"error": "No response from LLM"}

    def _display_test_results(self, results: Dict[str, Any]):
        """Display test results."""
        if "error" in results:
            self.console.print(f"[red]Error: {results['error']}[/red]")
            return

        # Display batch info
        batch_info = results.get("batch_info", {})
        self.console.print(f"\n[cyan]Batch {batch_info.get('batch_num', '?')} "
                           f"(Pages {batch_info.get('pages', '?')})[/cyan]")

        # Show where the full response was saved
        if "saved_to" in results:
            self.console.print(f"[yellow]Full response saved to: {results['saved_to']}[/yellow]")
            self.console.print("[dim]Use 'v' option to view the full saved file[/dim]")

        # Display raw JSON of final response (this might be truncated in console)
        self._display_raw_json(results["final_response"], "Final Response (Console Preview)")

    def _view_raw_batch_text(self):
        """View the raw text extracted from the current batch of pages."""
        batch_pages = self._get_current_batch()
        if not batch_pages:
            self.console.print("[red]No pages available for current batch[/red]")
            return

        # Display batch info
        self.console.print(f"\n[cyan]Raw Text for Batch {self.current_batch_index + 1}[/cyan]")
        self.console.print(f"[dim]Total pages in batch: {len(batch_pages)}[/dim]\n")

        # Display each page's raw text using base class method
        for page_data in batch_pages:
            self._display_content_preview(
                page_data.text,
                title=f"Page {page_data.page}",
                max_chars=2000
            )

        # Show statistics
        total_chars = sum(len(p.text) for p in batch_pages)
        self.console.print("\n[yellow]Batch Statistics:[/yellow]")
        self.console.print(f"  • Pages: {batch_pages[0].page} to {batch_pages[-1].page}")
        self.console.print(f"  • Total characters: {total_chars:,}")
        self.console.print(f"  • Average chars/page: {total_chars // len(batch_pages):,}")

        # Offer to save the raw text
        if Confirm.ask("\nSave raw text to file?", default=False):
            # Compile all text
            full_text = ""
            for page_data in batch_pages:
                full_text += "="*60 + "\n"
                full_text += f"PAGE {page_data.page}\n"
                full_text += "="*60 + "\n\n"
                full_text += page_data.text
                full_text += "\n\n"

            # Use base class method to save
            self._save_text_to_file(
                full_text,
                prefix=f"raw_text_batch_{self.current_batch_index + 1}"
            )

            # Also offer to view full text in console
            if Confirm.ask("View full text in console (may be long)?", default=False):
                for page_data in batch_pages:
                    self.console.print(f"\n[bold cyan]{'═'*10} Page {page_data.page} {'═'*10}[/bold cyan]\n")
                    self.console.print(page_data.text)

    def _view_filled_prompt(self):
        """View the filled prompt template for the current batch."""
        batch_pages = self._get_current_batch()
        if not batch_pages:
            self.console.print("[red]No pages available for current batch[/red]")
            return

        # Get the filled prompt
        filled_prompt = self._fill_prompt(batch_pages)

        # Display batch info
        self.console.print(f"\n[cyan]Filled Prompt for Batch {self.current_batch_index + 1}[/cyan]")
        self.console.print(f"[dim]Pages: {batch_pages[0].page}-{batch_pages[-1].page}[/dim]\n")

        # Use base class method for display
        self._display_text_with_syntax(
            filled_prompt,
            title="Filled Prompt Template",
            language="markdown",
            line_numbers=True,
            offer_save=True,
            save_prefix=f"filled_prompt_batch_{self.current_batch_index + 1}"
        )

    def _compare_prompts_with_previous(self):
        """Compare current prompt with the previously saved version."""
        # Load the saved prompt
        saved_prompt_file = self.prompt_dir / f"{self.prompt_name}.txt"
        if not saved_prompt_file.exists():
            self.console.print("[yellow]No saved prompt to compare with[/yellow]")
            return

        with open(saved_prompt_file, "r") as f:
            saved_prompt = f.read()

        self.console.print("\n[cyan]Comparing Prompts:[/cyan]")
        self._compare_prompts(saved_prompt, self.prompt_template)

    def _get_ignore_fields(self, response_model) -> set:
        """Get fields that should be ignored in comparisons based on ComparisonStrategy.IGNORE."""
        ignore_fields = set()

        # Check if it's a Pydantic model with model_fields
        if hasattr(response_model.__class__, 'model_fields'):
            for field_name, field_info in response_model.__class__.model_fields.items():
                # Check if field has IGNORE comparison strategy
                if hasattr(field_info, 'metadata'):
                    for meta in field_info.metadata:
                        if hasattr(meta, 'comparison') and meta.comparison == ComparisonStrategy.IGNORE:
                            ignore_fields.add(field_name)

        # No default ignored fields - only use what's explicitly marked with IGNORE strategy
        return ignore_fields

    def _view_consensus_disagreements(self):
        """View detailed diffs of consensus disagreements from the last test."""
        # Check if we have a stored result
        last_result = getattr(self, 'last_consensus_result', None)
        if not last_result:
            self.console.print("[yellow]No consensus result available. Run a test first.[/yellow]")
            return

        result = last_result

        # Properly typed access to rounds
        if not result.rounds:
            self.console.print("[yellow]No rounds information available[/yellow]")
            return

        rounds = result.rounds
        if not rounds:
            self.console.print("[green]No rounds to display[/green]")
            return

        self.console.print("\n[bold cyan]Consensus Disagreement Analysis[/bold cyan]\n")

        # Check each round for disagreements
        any_disagreements = False
        for round_idx, round_data in enumerate(rounds):
            round_num = round_idx + 1

            # Check if this round has responses
            if round_data.responses:
                # Extract unique responses to find disagreements
                responses_list = round_data.responses

                # Group by unique response values for comparison
                unique_responses = {}
                for model_response in responses_list:
                    # Each ModelResponse has id and content attributes (properly typed)
                    model_id = model_response.id
                    response = model_response.content

                    # Create a hash of the response for grouping (responses are Pydantic models)
                    response_str = str(response.model_dump())
                    response_hash = hash(response_str)

                    if response_hash not in unique_responses:
                        unique_responses[response_hash] = {'response': response, 'models': []}
                    unique_responses[response_hash]['models'].append(model_id)

                # If there are disagreements (more than 1 unique response)
                if len(unique_responses) > 1:
                    any_disagreements = True
                    self.console.print(
                        f"\n[yellow]Round {round_num} - {len(unique_responses)} different responses:[/yellow]")

                    # Show the differences between first two unique responses
                    unique_list = list(unique_responses.values())
                    if len(unique_list) >= 2:
                        resp1 = unique_list[0]['response']
                        resp2 = unique_list[1]['response']
                        models1 = ', '.join(unique_list[0]['models'])
                        models2 = ', '.join(unique_list[1]['models'])

                        # Convert responses to comparable format (they are Pydantic models)
                        dict1 = resp1.model_dump()
                        dict2 = resp2.model_dump()

                        # Get fields to ignore based on ComparisonStrategy.IGNORE
                        ignore_fields = self._get_ignore_fields(resp1)

                        # Find all differing fields
                        all_differing = []
                        ignored_differing = []
                        for field in dict1.keys():
                            if field in dict2 and dict1[field] != dict2[field]:
                                if field in ignore_fields:
                                    ignored_differing.append(field)
                                else:
                                    all_differing.append(field)

                        # Show note about ignored fields if any
                        if ignored_differing:
                            self.console.print(
                                f"  [dim]Ignoring fields with IGNORE strategy: {', '.join(ignored_differing)}[/dim]")

                        if all_differing:
                            self.console.print(
                                f"  [yellow]Significant disagreements in: {', '.join(all_differing)}[/yellow]")

                            # Show diffs for each differing field (excluding ignored ones)
                            for field in all_differing[:3]:  # Show first 3 significant fields
                                self._show_value_diff(
                                    dict1[field],
                                    dict2[field],
                                    field,
                                    f"Models: {models1}",
                                    f"Models: {models2}"
                                )

                    if len(unique_responses) > 2:
                        self.console.print(
                            f"  [dim]Total of {len(unique_responses)} unique responses in this round[/dim]")
                else:
                    self.console.print(f"[green]Round {round_num}: Full consensus achieved[/green]")
            else:
                self.console.print(f"[dim]Round {round_num}: No response data available[/dim]")

        if not any_disagreements:
            self.console.print("\n[green]No disagreements found across all rounds[/green]")

        # Show final consensus status
        if result.consensus_achieved:
            self.console.print(f"\n[green]✓ Final consensus reached after {result.total_rounds} rounds[/green]")
            self.console.print(f"[dim]Convergence score: {result.convergence_score:.2%}[/dim]")
        else:
            self.console.print(f"\n[red]✗ No consensus after {result.total_rounds} rounds[/red]")
            if result.dissenting_models:
                self.console.print(f"[dim]Dissenting models: {', '.join(result.dissenting_models)}[/dim]")

    async def _process_all_batches(self):
        """Process all batches sequentially and save results."""
        total_batches = (len(self.pages) + self.batch_size - 1) // self.batch_size

        self.console.print(f"\n[cyan]Processing all {total_batches} batches...[/cyan]")

        if not Confirm.ask(f"This will process {total_batches} batches. Continue?", default=True):
            return

        all_results = []
        original_batch_index = self.current_batch_index

        try:
            for batch_idx in range(total_batches):
                self.current_batch_index = batch_idx
                batch_pages = self._get_current_batch()

                self.console.print(f"\n[yellow]Processing Batch {batch_idx + 1}/{total_batches}[/yellow]")
                self.console.print(f"[dim]Pages: {batch_pages[0].page}-{batch_pages[-1].page}[/dim]")

                with self.console.status(f"Processing batch {batch_idx + 1}..."):
                    result = await self._test_prompt(self.prompt_template)

                if "error" not in result:
                    all_results.append(result)
                    self.console.print(f"[green]✓ Batch {batch_idx + 1} completed[/green]")
                else:
                    self.console.print(f"[red]✗ Batch {batch_idx + 1} failed: {result['error']}[/red]")
                    if not Confirm.ask("Continue with remaining batches?", default=True):
                        break

            # Save combined results
            if all_results:
                combined_data = {
                    "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                    "total_batches": len(all_results),
                    "prompt_used": self.prompt_template,
                    "batches": all_results
                }
                saved_file = self._save_response_to_file(combined_data, prefix="all_batches")

                # Show summary
                self.console.print("\n[green]Processing Complete![/green]")
                self.console.print(f"Processed {len(all_results)}/{total_batches} batches successfully")
                self.console.print(f"Results saved to: {saved_file}")

                # Show aggregate statistics
                total_sections = sum(len(r.get("sections", [])) for r in all_results)
                total_terms = sum(len(r.get("terms", [])) for r in all_results)
                total_concepts = sum(len(r.get("concepts", [])) for r in all_results)

                table = Table(title="Aggregate Statistics")
                table.add_column("Metric", style="cyan")
                table.add_column("Total", style="green")
                table.add_row("Sections Found", str(total_sections))
                table.add_row("Terms/Acronyms", str(total_terms))
                table.add_row("Concepts", str(total_concepts))
                self.console.print(table)

        finally:
            # Restore original batch index
            self.current_batch_index = original_batch_index
            self.console.print(f"\n[dim]Returned to batch {self.current_batch_index + 1}[/dim]")

    async def run(self):
        """Main CLI loop."""
        self.console.print(
            Panel.fit(
                f"[bold cyan]Technical Analysis Prompt Refinement[/bold cyan]\n"
                f"Document: {self.pdf_path.name}\n"
                f"Batch Size: {self.batch_size} pages\n"
                f"Responses saved to: {self.output_dir}",
                title="Phase 1: Technical Analysis",
            )
        )

        # Wait for document extraction
        if not self.pages:
            await self._extract_document()

        if not self.pages:
            self.console.print("[red]Failed to extract document[/red]")
            return

        total_batches = (len(self.pages) + self.batch_size - 1) // self.batch_size
        self.last_test_result: Optional[Dict[str, Any]] = None

        while True:
            self.console.print(
                f"\n[bold cyan]Options (Batch {self.current_batch_index + 1}/{total_batches}):[/bold cyan]")
            self.console.print("1. View current prompt template")
            self.console.print("2. Test on current batch")
            self.console.print("3. Next batch")
            self.console.print("4. Previous batch")
            self.console.print("5. Jump to batch")
            self.console.print("6. Refine prompt")
            self.console.print("7. Manually edit prompt")
            self.console.print("8. Save refined prompt")
            self.console.print("r. View raw text for current batch")
            self.console.print("f. View filled prompt for current batch")
            self.console.print("v. View saved response files")
            self.console.print("c. Compare prompts (current vs saved)")
            self.console.print("d. View consensus disagreements (diffs)")
            self.console.print("p. Process all batches")
            self.console.print("0. Exit")

            choice = Prompt.ask("Select option", default="2")

            try:
                if choice == "1":
                    self._view_prompt()
                elif choice == "2":
                    results: Dict[str, Any] = await self._test_prompt(self.prompt_template)
                    self.last_test_result = results
                    self._display_test_results(results)
                elif choice == "3":
                    if self.current_batch_index < total_batches - 1:
                        self.current_batch_index += 1
                        self.console.print(f"[green]Moved to batch {self.current_batch_index + 1}[/green]")
                    else:
                        self.console.print("[yellow]Already at last batch[/yellow]")
                elif choice == "4":
                    if self.current_batch_index > 0:
                        self.current_batch_index -= 1
                        self.console.print(f"[green]Moved to batch {self.current_batch_index + 1}[/green]")
                    else:
                        self.console.print("[yellow]Already at first batch[/yellow]")
                elif choice == "5":
                    batch_num = IntPrompt.ask(
                        f"Jump to batch (1-{total_batches})",
                        default=self.current_batch_index + 1
                    )
                    if 1 <= batch_num <= total_batches:
                        self.current_batch_index = batch_num - 1
                        self.console.print(f"[green]Jumped to batch {batch_num}[/green]")
                elif choice == "6":
                    await self._refine_prompt_with_principles()
                elif choice == "7":
                    self._manual_edit()
                elif choice == "8":
                    self._save_prompt_template(self.prompt_template)
                elif choice.lower() == "r":
                    self._view_raw_batch_text()
                elif choice.lower() == "f":
                    self._view_filled_prompt()
                elif choice.lower() == "v":
                    self._view_saved_responses()
                elif choice.lower() == "c":
                    self._compare_prompts_with_previous()
                elif choice.lower() == "d":
                    self._view_consensus_disagreements()
                elif choice.lower() == "p":
                    await self._process_all_batches()
                elif choice == "0":
                    break
            except Exception as e:
                self.console.print(f"[red]Error: {e}[/red]")
                if Confirm.ask("Show traceback?", default=False):
                    import traceback
                    traceback.print_exc()

    async def _refine_prompt_with_principles(self):
        """Refine prompt using ONLY principle-based approach (Google's guidelines method)."""
        if not self.prompt_aligner:
            self.console.print("[red]Prompt aligner not initialized[/red]")
            return

        self.console.print("\n[bold cyan]Principle-Based Refinement (Google's Guidelines Method)[/bold cyan]")
        self.console.print("[dim]This uses extraction of reusable principles from feedback[/dim]\n")

        # Check existing principles
        existing_principles = self.prompt_aligner.export_principles() if hasattr(self.prompt_aligner, 'export_principles') else []
        if existing_principles:
            self.console.print(f"[dim]Currently have {len(existing_principles)} learned principles[/dim]")
            if Confirm.ask("View existing principles?", default=False):
                for idx, p in enumerate(existing_principles[:5], 1):
                    if isinstance(p, dict):
                        self.console.print(f"  {idx}. {p.get('principle', str(p))}")
                    else:
                        self.console.print(f"  {idx}. {str(p)}")
                if len(existing_principles) > 5:
                    self.console.print(f"  [dim]... and {len(existing_principles) - 5} more[/dim]")

        # Step 1: Learn principles from good/bad examples
        self.console.print("\n[bold]Step 1: Extract Principles from Examples[/bold]")
        self.console.print("[dim]Provide examples of good extractions and what made them good[/dim]\n")

        learn_more = True
        total_principles = 0

        while learn_more:
            # Show current batch for context
            batch_pages = self._get_current_batch()
            if batch_pages:
                self.console.print(f"\n[dim]Current batch: Pages {batch_pages[0].page}-{batch_pages[-1].page}[/dim]")

            # Get example of what makes a good extraction
            self.console.print("\nDescribe a good extraction for this type of document:")

            example_type = Prompt.ask(
                "Example type",
                choices=["good", "bad", "skip"],
                default="good"
            )

            if example_type == "skip":
                break

            if example_type == "good":
                # Learn from positive example
                good_sections = IntPrompt.ask("Number of sections that should be found", default=5)
                good_keywords = IntPrompt.ask("Number of important keywords expected", default=10)
                good_acronyms = IntPrompt.ask("Number of acronyms expected", default=5)
                good_summary = Prompt.ask("Key aspects the summary should capture")

                ideal_response = f"""
                A good extraction should:
                - Find approximately {good_sections} sections with clear hierarchy
                - Identify around {good_keywords} important domain-specific keywords
                - Extract around {good_acronyms} acronyms with their full definitions
                - Summary should capture: {good_summary}
                - Each section properly classified (concept/procedure/requirement/example/data/narrative)
                - Include reasoning for each classification
                """

                # Extract principles from this ideal
                if hasattr(self.prompt_aligner, 'extract_principles_from_ideal'):
                    with self.console.status("Extracting principles from ideal example..."):
                        principles = await self.prompt_aligner.extract_principles_from_ideal(
                            self.prompt_template,
                            ideal_response
                        )

                    if principles:
                        total_principles += len(principles)
                        self.console.print(f"[green]✓ Extracted {len(principles)} principles[/green]")
                        for p in principles[:3]:
                            self.console.print(f"  • {p.principle}")

            else:  # bad example
                # Learn from negative example
                self.console.print("Describe what makes a BAD extraction:")
                bad_aspects = Prompt.ask("What problems should be avoided?")

                # Convert negative to positive principles
                positive_principles = f"""
                Avoid these issues:
                - {bad_aspects}
                Convert to positive principles for better extraction.
                """

                if hasattr(self.prompt_aligner, 'extract_principles_from_ideal'):
                    with self.console.status("Converting issues to positive principles..."):
                        principles = await self.prompt_aligner.extract_principles_from_ideal(
                            self.prompt_template,
                            positive_principles
                        )

                    if principles:
                        total_principles += len(principles)
                        self.console.print(f"[green]✓ Converted to {len(principles)} positive principles[/green]")

            learn_more = Confirm.ask("\nLearn from another example?", default=False)

        if total_principles == 0 and not existing_principles:
            self.console.print("\n[yellow]No principles available for refinement.[/yellow]")
            self.console.print("[dim]Principle-based refinement requires examples to learn from.[/dim]")
            return

        # Step 2: Apply principles to refine prompt
        self.console.print("\n[bold]Step 2: Apply Principles to Refine Prompt[/bold]")

        # Test current prompt first
        self.console.print("\n[cyan]Testing current prompt for baseline...[/cyan]")
        test_results = await self._test_prompt(self.prompt_template)
        self._display_test_results(test_results)

        # Get target behavior
        target = Prompt.ask(
            "\nDescribe the ideal behavior for the refined prompt",
            default="Extract comprehensive, well-structured information with clear classifications"
        )

        # Apply principle-based refinement
        from com_blockether_catalyst.prompt.PromptAlignmentCore import PromptConfiguration

        config = PromptConfiguration(
            initial_prompt=self.prompt_template,
            target_behavior=target,
            max_iterations=IntPrompt.ask("Max refinement iterations", default=3),
            score_threshold=0.85,
        )

        original_prompt = self.prompt_template

        with self.console.status("Applying principles to refine prompt..."):
            result = await self.prompt_aligner.align_prompt(config)

        # Show results
        self._display_alignment_result(result)

        # Show changes
        if result.aligned_prompt != original_prompt:
            self.console.print("\n[bold cyan]Prompt Changes:[/bold cyan]")
            self._compare_prompts(original_prompt, result.aligned_prompt)

            if Confirm.ask("\nView detailed diff?", default=True):
                diff_text = self._show_diff(
                    original_prompt,
                    result.aligned_prompt,
                    "Original",
                    "Refined (Principle-Based)",
                    context_lines=5
                )
                from rich.panel import Panel
                self.console.print(Panel(diff_text, title="[bold]Principle-Based Changes[/bold]",
                                         border_style="green", expand=False))

            if Confirm.ask("\nAccept principle-refined prompt?", default=True):
                self.prompt_template = result.aligned_prompt
                self.console.print("[green]✓ Prompt updated with principle-based refinements[/green]")
        else:
            self.console.print("\n[yellow]Prompt already optimal according to learned principles[/yellow]")


async def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        console.print("[red]Usage: uv run python TechnicalAnalysisRefinementCLI.py <pdf_path>[/red]")
        sys.exit(1)

    pdf_path = Path(sys.argv[1])
    if not pdf_path.exists():
        console.print(f"[red]File not found: {pdf_path}[/red]")
        sys.exit(1)

    cli = TechnicalAnalysisRefinementCLI(pdf_path)

    try:
        await cli.run()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    anyio.run(main)
