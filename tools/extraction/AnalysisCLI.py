#!/usr/bin/env python3
"""
Table of Contents Generator CLI.

Generates a hierarchical Table of Contents from batch technical analysis JSON files.
"""

import json
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import anyio
from pydantic import BaseModel, Field
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.table import Table
from rich.tree import Tree

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
from com_blockether_catalyst.prompt import PromptAlignmentCore
from com_blockether_catalyst.prompt.PromptAlignmentCLIBase import PromptAlignmentCLIBase
from com_blockether_catalyst.prompt.internal.PromptAlignmentTypes import (
    AlignmentFeedback,
    EvaluationResult,
)
from com_blockether_catalyst.utils.instructor.InstructorLLMCall import InstructorLLMCall

console = Console()


class TOCSection(BaseModel):
    """A section in the Table of Contents with hierarchical information."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    level: int
    start_page: int
    end_page: int
    summary: str
    type: str
    text: str
    children: List["TOCSection"] = []
    source_file: str
    batch_number: int


class TermInfo(BaseModel):
    """Information about a term/keyword."""

    term: str
    type: str  # 'acronym' or 'keyword'
    full_form: Optional[str]
    definition: Optional[str]
    page_found: int
    source_file: str
    batch_number: int


class TermSectionMapping(BaseModel):
    """Mapping of a term to sections where it appears."""

    term_infos: List[TermInfo]  # All variations of this term found
    section_ids: List[str]  # List of section UUIDs where any variation appears


class TOCDocument(BaseModel):
    """The complete Table of Contents document."""

    generated_at: str
    document_metadata: Optional[Dict[str, Any]] = (
        None  # Document metadata from batch files
    )
    total_batches: int
    total_sections: int
    total_terms: int
    sections: List[TOCSection]
    terms: Dict[str, TermSectionMapping]  # keyword -> section mapping


class DocumentOverview(TypedCallBaseForConsensus):
    """High-level overview of the document."""

    purpose: str = VotingField(
        description="The main purpose and objective of this document",
        comparison=ComparisonStrategy.SEMANTIC,
        min_length=50,
        threshold=0.8,
    )
    audience: List[str] = VotingField(
        default_factory=list,
        description="List of intended audience groups",
        comparison=ComparisonStrategy.SEQUENCE_UNORDERED_ALIKE,
    )
    concepts: List[str] = VotingField(
        default_factory=list,
        description="Key concepts, frameworks, and methodologies covered",
        comparison=ComparisonStrategy.SEQUENCE_UNORDERED_ALIKE,
    )
    summary: str = VotingField(
        description="Detailed summary with page references",
        comparison=ComparisonStrategy.SEMANTIC,
        min_length=200,
        threshold=0.8,
    )


class TOCGeneratorCLI(PromptAlignmentCLIBase):
    """CLI for generating Table of Contents from technical analysis batches."""

    def __init__(
        self,
        output_dir: Path = Path("output/technical_analysis_responses"),
        api_base_url: str = "http://localhost:3005/v1",
        api_key: str = "test-key",
    ):
        """
        Initialize the TOC Generator CLI.

        Args:
            output_dir: Directory containing technical analysis batch files
            api_base_url: Base URL for the Instructor API
            api_key: API key for the Instructor API
        """
        self.batch_files_dir = output_dir
        self.api_base_url = api_base_url
        self.api_key = api_key
        self.batch_files: List[Path] = []
        self.all_sections: List[Dict[str, Any]] = []
        self.all_terms: List[Dict[str, Any]] = []
        self.terms_mapping: Dict[str, TermSectionMapping] = {}
        self.section_lookup: Dict[str, TOCSection] = {}  # UUID -> TOCSection mapping
        self.document_metadata: Optional[Dict[str, Any]] = (
            None  # Document metadata from batch files
        )
        self.toc_data: Optional[TOCDocument] = None  # Loaded ToC data
        self.last_consensus_result: Optional[Any] = None  # Store last consensus result
        
        super().__init__(
            prompt_name="document_overview",
            prompt_dir=Path("tools/extraction/prompts"),
            output_dir=Path("output/analysis_responses"),
            console=console,
        )

    def _init_components(self) -> None:
        """Initialize LLM components and prompt aligner."""
        # Create model for document overview
        overview_model: InstructorLLMCall[DocumentOverview] = InstructorLLMCall(
            response_model=DocumentOverview,
            model="gpt-4o",
            temperature=0.3,
            base_url=self.api_base_url,
            api_key=self.api_key,
        )

        # Create consensus for overview generation
        self.overview_consensus: Consensus[DocumentOverview] = ConsensusCore[
            DocumentOverview
        ].consensus(
            models=[
                ConsensusCore[DocumentOverview].model(
                    id="overview_analyzer",
                    executor=overview_model,
                    perspective="Analyze document structure and provide comprehensive overview.",
                    weight_multiplier=1.0,
                ),
                ConsensusCore[DocumentOverview].model(
                    id="overview_analyzer_2",
                    executor=overview_model,
                    perspective="Extract key information about document purpose, audience, and organization.",
                    weight_multiplier=1.0,
                ),
            ],
            judge=overview_model,
            settings=ConsensusSettings(
                max_rounds=3, threshold=0.9, verbosity=VerbosityLevel.VERBOSE
            ),
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
                    perspective="Evaluate overview quality and completeness.",
                    weight_multiplier=1.0,
                ),
                ConsensusCore.model(
                    id="evaluator1",
                    executor=eval_model,
                    perspective="Evaluate overview quality and completeness. Take into account all of the rules and remember to be concise and strict.",
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
                    perspective="Provide feedback for improving overview prompts.",
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
            "{document_title}",
            "{document_author}",
            "{toc_summary}",
            "{terms_summary}",
        ]
        
    def _find_batch_files(self) -> None:
        """Find all batch JSON files in the output directory."""
        self.batch_files = sorted(
            self.batch_files_dir.glob("batch_*_pages_*.json"),
            key=lambda x: self._extract_batch_number(x),
        )

    @staticmethod
    def _extract_batch_number(file_path: Path) -> int:
        """Extract batch number from filename."""
        try:
            # Format: batch_1_pages_1-2_20250907_170640.json
            parts = file_path.stem.split("_")
            for i, part in enumerate(parts):
                if part == "batch" and i + 1 < len(parts):
                    return int(parts[i + 1])
            return 0
        except (ValueError, IndexError):
            return 0

    def _load_batch_files(self) -> None:
        """Load all batch files and extract sections and terms."""
        self.all_sections = []
        self.all_terms = []

        for batch_file in self.batch_files:
            try:
                with open(batch_file, "r") as f:
                    data = json.load(f)

                batch_num = self._extract_batch_number(batch_file)

                # Extract document metadata from first batch file
                if self.document_metadata is None and "document_metadata" in data:
                    self.document_metadata = data["document_metadata"]

                # Extract sections from the response
                if "response" in data and "sections" in data["response"]:
                    for section in data["response"]["sections"]:
                        section_copy = section.copy()
                        section_copy["source_file"] = batch_file.name
                        section_copy["batch_number"] = batch_num
                        self.all_sections.append(section_copy)

                # Extract terms from the response
                if "response" in data and "terms" in data["response"]:
                    for term in data["response"]["terms"]:
                        term_copy = term.copy()
                        term_copy["source_file"] = batch_file.name
                        term_copy["batch_number"] = batch_num
                        self.all_terms.append(term_copy)

            except Exception as e:
                console.print(f"[red]Error loading {batch_file}: {e}[/red]")

    def _build_hierarchical_toc(self) -> List[TOCSection]:
        """Build hierarchical TOC structure from flat sections list."""
        root_sections: List[TOCSection] = []
        section_stack: List[TOCSection] = []
        self.section_lookup = {}  # Reset lookup

        for section_data in self.all_sections:
            section = TOCSection(
                title=section_data["title"],
                level=section_data["level"],
                start_page=section_data["start_page"],
                end_page=section_data["end_page"],
                summary=section_data["summary"],
                type=section_data["type"],
                text=section_data["text"],
                source_file=section_data["source_file"],
                batch_number=section_data["batch_number"],
            )

            # Store in lookup dictionary
            self.section_lookup[section.id] = section

            # Find parent section
            while section_stack and section_stack[-1].level >= section.level:
                section_stack.pop()

            if section_stack:
                # Add as child to the last section in stack
                section_stack[-1].children.append(section)
            else:
                # Add as root section
                root_sections.append(section)

            section_stack.append(section)

        return root_sections

    def _build_terms_mapping(self) -> None:
        """Build mapping of terms to sections where they appear."""
        self.terms_mapping = {}

        for term_data in self.all_terms:
            term_key = term_data["term"].lower()  # Use lowercase for consistent mapping

            # Create TermInfo object
            term_info = TermInfo(
                term=term_data["term"],
                type=term_data["type"],
                full_form=term_data.get("full_form"),
                definition=term_data.get("definition"),
                page_found=term_data["page_found"],
                source_file=term_data["source_file"],
                batch_number=term_data["batch_number"],
            )

            # Find sections where this term appears (by UUID)
            section_ids_with_term = []
            for section in self.section_lookup.values():
                # Check if term appears in section (based on page range)
                if section.start_page <= term_data["page_found"] <= section.end_page:
                    section_ids_with_term.append(section.id)

            # If term already exists, add this variation
            if term_key in self.terms_mapping:
                # Check if this exact variation already exists
                existing_variations = self.terms_mapping[term_key].term_infos
                variation_exists = False

                for existing in existing_variations:
                    if (
                        existing.term == term_info.term
                        and existing.type == term_info.type
                        and existing.full_form == term_info.full_form
                        and existing.definition == term_info.definition
                    ):
                        variation_exists = True
                        break

                # Add new variation if it doesn't exist
                if not variation_exists:
                    self.terms_mapping[term_key].term_infos.append(term_info)

                # Add new section IDs (avoid duplicates)
                existing_ids = set(self.terms_mapping[term_key].section_ids)
                new_ids = [
                    sid for sid in section_ids_with_term if sid not in existing_ids
                ]
                self.terms_mapping[term_key].section_ids.extend(new_ids)
            else:
                # Create new mapping
                self.terms_mapping[term_key] = TermSectionMapping(
                    term_infos=[term_info], section_ids=section_ids_with_term
                )

    def _display_toc_tree(self, sections: List[TOCSection]) -> None:
        """Display TOC as a tree structure."""
        tree = Tree("[bold cyan]Table of Contents[/bold cyan]")

        def add_section_to_tree(section: TOCSection, parent_node: Any) -> None:
            """Recursively add sections to tree."""
            section_text = f"[bold]{section.title}[/bold] "
            section_text += f"[dim](p{section.start_page}"
            if section.end_page != section.start_page:
                section_text += f"-{section.end_page}"
            section_text += f", {section.type}"
            # Show abbreviated UUID for debugging (first 8 chars)
            section_text += f", id: {section.id[:8]}...)[/dim]"

            node = parent_node.add(section_text)

            # Add summary as a child node
            if section.summary:
                summary_lines = section.summary.split("\n")
                for line in summary_lines[:2]:  # Show first 2 lines
                    if line.strip():
                        node.add(f"[italic dim]{line.strip()[:80]}...[/italic dim]")

            # Recursively add children
            for child in section.children:
                add_section_to_tree(child, node)

        for section in sections:
            add_section_to_tree(section, tree)

        console.print(tree)

    def _display_terms_summary(self) -> None:
        """Display summary of terms and their section mappings."""
        console.print("\n[bold cyan]Terms & Keywords Summary[/bold cyan]")

        # Create table for top terms
        table = Table(title="Top Terms by Frequency")
        table.add_column("Term", style="cyan")
        table.add_column("Type", style="yellow")
        table.add_column("Sections", style="green")
        table.add_column("Pages", style="magenta")

        # Sort terms by number of sections they appear in
        sorted_terms = sorted(
            self.terms_mapping.items(),
            key=lambda x: len(x[1].section_ids),
            reverse=True,
        )[:10]  # Show top 10

        for term_key, mapping in sorted_terms:
            section_count = len(mapping.section_ids)

            # Get primary term info (first variation)
            primary_info = mapping.term_infos[0]

            # Collect all unique definitions/full forms
            all_full_forms = []
            all_pages = set()
            term_types = set()

            for info in mapping.term_infos:
                if (
                    info.type == "acronym"
                    and info.full_form
                    and info.full_form not in all_full_forms
                ):
                    all_full_forms.append(info.full_form)
                all_pages.add(info.page_found)
                term_types.add(info.type)

            # Format display term
            display_term = primary_info.term
            if all_full_forms:
                display_term += f" ({'; '.join(all_full_forms)})"

            # Show all page numbers where term appears
            pages = "p" + ", p".join(map(str, sorted(all_pages)))

            # Show term type(s)
            term_type_str = "/".join(sorted(term_types))

            table.add_row(display_term, term_type_str, str(section_count), pages)

        console.print(table)

    def _display_statistics(self) -> None:
        """Display statistics about the TOC."""
        table = Table(title="TOC Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total Batch Files", str(len(self.batch_files)))
        table.add_row("Total Sections", str(len(self.all_sections)))
        table.add_row("Total Terms", str(len(self.all_terms)))
        table.add_row("Unique Terms", str(len(self.terms_mapping)))

        # Count by level
        level_counts: Dict[int, int] = {}
        for section in self.all_sections:
            level = section["level"]
            level_counts[level] = level_counts.get(level, 0) + 1

        for level in sorted(level_counts.keys()):
            table.add_row(f"Level {level} Sections", str(level_counts[level]))

        # Count by section type
        type_counts: Dict[str, int] = {}
        for section in self.all_sections:
            section_type = section["type"]
            type_counts[section_type] = type_counts.get(section_type, 0) + 1

        for section_type in sorted(type_counts.keys()):
            table.add_row(
                f"{section_type.capitalize()} Sections", str(type_counts[section_type])
            )

        # Count by term type
        term_type_counts = {"acronym": 0, "keyword": 0}
        for term in self.all_terms:
            term_type = term.get("type", "keyword")
            term_type_counts[term_type] = term_type_counts.get(term_type, 0) + 1

        for term_type, count in term_type_counts.items():
            if count > 0:
                table.add_row(f"{term_type.capitalize()}s", str(count))

        console.print(table)

        # Show terms with multiple variations
        terms_with_variations = [
            (key, mapping)
            for key, mapping in self.terms_mapping.items()
            if len(mapping.term_infos) > 1
        ]

        if terms_with_variations:
            console.print(
                f"\n[yellow]Note: {len(terms_with_variations)} terms have multiple variations/definitions[/yellow]"
            )

    def _save_toc(self, toc_sections: List[TOCSection]) -> Path:
        """Save TOC to JSON file."""
        toc_doc = TOCDocument(
            generated_at=datetime.now().isoformat(),
            document_metadata=self.document_metadata,
            total_batches=len(self.batch_files),
            total_sections=len(self.all_sections),
            total_terms=len(self.all_terms),
            sections=toc_sections,
            terms=self.terms_mapping,
        )

        # Create output filename - save to analysis_responses directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("output/analysis_responses")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"table_of_contents_{timestamp}.json"

        # Save to file
        with open(output_file, "w") as f:
            json.dump(toc_doc.model_dump(), f, indent=2)

        return output_file

    def _load_toc_data(self) -> Optional[TOCDocument]:
        """Load the most recent Table of Contents file."""
        # Look in the new analysis_responses directory
        analysis_dir = Path("output/analysis_responses")
        toc_files = sorted(
            analysis_dir.glob("table_of_contents_*.json"), reverse=True
        )

        if not toc_files:
            console.print("[red]No Table of Contents files found[/red]")
            return None

        # Load the most recent ToC file
        latest_toc = toc_files[0]
        console.print(f"[green]Loading ToC from: {latest_toc.name}[/green]")

        try:
            with open(latest_toc, "r") as f:
                data = json.load(f)
                self.toc_data = TOCDocument(**data)
                return self.toc_data
        except Exception as e:
            console.print(f"[red]Error loading ToC: {e}[/red]")
            return None
            
    def _load_specific_toc(self, toc_file: Path) -> Optional[TOCDocument]:
        """Load a specific Table of Contents file."""
        console.print(f"[green]Loading ToC from: {toc_file.name}[/green]")
        
        try:
            with open(toc_file, "r") as f:
                data = json.load(f)
                self.toc_data = TOCDocument(**data)
                console.print("[green]✓ ToC loaded successfully[/green]")
                return self.toc_data
        except Exception as e:
            console.print(f"[red]Error loading ToC: {e}[/red]")
            return None

    async def _test_prompt(self, prompt: str) -> Dict[str, Any]:
        """Test the current prompt and return results."""
        # Load ToC data if not already loaded
        if not self.toc_data:
            if not self._load_toc_data():
                return {"error": "No Table of Contents data available"}
                
        # Prepare the prompt with ToC information
        toc_summary = self._prepare_toc_summary()
        terms_summary = self._prepare_terms_summary()

        # Extract document info safely
        doc_title = "Unknown"
        doc_author = "Unknown"
        if self.toc_data and self.toc_data.document_metadata:
            doc_title = self.toc_data.document_metadata.get("title", "Unknown")
            doc_author = self.toc_data.document_metadata.get("author", "Unknown")
        
        # Fill the prompt template
        values = {
            "document_title": doc_title,
            "document_author": doc_author,
            "toc_summary": toc_summary,
            "terms_summary": terms_summary,
        }
        
        filled_prompt = self._fill_template(prompt, values)
        
        # Run the consensus
        result = await self.overview_consensus.call(filled_prompt)
        
        if result and result.final_response:
            response = result.final_response
            
            # Store consensus result for diff viewing
            self.last_consensus_result = result
            
            # Create the response dictionary
            full_response_data = response.model_dump()
            
            # Save the full response to file
            saved_file = self._save_response_to_file(full_response_data, prefix="overview_test")
            
            return {
                "purpose": response.purpose,
                "audience": response.audience,
                "concepts": response.concepts,
                "summary": response.summary,
                "final_response": full_response_data,
                "saved_to": str(saved_file),
                "consensus_result": result,
            }
            
        return {"error": "No response from LLM"}
        
    def _display_test_results(self, results: Dict[str, Any]):
        """Display test results to the user."""
        if "error" in results:
            self.console.print(f"[red]Error: {results['error']}[/red]")
            return
            
        # Display overview information
        self.console.print("\n[bold cyan]Test Results - Document Overview[/bold cyan]\n")
        
        if "purpose" in results:
            self.console.print(f"[yellow]Purpose:[/yellow] {results['purpose']}")
            
        if "audience" in results:
            self.console.print("\n[yellow]Target Audience:[/yellow]")
            for audience in results["audience"]:
                self.console.print(f"  • {audience}")
                
        if "concepts" in results:
            self.console.print("\n[yellow]Key Concepts:[/yellow]")
            for concept in results["concepts"][:5]:
                self.console.print(f"  • {concept}")
            if len(results["concepts"]) > 5:
                self.console.print(f"  [dim]... and {len(results['concepts']) - 5} more[/dim]")
                
        # Show where the full response was saved
        if "saved_to" in results:
            self.console.print(f"\n[yellow]Full response saved to: {results['saved_to']}[/yellow]")
            
        # Display raw JSON of final response
        self._display_raw_json(results["final_response"], "Full Response")

    async def _generate_document_overview(self) -> None:
        """Generate a high-level document overview using LLM."""
        # Load ToC data if not already loaded
        if not self.toc_data:
            if not self._load_toc_data():
                return

        console.print("\n[cyan]Generating Document Overview...[/cyan]")

        # Test the prompt
        test_results = await self._test_prompt(self.prompt_template)
        
        if "error" not in test_results and "final_response" in test_results:
            response_data = test_results["final_response"]
            
            # Create DocumentOverview from the response data
            response = DocumentOverview(**response_data)
            
            # Display the overview
            self._display_document_overview(response)

            # Save the overview
            if Confirm.ask("\nSave document overview to file?", default=True):
                self._save_document_overview(response)
        else:
            console.print(f"[red]Error generating overview: {test_results.get('error', 'Unknown error')}[/red]")

    def _prepare_toc_summary(self) -> str:
        """Prepare a summary of the ToC structure for the prompt."""
        lines = []

        def add_section(section: TOCSection, indent: int = 0) -> None:
            """Recursively add sections to the summary."""
            prefix = "  " * indent
            lines.append(
                f"{prefix}• {section.title} (Pages {section.start_page}-{section.end_page}, Type: {section.type})"
            )
            lines.append(f"{prefix}  Summary: {section.summary}")

            for child in section.children:
                add_section(child, indent + 1)

        if self.toc_data:
            for section in self.toc_data.sections:
                add_section(section)

        return "\n".join(lines)

    def _prepare_terms_summary(self) -> str:
        """Prepare a summary of important terms for the prompt."""
        lines = []

        # Get top terms by frequency
        if not self.toc_data:
            return ""
        sorted_terms = sorted(
            self.toc_data.terms.items(),
            key=lambda x: len(x[1].section_ids),
            reverse=True,
        )[:20]  # Top 20 terms

        for term_key, mapping in sorted_terms:
            primary_info = mapping.term_infos[0]
            if primary_info.type == "acronym" and primary_info.full_form:
                lines.append(f"• {primary_info.term}: {primary_info.full_form}")
            else:
                lines.append(f"• {primary_info.term}")

        return "\n".join(lines)

    def _display_document_overview(self, overview: DocumentOverview) -> None:
        """Display the document overview."""
        console.print("\n[bold cyan]Document Overview[/bold cyan]\n")

        # Document info from metadata
        doc_author = "Unknown"
        doc_name = "Unknown"
        if self.toc_data and self.toc_data.document_metadata:
            doc_author = self.toc_data.document_metadata.get("author", "Unknown")
            doc_name = self.toc_data.document_metadata.get("title", "Unknown")

        # Document info
        table = Table(show_header=False, box=None)
        table.add_column("Field", style="cyan")
        table.add_column("Value", style="white")

        table.add_row("Author:", doc_author)
        table.add_row("Document Name:", doc_name)
        table.add_row("Purpose:", overview.purpose)

        console.print(table)

        # Audience
        console.print("\n[yellow]Target Audience:[/yellow]")
        for audience in overview.audience:
            console.print(f"  • {audience}")

        # Key concepts
        console.print("\n[yellow]Key Concepts:[/yellow]")
        for concept in overview.concepts:
            console.print(f"  • {concept}")

        # Detailed summary
        console.print("\n[yellow]Detailed Summary:[/yellow]")
        console.print(overview.summary)

    def _save_document_overview(self, overview: DocumentOverview) -> Path:
        """Save the document overview to a JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("output/analysis_responses")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"document_overview_{timestamp}.json"

        # Create the full document data
        overview_data = {
            "generated_at": datetime.now().isoformat(),
            "document_metadata": self.toc_data.document_metadata
            if self.toc_data
            else None,
            "purpose": overview.purpose,
            "audience": overview.audience,
            "concepts": overview.concepts,
            "summary": overview.summary,
        }

        # Save to file
        with open(output_file, "w") as f:
            json.dump(overview_data, f, indent=2)

        console.print(f"\n[green]✓ Document overview saved to: {output_file}[/green]")

        return output_file

    def _view_filled_prompt(self):
        """View the filled prompt template for document overview."""
        # Load ToC data if not already loaded
        if not self.toc_data:
            if not self._load_toc_data():
                self.console.print("[red]No Table of Contents data available[/red]")
                return
                
        # Prepare the prompt with ToC information
        toc_summary = self._prepare_toc_summary()
        terms_summary = self._prepare_terms_summary()

        # Extract document info safely
        doc_title = "Unknown"
        doc_author = "Unknown"
        if self.toc_data and self.toc_data.document_metadata:
            doc_title = self.toc_data.document_metadata.get("title", "Unknown")
            doc_author = self.toc_data.document_metadata.get("author", "Unknown")
        
        # Fill the prompt template
        values = {
            "document_title": doc_title,
            "document_author": doc_author,
            "toc_summary": toc_summary,
            "terms_summary": terms_summary,
        }
        
        filled_prompt = self._fill_template(self.prompt_template, values)
        
        # Use base class method for display
        self._display_text_with_syntax(
            filled_prompt,
            title="Filled Document Overview Prompt",
            language="markdown",
            line_numbers=True,
            offer_save=True,
            save_prefix="filled_overview_prompt"
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
        self.console.print("[dim]Provide examples of good document overviews and what made them good[/dim]\n")

        learn_more = True
        total_principles = 0

        while learn_more:
            # Show current ToC for context
            if self.toc_data:
                total_sections = len(self.toc_data.sections) if self.toc_data.sections else 0
                total_terms = len(self.toc_data.terms) if self.toc_data.terms else 0
                self.console.print(f"\n[dim]Current ToC has {total_sections} sections and {total_terms} terms[/dim]")

            # Get example of what makes a good overview
            self.console.print("\nDescribe a good overview for this type of document:")

            example_type = Prompt.ask(
                "Example type",
                choices=["good", "bad", "skip"],
                default="good"
            )

            if example_type == "skip":
                break

            if example_type == "good":
                # Learn from positive example
                good_purpose = Prompt.ask("What should the purpose capture?", default="Clear business objective and use case")
                good_audience = IntPrompt.ask("How many audience groups should be identified?", default=3)
                good_concepts = IntPrompt.ask("How many key concepts should be extracted?", default=5)
                good_summary_style = Prompt.ask("What style should the summary use?", default="Detailed with page references")

                ideal_response = f"""
                A good overview should:
                - Purpose: {good_purpose}
                - Identify approximately {good_audience} distinct audience groups
                - Extract around {good_concepts} key concepts and methodologies
                - Summary should be {good_summary_style}
                - Each section should be referenced with page numbers
                - Focus on business value and practical application
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
                self.console.print("Describe what makes a BAD overview:")
                bad_aspects = Prompt.ask("What problems should be avoided?")

                # Convert negative to positive principles
                positive_principles = f"""
                Avoid these issues:
                - {bad_aspects}
                Convert to positive principles for better overview generation.
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
            default="Generate comprehensive, well-structured document overviews with clear business context"
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
            
    def _generate_toc(self) -> None:
        """Generate the Table of Contents."""
        # Find and load batch files
        self._find_batch_files()

        if not self.batch_files:
            console.print("[red]No batch files found in output directory[/red]")
            return

        console.print(f"[green]Found {len(self.batch_files)} batch files[/green]")

        # Load all sections
        with console.status("Loading batch files..."):
            self._load_batch_files()

        if not self.all_sections:
            console.print("[red]No sections found in batch files[/red]")
            return

        console.print(f"[green]Loaded {len(self.all_sections)} sections[/green]\n")

        # Build hierarchical structure
        toc_sections = self._build_hierarchical_toc()

        # Build terms mapping
        self._build_terms_mapping()

        # Display statistics
        self._display_statistics()
        console.print()

        # Display document metadata if available
        if self.document_metadata:
            console.print("\n[bold cyan]Document Information[/bold cyan]")
            console.print(f"Title: {self.document_metadata.get('title', 'Unknown')}")
            console.print(f"Author: {self.document_metadata.get('author', 'Unknown')}")
            console.print(f"File: {self.document_metadata.get('file_path', 'Unknown')}")

        # Display TOC tree
        self._display_toc_tree(toc_sections)

        # Display terms summary if any
        if self.terms_mapping:
            self._display_terms_summary()

        # Save TOC
        if Confirm.ask("\nSave Table of Contents to file?", default=True):
            output_file = self._save_toc(toc_sections)
            console.print(
                f"\n[green]✓ Table of Contents saved to: {output_file}[/green]"
            )

    async def _document_overview_menu(self) -> bool:
        """Handle Document Overview sub-menu. Returns False to go back to main menu."""
        while True:
            console.print("\n[bold cyan]Document Overview Options:[/bold cyan]")
            console.print("1. Generate overview")
            console.print("2. View current prompt template")
            console.print("3. Test current prompt")
            console.print("4. Jump to saved file")
            console.print("5. Refine prompt")
            console.print("6. Manually edit prompt")
            console.print("7. Save refined prompt")
            console.print("f. View filled prompt for current batch")
            console.print("v. View saved response files")
            console.print("c. Compare prompts (current vs saved)")
            console.print("d. View consensus disagreements (diffs)")
            console.print("0. Back to main menu")
            
            choice = Prompt.ask("Select option", default="1")
            
            try:
                if choice == "1":
                    await self._generate_document_overview()
                elif choice == "2":
                    self._view_prompt()
                elif choice == "3":
                    results = await self._test_prompt(self.prompt_template)
                    self._display_test_results(results)
                elif choice == "4":
                    # Jump to a saved ToC file
                    analysis_dir = Path("output/analysis_responses")
                    toc_files = sorted(
                        analysis_dir.glob("table_of_contents_*.json"), reverse=True
                    )
                    if toc_files:
                        console.print("\n[cyan]Available ToC files:[/cyan]")
                        for idx, file in enumerate(toc_files[:10], 1):
                            console.print(f"{idx}. {file.name}")
                        file_num = IntPrompt.ask("Select file (0 to cancel)", default=0)
                        if file_num > 0 and file_num <= len(toc_files):
                            selected_file = toc_files[file_num - 1]
                            self._load_specific_toc(selected_file)
                    else:
                        console.print("[yellow]No ToC files found[/yellow]")
                elif choice == "5":
                    await self._refine_prompt_with_principles()
                elif choice == "6":
                    self._manual_edit()
                elif choice == "7":
                    self._save_prompt_template(self.prompt_template)
                elif choice.lower() == "f":
                    self._view_filled_prompt()
                elif choice.lower() == "v":
                    self._view_saved_responses()
                elif choice.lower() == "c":
                    self._compare_prompts_with_previous()
                elif choice.lower() == "d":
                    self._view_consensus_disagreements()
                elif choice == "0":
                    return False
                else:
                    console.print("[yellow]Invalid option[/yellow]")
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
                if Confirm.ask("Show traceback?", default=False):
                    import traceback
                    traceback.print_exc()

    async def run(self) -> None:
        """Main CLI loop."""
        console.print(
            Panel.fit(
                "[bold cyan]Table of Contents Generator[/bold cyan]\n"
                f"Output Directory: {self.output_dir}",
                title="TOC Generator",
            )
        )

        while True:
            console.print("\n[bold cyan]Options:[/bold cyan]")
            console.print("1. Generate ToC")
            console.print("2. Document Overview")
            console.print("0. Exit")

            choice = Prompt.ask("Select option", default="1")

            try:
                if choice == "1":
                    self._generate_toc()
                elif choice == "2":
                    # Enter document overview sub-menu
                    await self._document_overview_menu()
                elif choice == "0":
                    break
                else:
                    console.print("[yellow]Invalid option[/yellow]")
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
                if Confirm.ask("Show traceback?", default=False):
                    import traceback

                    traceback.print_exc()


async def main() -> None:
    """Main entry point."""
    # Use default output directory
    output_dir = Path("output/technical_analysis_responses")

    # Allow optional override via command line
    if len(sys.argv) > 1:
        output_dir = Path(sys.argv[1])

    if not output_dir.exists():
        console.print(f"[red]Output directory not found: {output_dir}[/red]")
        sys.exit(1)

    # Allow optional API configuration via environment or defaults
    api_base_url = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:3005/v1"
    api_key = sys.argv[3] if len(sys.argv) > 3 else "test-key"

    cli = TOCGeneratorCLI(output_dir, api_base_url=api_base_url, api_key=api_key)

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
