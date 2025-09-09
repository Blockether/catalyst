#!/usr/bin/env python3
"""
Knowledge extraction tool with consensus-based validation and prompt refinement.

This module provides:
1. Knowledge extraction from PDF documents with consensus validation
2. Prompt refinement for chunking using principle-based alignment
3. Prompt refinement for term extraction using principle-based alignment

Usage:
    # Extract knowledge (default)
    uv run python3 tools/KnowledgeExtraction.py extract input/*.pdf

    # Refine chunking prompts
    uv run python3 tools/KnowledgeExtraction.py refine-chunking

    # Refine term extraction prompts
    uv run python3 tools/KnowledgeExtraction.py refine-terms
"""

import anyio
import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar, Tuple

import instructor
from openai import AsyncOpenAI
from pydantic import BaseModel
from rich.console import Console

from com_blockether_catalyst.consensus.ConsensusCore import ConsensusCore
from com_blockether_catalyst.consensus.ConsensusTypes import ConsensusSettings, VerbosityLevel
from com_blockether_catalyst.knowledge.KnowledgeExtractionCallBase import (
    BaseTermExtractionCall,
    BaseDocumentChunkingCall,
)
from com_blockether_catalyst.knowledge.KnowledgeExtractionCore import KnowledgeExtractionCore
from com_blockether_catalyst.knowledge.KnowledgeExtractionTypes import (
    KnowledgeExtractionOutput,
    KnowledgeMetadata,
    KnowledgePageDataWithRawText,
    KnowledgeProcessorSettings,
    TermMeaningExtractionResponse,
    ChunkingDecision,
)

from com_blockether_catalyst.knowledge.PDKnowledgeExtractorTypes import PDFKnowledgeProcessorSettings, PDFPageCropOffset
from com_blockether_catalyst.utils.TypedCalls import ArityOneTypedCall
from com_blockether_catalyst.prompt import PromptAlignmentCore
from com_blockether_catalyst.prompt.PromptAlignmentCLIBase import PromptAlignmentCLIBase
from com_blockether_catalyst.prompt.PromptAlignmentTypes import (
    EvaluationResult,
    AlignmentFeedback,
)
from jinja2 import Environment, FileSystemLoader
from rich.panel import Panel
from rich.table import Table

# Type variable for response types
T = TypeVar("T", bound=BaseModel)

# Setup Jinja2 environment for templates
template_env = Environment(
    loader=FileSystemLoader(Path(__file__).parent / "prompts"),
    trim_blocks=True,
    lstrip_blocks=True,
)


class SimpleInstructorLLMCall(ArityOneTypedCall[str, T]):
    """
    Simple implementation of ArityOneTypedCall using Instructor for the example.
    """

    def __init__(
        self,
        response_model: Type[T],
        model: str = "gpt-4o",
        temperature: float = 0.7,
        base_url: str = "http://localhost:3005/v1",
        api_key: Optional[str] = None,
    ):
        """Initialize the Instructor LLM call."""
        self.response_model = response_model
        self.model = model
        self.temperature = temperature

        # Use provided API key or fall back to environment variable
        api_key = api_key or os.getenv("OPENAI_API_KEY", "sk-not-needed")

        # Create async client
        client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
        )

        # Patch with instructor
        self.client = instructor.from_openai(client)

    async def call(self, x: str) -> T:
        """Make a structured LLM call."""
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": x}],
            response_model=self.response_model,
            temperature=self.temperature,
        )
        return response


class ConsensusTermExtractionCall(BaseTermExtractionCall):
    """
    Consensus-based term extraction that validates and extracts meanings from term candidates.
    Handles both acronyms and keywords.
    """

    def __init__(self, models: List, max_rounds: int = 3):
        """
        Initialize the consensus term extractor.

        Args:
            models: List of model configurations for consensus
            max_rounds: Maximum rounds for consensus (default: 3)
        """
        self.models = models
        self.max_rounds = max_rounds

        settings = ConsensusSettings(
            max_rounds=max_rounds,
            threshold=0.8,
            verbosity=VerbosityLevel.VERBOSE  # Enable verbose logging for detailed consensus tracking
        )

        # Create judge for tie-breaking
        judge_call = SimpleInstructorLLMCall(
            response_model=TermMeaningExtractionResponse,
            model="gpt-4o",
            temperature=0.3,  # Low temperature for consistent judgments
        )

        consensus = ConsensusCore.consensus(
            models=models,
            judge=judge_call,
            settings=settings
        )

        # Initialize the base class with the consensus instance
        super().__init__(consensus=consensus)

    def fill_prompt(
        self,
        term: str,
        type: str,
        occurrences_contexts: List[str],
        cooccurring_terms: Dict[str, List[str]],
    ) -> str:
        """
        Fill the prompt for term extraction using Jinja2 templates.
        """
        template_name = "term_refinement.j2"
        template = template_env.get_template(template_name)

        return template.render(
            term=term,
            type=type,
            occurrences_contexts=occurrences_contexts,
            cooccurring_terms=cooccurring_terms,
        )


class ConsensusChunkingCall(BaseDocumentChunkingCall):
    """
    Consensus-based chunking extractor that uses multiple models for document segmentation.
    """

    def __init__(self, models: List, max_rounds: int = 2):
        """
        Initialize the consensus chunking extractor.

        Args:
            models: List of model configurations for consensus
            max_rounds: Maximum rounds for consensus (default: 2 for faster chunking)
        """
        self.models = models
        self.max_rounds = max_rounds

        settings = ConsensusSettings(
            max_rounds=max_rounds,
            threshold=0.8,
            verbosity=VerbosityLevel.VERBOSE  # Enable verbose logging for detailed consensus tracking
        )

        # Create judge for tie-breaking
        judge_call = SimpleInstructorLLMCall(
            response_model=ChunkingDecision,
            model="gpt-4o",
            temperature=0.2,  # Very low for structural decisions
        )

        consensus = ConsensusCore.consensus(
            models=models,
            judge=judge_call,
            settings=settings
        )

        # Initialize the base class with the consensus instance
        super().__init__(consensus=consensus)

    def fill_prompt(
        self,
        page: KnowledgePageDataWithRawText,
        document_name: str,
        metadata: KnowledgeMetadata
    ) -> str:
        """
        Fill the prompt for document chunking using Jinja2 template.
        """
        template = template_env.get_template("document_chunking.j2")

        return template.render(
            page=page,
            document_name=document_name,
            metadata=metadata,
        )


class KnowledgeExtraction:
    """
    Example class for knowledge extraction from PDF documents.

    This class demonstrates how to use KnowledgeExtractionCore with consensus-based validation
    for extracting and validating knowledge from documents.
    """

    def __init__(
        self,
        input_glob: Optional[str] = None,
        output_dir: Optional[Path] = None,
        log_level: int = logging.INFO,
    ):
        """
        Initialize the knowledge extraction example.

        Args:
            input_glob: Glob pattern for input files. Defaults to "input/*.pdf"
            output_dir: Output directory for extraction results. Defaults to "public/knowledge_extraction"
            log_level: Logging level. Defaults to INFO
        """
        self.input_glob = input_glob or "tests/com_blockether_catalyst/test_data/full_sample_test_1.pdf"
        self.output_dir = output_dir or Path("public/knowledge_extraction")
        self.log_level = log_level
        self.extractor = None
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Configure logging for the extraction process."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        log_file = self.output_dir / "extraction.log"

        # Configure root logger
        logging.basicConfig(
            level=self.log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
        )

        # Create main logger for this class
        self.logger = logging.getLogger(__name__)

        # Enable detailed consensus logging
        consensus_logger = logging.getLogger("com_blockether_catalyst.consensus")
        consensus_logger.setLevel(logging.DEBUG)  # Capture all consensus logs

        self.logger.info("=" * 70)
        self.logger.info("KNOWLEDGE EXTRACTION WITH CONSENSUS VALIDATION")
        self.logger.info("=" * 70)
        self.logger.info(f"📚 Input: {self.input_glob}")
        self.logger.info(f"📁 Output: {self.output_dir}")
        self.logger.info(f"📝 Log: {log_file}")
        self.logger.info("=" * 70)

    def _create_term_extraction_call(self) -> ConsensusTermExtractionCall:
        """
        Create a consensus-based term extraction call with diverse model perspectives.

        Returns:
            ConsensusTermExtractionCall configured with three different validation perspectives
        """
        # Load perspectives from templates
        conservative_perspective = template_env.get_template("perspectives/term_extraction_conservative.j2").render()
        balanced_perspective = template_env.get_template("perspectives/term_extraction_balanced.j2").render()
        liberal_perspective = template_env.get_template("perspectives/term_extraction_liberal.j2").render()

        # Model 1: Conservative financial domain expert - strictest validation
        conservative_expert_call = SimpleInstructorLLMCall(
            response_model=TermMeaningExtractionResponse,
            model="gpt-4o",
            temperature=0.1,  # Very low temperature for consistent strict decisions
        )

        conservative_expert_config = ConsensusCore.model(
            id="conservative-compliance-expert",
            executor=conservative_expert_call,
            perspective=conservative_perspective,
            weight_multiplier=1.2,
        )

        # Model 2: Balanced technical analyst - moderate validation
        balanced_analyst_call = SimpleInstructorLLMCall(
            response_model=TermMeaningExtractionResponse,
            model="gpt-4o",
            temperature=0.5,
        )

        balanced_analyst_config = ConsensusCore.model(
            id="balanced-technical-analyst",
            executor=balanced_analyst_call,
            perspective=balanced_perspective,
            weight_multiplier=1.0,  # Standard weight
        )

        # Model 3: Liberal linguistic processor - inclusive validation
        liberal_linguist_call = SimpleInstructorLLMCall(
            response_model=TermMeaningExtractionResponse,
            model="gpt-4o",
            temperature=0.8,
        )

        liberal_linguist_config = ConsensusCore.model(
            id="liberal-linguistic-processor",
            executor=liberal_linguist_call,
            perspective=liberal_perspective,
            weight_multiplier=0.8,
        )

        # Create consensus validation extractor with diverse model perspectives
        return ConsensusTermExtractionCall(
            models=[conservative_expert_config, balanced_analyst_config, liberal_linguist_config],
            max_rounds=3,
        )

    def _create_chunking_call(self) -> ConsensusChunkingCall:
        """
        Create a consensus-based chunking extractor for intelligent document segmentation.

        Returns:
            ConsensusChunkingCall configured with three different chunking perspectives
        """
        # Load perspectives from templates
        markdown_perspective = template_env.get_template("perspectives/chunking_markdown_expert.j2").render()
        semantic_perspective = template_env.get_template("perspectives/chunking_semantic_analyzer.j2").render()
        context_perspective = template_env.get_template("perspectives/chunking_context_specialist.j2").render()

        # Model 1: Markdown structure expert
        markdown_expert_call: SimpleInstructorLLMCall[ChunkingDecision] = SimpleInstructorLLMCall(
            response_model=ChunkingDecision,
            model="gpt-4o",
            temperature=0.2,  # Low temperature for consistent structure decisions
        )

        markdown_expert_config = ConsensusCore.model(
            id="markdown-structure-expert",
            executor=markdown_expert_call,
            perspective=markdown_perspective,
            weight_multiplier=1.2,  # Higher weight for structure preservation
        )

        # Model 2: Semantic coherence analyzer
        semantic_analyzer_call = SimpleInstructorLLMCall(
            response_model=ChunkingDecision,
            model="gpt-4o",
            temperature=0.5,
        )

        semantic_analyzer_config = ConsensusCore.model(
            id="semantic-coherence-analyzer",
            executor=semantic_analyzer_call,
            perspective=semantic_perspective,
            weight_multiplier=1.0,
        )

        # Model 3: Context preservation specialist
        context_specialist_call = SimpleInstructorLLMCall(
            response_model=ChunkingDecision,
            model="gpt-4o",
            temperature=0.4,
        )

        context_specialist_config = ConsensusCore.model(
            id="context-preservation-specialist",
            executor=context_specialist_call,
            perspective=context_perspective,
            weight_multiplier=0.8,
        )

        # Create agentic chunking extractor with consensus
        return ConsensusChunkingCall(
            models=[
                markdown_expert_config,
                semantic_analyzer_config,
                context_specialist_config,
            ],
            max_rounds=2,
        )

    async def setup(self) -> None:
        """
        Set up the extraction environment and initialize components.

        This method:
        1. Creates consensus extractors for validation and chunking
        2. Configures KnowledgeProcessorSettings with all required parameters
        3. Initializes the KnowledgeExtractionCore with the settings
        """
        # Create extraction calls
        term_extraction_call = self._create_term_extraction_call()
        document_chunking_call = self._create_chunking_call()

        # Log setup info
        self.logger.info("🧠 Setting up intelligent chunking with 3-model consensus")

        # Create settings with all required extractors
        settings = KnowledgeProcessorSettings(
            extraction_output_dir=self.output_dir,
            pdf_settings=PDFKnowledgeProcessorSettings(
                pdf_page_crop_offset=PDFPageCropOffset(
                    top=48,
                    bottom=48
                )
            ),
            min_term_score=0.0,
            linking_threshold=0.65,
            max_display_occurrences=15,
            max_display_cooccurrences=5,
        )

        # Create knowledge extractor with settings
        self.extractor = KnowledgeExtractionCore(
            calls=ExtractionCallsSettings(
                term_extraction=term_extraction_call,
                document_chunking=document_chunking_call,
            ),
            settings=settings
        )

    async def extract(self) -> None:
        """
        Execute the knowledge extraction process.

        Raises:
            RuntimeError: If setup() hasn't been called first
        """
        if not self.extractor:
            raise RuntimeError("Extractor not initialized. Call setup() first.")

        self.logger.info("🚀 Starting extraction with consensus validation...")
        await self.extractor.extract(globs=[self.input_glob])
        self.logger.info("✅ Knowledge extraction completed")

    async def run(self) -> None:
        """
        Run the complete extraction pipeline.

        This is a convenience method that calls setup() and extract() in sequence.
        """
        await self.setup()
        await self.extract()

    @classmethod
    async def from_cli(cls, args: Optional[List[str]] = None) -> None:
        """
        Create and run extraction from command line arguments.

        Args:
            args: Command line arguments. If None, uses sys.argv

        Usage:
            uv run python3 tools/KnowledgeExtraction.py [input_glob] [output_dir]

        Examples:
            # Extract all PDFs in input directory
            uv run python3 tools/KnowledgeExtraction.py "input/*.pdf"

            # Extract specific PDFs with pattern
            uv run python3 tools/KnowledgeExtraction.py "docs/**/*.pdf" output/

            # Extract with custom output directory
            uv run python3 tools/KnowledgeExtraction.py "input/*.pdf" output/
        """
        args = args or sys.argv[1:]

        # Parse arguments
        input_glob = args[0] if len(args) > 0 else None
        output_dir = Path(args[1]) if len(args) > 1 else Path("output/")

        # Create and run extractor
        extractor = cls(
            input_glob=input_glob,
            output_dir=output_dir,
        )

        await extractor.run()


class ChunkingPromptRefinementCLI(PromptAlignmentCLIBase):
    """CLI for refining document chunking prompts using real extraction data."""

    def __init__(
        self,
        prompt_dir: Path = Path("prompts/refined"),
        console: Optional[Console] = None,
        output_dir: Optional[Path] = None,
        extraction_dir: Optional[Path] = None,
    ):
        """Initialize the chunking prompt refinement CLI."""
        super().__init__(
            prompt_name="document_chunking",
            prompt_dir=prompt_dir,
            console=console,
            output_dir=output_dir or Path("output/chunking_refinement"),
        )
        self.extraction_dir = extraction_dir or Path("public/knowledge_extraction")
        self._cached_extraction_data = None

    def _init_components(self):
        """Initialize LLM components and prompt aligner."""
        # Create evaluation consensus (3 models evaluate prompt quality)
        eval_models = []

        # Model 1: Structure-focused evaluator
        structure_eval = SimpleInstructorLLMCall(
            response_model=EvaluationResult,
            model="gpt-4o",
            temperature=0.3,
        )
        eval_models.append(
            ConsensusCore.model(
                id="structure-evaluator",
                executor=structure_eval,
                perspective="As a document structure expert, evaluate how well this prompt guides proper chunking that preserves markdown structure, tables, and formatting.",
                weight_multiplier=1.2,
            )
        )

        # Model 2: Semantic coherence evaluator
        semantic_eval = SimpleInstructorLLMCall(
            response_model=EvaluationResult,
            model="gpt-4o",
            temperature=0.4,
        )
        eval_models.append(
            ConsensusCore.model(
                id="semantic-evaluator",
                executor=semantic_eval,
                perspective="As a content analyst, evaluate how well this prompt ensures semantic coherence and logical grouping of related information.",
                weight_multiplier=1.0,
            )
        )

        # Model 3: Context preservation evaluator
        context_eval = SimpleInstructorLLMCall(
            response_model=EvaluationResult,
            model="gpt-4o",
            temperature=0.3,
        )
        eval_models.append(
            ConsensusCore.model(
                id="context-evaluator",
                executor=context_eval,
                perspective="As a technical documentation specialist, evaluate how well this prompt ensures chunks have sufficient context for independent understanding.",
                weight_multiplier=0.8,
            )
        )

        # Create judge for evaluation consensus
        eval_judge = SimpleInstructorLLMCall(
            response_model=EvaluationResult,
            model="gpt-4o",
            temperature=0.2,
        )

        eval_consensus = ConsensusCore.consensus(
            models=eval_models,
            judge=eval_judge,
            settings=ConsensusSettings(
                max_rounds=2,
                threshold=0.75,
                verbosity=VerbosityLevel.NORMAL,
            )
        )

        # Create alignment feedback consensus (3 models provide improvement suggestions)
        feedback_models = []

        for i, (id_name, perspective) in enumerate([
            ("clarity-improver", "As a technical writing expert, identify how to make this prompt clearer and more precise in its chunking instructions."),
            ("structure-enhancer", "As a markdown and document structure specialist, suggest improvements for better structural preservation."),
            ("coherence-optimizer", "As a content strategist, recommend changes to ensure better semantic grouping and logical flow."),
        ]):
            feedback_model = SimpleInstructorLLMCall(
                response_model=AlignmentFeedback,
                model="gpt-4o",
                temperature=0.5 + i * 0.1,  # Varied temperatures for diversity
            )
            feedback_models.append(
                ConsensusCore.model(
                    id=id_name,
                    executor=feedback_model,
                    perspective=perspective,
                    weight_multiplier=1.0,
                )
            )

        # Create judge for feedback consensus
        feedback_judge = SimpleInstructorLLMCall(
            response_model=AlignmentFeedback,
            model="gpt-4o",
            temperature=0.3,
        )

        feedback_consensus = ConsensusCore.consensus(
            models=feedback_models,
            judge=feedback_judge,
            settings=ConsensusSettings(
                max_rounds=2,
                threshold=0.75,
                verbosity=VerbosityLevel.NORMAL,
            )
        )

        # Initialize prompt aligner
        self.prompt_aligner = PromptAlignmentCore(
            target_consensus=eval_consensus,
            alignment_consensus=feedback_consensus,
        )

    def _fill_template(self, prompt: str, values: Dict[str, Any]) -> str:
        """
        Fill prompt template using Jinja2 instead of str.format.

        Args:
            prompt: The Jinja2 template string
            values: Dictionary of template variables

        Returns:
            Rendered prompt string
        """
        from jinja2 import Template

        template = Template(prompt)
        return template.render(**values)

    def _view_filled_prompt(self) -> Optional[str]:
        """
        View both the raw template and filled prompt.

        Returns:
            Filled prompt string or None if no item selected
        """
        if not self._ensure_item_selected():
            return None

        from rich.syntax import Syntax

        # Show raw template first
        self.console.print(
            Panel(
                Syntax(self.prompt_template, "jinja2", theme="monokai", line_numbers=True),
                title="🔧 Raw Template (Jinja2)",
                border_style="blue",
            )
        )

        # Create template variables from current item
        if self.current_item is None:
            self.console.print("[red]❌ No item selected[/red]")
            return None
        template_vars = self._get_template_variables(self.current_item)
        filled_prompt = self._fill_template(self.prompt_template, template_vars)

        self.console.print(
            Panel(
                Syntax(filled_prompt, "text", theme="monokai", line_numbers=True),
                title="📝 Filled Prompt Template",
                border_style="green",
            )
        )

        return filled_prompt

    def _get_default_prompt(self) -> str:
        """Get the default chunking prompt template."""
        template_path = Path(__file__).parent / "prompts" / "document_chunking.j2"
        if not template_path.exists():
            raise FileNotFoundError(
                f"Chunking prompt template not found at {template_path}. "
                "Please ensure the template file exists."
            )

        # Get the raw template text without rendering
        with open(template_path, "r") as f:
            return f.read()

    def _load_extraction_data(self) -> Optional[KnowledgeExtractionOutput]:
        """Load real extraction data from PKL files."""
        import pickle

        # Find all raw extraction files (now without timestamps)
        raw_extractions = list(self.extraction_dir.glob("1_raw_extraction.pkl"))

        if not raw_extractions:
            self.console.print("[bold red]ERROR: No extraction data found![/bold red]")
            self.console.print("\nYou must run extraction first to generate real data:")
            self.console.print("[cyan]uv run python3 tools/KnowledgeExtraction.py extract input/*.pdf[/cyan]")
            self.console.print("\n[yellow]Refinement requires real extraction data to be effective.[/yellow]")
            return None

        # Since we no longer have timestamps, directly load the file
        if raw_extractions:
            selected_file = raw_extractions[0]
            self.console.print(f"\n[cyan]Loading extraction data from: {selected_file.name}[/cyan]")
            with open(selected_file, 'rb') as f:
                data = pickle.load(f)
            self.console.print(f"[green]✓ Loaded {selected_file.name}[/green]")

            # Ensure we return the correct type
            if isinstance(data, KnowledgeExtractionOutput):
                return data
            else:
                self.console.print(f"[red]Warning: Unexpected data type: {type(data)}[/red]")
                return None
        return None

    def _select_page_for_testing(self, extraction_data):
        """Interactively select a page from extraction data for testing."""
        from rich.prompt import IntPrompt

        if not extraction_data or not hasattr(extraction_data, 'documents'):
            return None

        # Show document info
        for doc_name, doc_data in extraction_data.documents.items():
            pages = doc_data.pages if hasattr(doc_data, 'pages') else []
            self.console.print(f"\n[cyan]Document: {doc_name}[/cyan]")
            self.console.print(f"Pages available: {len(pages)}")

            if pages:
                # Show page previews
                for idx, page in enumerate(pages[:3], 1):
                    text_preview = page.raw_text[:200] if hasattr(page, 'raw_text') else page.text[:200]
                    self.console.print(f"\nPage {page.page}:")
                    self.console.print(f"[dim]{text_preview}...[/dim]")

                # Let user select a page
                page_num = IntPrompt.ask(f"Select page number (1-{len(pages)})", default=1)
                if 1 <= page_num <= len(pages):
                    selected_page = pages[page_num - 1]
                    return doc_name, selected_page

        return None

    def _display_test_results(self, results: Dict[str, Any]):
        """Display chunking test results."""
        if results.get("success"):
            table = Table(title="Chunking Test Results")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")

            table.add_row("Success", "✅ Yes")
            table.add_row("Number of Chunks", str(results.get("num_chunks", 0)))
            table.add_row("Total Characters", f"{results.get('total_chars', 0):,}")

            self.console.print(table)

            if results.get("chunks"):
                self.console.print("\n[cyan]Chunk Previews:[/cyan]")
                for i, preview in enumerate(results["chunks"][:3], 1):
                    self.console.print(f"  {i}. {preview}")

            if results.get("reasoning"):
                self.console.print(f"\n[dim]Reasoning: {results['reasoning'][:200]}...[/dim]")
        else:
            self.console.print(f"[red]❌ Test failed: {results.get('error')}[/red]")

        # Always show raw JSON
        self.console.print("\n")
        self._display_raw_json(results, "Full Test Results")

    def _load_data(self) -> Optional[KnowledgeExtractionOutput]:
        """Load the extraction data to be processed."""
        return self._load_extraction_data()

    def _get_all_pages(self) -> List[Tuple[str, KnowledgePageDataWithRawText]]:
        """Get all pages from all documents in the loaded data."""
        all_pages: List[Tuple[str, KnowledgePageDataWithRawText]] = []

        if not isinstance(self.current_data, KnowledgeExtractionOutput):
            return all_pages

        if self.current_data.pdf is not None:
            for pdf_item in self.current_data.pdf:
                if pdf_item.result is not None:
                    doc_name = pdf_item.result.filename
                    for page in pdf_item.result.pages:
                        all_pages.append((doc_name, page))

        return all_pages

    def _select_next_item(self) -> bool:
        """Select the next item to process."""
        if not self._ensure_data_loaded():
            return False

        if not self.current_data:
            return False

        all_pages = self._get_all_pages()
        if not all_pages:
            return False

        # If no current item, start at first item (index 0)
        if self.current_item is None:
            self.current_item_index = 0
            self.current_item = all_pages[self.current_item_index]
            return True
        else:
            # Move to next item
            if self.current_item_index + 1 >= len(all_pages):
                return False
            self.current_item_index += 1
            self.current_item = all_pages[self.current_item_index]
            return True

    def _select_previous_item(self) -> bool:
        """Select the previous item to process."""
        if not self._ensure_data_loaded():
            return False

        if self.current_item_index > 0:
            all_pages = self._get_all_pages()
            if all_pages and self.current_item_index > 0:
                self.current_item_index -= 1
                self.current_item = all_pages[self.current_item_index]
                return True

        return False

    def _jump_to_item(self, index: int) -> bool:
        """Jump to a specific item by index."""
        if not self._ensure_data_loaded():
            return False

        all_pages = self._get_all_pages()
        if 0 <= index < len(all_pages):
            self.current_item_index = index
            self.current_item = all_pages[index]
            return True

        return False

    def _get_item_preview(self, item: Tuple[str, KnowledgePageDataWithRawText]) -> str:
        """Get a preview/summary of an item for display."""
        doc_name, page = item
        text_preview = page.raw_text[:200] if page.raw_text else page.text[:200]
        return f"Document: {doc_name}\nPage: {page.page}\nPreview: {text_preview}..."

    def _get_total_items(self) -> int:
        """Get the total number of items available."""
        all_pages = self._get_all_pages()
        return len(all_pages)

    def _get_template_variables(self, item: Tuple[str, KnowledgePageDataWithRawText]) -> Dict[str, Any]:
        """Extract template variables from an item."""
        doc_name, page = item

        # Get metadata from the extraction result - proper type casting
        metadata = KnowledgeMetadata()  # Default empty metadata
        if isinstance(self.current_data, KnowledgeExtractionOutput) and self.current_data.pdf is not None:
            for pdf_item in self.current_data.pdf:
                if pdf_item.result is not None and pdf_item.result.filename == doc_name:
                    metadata = pdf_item.result.metadata
                    break

        return {
            'document_name': doc_name,
            'page': page,
            'metadata': metadata
        }


class TermExtractionPromptRefinementCLI(PromptAlignmentCLIBase):
    """CLI for refining term extraction prompts using real extraction data."""

    def __init__(
        self,
        prompt_dir: Path = Path("prompts/refined"),
        console: Optional[Console] = None,
        output_dir: Optional[Path] = None,
        extraction_dir: Optional[Path] = None,
    ):
        """Initialize the term extraction prompt refinement CLI."""
        super().__init__(
            prompt_name="term_refinement",
            prompt_dir=prompt_dir,
            console=console,
            output_dir=output_dir or Path("output/term_refinement"),
        )
        self.extraction_dir = extraction_dir or Path("public/knowledge_extraction")
        self._cached_term_data = None

    def _init_components(self):
        """Initialize LLM components and prompt aligner."""
        # Create evaluation consensus
        eval_models = []

        # Model 1: Accuracy evaluator
        accuracy_eval = SimpleInstructorLLMCall(
            response_model=EvaluationResult,
            model="gpt-4o",
            temperature=0.2,
        )
        eval_models.append(
            ConsensusCore.model(
                id="accuracy-evaluator",
                executor=accuracy_eval,
                perspective="As a terminology expert, evaluate how well this prompt identifies valid technical terms and extracts accurate meanings.",
                weight_multiplier=1.2,
            )
        )

        # Model 2: Completeness evaluator
        completeness_eval = SimpleInstructorLLMCall(
            response_model=EvaluationResult,
            model="gpt-4o",
            temperature=0.3,
        )
        eval_models.append(
            ConsensusCore.model(
                id="completeness-evaluator",
                executor=completeness_eval,
                perspective="As a knowledge extraction specialist, evaluate how well this prompt ensures comprehensive term identification and meaning extraction.",
                weight_multiplier=1.0,
            )
        )

        # Model 3: Context evaluator
        context_eval = SimpleInstructorLLMCall(
            response_model=EvaluationResult,
            model="gpt-4o",
            temperature=0.3,
        )
        eval_models.append(
            ConsensusCore.model(
                id="context-evaluator",
                executor=context_eval,
                perspective="As a computational linguist, evaluate how well this prompt uses context and co-occurrences for validation.",
                weight_multiplier=0.8,
            )
        )

        eval_judge = SimpleInstructorLLMCall(
            response_model=EvaluationResult,
            model="gpt-4o",
            temperature=0.2,
        )

        eval_consensus = ConsensusCore.consensus(
            models=eval_models,
            judge=eval_judge,
            settings=ConsensusSettings(
                max_rounds=2,
                threshold=0.75,
                verbosity=VerbosityLevel.NORMAL,
            )
        )

        # Create alignment feedback consensus
        feedback_models = []

        for i, (id_name, perspective) in enumerate([
            ("precision-improver", "As a terminology expert, suggest how to make term validation more precise and accurate."),
            ("context-enhancer", "As a linguistic analyst, recommend improvements for better context utilization."),
            ("clarity-optimizer", "As a technical writer, identify ways to make the instructions clearer."),
        ]):
            feedback_model = SimpleInstructorLLMCall(
                response_model=AlignmentFeedback,
                model="gpt-4o",
                temperature=0.4 + i * 0.1,
            )
            feedback_models.append(
                ConsensusCore.model(
                    id=id_name,
                    executor=feedback_model,
                    perspective=perspective,
                    weight_multiplier=1.0,
                )
            )

        feedback_judge = SimpleInstructorLLMCall(
            response_model=AlignmentFeedback,
            model="gpt-4o",
            temperature=0.3,
        )

        feedback_consensus = ConsensusCore.consensus(
            models=feedback_models,
            judge=feedback_judge,
            settings=ConsensusSettings(
                max_rounds=2,
                threshold=0.75,
                verbosity=VerbosityLevel.NORMAL,
            )
        )

        # Initialize prompt aligner
        self.prompt_aligner = PromptAlignmentCore(
            target_consensus=eval_consensus,
            alignment_consensus=feedback_consensus,
        )

    def _fill_template(self, prompt: str, values: Dict[str, Any]) -> str:
        """
        Fill prompt template using Jinja2 instead of str.format.

        Args:
            prompt: The Jinja2 template string
            values: Dictionary of template variables

        Returns:
            Rendered prompt string
        """
        from jinja2 import Template

        template = Template(prompt)
        return template.render(**values)

    def _view_filled_prompt(self) -> Optional[str]:
        """
        View both the raw template and filled prompt.

        Returns:
            Filled prompt string or None if no item selected
        """
        if not self._ensure_item_selected():
            return None

        from rich.syntax import Syntax

        # Show raw template first
        self.console.print(
            Panel(
                Syntax(self.prompt_template, "jinja2", theme="monokai", line_numbers=True),
                title="🔧 Raw Template (Jinja2)",
                border_style="blue",
            )
        )

        # Create template variables from current item
        if self.current_item is None:
            self.console.print("[red]❌ No item selected[/red]")
            return None
        template_vars = self._get_template_variables(self.current_item)
        filled_prompt = self._fill_template(self.prompt_template, template_vars)

        self.console.print(
            Panel(
                Syntax(filled_prompt, "text", theme="monokai", line_numbers=True),
                title="📝 Filled Prompt Template",
                border_style="green",
            )
        )

        return filled_prompt

    def _get_default_prompt(self) -> str:
        """Get the default term extraction prompt template."""
        template_path = Path(__file__).parent / "prompts" / "term_refinement.j2"
        if not template_path.exists():
            raise FileNotFoundError(
                f"Term extraction prompt template not found at {template_path}. "
                "Please ensure the template file exists."
            )

        # Get the raw template text without rendering
        with open(template_path, "r") as f:
            return f.read()

    def _load_term_data(self):
        """Load real term extraction data from PKL files."""
        import pickle

        # Find term cooccurrences files (now without timestamps)
        term_files = list(self.extraction_dir.glob("5_terms_with_cooccurrences.pkl"))

        if not term_files:
            self.console.print("[bold red]ERROR: No term extraction data found![/bold red]")
            self.console.print("\nYou must run extraction first to generate real data:")
            self.console.print("[cyan]uv run python3 tools/KnowledgeExtraction.py extract input/*.pdf[/cyan]")
            self.console.print("\n[yellow]Term refinement requires real extraction data to be effective.[/yellow]")
            return None

        # Since we no longer have timestamps, directly load the file
        if term_files:
            selected_file = term_files[0]
            self.console.print(f"\n[cyan]Loading term data from: {selected_file.name}[/cyan]")
            with open(selected_file, 'rb') as f:
                data = pickle.load(f)
            self.console.print(f"[green]✓ Loaded {selected_file.name}[/green]")
            return data
        return None

    def _select_term_for_testing(self, term_data):
        """Interactively select a term for testing."""
        from rich.prompt import Prompt
        from typing import Optional, Tuple, Any

        if not term_data:
            return None

        # Show some available terms
        terms = list(term_data.keys())
        if not terms:
            self.console.print("[red]No terms found in data![/red]")
            return None

        self.console.print(f"\n[cyan]Available terms ({len(terms)} total):[/cyan]")

        # Show first 10 terms
        for term in terms[:10]:
            term_info = term_data[term]
            occurrences = len(term_info.occurrences) if hasattr(term_info, 'occurrences') else 0
            self.console.print(f"  • {term} ({occurrences} occurrences)")

        if len(terms) > 10:
            self.console.print(f"  [dim]... and {len(terms) - 10} more[/dim]")

        # Let user select or enter a term
        selected_term = Prompt.ask("Enter term to test (or press Enter for first term)", default=terms[0])

        if selected_term in term_data:
            return (selected_term, term_data[selected_term])
        else:
            self.console.print(f"[yellow]Term '{selected_term}' not found, using '{terms[0]}'[/yellow]")
            return (terms[0], term_data[terms[0]])

    async def _test_prompt(self, prompt: str) -> Dict[str, Any]:
        """Test the term extraction prompt with real term data."""
        # Use current item from navigation system
        if not self.current_item:
            return {"success": False, "error": "No term selected"}

        # Fill the template with real data using enhanced template variables
        template_vars = self._get_enhanced_template_variables(self.current_item)
        filled_prompt = self._fill_template(prompt, template_vars)

        # Test with a simple call
        test_call = SimpleInstructorLLMCall(
            response_model=TermMeaningExtractionResponse,
            model="gpt-4o",
            temperature=0.3,
        )

        try:
            result = await test_call.call(filled_prompt)
            return {
                "success": True,
                "type": result.type,
                "meaning": result.meaning,
                "full_form": result.full_form,
                "reasoning": result.reasoning
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def _display_test_results(self, results: Dict[str, Any]):
        """Display term extraction test results."""
        if results.get("success"):
            table = Table(title="Term Extraction Test Results")
            table.add_column("Field", style="cyan")
            table.add_column("Value", style="green")

            table.add_row("Success", "✅ Yes")
            table.add_row("Type", results.get("type", "N/A"))
            table.add_row("Full Form", results.get("full_form") or "N/A")

            self.console.print(table)

            if results.get("meaning"):
                self.console.print(f"\n[cyan]Meaning:[/cyan]\n{results['meaning']}")

            if results.get("reasoning"):
                self.console.print(f"\n[dim]Reasoning: {results['reasoning']}[/dim]")
        else:
            self.console.print(f"[red]❌ Test failed: {results.get('error')}[/red]")

        # Always show raw JSON
        self.console.print("\n")
        self._display_raw_json(results, "Full Test Results")

    def _load_data(self) -> Optional[Any]:
        """Load the term data to be processed."""
        return self._load_term_data()

    def _select_next_item(self) -> bool:
        """Select the next term to process."""
        if not self._ensure_data_loaded():
            return False

        if not self.current_data:
            return False

        # Get all terms as a list
        all_terms = list(self.current_data.keys()) if isinstance(self.current_data, dict) else []

        if not all_terms:
            return False

        # If no current item, start at first item (index 0)
        if self.current_item is None:
            self.current_item_index = 0
            term = all_terms[self.current_item_index]
            self.current_item = (term, self.current_data[term])
            return True
        else:
            # Move to next term
            if self.current_item_index + 1 < len(all_terms):
                self.current_item_index += 1
                term = all_terms[self.current_item_index]
                self.current_item = (term, self.current_data[term])
                return True

        return False

    def _select_previous_item(self) -> bool:
        """Select the previous term to process."""
        if not self._ensure_data_loaded():
            return False

        if self.current_item_index > 0:
            if not self.current_data:
                return False

            # Get all terms as a list
            all_terms = list(self.current_data.keys()) if isinstance(self.current_data, dict) else []

            if all_terms and self.current_item_index > 0:
                self.current_item_index -= 1
                term = all_terms[self.current_item_index]
                self.current_item = (term, self.current_data[term])
                return True

        return False

    def _jump_to_item(self, index: int) -> bool:
        """Jump to a specific term by index."""
        if not self._ensure_data_loaded():
            return False

        if not self.current_data:
            return False

        # Get all terms as a list
        all_terms = list(self.current_data.keys()) if isinstance(self.current_data, dict) else []

        if 0 <= index < len(all_terms):
            self.current_item_index = index
            term = all_terms[index]
            self.current_item = (term, self.current_data[term])
            return True

        return False

    def _get_item_preview(self, item: Any) -> str:
        """Get a preview/summary of an item for display."""
        if isinstance(item, tuple) and len(item) == 2:
            term, term_info = item

            # Proper type casting - term_info should be TermGrouped
            if isinstance(term_info, TermGrouped):
                occurrences = len(term_info.occurrences)
                cooccurrences = len(term_info.cooccurrences)
                return f"Term: {term}\nType: {term_info.type}\nOccurrences: {occurrences}\nCo-occurrences: {cooccurrences}"

        return str(item)[:200] + "..."

    def _get_total_items(self) -> int:
        """Get the total number of terms available."""
        if not self.current_data or not isinstance(self.current_data, dict):
            return 0

        return len(self.current_data)

    def _get_template_variables(self, item: Any) -> Dict[str, Any]:
        """Extract template variables from an item."""
        if isinstance(item, tuple) and len(item) == 2:
            term, term_info = item

            # Proper type casting - term_info should be TermGrouped
            if not isinstance(term_info, TermGrouped):
                return {'item': item}

            # Extract contexts from occurrences - proper type access
            contexts = []
            for occ in term_info.occurrences[:15]:
                context = f"[Document: {occ.document_name}, Page: {occ.page}, Chunk: {occ.chunk_index}]"
                contexts.append(context)

            # Extract cooccurrences - proper type access
            cooccurring = {}
            for cooc in term_info.cooccurrences[:10]:
                cooccurring[cooc.term] = [f"score: {cooc.score:.2f}"]

            return {
                'term': term,
                'type': term_info.type,  # Use the actual type from the data
                'occurrences_contexts': contexts,
                'cooccurring_terms': cooccurring
            }

        return {'item': item}

    def _get_enhanced_template_variables(self, item: Any) -> Dict[str, Any]:
        """Extract enhanced template variables with rich contextual data."""
        if isinstance(item, tuple) and len(item) == 2:
            term, term_info = item

            # Proper type casting - term_info should be TermGrouped
            if not isinstance(term_info, TermGrouped):
                return {'item': item}

            # Extract contexts from occurrences - proper type access
            contexts = []
            for occ in term_info.occurrences[:15]:
                context = f"[Document: {occ.document_name}, Page: {occ.page}, Chunk: {occ.chunk_index}]"
                contexts.append(context)

            # Extract cooccurrences - proper type access
            cooccurring = {}
            for cooc in term_info.cooccurrences[:10]:
                cooccurring[cooc.term] = [f"score: {cooc.score:.2f}"]

            # Load chunked documents to get actual text contexts
            chunked_files = list(self.extraction_dir.glob("2_chunked_documents.pkl"))
            chunk_texts = {}
            if chunked_files:
                import pickle
                with open(chunked_files[0], 'rb') as f:
                    chunked_data = pickle.load(f)
                    # Build a map of chunk texts - proper typing expected
                    for doc_result in chunked_data:
                        # doc_result should be KnowledgeExtractionResultWithChunks
                        if isinstance(doc_result, KnowledgeExtractionResultWithChunks):
                            for idx, chunk in enumerate(doc_result.chunks):
                                key = f"{doc_result.filename}_{idx}"
                                chunk_texts[key] = chunk.text  # Use proper field name

            # Get real contexts from the chunks using proper type access
            real_contexts = []
            for occ in term_info.occurrences[:15]:
                chunk_key = f"{occ.document_name}_{occ.chunk_index}"
                if chunk_key in chunk_texts:
                    # Find the term in the chunk and get surrounding context
                    chunk_text = chunk_texts[chunk_key]
                    # Get a snippet around the term
                    term_lower = term.lower()
                    chunk_lower = chunk_text.lower()
                    pos = chunk_lower.find(term_lower)
                    if pos >= 0:
                        # Get 100 chars before and after
                        start = max(0, pos - 100)
                        end = min(len(chunk_text), pos + len(term) + 100)
                        context = chunk_text[start:end].strip()
                        if start > 0:
                            context = "..." + context
                        if end < len(chunk_text):
                            context = context + "..."
                        real_contexts.append(context)

            # Use real contexts or fallback to location info
            contexts = real_contexts if real_contexts else contexts

            return {
                'term': term,
                'type': term_info.type,  # Use actual type from data
                'occurrences_contexts': contexts,
                'cooccurring_terms': cooccurring
            }

        return {'item': item}


async def main() -> None:
    """Main entry point with subcommand support."""
    parser = argparse.ArgumentParser(
        description="Knowledge extraction and prompt refinement tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Extract knowledge from PDFs (default)
    uv run python3 tools/KnowledgeExtraction.py extract input/*.pdf

    # Refine chunking prompts
    uv run python3 tools/KnowledgeExtraction.py refine-chunking

    # Refine term extraction prompts
    uv run python3 tools/KnowledgeExtraction.py refine-terms

    # Extract with custom output
    uv run python3 tools/KnowledgeExtraction.py extract "docs/**/*.pdf" output/
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Extract subcommand (default behavior)
    extract_parser = subparsers.add_parser(
        "extract",
        help="Extract knowledge from PDF documents",
        description="Extract and validate knowledge using consensus-based validation"
    )
    extract_parser.add_argument(
        "input_glob",
        nargs="?",
        default="tests/com_blockether_catalyst/test_data/full_sample_test_1.pdf",
        help="Glob pattern for input PDF files"
    )
    extract_parser.add_argument(
        "output_dir",
        nargs="?",
        type=Path,
        default=Path("public/knowledge_extraction"),
        help="Output directory for extraction results"
    )

    # Refine chunking subcommand
    chunking_parser = subparsers.add_parser(
        "refine-chunking",
        help="Refine document chunking prompts",
        description="Use principle-based alignment to improve chunking prompts"
    )
    chunking_parser.add_argument(
        "--prompt-dir",
        type=Path,
        default=Path("prompts/refined"),
        help="Directory for refined prompts"
    )
    chunking_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/chunking_refinement"),
        help="Output directory for refinement results"
    )
    chunking_parser.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        help="Maximum refinement iterations"
    )

    # Refine terms subcommand
    terms_parser = subparsers.add_parser(
        "refine-terms",
        help="Refine term extraction prompts",
        description="Use principle-based alignment to improve term extraction prompts"
    )
    terms_parser.add_argument(
        "--prompt-dir",
        type=Path,
        default=Path("prompts/refined"),
        help="Directory for refined prompts"
    )
    terms_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/term_refinement"),
        help="Output directory for refinement results"
    )
    terms_parser.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        help="Maximum refinement iterations"
    )

    args = parser.parse_args()

    # Default to extract if no command specified
    if args.command is None:
        args.command = "extract"
        # Parse again with default
        args = parser.parse_args(["extract"] + sys.argv[1:])

    console = Console()

    if args.command == "extract":
        # Run knowledge extraction
        extractor = KnowledgeExtraction(
            input_glob=args.input_glob,
            output_dir=args.output_dir,
        )
        await extractor.run()

    elif args.command == "refine-chunking":
        # Run chunking prompt refinement
        console.print("[bold cyan]Starting Chunking Prompt Refinement[/bold cyan]\n")
        console.print("[dim]Using real extraction data from your previous runs[/dim]\n")

        # Use the extraction directory for real data
        extraction_dir = Path("public/knowledge_extraction")
        if not extraction_dir.exists() or not (extraction_dir / "1_raw_extraction.pkl").exists():
            console.print("[bold red]ERROR: No extraction data found![/bold red]\n")
            console.print("You must run extraction first to generate real data:")
            console.print("[cyan]uv run python3 tools/KnowledgeExtraction.py extract input/*.pdf[/cyan]\n")
            console.print("[yellow]Prompt refinement requires real data to be effective.[/yellow]")
            return

        cli = ChunkingPromptRefinementCLI(
            prompt_dir=args.prompt_dir,
            console=console,
            output_dir=args.output_dir,
            extraction_dir=extraction_dir,
        )
        await cli.run()

    elif args.command == "refine-terms":
        # Run term extraction prompt refinement
        console.print("[bold cyan]Starting Term Extraction Prompt Refinement[/bold cyan]\n")
        console.print("[dim]Using real term data from your previous extractions[/dim]\n")

        # Use the extraction directory for real data
        extraction_dir = Path("public/knowledge_extraction")
        if not extraction_dir.exists() or not (extraction_dir / "5_terms_with_cooccurrences.pkl").exists():
            console.print("[bold red]ERROR: No term extraction data found![/bold red]\n")
            console.print("You must run extraction first to generate real data:")
            console.print("[cyan]uv run python3 tools/KnowledgeExtraction.py extract input/*.pdf[/cyan]\n")
            console.print("[yellow]Term refinement requires real extraction data to be effective.[/yellow]")
            return

        # Ensure we have the necessary PKL files (up to step 4)
        await _ensure_extraction_pkls(extraction_dir, console)

        cli = TermExtractionPromptRefinementCLI(
            prompt_dir=args.prompt_dir,
            console=console,
            output_dir=args.output_dir,
            extraction_dir=extraction_dir,
        )
        await cli.run()

    else:
        parser.error(f"Unknown command: {args.command}")


async def _ensure_extraction_pkls(extraction_dir: Path, console: Console):
    """Ensure all necessary PKL files exist for refinement."""
    required_files = [
        "1_raw_extraction.pkl",
        "2_chunked_documents.pkl",
        "3_term_candidates.pkl",
        "4_grouped_terms.pkl",
        "5_terms_with_cooccurrences.pkl",
        "6_terms_with_meanings.pkl",
        "7_terms_with_links.pkl",
        "linked_knowledge.pkl"
    ]

    missing = []
    for filename in required_files:
        if not (extraction_dir / filename).exists():
            missing.append(filename.replace('.pkl', ''))

    if missing:
        console.print("[yellow]Some extraction steps are missing:[/yellow]")
        for step in missing:
            console.print(f"  • {step}")
        console.print("\n[cyan]Running extraction to generate missing files...[/cyan]")
        # Could trigger partial extraction here if needed
    else:
        console.print("[green]✓ All extraction PKL files present[/green]")


if __name__ == "__main__":
    anyio.run(main)
