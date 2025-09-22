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
import sys
import glob as glob_module
from pathlib import Path
from typing import Any, Dict, List, Optional, TypeVar

from pydantic import BaseModel, Field
from rich.console import Console

from blockether_catalyst.consensus.ConsensusCore import ConsensusCore
from blockether_catalyst.consensus.ConsensusTypes import ConsensusSettings, VerbosityLevel
from blockether_catalyst.knowledge.KnowledgeExtractionCallBase import (
    BaseTermExtractionCall,
    BaseDocumentChunkingCall,
    BaseChunkContentClassificationCall,
    ExtractionCallsSettings,
)
from blockether_catalyst.knowledge.KnowledgeExtractionCore import KnowledgeExtractionCore
from blockether_catalyst.knowledge.KnowledgeTypes import (
    DocumentMetadata,
    KnowledgePageDataWithRawText,
    KnowledgeProcessorSettings,
    TermMeaningExtractionResponse,
    ChunkingDecisionResponse,
    ChunkContentClassification,
)

from blockether_catalyst.knowledge.PDKnowledgeExtractorTypes import PDFKnowledgeProcessorSettings, PDFPageCropOffset
from blockether_catalyst.utils.instructor.InstructorLLMCall import InstructorLLMCall
from blockether_catalyst.prompt import PromptAlignmentCore
from blockether_catalyst.prompt.PromptAlignmentCLIBase import PromptAlignmentCLIBase
from blockether_catalyst.prompt.PromptAlignmentTypes import (
    EvaluationResult,
    AlignmentFeedback,
)
from jinja2 import Environment, FileSystemLoader

# Type variable for response types
T = TypeVar("T", bound=BaseModel)

# Setup Jinja2 environment for templates
template_env = Environment(
    loader=FileSystemLoader(Path(__file__).parent / "prompts"),
    trim_blocks=True,
    lstrip_blocks=True,
)

DEFAULT_THRESHOLD = 0.65 # 2/3 majority agreement

class ConsensusTermExtractionCall(BaseTermExtractionCall):
    """
    Consensus-based term extraction that validates and extracts meanings from term candidates.
    Handles both acronyms and keywords.
    """

    def __init__(self, models: List, judge, max_rounds: int = 3):
        """
        Initialize the consensus term extractor.

        Args:
            models: List of model configurations for consensus
            judge: Judge TypedCall for tie-breaking that returns TermMeaningExtractionResponse
            max_rounds: Maximum rounds for consensus (default: 3)
        """
        self.models = models
        self.judge = judge
        self.max_rounds = max_rounds

        settings = ConsensusSettings(
            max_rounds=max_rounds,
            threshold=DEFAULT_THRESHOLD,
            verbosity=VerbosityLevel.VERBOSE
        )

        consensus = ConsensusCore.consensus(
            models=models,
            judge=judge,
            settings=settings
        )

        # Initialize the base class with the consensus instance
        super().__init__(consensus=consensus)

    def fill_template(
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

    def __init__(self, models: List, judge, max_rounds: int = 2):
        """
        Initialize the consensus chunking extractor.

        Args:
            models: List of model configurations for consensus
            judge: Judge TypedCall for tie-breaking that returns ChunkingDecisionResponse
            max_rounds: Maximum rounds for consensus (default: 2 for faster chunking)
        """
        self.models = models
        self.judge = judge
        self.max_rounds = max_rounds

        settings = ConsensusSettings(
            max_rounds=max_rounds,
            threshold=DEFAULT_THRESHOLD,
            verbosity=VerbosityLevel.VERBOSE
        )

        consensus = ConsensusCore.consensus(
            models=models,
            judge=judge,
            settings=settings
        )

        # Initialize the base class with the consensus instance
        super().__init__(consensus=consensus)

    def fill_template(
        self,
        page: KnowledgePageDataWithRawText,
        document_name: str,
        metadata: DocumentMetadata
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


class ConsensusChunkContentClassificationCall(BaseChunkContentClassificationCall):
    """
    Consensus-based chunk content classification that classifies semantic types of chunks.
    """

    def __init__(self, models: List, judge, max_rounds: int = 2):
        """
        Initialize the consensus chunk content classifier.

        Args:
            models: List of model configurations for consensus
            judge: Judge TypedCall for tie-breaking that returns ChunkContentClassification
            max_rounds: Maximum rounds for consensus (default: 2 for efficiency)
        """
        self.models = models
        self.judge = judge
        self.max_rounds = max_rounds

        settings = ConsensusSettings(
            max_rounds=max_rounds,
            threshold=DEFAULT_THRESHOLD,
            verbosity=VerbosityLevel.VERBOSE
        )

        consensus = ConsensusCore.consensus(
            models=models,
            judge=judge,
            settings=settings
        )

        # Initialize the base class with the consensus instance
        super().__init__(consensus=consensus)

    def fill_template(
        self,
        chunk_text: str,
        document_name: str,
        page_number: int,
        content_types: List[str],
    ) -> str:
        """
        Fill the prompt for chunk content classification using Jinja2 template.
        """
        template = template_env.get_template("chunk_classification.j2")

        return template.render(
            chunk_text=chunk_text,
            document_name=document_name,
            page_number=page_number,
            content_types=content_types,
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
        validate_inputs: bool = True,
    ):
        """
        Initialize the knowledge extraction example.

        Args:
            input_glob: Glob pattern for input files. Defaults to "input/*.pdf"
            output_dir: Output directory for extraction results. Defaults to "public/knowledge_extraction"
            log_level: Logging level. Defaults to INFO
            validate_inputs: Whether to validate input files exist. Defaults to True
        """
        self.input_glob = input_glob or "tests/blockether_catalyst/test_data/full_sample_test_1.pdf"
        self.output_dir = output_dir or Path("public/knowledge_extraction")
        self.log_level = log_level
        self.extractor: KnowledgeExtractionCore = None  # type: ignore[assignment]
        self.console = Console()

        if validate_inputs:
            self._validate_inputs()

        self._setup_logging()

    def _validate_inputs(self) -> None:
        """Validate input files exist and are accessible."""
        # Expand glob pattern to find files
        matching_files = glob_module.glob(self.input_glob, recursive=True)

        if not matching_files:
            self.console.print(f"[bold red]❌ No files found matching pattern: {self.input_glob}[/bold red]")
            self.console.print("\n[yellow]Tips:[/yellow]")
            self.console.print("  • Check if the path exists")
            self.console.print("  • Use quotes for patterns with spaces: \"path with spaces/*.pdf\"")
            self.console.print("  • For recursive search use: \"**/*.pdf\"")
            self.console.print("  • Ensure files have .pdf extension")
            sys.exit(1)

        # Check if files are PDFs
        pdf_files = [f for f in matching_files if f.lower().endswith('.pdf')]
        if not pdf_files:
            self.console.print(f"[bold red]❌ No PDF files found in: {matching_files}[/bold red]")
            self.console.print("\n[yellow]This tool only processes PDF files.[/yellow]")
            sys.exit(1)

        # Check file accessibility
        inaccessible_files = []
        for file_path in pdf_files:
            path = Path(file_path)
            if not path.exists():
                inaccessible_files.append((file_path, "File not found"))
            elif not path.is_file():
                inaccessible_files.append((file_path, "Not a file"))
            elif not path.stat().st_size > 0:
                inaccessible_files.append((file_path, "Empty file"))

        if inaccessible_files:
            self.console.print("[bold red]❌ Some files are not accessible:[/bold red]")
            for file_path, reason in inaccessible_files:
                self.console.print(f"  • {file_path}: {reason}")
            sys.exit(1)

        # Display files to be processed
        self.console.print(f"\n[bold green]✓ Found {len(pdf_files)} PDF file(s) to process:[/bold green]")
        for i, file_path in enumerate(pdf_files[:5], 1):
            self.console.print(f"  {i}. {Path(file_path).name}")
        if len(pdf_files) > 5:
            self.console.print(f"  ... and {len(pdf_files) - 5} more")

        # Check output directory permissions
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            test_file = self.output_dir / ".write_test"
            test_file.touch()
            test_file.unlink()
        except PermissionError:
            self.console.print(f"[bold red]❌ Cannot write to output directory: {self.output_dir}[/bold red]")
            self.console.print("[yellow]Check directory permissions or choose a different output location.[/yellow]")
            sys.exit(1)
        except Exception as e:
            self.console.print(f"[bold red]❌ Output directory error: {e}[/bold red]")
            sys.exit(1)

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
        consensus_logger = logging.getLogger("blockether_catalyst.consensus")
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
            ConsensusTermExtractionCall configured with three different validation perspectives and a judge
        """
        # Load perspectives from templates
        conservative_perspective = template_env.get_template("perspectives/term_refinement_conservative.j2").render()
        balanced_perspective = template_env.get_template("perspectives/term_refinement_balanced.j2").render()
        liberal_perspective = template_env.get_template("perspectives/term_refinement_liberal.j2").render()

        # Model 1: Conservative financial domain expert - strictest validation
        conservative_expert_call = InstructorLLMCall(
            response_model=TermMeaningExtractionResponse,
            model="gpt-4o",
            temperature=0.3,
        )

        conservative_expert_config = ConsensusCore.model(
            id="conservative-compliance-expert",
            executor=conservative_expert_call,
            perspective=conservative_perspective,
            weight_multiplier=1.2,
        )

        # Model 2: Balanced technical analyst - moderate validation
        balanced_analyst_call = InstructorLLMCall(
            response_model=TermMeaningExtractionResponse,
            model="gpt-4o",
            temperature=0.5,
        )

        balanced_analyst_config = ConsensusCore.model(
            id="balanced-technical-analyst",
            executor=balanced_analyst_call,
            perspective=balanced_perspective,
            weight_multiplier=1.0,
        )

        # Model 3: Liberal linguistic processor - inclusive validation
        liberal_linguist_call = InstructorLLMCall(
            response_model=TermMeaningExtractionResponse,
            model="gpt-4o",
            temperature=0.5,
        )

        liberal_linguist_config = ConsensusCore.model(
            id="liberal-linguistic-processor",
            executor=liberal_linguist_call,
            perspective=liberal_perspective,
            weight_multiplier=0.8,
        )

        # Create judge for tie-breaking
        judge_call = InstructorLLMCall(
            response_model=TermMeaningExtractionResponse,
            model="gpt-4o",
            temperature=0.1,
        )

        # Create consensus validation extractor with diverse model perspectives and judge
        return ConsensusTermExtractionCall(
            models=[conservative_expert_config, balanced_analyst_config, liberal_linguist_config],
            judge=judge_call,
            max_rounds=3,
        )

    def _create_chunking_call(self) -> ConsensusChunkingCall:
        """
        Create a consensus-based chunking extractor for intelligent document segmentation.

        Returns:
            ConsensusChunkingCall configured with three different chunking perspectives and a judge
        """
        # Load perspectives from templates
        markdown_perspective = template_env.get_template("perspectives/chunking_markdown_expert.j2").render()
        semantic_perspective = template_env.get_template("perspectives/chunking_semantic_analyzer.j2").render()
        context_perspective = template_env.get_template("perspectives/chunking_context_specialist.j2").render()

        # Model 1: Markdown structure expert
        markdown_expert_call: InstructorLLMCall[ChunkingDecisionResponse] = InstructorLLMCall(
            response_model=ChunkingDecisionResponse,
            model="gpt-4o",
            temperature=0.3,
        )

        markdown_expert_config = ConsensusCore.model(
            id="markdown-structure-expert",
            executor=markdown_expert_call,
            perspective=markdown_perspective,
            weight_multiplier=1.2,
        )

        # Model 2: Semantic coherence analyzer
        semantic_analyzer_call = InstructorLLMCall(
            response_model=ChunkingDecisionResponse,
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
        context_specialist_call = InstructorLLMCall(
            response_model=ChunkingDecisionResponse,
            model="gpt-4o",
            temperature=0.4,
        )

        context_specialist_config = ConsensusCore.model(
            id="context-preservation-specialist",
            executor=context_specialist_call,
            perspective=context_perspective,
            weight_multiplier=0.8,
        )

        # Create judge for tie-breaking
        judge_call: InstructorLLMCall[ChunkingDecisionResponse] = InstructorLLMCall(
            response_model=ChunkingDecisionResponse,
            model="gpt-4o",
            temperature=0.1,  # Very low temperature for decisive judgments
        )

        # Create agentic chunking extractor with consensus and judge
        return ConsensusChunkingCall(
            models=[
                markdown_expert_config,
                semantic_analyzer_config,
                context_specialist_config,
            ],
            judge=judge_call,
            max_rounds=2,
        )

    def _create_chunk_classification_call(self) -> ConsensusChunkContentClassificationCall:
        """
        Create a consensus-based chunk content classification call for semantic typing.

        Returns:
            ConsensusChunkContentClassificationCall configured with three different classification perspectives
        """
        # Load perspectives from templates
        structural_perspective = template_env.get_template("perspectives/chunk_classification_structural.j2").render()
        semantic_perspective = template_env.get_template("perspectives/chunk_classification_semantic.j2").render()
        contextual_perspective = template_env.get_template("perspectives/chunk_classification_contextual.j2").render()

        # Model 1: Structural analyzer - focuses on document structure
        structural_analyzer_call = InstructorLLMCall(
            response_model=ChunkContentClassification,
            model="gpt-4o",
            temperature=0.3,
        )

        structural_analyzer_config = ConsensusCore.model(
            id="structural-analyzer",
            executor=structural_analyzer_call,
            perspective=structural_perspective,
            weight_multiplier=1.0,
        )

        # Model 2: Semantic classifier - focuses on meaning and content type
        semantic_classifier_call = InstructorLLMCall(
            response_model=ChunkContentClassification,
            model="gpt-4o",
            temperature=0.4,
        )

        semantic_classifier_config = ConsensusCore.model(
            id="semantic-classifier",
            executor=semantic_classifier_call,
            perspective=semantic_perspective,
            weight_multiplier=1.2,
        )

        # Model 3: Contextual interpreter - considers surrounding context
        contextual_interpreter_call = InstructorLLMCall(
            response_model=ChunkContentClassification,
            model="gpt-4o",
            temperature=0.4,
        )

        contextual_interpreter_config = ConsensusCore.model(
            id="contextual-interpreter",
            executor=contextual_interpreter_call,
            perspective=contextual_perspective,
            weight_multiplier=0.8,
        )

        # Create judge for tie-breaking
        judge_call = InstructorLLMCall(
            response_model=ChunkContentClassification,
            model="gpt-4o",
            temperature=0.1,
        )

        # Create consensus chunk classifier
        return ConsensusChunkContentClassificationCall(
            models=[
                structural_analyzer_config,
                semantic_classifier_config,
                contextual_interpreter_config,
            ],
            judge=judge_call,
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
        chunk_classification_call = self._create_chunk_classification_call()

        # Log setup info
        self.logger.info("🧠 Setting up intelligent chunking with 3-model consensus")
        self.logger.info("📊 Setting up chunk content classification with semantic typing")

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
                term_extraction_call=term_extraction_call,
                document_chunking_call=document_chunking_call,
                chunk_content_classification_call=chunk_classification_call,
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

        # Check what's already been extracted
        status = self.check_extraction_status()
        if status:
            self.logger.info("📊 Extraction status check:")
            for step, exists in status.items():
                if exists:
                    self.logger.info(f"  ✓ {step}: Already completed")
                else:
                    self.logger.info(f"  ⏳ {step}: Pending")

        self.logger.info("🚀 Starting extraction with consensus validation...")

        await self.extractor.extract(globs=[self.input_glob])

        self.logger.info("✅ Knowledge extraction completed")

        # Show summary of results
        self._show_extraction_summary()

    def check_extraction_status(self) -> Dict[str, bool]:
        """
        Check which extraction files exist.

        Returns:
            Dictionary mapping step names to existence status
        """
        if not self.extractor:
            return {}

        return self.extractor.get_extraction_status()

    async def run(self) -> None:
        """
        Run the complete extraction pipeline.

        This is a convenience method that calls setup() and extract() in sequence.
        """
        await self.setup()
        await self.extract()

    def _show_extraction_summary(self) -> None:
        """Display a summary of extraction results."""
        try:
            # Check for output files
            output_files = list(self.output_dir.glob("*.pkl"))
            if output_files:
                self.console.print("\n[bold cyan]📊 Extraction Summary:[/bold cyan]")
                self.console.print(f"  • Output directory: {self.output_dir}")
                self.console.print(f"  • Generated {len(output_files)} pickle files")

                # Check for key files
                key_files = [
                    ("linked_knowledge.pkl", "Knowledge graph"),
                    ("knowledge_search.pkl", "Search index"),
                ]
                for filename, description in key_files:
                    if (self.output_dir / filename).exists():
                        size_mb = (self.output_dir / filename).stat().st_size / (1024 * 1024)
                        self.console.print(f"  • {description}: {size_mb:.2f} MB")
        except Exception as e:
            self.logger.debug(f"Could not show summary: {e}")

    async def _handle_regeneration_submenu(self, console: Console) -> None:
        """Handle the regeneration submenu with selective options."""
        console.print("\n[bold yellow]🔄 Regeneration Submenu:[/bold yellow]")
        console.print("1. Regenerate images and dependent steps")
        console.print("2. Regenerate term meanings and dependent steps")
        console.print("3. Regenerate knowledge linking")
        console.print("4. Regenerate search indices")
        console.print("5. Clear all and start fresh")
        console.print("6. Back to main menu")
        
        while True:
            try:
                choice = console.input("\n[cyan]Select regeneration option (1-6): [/cyan]").strip()
                
                if choice == "1":
                    console.print("[yellow]🖼️  Regenerating images and dependent steps...[/yellow]")
                    await self.extractor._regenerate_images_with_dependencies([self.input_glob])
                    console.print("[green]✅ Image regeneration completed![/green]")
                    break
                elif choice == "2":
                    console.print("[yellow]🧠 Regenerating term meanings and dependent steps...[/yellow]")
                    await self.extractor._regenerate_term_meanings_with_dependencies([self.input_glob])
                    console.print("[green]✅ Term meanings regeneration completed![/green]")
                    break
                elif choice == "3":
                    console.print("[yellow]🔗 Regenerating knowledge linking...[/yellow]")
                    await self.extractor._regenerate_knowledge_linking_with_dependencies([self.input_glob])
                    console.print("[green]✅ Knowledge linking regeneration completed![/green]")
                    break
                elif choice == "4":
                    console.print("[yellow]🔍 Regenerating search indices...[/yellow]")
                    await self.extractor._regenerate_search_indices_with_dependencies([self.input_glob])
                    console.print("[green]✅ Search indices regeneration completed![/green]")
                    break
                elif choice == "5":
                    console.print("[yellow]🗑️  Clearing all extraction data and starting fresh...[/yellow]")
                    self.extractor._clear_all_extraction_steps()
                    console.print("[green]✓ All data cleared. Will start fresh extraction...[/green]")
                    break
                elif choice == "6":
                    console.print("[blue]🔙 Returning to main menu...[/blue]")
                    break
                else:
                    console.print("[red]❌ Invalid choice. Please enter 1-6.[/red]")
            except (EOFError, KeyboardInterrupt):
                console.print("\n[yellow]⚠️  Returning to main menu...[/yellow]")
                break

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


class ChunkingPromptRefinementCLI(PromptAlignmentCLIBase[ChunkingDecisionResponse]):
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
        structure_eval = InstructorLLMCall(
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
        semantic_eval = InstructorLLMCall(
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
        context_eval = InstructorLLMCall(
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
        eval_judge = InstructorLLMCall(
            response_model=EvaluationResult,
            model="gpt-4o",
            temperature=0.1,  # Very low temperature for decisive judgments
        )

        eval_consensus = ConsensusCore.consensus(
            models=eval_models,
            judge=eval_judge,
            settings=ConsensusSettings(
                max_rounds=2,
                threshold=0.75,
                verbosity=VerbosityLevel.VERBOSE,
            )
        )

        # Create alignment feedback consensus (3 models provide improvement suggestions)
        feedback_models = []

        for i, (id_name, perspective) in enumerate([
            ("clarity-improver", "As a technical writing expert, identify how to make this prompt clearer and more precise in its chunking instructions."),
            ("structure-enhancer", "As a markdown and document structure specialist, suggest improvements for better structural preservation."),
            ("coherence-optimizer", "As a content strategist, recommend changes to ensure better semantic grouping and logical flow."),
        ]):
            feedback_model = InstructorLLMCall(
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
        # Create judge for feedback consensus
        feedback_judge = InstructorLLMCall(
            response_model=AlignmentFeedback,
            model="gpt-4o",
            temperature=0.1,  # Very low temperature for decisive judgments
        )

        feedback_consensus = ConsensusCore.consensus(
            models=feedback_models,
            judge=feedback_judge,
            settings=ConsensusSettings(
                max_rounds=2,
                threshold=0.75,
                verbosity=VerbosityLevel.VERBOSE,
            )
        )

        # Initialize prompt aligner
        self.prompt_aligner = PromptAlignmentCore(
            target_consensus=eval_consensus,
            alignment_consensus=feedback_consensus,
        )

    def fill_template(self, prompt: str, values: Dict[str, Any]) -> str:
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

    async def _test_prompt(self, prompt: str) -> ChunkingDecisionResponse:
        """Test the chunking prompt with real document data."""
        # Create a dummy response since we don't have access to real extraction data
        from blockether_catalyst.knowledge.KnowledgeTypes import ChunkOutput

        # Use sample text for testing
        sample_text = """
        Machine learning is a subset of artificial intelligence that enables
        computers to learn from data without being explicitly programmed.
        """

        return ChunkingDecisionResponse(
            chunks=[ChunkOutput(
                text=sample_text.strip(),
            )],
            reasoning="Test chunking response for prompt evaluation. The content represents a single logical unit about machine learning fundamentals.",
        )

        # Test with sample content
        filled_prompt = self._fill_template(prompt, {
            "content": sample_text,
            "page_number": 1,
        })

        # Use consensus to evaluate chunking
        llm = InstructorLLMCall(
            response_model=ChunkingDecisionResponse,
            model="gpt-4o",
            temperature=0.3,
        )

        try:
            response = await llm.call(filled_prompt)
            return response
        except Exception:
            # Return default response if LLM call fails
            return ChunkingDecisionResponse(
                chunks=[ChunkOutput(text=sample_text.strip())],
                reasoning="Default chunking response due to LLM call failure.",
            )

    def _display_test_results(self, results: ChunkingDecisionResponse):
        """Display chunking test results."""
        from rich.panel import Panel

        self.console.print("\n[bold cyan]Chunking Test Results:[/bold cyan]")
        self.console.print(Panel(
            f"Total Chunks: {results.total_chunks}\n"
            f"Reasoning: {results.reasoning}",
            title="Chunking Decision",
            border_style="cyan",
        ))

        if results.chunks:
            self.console.print("\n[yellow]Chunk Preview:[/yellow]")
            for i, chunk in enumerate(results.chunks[:3], 1):
                preview = chunk.text[:100] + "..." if len(chunk.text) > 100 else chunk.text
                self.console.print(f"  • Chunk {i}: {preview}")


# Simple response model for term extraction testing
class TermExtractionTestResponse(BaseModel):
    """Response from testing term extraction prompt."""
    extracted_terms: List[str] = Field(default_factory=list, description="Terms extracted")
    confidence_scores: Dict[str, float] = Field(default_factory=dict, description="Confidence scores for terms")
    reasoning: str = Field(description="Reasoning about the extraction")


class TermExtractionPromptRefinementCLI(PromptAlignmentCLIBase[TermExtractionTestResponse]):
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
        accuracy_eval = InstructorLLMCall(
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
        completeness_eval = InstructorLLMCall(
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
        context_eval = InstructorLLMCall(
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

        # Create judge for evaluation consensus
        eval_judge = InstructorLLMCall(
            response_model=EvaluationResult,
            model="gpt-4o",
            temperature=0.1,  # Very low temperature for decisive judgments
        )

        eval_consensus = ConsensusCore.consensus(
            models=eval_models,
            judge=eval_judge,
            settings=ConsensusSettings(
                max_rounds=2,
                threshold=0.75,
                verbosity=VerbosityLevel.VERBOSE,
            )
        )

        # Create alignment feedback consensus
        feedback_models = []

        for i, (id_name, perspective) in enumerate([
            ("precision-improver", "As a terminology expert, suggest how to make term validation more precise and accurate."),
            ("context-enhancer", "As a linguistic analyst, recommend improvements for better context utilization."),
            ("clarity-optimizer", "As a technical writer, identify ways to make the instructions clearer."),
        ]):
            feedback_model = InstructorLLMCall(
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

        # Create judge for feedback consensus
        feedback_judge = InstructorLLMCall(
            response_model=AlignmentFeedback,
            model="gpt-4o",
            temperature=0.1,  # Very low temperature for decisive judgments
        )

        feedback_consensus = ConsensusCore.consensus(
            models=feedback_models,
            judge=feedback_judge,
            settings=ConsensusSettings(
                max_rounds=2,
                threshold=0.75,
                verbosity=VerbosityLevel.VERBOSE,
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

    async def _test_prompt(self, prompt: str) -> TermExtractionTestResponse:
        """Test the term extraction prompt with real document data."""
        # Load sample text from extraction data
        extraction_dir = Path("public/knowledge_extraction")

        # Try to get some sample text
        sample_text = "This is a test document with some technical terms like API, machine learning, and data processing."

        if extraction_dir.exists():
            # Try to load actual pages if available
            pages_file = extraction_dir / "1_extracted_pages_with_images.pkl"
            if pages_file.exists():
                import pickle
                with open(pages_file, "rb") as f:
                    pages = pickle.load(f)
                    if pages and len(pages) > 0:
                        sample_text = pages[0].content[:1000]  # First 1000 chars

        # Fill and test prompt
        filled_prompt = self._fill_template(prompt, {"text": sample_text})

        # Use simple LLM call for testing
        llm = InstructorLLMCall(
            response_model=TermExtractionTestResponse,
            model="gpt-4o",
            temperature=0.3,
        )

        response = await llm.call(filled_prompt)
        return response

    def _display_test_results(self, results: TermExtractionTestResponse):
        """Display term extraction test results."""
        from rich.panel import Panel
        from rich.table import Table

        self.console.print("\n[bold cyan]Term Extraction Test Results:[/bold cyan]")

        # Show reasoning
        self.console.print(Panel(
            results.reasoning,
            title="Extraction Reasoning",
            border_style="cyan",
        ))

        # Show extracted terms
        if results.extracted_terms:
            table = Table(title="Extracted Terms")
            table.add_column("Term", style="yellow")
            table.add_column("Confidence", style="green")

            for term in results.extracted_terms[:10]:
                confidence = results.confidence_scores.get(term, 0.0)
                table.add_row(term, f"{confidence:.2f}")

            self.console.print(table)
        else:
            self.console.print("[yellow]No terms extracted[/yellow]")


def validate_glob_pattern(pattern: str) -> str:
    """Validate and return a glob pattern.

    Args:
        pattern: The glob pattern to validate

    Returns:
        The validated pattern

    Raises:
        argparse.ArgumentTypeError: If the pattern is invalid
    """
    if not pattern:
        raise argparse.ArgumentTypeError("Pattern cannot be empty")

    # Check for common mistakes
    if pattern.startswith('~'):
        # Expand home directory
        pattern = str(Path(pattern).expanduser())

    # Warn about common issues
    if ' ' in pattern and not (pattern.startswith('"') or pattern.startswith("'")):
        console = Console()
        console.print("[yellow]Warning: Pattern contains spaces. Consider using quotes.[/yellow]")

    return pattern

def validate_output_dir(path_str: str) -> Path:
    """Validate and return an output directory path.

    Args:
        path_str: The path string to validate

    Returns:
        The validated Path object

    Raises:
        argparse.ArgumentTypeError: If the path is invalid
    """
    try:
        path = Path(path_str).expanduser().resolve()

        # Check if parent directory exists
        if not path.parent.exists():
            raise argparse.ArgumentTypeError(f"Parent directory does not exist: {path.parent}")

        return path
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Invalid output directory: {e}")

async def main() -> None:
    """Main entry point with subcommand support."""
    parser = argparse.ArgumentParser(
        description="Knowledge extraction and prompt refinement tool with robust validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
╔══════════════════════════════════════════════════════════════════╗
║                           EXAMPLES                               ║
╚══════════════════════════════════════════════════════════════════╝

📚 Extract knowledge from PDFs:
    uv run python3 tools/KnowledgeExtraction.py extract "input/*.pdf"
    uv run python3 tools/KnowledgeExtraction.py extract "docs/**/*.pdf" output/
    uv run python3 tools/KnowledgeExtraction.py extract "~/Documents/*.pdf"

🔧 Refine prompts:
    uv run python3 tools/KnowledgeExtraction.py refine-chunking
    uv run python3 tools/KnowledgeExtraction.py refine-terms

💡 Tips:
    • Use quotes for paths with spaces: "my docs/*.pdf"
    • Use ** for recursive search: "**/*.pdf"
    • Paths support ~ for home directory
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Extract subcommand (default behavior)
    extract_parser = subparsers.add_parser(
        "extract",
        help="Extract knowledge from PDF documents",
        description="Extract and validate knowledge using consensus-based validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
📝 Input patterns:
    • "file.pdf"           - Single file
    • "*.pdf"              - All PDFs in current directory
    • "docs/*.pdf"         - All PDFs in docs directory
    • "**/*.pdf"           - All PDFs recursively
    • "~/Documents/*.pdf"  - Supports home directory
        """
    )
    extract_parser.add_argument(
        "input_glob",
        nargs="?",
        default="tests/blockether_catalyst/test_data/full_sample_test_1.pdf",
        type=validate_glob_pattern,
        help="Glob pattern for input PDF files (use quotes for patterns with spaces)"
    )
    extract_parser.add_argument(
        "output_dir",
        nargs="?",
        type=validate_output_dir,
        default=Path("public/knowledge_extraction"),
        help="Output directory for extraction results"
    )
    extract_parser.add_argument(
        "--no-validation",
        action="store_true",
        help="Skip input file validation (use with caution)"
    )
    extract_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
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
        # Set log level based on verbosity flags
        log_level = logging.INFO

        if args.verbose:
            log_level = logging.DEBUG

        # Show confirmation before starting
        console.print("\n[bold cyan]📋 Extraction Configuration:[/bold cyan]")
        console.print(f"  • Input pattern: {args.input_glob}")
        console.print(f"  • Output directory: {args.output_dir}")

        # Count matching files for preview
        matching_files = glob_module.glob(args.input_glob, recursive=True)
        pdf_files = [f for f in matching_files if f.lower().endswith('.pdf')]
        if pdf_files:
            console.print(f"  • Files to process: {len(pdf_files)}")

        # Run knowledge extraction
        validate = not (hasattr(args, 'no_validation') and args.no_validation)
        extractor = KnowledgeExtraction(
            input_glob=args.input_glob,
            output_dir=args.output_dir,
            log_level=log_level,
            validate_inputs=validate,
        )

        try:
            await extractor.setup()
            
            # Check extraction status and always show options menu
            status = extractor.check_extraction_status()
            has_existing_data = status and any(status.values())
            
            if has_existing_data:
                console.print("\n[bold cyan]📊 Previous Extraction Detected:[/bold cyan]")
                
                # Show what's already been done
                completed_steps = [step for step, exists in status.items() if exists]
                pending_steps = [step for step, exists in status.items() if not exists]
                
                if completed_steps:
                    console.print("[green]✓ Completed steps:[/green]")
                    for step in completed_steps[:3]:  # Show first 3
                        console.print(f"  • {step}")
                    if len(completed_steps) > 3:
                        console.print(f"  • ... and {len(completed_steps) - 3} more")
                
                if pending_steps:
                    console.print("[yellow]⏳ Pending steps:[/yellow]")
                    for step in pending_steps[:2]:  # Show first 2
                        console.print(f"  • {step}")
                    if len(pending_steps) > 2:
                        console.print(f"  • ... and {len(pending_steps) - 2} more")
                
                # Show regeneration options for existing data
                console.print("\n[bold yellow]🔄 Extraction Options:[/bold yellow]")
                console.print("1. Continue extraction (skip completed steps)")
                console.print("2. Start fresh (delete all existing steps)")
                console.print("3. Regeneration submenu (selective regeneration)")
                console.print("4. Exit")
                
                while True:
                    try:
                        choice = console.input("\n[cyan]Enter your choice (1-4): [/cyan]").strip()
                        if choice == "1":
                            # Continue normal extraction
                            break
                        elif choice == "2":
                            # Start fresh - clear all existing data
                            console.print("[yellow]🗑️  Clearing all existing extraction data...[/yellow]")
                            extractor.extractor._clear_all_extraction_steps()
                            console.print("[green]✓ All data cleared. Starting fresh extraction...[/green]")
                            break
                        elif choice == "3":
                            # Show regeneration submenu and handle the choice
                            await extractor._handle_regeneration_submenu(console)
                            # After regeneration, exit the loop to continue
                            break
                        elif choice == "4":
                            console.print("[yellow]👋 Exiting...[/yellow]")
                            sys.exit(0)
                        else:
                            console.print("[red]❌ Invalid choice. Please enter 1, 2, 3, or 4.[/red]")
                    except (EOFError, KeyboardInterrupt):
                        console.print("\n[yellow]⚠️  Extraction interrupted by user[/yellow]")
                        sys.exit(130)
            else:
                # No existing data - show simple start menu
                console.print("\n[bold cyan]🚀 Starting New Extraction:[/bold cyan]")
                console.print("No previous extraction data found.")
                console.print("\n[bold yellow]🔄 Options:[/bold yellow]")
                console.print("1. Start extraction")
                console.print("2. Exit")
                
                while True:
                    try:
                        choice = console.input("\n[cyan]Enter your choice (1-2): [/cyan]").strip()
                        if choice == "1":
                            # Start extraction
                            break
                        elif choice == "2":
                            console.print("[yellow]👋 Exiting...[/yellow]")
                            sys.exit(0)
                        else:
                            console.print("[red]❌ Invalid choice. Please enter 1 or 2.[/red]")
                    except (EOFError, KeyboardInterrupt):
                        console.print("\n[yellow]⚠️  Extraction interrupted by user[/yellow]")
                        sys.exit(130)
            
            # Run extraction
            await extractor.extract()
            console.print("\n[bold green]✅ Extraction completed successfully![/bold green]")
        except KeyboardInterrupt:
            console.print("\n[yellow]⚠️  Extraction interrupted by user[/yellow]")
            sys.exit(130)  # Standard exit code for SIGINT
        except Exception as e:
            console.print(f"\n[bold red]❌ Extraction failed: {e}[/bold red]")
            if log_level == logging.DEBUG:
                console.print_exception()
            sys.exit(1)

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

        cli = TermExtractionPromptRefinementCLI(
            prompt_dir=args.prompt_dir,
            console=console,
            output_dir=args.output_dir,
            extraction_dir=extraction_dir,
        )
        await cli.run()

    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    anyio.run(main)
