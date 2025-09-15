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
from typing import Any, Dict, List, Optional, TypeVar, Tuple

from pydantic import BaseModel
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.prompt import Confirm

from com_blockether_catalyst.consensus.ConsensusCore import ConsensusCore
from com_blockether_catalyst.consensus.ConsensusTypes import ConsensusSettings, VerbosityLevel
from com_blockether_catalyst.knowledge.KnowledgeExtractionCallBase import (
    BaseTermExtractionCall,
    BaseDocumentChunkingCall,
    ExtractionCallsSettings,
)
from com_blockether_catalyst.knowledge.KnowledgeExtractionCore import KnowledgeExtractionCore
from com_blockether_catalyst.knowledge.KnowledgeTypes import (
    KnowledgeExtractionOutput,
    DocumentMetadata,
    KnowledgePageDataWithRawText,
    KnowledgeProcessorSettings,
    TermMeaningExtractionResponse,
    ChunkingDecisionResponse,
)

from com_blockether_catalyst.knowledge.PDKnowledgeExtractorTypes import PDFKnowledgeProcessorSettings, PDFPageCropOffset
from com_blockether_catalyst.utils.instructor.InstructorLLMCall import InstructorLLMCall
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
        self.input_glob = input_glob or "tests/com_blockether_catalyst/test_data/full_sample_test_1.pdf"
        self.output_dir = output_dir or Path("public/knowledge_extraction")
        self.log_level = log_level
        self.extractor = None
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
            ConsensusTermExtractionCall configured with three different validation perspectives and a judge
        """
        # Load perspectives from templates
        conservative_perspective = template_env.get_template("perspectives/term_extraction_conservative.j2").render()
        balanced_perspective = template_env.get_template("perspectives/term_extraction_balanced.j2").render()
        liberal_perspective = template_env.get_template("perspectives/term_extraction_liberal.j2").render()

        # Model 1: Conservative financial domain expert - strictest validation
        conservative_expert_call = InstructorLLMCall(
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
            temperature=0.8,
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
            temperature=0.2,
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
                term_extraction_call=term_extraction_call,
                document_chunking_call=document_chunking_call,
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

        # Show summary of results
        self._show_extraction_summary()

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
        default="tests/com_blockether_catalyst/test_data/full_sample_test_1.pdf",
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
    extract_parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress non-essential output"
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
        if hasattr(args, 'verbose') and args.verbose:
            log_level = logging.DEBUG
        elif hasattr(args, 'quiet') and args.quiet:
            log_level = logging.WARNING

        # Show confirmation before starting
        console.print("\n[bold cyan]📋 Extraction Configuration:[/bold cyan]")
        console.print(f"  • Input pattern: {args.input_glob}")
        console.print(f"  • Output directory: {args.output_dir}")

        # Count matching files for preview
        matching_files = glob_module.glob(args.input_glob, recursive=True)
        pdf_files = [f for f in matching_files if f.lower().endswith('.pdf')]
        if pdf_files:
            console.print(f"  • Files to process: {len(pdf_files)}")

        # Ask for confirmation if output directory exists and has files
        if args.output_dir.exists() and any(args.output_dir.iterdir()):
            console.print("\n[yellow]⚠️  Output directory exists and contains files[/yellow]")
            # Check if we're in an interactive terminal
            if sys.stdin.isatty():
                if not Confirm.ask("Continue and potentially overwrite existing files?", default=False):
                    console.print("[red]Extraction cancelled.[/red]")
                    return
            else:
                console.print("[dim]Non-interactive mode: proceeding with extraction[/dim]")

        # Run knowledge extraction
        validate = not (hasattr(args, 'no_validation') and args.no_validation)
        extractor = KnowledgeExtraction(
            input_glob=args.input_glob,
            output_dir=args.output_dir,
            log_level=log_level,
            validate_inputs=validate,
        )

        try:
            await extractor.run()
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
        # "1_raw_extraction.pkl",
        # "2_chunked_documents.pkl",
        # "3_term_candidates.pkl",
        # "4_grouped_terms.pkl",
        # "5_terms_with_cooccurrences.pkl",
        # "6_terms_with_meanings.pkl",
        # "7_terms_with_links.pkl",
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
