#!/usr/bin/env python3
"""
Knowledge extraction example using KnowledgeExtractionCore with consensus-based validation.

This module provides a class-based approach to extracting knowledge from PDF documents
with consensus-based validation using multiple AI models.

Usage:
    from examples.KnowledgeExtractionExample import KnowledgeExtractionExample

    extractor = KnowledgeExtractionExample()
    await extractor.run()
"""

import anyio
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Type, TypeVar

import instructor
from openai import AsyncOpenAI
from pydantic import BaseModel

from com_blockether_catalyst.consensus import ConsensusCore, ConsensusSettings
from com_blockether_catalyst.consensus.internal.ConsensusTypes import ConsensusResult
from com_blockether_catalyst.knowledge import (
    KnowledgeExtractionCore,
)
from com_blockether_catalyst.knowledge.internal import (
    ChunkingDecision,
    KnowledgeProcessorSettings,
)
from com_blockether_catalyst.knowledge.internal.KnowledgeExtractionCallBase import (
    BaseAcronymExtractionCall,
    BaseKeywordExtractionCall,
    BaseChunkingCall,
    BaseChunkAcronymExtractionCall,
    BaseChunkKeywordExtractionCall,
)
# Import from base types
from com_blockether_catalyst.knowledge.internal.KnowledgeExtractionBaseTypes import (
    AcronymMeaningExtractionResponse,
    ChunkAcronymExtractionResponse,
    ChunkKeywordExtractionResponse,
    ChunkOutput,
    KeywordMeaningExtractionResponse,
    KnowledgeMetadata,
    KnowledgePageData,
)
# Import from main types
from com_blockether_catalyst.knowledge.internal.KnowledgeExtractionTypes import (
    TermCooccurrence,
)
from com_blockether_catalyst.knowledge.internal.PDKnowledgeExtractorTypes import PDFPageCropOffset
from com_blockether_catalyst.utils.TypedCalls import ArityOneTypedCall

# Type variable for response types
T = TypeVar("T", bound=BaseModel)


class SimpleInstructorLLMCall(ArityOneTypedCall[str, T]):
    """
    Simple implementation of ArityOneTypedCall using Instructor for the example.
    """

    def __init__(
        self,
        response_model: Type[T],
        model: str = "gpt-4.1",
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


class ConsensusAcronymValidationExtractor(BaseAcronymExtractionCall):
    """
    Consensus-based acronym validation extractor that validates and extracts meanings from acronym candidates.
    """

    def __init__(self, models: List, max_rounds: int = 3):
        """
        Initialize the consensus acronym extractor.

        Args:
            models: List of model configurations for consensus
            max_rounds: Maximum rounds for consensus (default: 3)
        """
        self.models = models
        self.max_rounds = max_rounds

        settings = ConsensusSettings(max_rounds=max_rounds, threshold=0.8)

        # Create judge for tie-breaking
        judge_call = SimpleInstructorLLMCall(
            response_model=AcronymMeaningExtractionResponse,
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
        acronym: str,
        contexts: List[str],
        cooccurrences_with_contexts: Optional[List[Tuple[TermCooccurrence, List[str]]]] = None,
        max_display_occurrences: int = 15,
        max_display_cooccurrences: int = 10,
    ) -> str:
        """
        Fill the prompt for acronym extraction.
        """
        prompt = f"""Analyze if '{acronym}' is a valid acronym and extract its full form.

Contexts where the term appears:
"""
        for i, ctx in enumerate(contexts[:max_display_occurrences]):
            prompt += f"{i+1}. {ctx}\n\n"

        if cooccurrences_with_contexts and len(cooccurrences_with_contexts) > 0:
            prompt += "\n\nCo-occurring terms:"
            for cooc, cooc_contexts in cooccurrences_with_contexts[:max_display_cooccurrences]:
                prompt += f"\n- {cooc.term} (confidence: {cooc.confidence:.2f})"

        prompt += """\n\nDetermine:
1. Is this a valid acronym?
2. If yes, what is the full form?
3. What is its meaning in this context?
4. Provide reasoning for your decision."""

        return prompt



class ConsensusKeywordValidationExtractor(BaseKeywordExtractionCall):
    """
    Consensus-based keyword validation extractor that validates and extracts meanings from keyword candidates.
    """

    def __init__(self, models: List, max_rounds: int = 3):
        """
        Initialize the consensus keyword extractor.

        Args:
            models: List of model configurations for consensus
            max_rounds: Maximum rounds for consensus (default: 3)
        """
        self.models = models
        self.max_rounds = max_rounds

        settings = ConsensusSettings(max_rounds=max_rounds, threshold=0.8)

        # Create judge for tie-breaking
        judge_call = SimpleInstructorLLMCall(
            response_model=KeywordMeaningExtractionResponse,
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
        contexts: List[str],
        cooccurrences_with_contexts: Optional[List[Tuple[TermCooccurrence, List[str]]]] = None,
        max_display_occurrences: int = 15,
        max_display_cooccurrences: int = 10,
    ) -> str:
        """
        Fill the prompt for keyword extraction.
        """
        prompt = f"""Analyze if '{term}' is a valid keyword and extract its meaning.

Contexts where the term appears:
"""
        for i, ctx in enumerate(contexts[:max_display_occurrences]):
            prompt += f"{i+1}. {ctx}\n\n"

        if cooccurrences_with_contexts and len(cooccurrences_with_contexts) > 0:
            prompt += "\n\nCo-occurring terms:"
            for cooc, cooc_contexts in cooccurrences_with_contexts[:max_display_cooccurrences]:
                prompt += f"\n- {cooc.term} (confidence: {cooc.confidence:.2f})"

        prompt += """\n\nDetermine:
1. Is this a valid keyword?
2. What is its meaning in this context?
3. Provide reasoning for your decision."""

        return prompt



class ConsensusChunkingExtractor(BaseChunkingCall):
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

        settings = ConsensusSettings(max_rounds=max_rounds, threshold=0.8)

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
        page: KnowledgePageData,
        document_name: str,
        metadata: KnowledgeMetadata
    ) -> str:
        """
        Fill the prompt for document chunking.
        """
        prompt = f"""Chunk this document page into semantic segments.

Document: {document_name}
Page {page.page}:

{page.text[:5000]}  # Limit to avoid token overflow

Please split this text into logical, self-contained chunks that:
1. Preserve markdown structure (don't break tables, lists, or code blocks)
2. Maintain semantic coherence
3. Include sufficient context for independent understanding
4. Keep related information together

Return a ChunkingDecision with the chunks."""

        return prompt



class ConsensusChunkAcronymDiscoveryExtractor(BaseChunkAcronymExtractionCall):
    """
    Consensus-based chunk acronym discovery extractor that finds acronym candidates in text chunks.
    """

    def __init__(self, models: List, max_rounds: int = 2):
        """
        Initialize the consensus chunk acronym discovery extractor.

        Args:
            models: List of model configurations for consensus
            max_rounds: Maximum rounds for consensus (default: 2 for faster discovery)
        """
        self.models = models
        self.max_rounds = max_rounds

        settings = ConsensusSettings(max_rounds=max_rounds, threshold=0.8)

        # Create judge for tie-breaking
        judge_call = SimpleInstructorLLMCall(
            response_model=ChunkAcronymExtractionResponse,
            model="gpt-4o",
            temperature=0.3,
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
        chunk_text: str,
        document_name: str,
        page_number: int,
        chunk_index: int,
    ) -> str:
        """
        Fill the prompt for acronym discovery in a chunk.
        """
        prompt = f"""Find all acronyms in this text chunk from document '{document_name}' (page {page_number}, chunk {chunk_index}).

Text to analyze:
{chunk_text[:3000]}  # Limit to avoid token overflow

Look for:
1. Capitalized abbreviations (2-8 letters like API, HTTP, NASA)
2. Terms that appear to be shortened forms of longer phrases
3. Industry-standard acronyms and abbreviations

For each acronym found:
- Provide the acronym itself
- Identify its full form if mentioned in the text
- Explain why you think it's an acronym

Return a list of acronyms with their full forms and rationale."""

        return prompt



class ConsensusChunkKeywordDiscoveryExtractor(BaseChunkKeywordExtractionCall):
    """
    Consensus-based chunk keyword discovery extractor that finds keyword candidates in text chunks.
    """

    def __init__(self, models: List, max_rounds: int = 2):
        """
        Initialize the consensus chunk keyword discovery extractor.

        Args:
            models: List of model configurations for consensus
            max_rounds: Maximum rounds for consensus (default: 2 for faster discovery)
        """
        self.models = models
        self.max_rounds = max_rounds

        settings = ConsensusSettings(max_rounds=max_rounds, threshold=0.8)

        # Create judge for tie-breaking
        judge_call = SimpleInstructorLLMCall(
            response_model=ChunkKeywordExtractionResponse,
            model="gpt-4o",
            temperature=0.3,
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
        chunk_text: str,
        document_name: str,
        page_number: int,
        chunk_index: int,
    ) -> str:
        """
        Fill the prompt for keyword discovery in a chunk.
        """
        prompt = f"""Extract important keywords and technical terms from this text chunk from document '{document_name}' (page {page_number}, chunk {chunk_index}).

Text to analyze:
{chunk_text[:3000]}  # Limit to avoid token overflow

Look for:
1. Domain-specific terminology
2. Technical concepts and processes
3. Important business terms
4. Key entities and technologies mentioned
5. Specialized vocabulary relevant to the document's subject matter

For each keyword found:
- Provide the term itself
- Explain why it's important in this context

Focus on terms that would be valuable for understanding, searching, or categorizing this content.
Avoid common words unless they have special meaning in this context.

Return a list of keywords with rationale for each."""

        return prompt



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
        self.input_glob = input_glob or "input/*.pdf"
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

        self.logger.info("=" * 70)
        self.logger.info("KNOWLEDGE EXTRACTION WITH CONSENSUS VALIDATION")
        self.logger.info("=" * 70)
        self.logger.info(f"📚 Input: {self.input_glob}")
        self.logger.info(f"📁 Output: {self.output_dir}")
        self.logger.info(f"📝 Log: {log_file}")
        self.logger.info("=" * 70)

    def _create_acronym_validation_extractor(self) -> ConsensusAcronymValidationExtractor:
        """
        Create a consensus-based acronym validation extractor with diverse model perspectives.

        Returns:
            ConsensusAcronymValidationExtractor configured with three different validation perspectives
        """
        # Model 1: Conservative financial domain expert - strictest validation
        conservative_expert_call = SimpleInstructorLLMCall(
            response_model=AcronymMeaningExtractionResponse,
            model="gpt-4o",
            temperature=0.1,  # Very low temperature for consistent strict decisions
        )

        conservative_expert_config = ConsensusCore.configuration(
            id="conservative-compliance-expert",
            executor=conservative_expert_call,
            perspective="As a senior compliance officer You are extremely conservative and only accept acronyms that are unambiguously defined in documents with clear expansions. Reject anything that could be a document code, version number, or regular word used in caps.",
            weight_multiplier=1.2,
        )

        # Model 2: Balanced technical analyst - moderate validation
        balanced_analyst_call = SimpleInstructorLLMCall(
            response_model=AcronymMeaningExtractionResponse,
            model="gpt-4o",
            temperature=0.5,
        )

        balanced_analyst_config = ConsensusCore.configuration(
            id="balanced-technical-analyst",
            executor=balanced_analyst_call,
            perspective="As an experienced technical writer who specializes in corporate documentation. You validate acronyms that have clear definitions or contextual evidence, balancing precision with practical document analysis. You consider industry-standard abbreviations but require reasonable evidence.",
            weight_multiplier=1.0,  # Standard weight
        )

        # Model 3: Liberal linguistic processor - inclusive validation
        liberal_linguist_call = SimpleInstructorLLMCall(
            response_model=AcronymMeaningExtractionResponse,
            model="gpt-4o",
            temperature=0.8,
        )

        liberal_linguist_config = ConsensusCore.configuration(
            id="liberal-linguistic-processor",
            executor=liberal_linguist_call,
            perspective="As a computational linguist who studies abbreviation patterns in corporate texts. You identify acronyms that may have implicit meanings or follow common organizational naming patterns, even if not explicitly defined. You're more inclusive but still require logical reasoning for acronym status.",
            weight_multiplier=0.8,
        )

        # Create consensus validation extractor with diverse model perspectives
        return ConsensusAcronymValidationExtractor(
            models=[conservative_expert_config, balanced_analyst_config, liberal_linguist_config],
            max_rounds=3,
        )

    def _create_keyword_validation_extractor(self) -> ConsensusKeywordValidationExtractor:
        """
        Create a consensus-based keyword validation extractor with diverse model perspectives.

        Returns:
            ConsensusKeywordValidationExtractor configured with three different validation perspectives
        """
        # Model 1: Domain expert - strict validation
        domain_expert_call = SimpleInstructorLLMCall(
            response_model=KeywordMeaningExtractionResponse,
            model="gpt-4o",
            temperature=0.2,
        )

        domain_expert_config = ConsensusCore.configuration(
            id="domain-expert",
            executor=domain_expert_call,
            perspective="As a domain expert, validate if this is a significant keyword in the document context. Only accept terms that are clearly important concepts, technologies, or domain-specific terminology.",
            weight_multiplier=1.2,
        )

        # Model 2: Technical writer - balanced validation
        technical_writer_call = SimpleInstructorLLMCall(
            response_model=KeywordMeaningExtractionResponse,
            model="gpt-4o",
            temperature=0.5,
        )

        technical_writer_config = ConsensusCore.configuration(
            id="technical-writer",
            executor=technical_writer_call,
            perspective="As a technical writer, identify keywords that would be important for documentation and understanding. Balance between common terms and specialized vocabulary.",
            weight_multiplier=1.0,
        )

        # Model 3: Content analyst - inclusive validation
        content_analyst_call = SimpleInstructorLLMCall(
            response_model=KeywordMeaningExtractionResponse,
            model="gpt-4o",
            temperature=0.7,
        )

        content_analyst_config = ConsensusCore.configuration(
            id="content-analyst",
            executor=content_analyst_call,
            perspective="As a content analyst, identify keywords that help understand the document's main topics and themes. Include terms that appear frequently or in important contexts.",
            weight_multiplier=0.8,
        )

        # Create consensus validation extractor with diverse model perspectives
        return ConsensusKeywordValidationExtractor(
            models=[domain_expert_config, technical_writer_config, content_analyst_config],
            max_rounds=3,
        )

    def _create_chunking_extractor(self) -> ConsensusChunkingExtractor:
        """
        Create a consensus-based chunking extractor for intelligent document segmentation.

        Returns:
            ConsensusChunkingExtractor configured with three different chunking perspectives
        """
        # Model 1: Markdown structure expert
        markdown_expert_call: SimpleInstructorLLMCall[ChunkingDecision] = SimpleInstructorLLMCall(
            response_model=ChunkingDecision,
            model="gpt-4o",
            temperature=0.2,  # Low temperature for consistent structure decisions
        )

        markdown_expert_config = ConsensusCore.configuration(
            id="markdown-structure-expert",
            executor=markdown_expert_call,
            perspective="As a markdown formatting expert who prioritizes valid markdown structure. You ensure tables, lists, code blocks, and links are NEVER broken. Breaking in the middle of any markdown structure is completely unacceptable.",
            weight_multiplier=1.2,  # Higher weight for structure preservation
        )

        # Model 2: Semantic coherence analyzer
        semantic_analyzer_call = SimpleInstructorLLMCall(
            response_model=ChunkingDecision,
            model="gpt-4o",
            temperature=0.5,
        )

        semantic_analyzer_config = ConsensusCore.configuration(
            id="semantic-coherence-analyzer",
            executor=semantic_analyzer_call,
            perspective="As a content analyst who ensures each chunk forms a complete, coherent unit of information. You prefer breaking at natural semantic boundaries like section ends, topic transitions, and complete thoughts.",
            weight_multiplier=1.0,
        )

        # Model 3: Context preservation specialist
        context_specialist_call = SimpleInstructorLLMCall(
            response_model=ChunkingDecision,
            model="gpt-4o",
            temperature=0.4,
        )

        context_specialist_config = ConsensusCore.configuration(
            id="context-preservation-specialist",
            executor=context_specialist_call,
            perspective="As a technical documentation specialist who ensures each chunk has sufficient context to be understood independently. You consider whether references, definitions, and explanations are complete within the chunk.",
            weight_multiplier=0.8,
        )

        # Create agentic chunking extractor with consensus
        return ConsensusChunkingExtractor(
            models=[
                markdown_expert_config,
                semantic_analyzer_config,
                context_specialist_config,
            ],
            max_rounds=2,
        )

    def _create_chunk_acronym_discovery_extractor(self) -> ConsensusChunkAcronymDiscoveryExtractor:
        """
        Create a consensus-based chunk acronym discovery extractor.

        Returns:
            ConsensusChunkAcronymDiscoveryExtractor for finding acronyms in chunks
        """
        # Model 1: Technical document expert
        tech_expert_call = SimpleInstructorLLMCall(
            response_model=ChunkAcronymExtractionResponse,
            model="gpt-4o",
            temperature=0.3,
        )

        tech_expert_config = ConsensusCore.configuration(
            id="technical-acronym-expert",
            executor=tech_expert_call,
            perspective="As a technical documentation expert, identify acronyms commonly used in technical, business, and academic contexts. Focus on industry-standard abbreviations and domain-specific acronyms.",
            weight_multiplier=1.2,
        )

        # Model 2: Pattern recognition specialist
        pattern_specialist_call = SimpleInstructorLLMCall(
            response_model=ChunkAcronymExtractionResponse,
            model="gpt-4o",
            temperature=0.4,
        )

        pattern_specialist_config = ConsensusCore.configuration(
            id="pattern-recognition-specialist",
            executor=pattern_specialist_call,
            perspective="As a pattern recognition specialist, identify capitalized abbreviations and shortened forms that follow acronym patterns. Look for consistent usage and contextual clues.",
            weight_multiplier=1.0,
        )

        return ConsensusChunkAcronymDiscoveryExtractor(
            models=[tech_expert_config, pattern_specialist_config],
            max_rounds=2,
        )

    def _create_chunk_keyword_discovery_extractor(self) -> ConsensusChunkKeywordDiscoveryExtractor:
        """
        Create a consensus-based chunk keyword discovery extractor.

        Returns:
            ConsensusChunkKeywordDiscoveryExtractor for finding keywords in chunks
        """
        # Model 1: Domain terminology expert
        domain_expert_call = SimpleInstructorLLMCall(
            response_model=ChunkKeywordExtractionResponse,
            model="gpt-4o",
            temperature=0.3,
        )

        domain_expert_config = ConsensusCore.configuration(
            id="domain-terminology-expert",
            executor=domain_expert_call,
            perspective="As a domain terminology expert, identify specialized vocabulary, technical terms, and domain-specific concepts that are key to understanding the content.",
            weight_multiplier=1.2,
        )

        # Model 2: Information extraction specialist
        info_extraction_call = SimpleInstructorLLMCall(
            response_model=ChunkKeywordExtractionResponse,
            model="gpt-4o",
            temperature=0.4,
        )

        info_extraction_config = ConsensusCore.configuration(
            id="information-extraction-specialist",
            executor=info_extraction_call,
            perspective="As an information extraction specialist, identify terms that would be valuable for search, categorization, and content understanding. Focus on entities, processes, and concepts.",
            weight_multiplier=1.0,
        )

        return ConsensusChunkKeywordDiscoveryExtractor(
            models=[domain_expert_config, info_extraction_config],
            max_rounds=2,
        )

    async def setup(self) -> None:
        """
        Set up the extraction environment and initialize components.

        This method:
        1. Creates consensus extractors for validation, discovery, and chunking
        2. Configures KnowledgeProcessorSettings with all required parameters
        3. Initializes the KnowledgeExtractionCore with the settings
        """
        # Create validation extractors (for meaning extraction)
        acronym_validation_extractor = self._create_acronym_validation_extractor()
        keyword_validation_extractor = self._create_keyword_validation_extractor()

        # Create discovery extractors (for finding candidates in chunks)
        chunk_acronym_discovery_extractor = self._create_chunk_acronym_discovery_extractor()
        chunk_keyword_discovery_extractor = self._create_chunk_keyword_discovery_extractor()

        # Create chunking extractor
        chunking_extractor = self._create_chunking_extractor()

        # Log setup info
        self.logger.info("🧠 Setting up intelligent chunking with 3-model consensus")

        # Create settings with all required extractors (validation and discovery)
        settings = KnowledgeProcessorSettings(
            extraction_output_dir=self.output_dir,
            pdf_page_crop_offset=PDFPageCropOffset(
                top=48,
                bottom=48
            ),
            # Validation extractors (for meaning extraction from candidates)
            acronym_extraction_call=acronym_validation_extractor,
            keyword_extraction_call=keyword_validation_extractor,
            # Discovery extractors (for finding candidates in chunks)
            chunk_acronym_extraction_call=chunk_acronym_discovery_extractor,
            chunk_keyword_extraction_call=chunk_keyword_discovery_extractor,
            # Document processing
            chunking_call=chunking_extractor,
            min_term_score=0.0,
            linking_threshold=0.65,
            max_display_occurrences=15,
            max_display_cooccurrences=5,
        )

        # Create knowledge extractor with settings
        self.extractor = KnowledgeExtractionCore(settings=settings)

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
            uv run python3 examples/KnowledgeExtractionExample.py [input_glob] [output_dir]

        Examples:
            # Extract all PDFs in input directory
            uv run python3 examples/KnowledgeExtractionExample.py "input/*.pdf"

            # Extract specific PDFs with pattern
            uv run python3 examples/KnowledgeExtractionExample.py "docs/**/*.pdf" output/

            # Extract with custom output directory
            uv run python3 examples/KnowledgeExtractionExample.py "input/*.pdf" output/
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


async def main() -> None:
    """Main entry point for command line execution."""
    await KnowledgeExtraction.from_cli()


if __name__ == "__main__":
    anyio.run(main)
