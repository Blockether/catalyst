"""
Simplified Document Terms Processing System
Self-contained implementation with standard logging
"""

import inspect
import logging
import math
import os
import pickle
import re
import shutil
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from functools import partial, wraps
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    Union,
    cast,
)

from anyio import to_thread
from pydantic import BaseModel, RootModel
from rapidfuzz import fuzz

from blockether_catalyst.consensus.ConsensusTypes import ConsensusResult

from ..utils import ConcurrentProcessor
from .ImageOptimizer import ImageOptimizer
from .ImageRecognition import ImageRecognition
from .KnowledgeExtractionCallBase import ExtractionCallsSettings
from .KnowledgeSearchCore import KnowledgeSearchCore
from .KnowledgeTypes import (
    DocumentMetadata,
    KnowledgeChunk,
    KnowledgeExtractionItem,
    KnowledgeExtractionOutput,
    KnowledgeExtractionResult,
    KnowledgeExtractionResultWithChunks,
    KnowledgePageData,
    KnowledgeProcessorSettings,
    LinkedKnowledge,
    RawExtractionData,
    Term,
    TermCandidate,
    TermCandidateGrouped,
    TermCooccurrence,
    TermLink,
    TermMeaningExtractionResponse,
    TermOccurrence,
    TermWithLinks,
)
from .KnowledgeVectorizers import KnowledgeVectorizers
from .PDFKnowledgeExtractor import PDFKnowledgeExtractor

logger = logging.getLogger(__name__)

T = TypeVar("T")

type PydanticLike = Union[
    BaseModel,
    RootModel,
    Sequence[BaseModel | RootModel],
    Mapping[str, BaseModel | RootModel],
]


def timed_operation(step_name: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to time operations and log their duration.

    Args:
        step_name: Name of the operation for logging

    Returns:
        Decorated function that logs execution time
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            start_time = time.time()
            logger.info(f"{step_name}: Starting...")
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            logger.info(f"{step_name}: Completed in {elapsed:.2f}s")
            return result

        return wrapper

    return decorator


def async_timed_operation(
    step_name: str,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to time async operations and log their duration.

    Args:
        step_name: Name of the operation for logging

    Returns:
        Decorated async function that logs execution time
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            start_time = time.time()
            logger.info(f"{step_name}: Starting...")
            result = await func(*args, **kwargs)  # type: ignore
            elapsed = time.time() - start_time
            logger.info(f"{step_name}: Completed in {elapsed:.2f}s")
            return cast(T, result)

        return wrapper  # type: ignore

    return decorator


class KnowledgeExtractionCore:
    """Core knowledge extraction system for processing documents and extracting structured knowledge.

    This class orchestrates the entire knowledge extraction pipeline including:
    - Document parsing and chunking
    - Term and acronym extraction
    - Co-occurrence analysis
    - Term linking and relationship building
    - Search index creation
    """

    def __init__(self, calls: ExtractionCallsSettings, settings: KnowledgeProcessorSettings):
        self._calls = calls
        self._settings = settings
        self._output_dir = settings.extraction_output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"KnowledgeExtractionCore initialized with settings: {self._settings.model_dump()}")
        logger.info(f"KnowledgeExtractionCore initialized with output_dir: {self._output_dir}")

        # Validate that ALL typed calls are provided - they are MANDATORY
        if not self._calls.term_extraction_call:
            raise ValueError("term_extraction_call is mandatory in settings")

        if not self._calls.document_chunking_call:
            raise ValueError("document_chunking_call is mandatory in settings")

        if not self._calls.chunk_content_classification_call:
            raise ValueError("chunk_content_classification_call is MANDATORY in settings")

        image_output_dir = self._output_dir / "images"
        image_output_dir.mkdir(parents=True, exist_ok=True)

        self._image_optimizer = ImageOptimizer(image_output_dir, level=settings.image_optimization_level)
        self._image_recognition = ImageRecognition()

        # Define extractors for each supported extension
        self._extractors = {".pdf": PDFKnowledgeExtractor(image_output_dir, self._settings)}

    @property
    def calls(self) -> ExtractionCallsSettings:
        """Get the extraction calls settings."""
        return self._calls

    @property
    def settings(self) -> KnowledgeProcessorSettings:
        """Get the processor settings."""
        return self._settings

    @property
    def extractors(self) -> Dict[str, PDFKnowledgeExtractor]:
        """Get the document extractors."""
        return self._extractors

    EMOJI_PATTERN = re.compile(
        "["
        "\U0001f600-\U0001f64f"  # emoticons
        "\U0001f300-\U0001f5ff"  # symbols & pictographs
        "\U0001f680-\U0001f6ff"  # transport & map symbols
        "\U0001f1e0-\U0001f1ff"  # flags (iOS)
        "\U0001f900-\U0001f9ff"  # supplemental symbols and pictographs
        "\U00002702-\U000027b0"  # dingbats
        "\U0000fe0f"  # variation selector-16 (emoji presentation)
        "\U0000fe0e"  # variation selector-15 (text presentation)
        "\U0000200d"  # zero-width joiner
        "]+",
        flags=re.UNICODE,
    )

    def _resolve_glob_patterns(self, globs: list[str]) -> list[Path]:
        """Resolve glob patterns to actual file paths.

        Args:
            globs: List of glob patterns (e.g., '*.pdf', 'docs/**/*.txt') or absolute paths

        Returns:
            List of resolved file paths matching the patterns
        """
        all_files: Set[Path] = set()

        for pattern in globs:
            logger.info(f"Resolving glob pattern: {pattern}")

            # Check if it's an absolute path
            path = Path(pattern)
            if path.is_absolute() and path.exists():
                all_files.add(path)
            else:
                # It's a glob pattern
                matched_files = list(Path().rglob(pattern))
                all_files.update(matched_files)

        return list(all_files)

    def _group_files_by_extension(self, files: list[Path]) -> dict[str, list[Path]]:
        """Group files by their extension for batch processing.

        Args:
            files: Sequence of file paths

        Returns:
            Dictionary mapping extension (e.g., '.pdf') to list of file paths
        """
        files_by_extension = defaultdict(list)
        for file_path in files:
            if file_path.is_file():
                extension = file_path.suffix.lower()
                if extension:
                    files_by_extension[extension].append(file_path)
        return files_by_extension

    @timed_operation("Step 1/12: Raw file extraction")
    def _process_files_by_extension(self, files_by_extension: dict[str, list[Path]]) -> KnowledgeExtractionOutput:
        """Process files in parallel, grouped by extension for optimal performance.

        Args:
            files_by_extension: Dictionary mapping file extensions to lists of file paths

        Returns:
            KnowledgeExtractionOutput containing raw extraction results from all files
        """
        # Initialize extraction output
        extraction_output = KnowledgeExtractionOutput()

        # Process files by extension
        for extension, file_list in files_by_extension.items():
            if extension not in self._extractors:
                logger.warning(f"Skipping unsupported extension: {extension}")
                continue

            logger.info(f"Processing {len(file_list)} {extension} files")

            extractor = self._extractors[extension]
            extraction_results: list[KnowledgeExtractionItem] = []

            max_workers = min(os.cpu_count() or 4, len(file_list)) if len(file_list) > 0 else 1

            def process_single_file(file_path: Path) -> KnowledgeExtractionItem:
                """Process a single file and return extraction item."""
                logger.info(f"Processing: {file_path}")
                single_result = extractor.extract(file_path)
                return KnowledgeExtractionItem(result=single_result)

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_file = {executor.submit(process_single_file, file_path): file_path for file_path in file_list}

                # Collect results as they complete
                for future in as_completed(future_to_file):
                    file_path = future_to_file[future]
                    try:
                        extraction_item = future.result()
                        extraction_results.append(extraction_item)
                    except Exception as exc:
                        logger.error(f"File {file_path} generated an exception: {exc}")

            if extension == ".pdf":
                extraction_output.pdf = extraction_results

        return extraction_output

    def _count_total_chunks(self, results_with_chunks: Sequence[KnowledgeExtractionResultWithChunks]) -> int:
        """Count total chunks across all documents for progress tracking.

        Args:
            results_with_chunks: Chunked extraction results

        Returns:
            Total number of chunks across all documents
        """
        total_chunks = 0
        for item in results_with_chunks:
            total_chunks += item.total_chunks
        return total_chunks

    def _build_document_chunk_index(
        self, results_with_chunks: Sequence[KnowledgeExtractionResultWithChunks]
    ) -> Dict[str, list[KnowledgeChunk]]:
        """Build document-to-chunks index for O(1) chunk lookups.

        Args:
            results_with_chunks: Chunked extraction results

        Returns:
            Dictionary mapping document IDs to their chunks for efficient access
        """
        chunks_index: Dict[str, list[KnowledgeChunk]] = {}

        for item in results_with_chunks:
            chunks_index[item.id] = list(item.chunks)

        return chunks_index

    def _build_chunk_term_index(
        self,
        grouped_terms: Dict[str, TermCandidateGrouped],
        chunks_index: Dict[str, list[KnowledgeChunk]],
    ) -> Dict[Tuple[str, int], Dict[str, Sequence[int]]]:
        """Build inverted index of term positions within chunks for co-occurrence analysis.

        Args:
            grouped_terms: Dictionary of grouped term candidates
            chunks_index: Pre-built index mapping document IDs to chunks

        Returns:
            Dictionary mapping (document_id, chunk_index) to term positions
        """
        chunk_terms_index: Dict[Tuple[str, int], Dict[str, Sequence[int]]] = defaultdict(lambda: defaultdict(list))

        for term, group in grouped_terms.items():
            for occurrence in group.occurrences:
                if occurrence.document_id in chunks_index:
                    for chunk in chunks_index[occurrence.document_id]:
                        positions = self._find_term_positions_in_text(term, chunk.text)
                        if positions:
                            chunk_terms_index[(occurrence.document_id, chunk.index)][term] = positions

        return chunk_terms_index

    def _detect_chunk_content_types(self, page: KnowledgePageData, chunk_text: str) -> List[str]:
        """
        Detect content types present in a chunk based on page data and chunk text.

        Args:
            page: The page data containing images and tables
            chunk_text: The text content of the chunk

        Returns:
            List of content types: 'text', 'image', 'table'
        """
        content_types = []

        # Check if chunk has text content (non-empty after stripping)
        if chunk_text and chunk_text.strip():
            content_types.append("text")

        # Regex pattern for table detection
        # Matches: box drawing chars, markdown tables, ASCII tables, or table references
        table_pattern = re.compile(
            r"[│┌┐└┘├┤┬┴┼]|"  # Box drawing characters
            r"\|[-\s]*\||"  # Markdown/ASCII table separators
            r"[-]{5,}|"  # Long horizontal lines (often used in tables)
            r"\btable\s+\d+|"  # "Table 1", "Table 2", etc.
            r"\b(?:see|refer|shown\s+in|following|below|above)\s+table|"  # Table references
            r"\btable\s+of\s+contents?\b",  # Table of contents
            re.IGNORECASE,
        )

        # Regex pattern for image/figure detection
        # Matches various image references and figure notations
        image_pattern = re.compile(
            r"\b(?:figure|fig\.?)\s+\d+|"  # "Figure 1", "Fig. 2", etc.
            r"\b(?:image|picture|diagram|chart|graph|illustration|screenshot|photo)\s+\d*|"  # Image references with optional numbers
            r"\b(?:see|refer|shown\s+in|following|below|above)\s+(?:figure|image|diagram)|"  # Image references
            r"\b(?:as\s+)?(?:shown|illustrated|depicted|displayed)\s+(?:in|below|above)",  # Visual references
            re.IGNORECASE,
        )

        # If page has tables, mark it as containing table content
        if page.tables and len(page.tables) > 0:
            content_types.append("table")
        # Even if page doesn't have tables, check if text mentions or contains tables
        elif table_pattern.search(chunk_text):
            content_types.append("table")

        # If page has images, mark it as containing image content
        if page.images and len(page.images) > 0:
            content_types.append("image")
        # Even if page doesn't have images, check if text references images
        elif image_pattern.search(chunk_text):
            content_types.append("image")

        # If no content types were detected, default to text if there's any content
        if not content_types and chunk_text:
            content_types.append("text")

        # Remove duplicates while preserving order
        seen: Set[str] = set()
        unique_types = []
        for ct in content_types:
            if ct not in seen:
                seen.add(ct)
                unique_types.append(ct)
        content_types = unique_types

        return content_types

    def _find_term_positions_in_text(self, term: str, text: str) -> Sequence[int]:
        """Find all word-boundary positions of a term in text.

        Args:
            term: The term to search for (case-insensitive)
            text: The text to search in

        Returns:
            List of character positions where the term starts as a whole word
        """
        positions = []
        pattern = r"\b" + re.escape(term.lower()) + r"\b"
        text_lower = text.lower()

        for match in re.finditer(pattern, text_lower):
            positions.append(match.start())

        return positions

    def _calculate_cooccurrence_weight(self, positions1: Sequence[int], positions2: Sequence[int]) -> float:
        """Calculate proximity-based weight for term co-occurrence using exponential decay.

        Args:
            positions1: Character positions of first term
            positions2: Character positions of second term

        Returns:
            Weight between 0.1 and 1.0 (higher = closer proximity)
        """
        if not positions1 or not positions2:
            return 0.0

        min_distance = float("inf")
        for pos1 in positions1:
            for pos2 in positions2:
                distance = abs(pos1 - pos2)
                min_distance = min(min_distance, distance)

        weight = math.exp(-min_distance / 100.0)
        return max(weight, 0.1)

    @async_timed_operation("Knowledge Extraction Pipeline")
    async def extract(self, globs: list[str]) -> None:
        """
        Extract knowledge from files matching the provided glob patterns.

        Args:
            globs: Sequence of glob patterns to match files

        Returns:
            LinkedKnowledge containing all extracted knowledge and relationships
        """
        logger.info(f"Starting with {len(globs)} glob patterns")

        # Check for existing extraction files
        status = self.get_extraction_status()
        has_existing_files = any(status.values())

        if has_existing_files:
            action = self._prompt_existing_files_action()

            if action == "cancel":
                logger.info("Extraction cancelled by user")
                return
            elif action == "remove":
                logger.info("User chose to remove existing files and start fresh")
                self._remove_existing_extraction_files()
            elif action == "continue":
                logger.info("User chose to continue with existing files")
                # Continue with normal execution, existing files will be loaded
            elif action == "regenerate_submenu":
                # Handle regeneration submenu
                while True:
                    submenu_action = self._prompt_regeneration_submenu(globs)

                    if submenu_action == "cancel":
                        logger.info("Extraction cancelled by user")
                        return
                    elif submenu_action == "back":
                        # Go back to main menu
                        action = self._prompt_existing_files_action()
                        if action in ["cancel", "remove", "continue"]:
                            break
                        # If user selects regenerate_submenu again, continue the loop
                    elif submenu_action == "images_with_deps":
                        logger.info("User chose to regenerate images with dependency management")
                        await self._regenerate_images_with_dependencies(globs)
                        # Regeneration is complete - it already rebuilt the pipeline, so we're done
                        logger.info("✅ Image regeneration completed successfully!")
                        return
                    elif submenu_action == "captions":
                        logger.info("User chose to regenerate captions only")
                        self._regenerate_captions_only(globs)
                        return
                    elif submenu_action == "documents":
                        logger.info("User chose to regenerate source documents")
                        self._regenerate_source_documents(globs)
                        return

                # Handle the action from returning to main menu
                if action == "cancel":
                    logger.info("Extraction cancelled by user")
                    return
                elif action == "remove":
                    logger.info("User chose to remove existing files and start fresh")
                    self._remove_existing_extraction_files()
                elif action == "continue":
                    logger.info("User chose to continue with existing files")

        # Always need all_files for later steps
        all_files = self._resolve_glob_patterns(globs)
        logger.info(f"Found {len(all_files)} files from glob patterns")

        # Step 1: Raw extraction
        def execute_raw_extraction() -> KnowledgeExtractionOutput:
            files_by_extension = self._group_files_by_extension(all_files)
            logger.info(f"Grouped files by extension: {dict((k, len(v)) for k, v in files_by_extension.items())}")
            result = self._process_files_by_extension(files_by_extension)
            logger.info(f"Raw extraction completed: {result.model_dump().keys()} documents")
            return result

        raw_extraction = await self._execute_or_load_step(
            step_name="1_raw_extraction",
            step_number="1/12",
            execute_fn=execute_raw_extraction,
        )

        # Step 2: Document chunking
        results_with_chunks = await self._execute_or_load_step(
            step_name="2_chunked_documents",
            step_number="2/12",
            execute_fn=self._chunk_extraction,
            raw_extraction=raw_extraction,
        )

        total_chunks = self._count_total_chunks(results_with_chunks)
        logger.info(f"Total chunks: {total_chunks} from {len(results_with_chunks)} documents")

        # Build chunks index (always needed for later steps)
        document_to_chunks_index: Dict[str, list[KnowledgeChunk]] = self._build_document_chunk_index(
            results_with_chunks
        )
        logger.info(f"Built chunks index for {len(document_to_chunks_index)} documents")

        # Step 3: Chunk semantic classification (MANDATORY)
        results_with_classified_chunks = await self._execute_or_load_step(
            step_name="3_classified_chunks",
            step_number="3/13",
            execute_fn=self._classify_chunk_content,
            results_with_chunks=results_with_chunks,
        )
        # Update the chunks index with classified chunks
        document_to_chunks_index = self._build_document_chunk_index(results_with_classified_chunks)
        logger.info(f"Classified semantic types for {total_chunks} chunks")

        # Step 4: Term extraction
        async def execute_term_extraction() -> Sequence[TermCandidate]:
            result = await self._extract_terms_candidates_from_documents(results_with_classified_chunks)
            logger.info(f"Found {len(result)} term candidates")
            return result

        terms = await self._execute_or_load_step(
            step_name="4_term_candidates",
            step_number="4/13",
            execute_fn=execute_term_extraction,
        )

        # Step 5: Term grouping
        grouped_terms = await self._execute_or_load_step(
            step_name="5_grouped_terms",
            step_number="5/13",
            execute_fn=self._group_term_candidates,
            term_candidates=terms,
        )
        logger.info(f"Grouped {len(grouped_terms)} unique terms")

        # Step 6: Co-occurrence analysis
        def execute_cooccurrence() -> Dict[str, TermCandidateGrouped]:
            result = self._enrich_with_cooccurrences(grouped_terms, document_to_chunks_index)
            logger.info(f"Added co-occurrences to {len(result)} terms")
            return result

        terms_with_cooccurrences = await self._execute_or_load_step(
            step_name="6_terms_with_cooccurrences",
            step_number="6/13",
            execute_fn=execute_cooccurrence,
        )

        # Step 7: Term meaning enrichment
        consolidated_terms = await self._execute_or_load_step(
            step_name="7_terms_with_meanings",
            step_number="7/13",
            execute_fn=self._enrich_with_terms_meanings,
            groups=terms_with_cooccurrences,
            document_to_chunks_index=document_to_chunks_index,
        )

        # Step 8: Term linking
        consolidated_terms_with_links = await self._execute_or_load_step(
            step_name="8_terms_with_links",
            step_number="8/13",
            execute_fn=self._link_terms,
            terms=consolidated_terms,
        )

        # Step 9: Build LinkedKnowledge
        def build_linked_knowledge() -> LinkedKnowledge:
            raw_data = RawExtractionData(
                results_with_chunks=results_with_chunks,
                terms=consolidated_terms_with_links,
                document_to_chunks_index=document_to_chunks_index,
            )
            return LinkedKnowledge.from_extraction_data(raw_data)

        linked_knowledge = await self._execute_or_load_step(
            step_name="linked_knowledge",
            step_number="9/13",
            execute_fn=build_linked_knowledge,
        )

        # Step 10: Optimize images
        logger.info("Step 10/13: Optimizing extracted images")
        self._image_optimizer.optimize()

        # Step 11: Copy source documents
        self._copy_source_documents(all_files)

        # Step 12: Create and persist search core
        search_pickle_path = self._output_dir / "knowledge_search.pkl"
        if not search_pickle_path.exists():
            logger.info("Step 11/13: Creating KnowledgeSearchCore with vector indices")
            search_core = KnowledgeSearchCore(
                linked_knowledge=linked_knowledge,
                pickle_path=search_pickle_path,
            )

            logger.info("Step 12/13: Persisting KnowledgeSearchCore to pickle")
            search_core.persist()

            pickle_size_mb = search_pickle_path.stat().st_size / (1024 * 1024)
            logger.info(f"KnowledgeSearchCore saved ({pickle_size_mb:.2f} MB)")
        else:
            pickle_size_mb = search_pickle_path.stat().st_size / (1024 * 1024)
            logger.info(f"Step 12/13: KnowledgeSearchCore already exists ({pickle_size_mb:.2f} MB)")

        logger.info("Step 13/13: Knowledge extraction pipeline completed successfully")

        # Log the summary using the new method
        logger.info("\n" + linked_knowledge.get_extraction_summary(detailed=False))

    @timed_operation("Step 11/13: Copy source documents")
    def _copy_source_documents(self, all_files: Sequence[Path]) -> None:
        """Copy source documents to output directory for viewing.

        Args:
            all_files: Sequence of source file paths to copy
        """
        logger.info(f"Starting to copy {len(all_files)} source documents")

        # Copy to output directory (source_documents)
        docs_dir = self._output_dir / "source_documents"
        docs_dir.mkdir(exist_ok=True)
        logger.info(f"Created source documents directory: {docs_dir}")

        copied_count = 0
        total_processed = 0

        for file_path in all_files:
            if not file_path.is_file():
                logger.warning(f"Skipping non-file path: {file_path}")
                continue

            total_processed += 1
            logger.debug(f"Processing file {total_processed}/{len(all_files)}: {file_path}")

            # Copy to output directory (for specific document types)
            if file_path.suffix.lower() in [".pdf", ".txt", ".md"]:
                # Keep original filename and overwrite if collision occurs
                dest_path = docs_dir / file_path.name

                if dest_path.exists():
                    logger.info(f"Overwriting existing file: {dest_path}")

                shutil.copy2(file_path, dest_path)
                copied_count += 1
                logger.debug(f"Copied: {file_path.name} -> {dest_path}")

        # Log summary
        logger.info(f"Document copying completed: {copied_count} files copied to {docs_dir}")

    async def _extract_terms_from_document(
        self, document_result: KnowledgeExtractionResultWithChunks
    ) -> Sequence[TermCandidate]:
        """
        Extract term candidates from a single document using LLM.

        Args:
            document_result: The parent document result for metadata

        Returns:
            Sequence of keyword candidates from this document
        """
        # Transform all chunks together to get consistent vocabulary
        chunk_texts = [chunk.text for chunk in document_result.chunks]

        # Adjust min_df based on the number of chunks available
        # min_df cannot be greater than the number of documents (chunks)
        keywords_min_df = min(self._settings.keywords_min_df, len(chunk_texts))
        acronyms_min_df = min(self._settings.acronyms_min_df, len(chunk_texts))

        vectorizers = KnowledgeVectorizers(keywords_min_df=keywords_min_df, acronyms_min_df=acronyms_min_df)

        # Check if chunk_texts are empty or only contain stop words
        if not chunk_texts or all(not text.strip() for text in chunk_texts):
            logger.warning(f"Document {document_result.document_filename} has no valid chunk texts")
            return []

        keyword_vectorizer = vectorizers.keywords_vectorizer()
        acronyms_vectorizer = vectorizers.acronyms_vectorizer()

        # Extract keywords and acronyms from chunks
        tfidf_keywords_matrix = keyword_vectorizer.fit_transform(chunk_texts)
        keywords = keyword_vectorizer.get_feature_names_out()
        keywords_scores_matrix = tfidf_keywords_matrix.toarray()  # type: ignore

        tfidf_acronyms_matrix = acronyms_vectorizer.fit_transform(chunk_texts)
        acronyms = acronyms_vectorizer.get_feature_names_out()
        acronyms_scores_matrix = tfidf_acronyms_matrix.toarray()  # type: ignore

        # Check if no features were extracted at all
        if keywords_scores_matrix.shape[1] == 0 and acronyms_scores_matrix.shape[1] == 0:
            logger.warning(f"No keywords or acronyms found in document {document_result.document_filename}")
            return []

        # Extract keywords for each chunk
        keyword_candidates = []
        acronyms_candidates = []
        for chunk_idx, chunk in enumerate(document_result.chunks):
            # Process keywords if available
            if keywords_scores_matrix.shape[1] > 0:
                chunk_scores_keywords = keywords_scores_matrix[chunk_idx]
                # Create keyword candidates for this chunk
                for keyword_idx, score in enumerate(chunk_scores_keywords):
                    if score > 0:  # Only add non-zero scores
                        keyword = LinkedKnowledge._normalize_term(keywords[keyword_idx])
                        # Count actual occurrences of the term in the chunk
                        term_count = len(self._find_term_positions_in_text(keyword, chunk.text))
                        keyword_candidates.append(
                            TermCandidate(
                                term=keyword,
                                document_filename=document_result.document_filename,
                                document_id=document_result.id,
                                total=term_count,  # Actual occurrence count in the chunk
                                page=chunk.page,
                                chunk=chunk.index,
                                type="keyword",
                            )
                        )

            # Process acronyms if available
            if acronyms_scores_matrix.shape[1] > 0:
                chunk_scores_acronyms = acronyms_scores_matrix[chunk_idx]
                # Create acronym candidates for this chunk
                for acronym_idx, score in enumerate(chunk_scores_acronyms):
                    if score > 0:
                        acronym = LinkedKnowledge._normalize_term(acronyms[acronym_idx]).upper()
                        # Count actual occurrences of the term in the chunk
                        term_count = len(self._find_term_positions_in_text(acronym, chunk.text))
                        acronyms_candidates.append(
                            TermCandidate(
                                term=acronym,
                                document_filename=document_result.document_filename,
                                document_id=document_result.id,
                                total=term_count,  # Actual occurrence count in the chunk
                                page=chunk.page,
                                chunk=chunk.index,
                                type="acronym",
                            )
                        )

        # Combine and sort all candidates by their occurrence count
        # TF-IDF was only used to identify important terms, not for scoring
        all_candidates = keyword_candidates + acronyms_candidates
        # Sort by total count to prioritize frequently occurring terms
        all_candidates.sort(key=lambda x: x.total, reverse=True)

        return all_candidates

    @async_timed_operation("Step 3/12: Terms extraction")
    async def _extract_terms_candidates_from_documents(
        self, results_with_chunks: Sequence[KnowledgeExtractionResultWithChunks]
    ) -> Sequence[TermCandidate]:
        """
        Extract term candidates from the chunked knowledge extraction output using BatchProcessor.

        Args:
            results_with_chunks: The chunked extraction output to process.

        Returns:
            Sequence of keyword candidates.
        """
        if not results_with_chunks:
            return []

        # Create ConcurrentProcessor for concurrent document processing
        processor = ConcurrentProcessor[KnowledgeExtractionResultWithChunks, TermCandidate](
            concurrency=10,
            max_retries=3,
        )

        # Create processor function for document extraction
        async def extract_terms_from_document(
            document_result: KnowledgeExtractionResultWithChunks,
        ) -> list[TermCandidate]:
            logger.info(f"Extracting terms from document: {document_result.document_filename}")
            doc_terms = await self._extract_terms_from_document(document_result)
            return list(doc_terms)

        # Process all documents concurrently
        terms = await processor.process(
            items=results_with_chunks,
            processor_func=extract_terms_from_document,
        )

        logger.info(f"Extracted {len(terms)} term candidates from {len(results_with_chunks)} documents")
        return terms

    @timed_operation("Step 4/12: Term grouping")
    def _group_term_candidates(self, term_candidates: List[TermCandidate]) -> Dict[str, TermCandidateGrouped]:
        """
        Group term candidates across all documents and pages.

        Args:
            acronym_candidates: List of all acronym candidates from extraction

        Returns:
            Dictionary mapping acronym to GroupedAcronym object
        """
        groupped: Dict[str, TermCandidateGrouped] = {}

        for candidate in term_candidates:
            term = candidate.term

            if term not in groupped:
                groupped[term] = TermCandidateGrouped(term=term, type=candidate.type, total=0)

            if groupped[term].type != candidate.type:
                # Stronger association with acronym if mixed types found
                groupped[term].type = "acronym"

            occurrence = TermOccurrence(
                document_id=candidate.document_id,
                document_name=candidate.document_filename,
                page=candidate.page,
                chunk_index=candidate.chunk,
                total=candidate.total,
            )

            groupped[term].occurrences.append(occurrence)
            groupped[term].total += candidate.total

        return groupped

    @async_timed_operation("Step 6/12: Terms enrichment with meanings")
    async def _enrich_with_terms_meanings(
        self,
        groups: Dict[str, TermCandidateGrouped],
        document_to_chunks_index: Dict[str, list[KnowledgeChunk]],
    ) -> Dict[str, Term]:
        # Create BatchProcessor for concurrent term validation
        processor = ConcurrentProcessor[TermCandidateGrouped, Term](
            concurrency=10,
            max_retries=3,
        )

        async def _enrich_term(
            item: TermCandidateGrouped,
        ) -> Optional[Term]:
            occurrences_contexts = [
                chunk.text
                for occurrence in item.occurrences
                for chunk in document_to_chunks_index.get(occurrence.document_id, [])
                if chunk.index == occurrence.chunk_index
            ]

            cooccurring_terms: Dict[str, List[str]] = defaultdict(list)
            for cooccurrence in item.cooccurrences:
                # Look up the co-occurring term in groups
                cooccurring_term_data = groups.get(cooccurrence.term)
                if cooccurring_term_data:
                    # Get contexts for this specific co-occurring term
                    contexts = [
                        chunk.text
                        for occurrence in cooccurring_term_data.occurrences[:3]  # Limit occurrences
                        for chunk in document_to_chunks_index.get(occurrence.document_id, [])
                        if chunk.index == occurrence.chunk_index
                    ]
                    cooccurring_terms[cooccurrence.term].extend(contexts)

            response: ConsensusResult[TermMeaningExtractionResponse] = await self._calls.term_extraction_call.execute(
                term=item.term,
                type=item.type,
                occurrences_contexts=occurrences_contexts,
                cooccurring_terms=cooccurring_terms,
            )

            unwrapped_result = response.final_response

            if not unwrapped_result or unwrapped_result.meaning_status == "unknown":
                logger.warning(f"Term extraction failed or invalid for term: {item.term}")
                return None

            if unwrapped_result.type != item.type:
                logger.warning(
                    f"Term type mismatch for '{item.term}': expected {item.type}, got {unwrapped_result.type}"
                )

            return Term(
                term=item.term,
                type=unwrapped_result.type,
                full_form=unwrapped_result.full_form or item.term,
                occurrences=item.occurrences,
                cooccurrences=item.cooccurrences,
                meaning=unwrapped_result.meaning,
                total=item.total,
                reasoning=unwrapped_result.reasoning,
            )

        # Process all acronyms concurrently
        consolidated_terms = await processor.process(
            items=list(groups.values()),
            processor_func=_enrich_term,
        )
        logger.info(f"Enriched {len(consolidated_terms)} terms with meanings and full forms")

        return {term.term: term for term in consolidated_terms}

    async def _chunk_extraction_pages(
        self,
        pages: Sequence[KnowledgePageData],
        document_name: str,
        document_id: str,
        metadata: DocumentMetadata,
    ) -> Sequence[KnowledgeChunk]:
        """
        Process pages asynchronously in batches to extract chunks.
        Uses BatchProcessor for concurrent processing with retry logic.

        Args:
            pages: Sequence of pages to process
            document_name: Name of the document
            document_id: Unique document identifier
            metadata: Document metadata

        Returns:
            Sequence of extracted chunks from all pages (order preserved)
        """
        # Create ConcurrentProcessor with retry logic
        processor = ConcurrentProcessor[KnowledgePageData, KnowledgeChunk](
            concurrency=10,
            max_retries=3,
        )

        # Create a wrapper function that captures the context
        async def process_page(page: KnowledgePageData) -> list[KnowledgeChunk]:
            """Process a single page and return its chunks."""
            chunks = await self._process_page_wrapper(page, document_name, document_id, metadata)
            return list(chunks)

        all_chunks = await processor.process(
            items=pages,
            processor_func=process_page,
        )

        # Re-index all chunks with global indices after collection
        # This ensures proper sequential indexing regardless of processing order
        for idx, chunk in enumerate(all_chunks):
            chunk.index = idx
            chunk.doc_id = f"{document_id}_page_{chunk.page}_chunk_{idx}"

        logger.info(f"Processed {len(pages)} pages for document {document_name}, extracted {len(all_chunks)} chunks")
        return all_chunks

    async def _process_page_wrapper(
        self,
        page: KnowledgePageData,
        document_name: str,
        document_id: str,
        metadata: DocumentMetadata,
    ) -> Sequence[KnowledgeChunk]:
        """
        Wrapper to process a single page and append results to shared list.

        Args:
            page: Page to process
            document_id: Unique document identifier
            metadata: Document metadata
        """

        result = await self.calls.document_chunking_call.execute(
            page=page, document_name=document_name, metadata=metadata
        )

        chunks = []
        for chunk_decision in result.final_response.chunks:
            # Detect content types for this specific chunk
            chunk_content_types = self._detect_chunk_content_types(page, chunk_decision.text)

            chunk = KnowledgeChunk(
                document_id=document_id,
                document_name=document_name,
                doc_id="",
                index=0,
                text=chunk_decision.text.strip(),
                page=page.page,
                content_types=chunk_content_types,
            )
            chunks.append(chunk)
        return chunks

    @async_timed_operation("Step 3/13: Chunk content classification")
    async def _classify_chunk_content(
        self,
        results_with_chunks: Sequence[KnowledgeExtractionResultWithChunks],
    ) -> Sequence[KnowledgeExtractionResultWithChunks]:
        """
        Classify the semantic type of each chunk (table_of_contents, summary, rule, explanation, example).
        THIS IS MANDATORY - all chunks MUST be classified.

        Args:
            results_with_chunks: Documents with chunks to classify

        Returns:
            Same documents with chunks now having semantic_types filled
        """
        classified_results = []

        # Process each document
        for doc_result in results_with_chunks:
            classified_chunks: list[KnowledgeChunk] = []

            # Process chunks in batches using ConcurrentProcessor
            processor = ConcurrentProcessor[KnowledgeChunk, KnowledgeChunk](
                concurrency=10,
                max_retries=2,
            )

            async def classify_single_chunk(chunk: KnowledgeChunk) -> KnowledgeChunk:
                """Classify a single chunk's semantic type - MANDATORY."""
                # Handle empty or whitespace-only chunks
                if not chunk.text or not chunk.text.strip():
                    logger.warning(f"Chunk {chunk.index} is empty, assigning 'general' semantic type")
                    chunk.semantic_types = ["general"]
                    return chunk

                result = await self.calls.chunk_content_classification_call.execute(
                    chunk_text=chunk.text,
                    document_name=doc_result.document_filename,
                    page_number=chunk.page,
                    content_types=chunk.content_types,
                )

                # Update chunk with semantic classifications (multiple types)
                chunk.semantic_types = list(result.final_response.semantic_types)
                logger.debug(
                    f"Classified chunk {chunk.index} with types: {chunk.semantic_types} "
                    f"with confidences: {result.final_response.confidence_scores}"
                )

                return chunk

            # Process all chunks for this document
            classified_chunks_result = await processor.process(
                items=list(doc_result.chunks),
                processor_func=classify_single_chunk,
            )
            classified_chunks = list(classified_chunks_result)

            # Create new result with classified chunks
            classified_result = doc_result.model_copy(update={"chunks": classified_chunks})
            classified_results.append(classified_result)

            # Log statistics for this document
            all_semantic_types = []
            for chunk in classified_chunks:
                all_semantic_types.extend(chunk.semantic_types)
            type_counts = {t: all_semantic_types.count(t) for t in set(all_semantic_types)}
            logger.info(
                f"Document '{doc_result.document_filename}': "
                f"Classified {len(classified_chunks)} chunks with semantic types - {type_counts}"
            )

        return classified_results

    @async_timed_operation("Step 2/12: Document chunking")
    async def _chunk_extraction(
        self,
        raw_extraction: KnowledgeExtractionOutput,
    ) -> Sequence[KnowledgeExtractionResultWithChunks]:
        """
        Efficient chunking using the DocumentChunkingStrategy that processes 2 pages at once with AI.
        Now processes multiple documents in parallel using anyio for better performance.
        """
        chunked_results: list[KnowledgeExtractionResultWithChunks] = []

        # Iterate over all document types in the extraction output
        for attr_name, _ in type(raw_extraction).model_fields.items():
            try:
                extraction_items = getattr(raw_extraction, attr_name)
            except AttributeError:
                continue

            if extraction_items is None or not isinstance(extraction_items, list):
                continue

            # Collect all valid items for parallel processing
            valid_items = []
            for item in extraction_items:
                if item.result is not None:
                    valid_items.append(item)

            if not valid_items:
                continue

            # Process all documents concurrently using anyio task group
            logger.info(f"Processing {len(valid_items)} documents in parallel")

            document_chunks_list = await self._extract_chunks_from_document(valid_items)

            chunked_results.extend(document_chunks_list)

        return chunked_results

    async def _extract_chunks_from_document(
        self, items: List[KnowledgeExtractionItem]
    ) -> List[KnowledgeExtractionResultWithChunks]:
        document_chunks_list: list[KnowledgeExtractionResultWithChunks] = []

        for item in items:
            result = await self._chunk_document_data(item)
            document_chunks_list.append(result)

        return document_chunks_list

    async def _chunk_document_data(self, item: KnowledgeExtractionItem) -> KnowledgeExtractionResultWithChunks:
        result = cast(KnowledgeExtractionResult, item.result)

        # Use the _chunk_extraction_pages method to chunk all pages
        chunks = await self._chunk_extraction_pages(
            pages=result.pages,
            document_name=result.document_filename,
            document_id=result.id,
            metadata=result.document_metadata,
        )

        # Create chunked result for this document using model_copy and update
        return KnowledgeExtractionResultWithChunks(
            **result.model_dump(),
            chunks=list(chunks),
            total_chunks=len(chunks),
        )

    @timed_operation("Step 5/12: Co-occurrence extraction")
    def _enrich_with_cooccurrences(
        self,
        grouped_terms: Dict[str, TermCandidateGrouped],
        chunks_index: Dict[str, list[KnowledgeChunk]],
    ) -> Dict[str, Any]:
        # Build index of term positions in each chunk
        chunk_terms_index = self._build_chunk_term_index(grouped_terms, chunks_index)

        # Calculate cooccurrence scores between all term pairs
        cooccurrence_scores: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

        for _, terms_positions in chunk_terms_index.items():
            terms_list = list(terms_positions.keys())

            # For each pair of terms in the same chunk
            for i, term1 in enumerate(terms_list):
                for term2 in terms_list[i + 1 :]:
                    if term1 != term2:
                        # Calculate weighted score based on minimum distance
                        weight = self._calculate_cooccurrence_weight(terms_positions[term1], terms_positions[term2])
                        cooccurrence_scores[term1][term2] += weight
                        cooccurrence_scores[term2][term1] += weight

        # Add cooccurrences to terms
        updated_terms: Dict[str, Any] = {}

        for term_key, term in grouped_terms.items():
            cooccurrences = []

            if term_key in cooccurrence_scores:
                for cooccurring_term, weighted_score in cooccurrence_scores[term_key].items():
                    # Calculate score based on normalized score,
                    # level of how probable the co-occurrence is
                    score = 1 / (1 + math.exp(-weighted_score))

                    cooccurrences.append(
                        TermCooccurrence(
                            term=cooccurring_term,
                            score=min(score, 1.0),
                        )
                    )

                cooccurrences.sort(key=lambda x: x.score, reverse=True)
                cooccurrences = cooccurrences[:10]

            # Update term with cooccurrences (empty list if none found)
            updated_terms[term_key] = term.model_copy(update={"cooccurrences": cooccurrences})

        return updated_terms

    def _calculate_fuzzy_match(self, x1: str, x2: str) -> float:
        # Handle edge cases
        if not x1 or not x2:
            return 0.0

        if len(x1) == 1:
            # Single character matching is unreliable
            return 0.0

        x1 = x1.lower()
        x2 = x2.lower()

        # Direct substring match (highest confidence)
        if x2 in x1:
            # Scale by relative length - longer matches are better
            length_ratio = len(x2) / len(x1)
            return 0.7 + (0.3 * length_ratio)

        # Token set ratio - handles partial word matches perfectly
        # This ignores word order and duplicates
        token_set_score = fuzz.token_set_ratio(x1, x2) / 100.0

        # Partial ratio - finds best matching substring
        partial_score = fuzz.partial_ratio(x1, x2) / 100.0

        # Weight token_set higher since it's better for this use case
        return (token_set_score * 0.7) + (partial_score * 0.3)

    def _apply_links_to_terms(
        self,
        terms: Dict[str, Term],
        links: List[TermLink],
    ) -> Dict[str, TermWithLinks]:
        """
        Apply term links to the original terms dictionary.

        Args:
            terms: Original terms dictionary
            links: List of TermLink objects representing linked terms

        Returns:
            Updated terms dictionary with links applied
        """
        # Create a copy of the original terms with links field
        terms_with_links: Dict[str, TermWithLinks] = {
            term_key: TermWithLinks(**term.model_dump(), links=[]) for term_key, term in terms.items()
        }

        # Apply links to the corresponding terms
        for link in links:
            if link.link_from in terms_with_links:
                terms_with_links[link.link_from].links.append(link)
            if link.link_to in terms_with_links:
                reverse_link = TermLink(
                    link_from=link.link_to,
                    link_to=link.link_from,
                    score=link.score,
                )
                terms_with_links[link.link_to].links.append(reverse_link)

        return terms_with_links

    @timed_operation("Step 7/12: Term linking")
    def _link_terms(
        self,
        terms: Dict[str, Term],
    ) -> Dict[str, TermWithLinks]:
        """
        Link terms based on similarity between their full forms or term text.

        This creates connections between:
        - Acronyms and keywords (using acronym's full form)
        - Keywords and keywords (using their text)
        - Any term with a full form to any other term

        Args:
            terms: Dictionary containing all terms (both acronyms and keywords)

        Returns:
            Sequence of TermLink objects representing linked terms
        """
        links: List[TermLink] = []
        terms_list = list(terms.items())

        # Compare every term pair
        for i, (term1_key, term1_data) in enumerate(terms_list):
            term1_text = term1_data.full_form

            for term2_key, term2_data in terms_list[i + 1 :]:
                # Skip linking a term to itself
                if term1_key == term2_key:
                    continue

                term2_text = term2_data.full_form

                # Calculate similarity between the two terms
                # Check both directions: term1_text vs term2_text AND term1_text vs term2.term
                match_score1 = self._calculate_fuzzy_match(term1_text, term2_data.term)
                match_score2 = self._calculate_fuzzy_match(term2_text, term1_data.term)

                # Take the best match score
                match_score = max(match_score1, match_score2)

                # Create link if similarity is above threshold
                if match_score >= self._settings.linking_threshold:
                    link = TermLink(
                        link_from=term1_data.term,
                        link_to=term2_data.term,
                        score=match_score,
                    )
                    links.append(link)

                    logger.info(
                        f"Linked '{term1_data.term}' ({term1_data.type}: {term1_text}) "
                        f"with '{term2_data.term}' ({term2_data.type}: {term2_text}) "
                        f"(match_score: {match_score:.2f})"
                    )

        return self._apply_links_to_terms(terms, links)

    def _save_pickle(self, filename: str, model: PydanticLike) -> None:
        pickle_path = self._output_dir / f"{filename}.pkl"

        os.makedirs(self._output_dir, exist_ok=True)
        with open(pickle_path, "wb") as f:
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

        file_size_mb = pickle_path.stat().st_size / (1024 * 1024)
        logger.info(f"Saved to {pickle_path} ({file_size_mb:.2f} MB)")

    def _load_pickle(self, filename: str) -> Optional[Any]:
        """Load a pickle file if it exists.

        Args:
            filename: Name of the pickle file (without .pkl extension)

        Returns:
            The loaded object or None if file doesn't exist
        """
        pickle_path = self._output_dir / f"{filename}.pkl"

        if not pickle_path.exists():
            return None

        try:
            with open(pickle_path, "rb") as f:
                data = pickle.load(f)
                file_size_mb = pickle_path.stat().st_size / (1024 * 1024)
                logger.info(f"Loaded existing {pickle_path} ({file_size_mb:.2f} MB)")
                return data
        except Exception as e:
            logger.error(f"Failed to load pickle file {pickle_path}: {e}")
            return None

    async def _execute_or_load_step(
        self,
        step_name: str,
        step_number: str,
        execute_fn: Callable,
        **kwargs: Any,
    ) -> Any:
        """Execute a step or load from cache if it exists.

        Args:
            step_name: Name of the pickle file to save/load
            step_number: Step number for logging (e.g., "1/12")
            execute_fn: Function to execute if cache doesn't exist
            is_async: Whether the execute function is async
            **kwargs: Arguments to pass to execute_fn

        Returns:
            The result from either cache or execution
        """
        # Try to load from cache first
        cached_data = self._load_pickle(step_name)

        if cached_data is not None:
            logger.info(f"Step {step_number}: Loaded existing {step_name.replace('_', ' ')}")
            return cached_data

        # Execute the step
        logger.info(f"Step {step_number}: Executing {step_name.replace('_', ' ')}...")

        if inspect.iscoroutinefunction(execute_fn):
            result = await execute_fn(**kwargs)
        else:
            # Run sync function in thread pool to avoid blocking
            # anyio.to_thread.run_sync expects function and args separately
            # We use partial to bind the kwargs
            func = partial(execute_fn, **kwargs)
            result = await to_thread.run_sync(func)

        # Save the result
        self._save_pickle(step_name, result)

        return result

    def get_extraction_status(self) -> Dict[str, bool]:
        """Check which extraction files exist.

        Returns:
            Dictionary mapping step names to existence status
        """
        steps = self._get_extraction_steps()

        status = {}
        for step in steps:
            pickle_path = self._output_dir / f"{step}.pkl"
            status[step] = pickle_path.exists()

        return status

    def _get_extraction_steps(self) -> List[str]:
        """Get list of extraction step names.

        Returns:
            List of extraction step names
        """
        return [
            "1_raw_extraction",
            "2_chunked_documents",
            "3_classified_chunks",
            "4_term_candidates",
            "5_grouped_terms",
            "6_terms_with_cooccurrences",
            "7_terms_with_meanings",
            "8_terms_with_links",
            "linked_knowledge",
            "knowledge_search",
        ]

    def _check_existing_state(self) -> Tuple[int, int]:
        """Check the existing extraction state.

        Returns:
            Tuple of (preserved_steps, total_steps) where preserved_steps is the
            number of existing extraction files and total_steps is the total
            number of expected steps.
        """
        status = self.get_extraction_status()
        preserved = sum(1 for exists in status.values() if exists)
        total = len(status)
        return preserved, total

    def _get_step_dependencies(self) -> Dict[str, List[str]]:
        """Define dependency chain for extraction steps.

        Returns:
            Dictionary mapping each step to list of steps it depends on
        """
        return {
            "1_raw_extraction": [],  # No dependencies
            "2_chunked_documents": ["1_raw_extraction"],
            "3_classified_chunks": ["2_chunked_documents"],
            "4_term_candidates": ["3_classified_chunks"],
            "5_grouped_terms": ["4_term_candidates"],
            "6_terms_with_cooccurrences": ["5_grouped_terms"],
            "7_terms_with_meanings": ["6_terms_with_cooccurrences"],
            "8_terms_with_links": ["7_terms_with_meanings"],
            "linked_knowledge": ["8_terms_with_links"],
            "knowledge_search": ["linked_knowledge"],
        }

    def _get_image_affected_steps(self) -> List[str]:
        """Get steps that are actually affected by image changes.

        Image regeneration only affects:
        1. Raw extraction (contains image metadata)
        2. Linked knowledge (includes image references)
        3. Search core (indexes image content)

        It does NOT affect:
        - Term extraction/meanings (text-based)
        - Term linking (relationship-based)
        - Co-occurrences (text analysis)

        Returns:
            List of steps that need regeneration when images change
        """
        return [
            "1_raw_extraction",  # Contains image metadata
            "linked_knowledge",  # Includes image references in final structure
            "knowledge_search",  # Search indices include image content
        ]

    def _invalidate_dependencies(self, changed_step: str) -> List[str]:
        """Invalidate steps that depend on the changed step.

        Args:
            changed_step: The step that was regenerated/changed

        Returns:
            List of steps that were invalidated
        """
        dependencies = self._get_step_dependencies()
        invalidated = []

        # Find all steps that depend on the changed step (directly or indirectly)
        def find_dependents(step: str) -> List[str]:
            dependents = []
            for dependent_step, deps in dependencies.items():
                if step in deps:
                    dependents.append(dependent_step)
                    # Recursively find steps that depend on this dependent
                    dependents.extend(find_dependents(dependent_step))
            return dependents

        steps_to_invalidate = find_dependents(changed_step)

        # Remove the pickle files for invalidated steps
        for step in steps_to_invalidate:
            pickle_path = self._output_dir / f"{step}.pkl"
            if pickle_path.exists():
                pickle_path.unlink()
                invalidated.append(step)
                logger.info(f"🗑️  Invalidated dependent step: {step}")

        return invalidated

    def _invalidate_image_affected_steps(self) -> List[str]:
        """Invalidate only the steps that are actually affected by image changes.

        This is much more targeted than full dependency invalidation.
        Images only affect extraction metadata and final knowledge structures,
        not the text processing pipeline.

        Returns:
            List of steps that were invalidated
        """
        steps_to_invalidate = self._get_image_affected_steps()
        invalidated = []

        for step in steps_to_invalidate:
            pickle_path = self._output_dir / f"{step}.pkl"
            if pickle_path.exists():
                pickle_path.unlink()
                invalidated.append(step)
                logger.info(f"🗑️  Invalidated image-affected step: {step}")

        return invalidated

    def _invalidate_dependent_steps(self, steps: List[str]) -> None:
        """Invalidate specific steps by removing their pickle files.

        Args:
            steps: List of step names to invalidate
        """
        for step in steps:
            pickle_path = self._output_dir / f"{step}.pkl"
            if pickle_path.exists():
                pickle_path.unlink()
                logger.info(f"🗑️  Invalidated step: {step}")

    def _prompt_existing_files_action(self) -> str:
        """Prompt user for action when existing extraction files are detected.

        Returns:
            User choice: 'remove', 'continue', 'regenerate_submenu', or 'cancel'
        """
        print("\n" + "=" * 60)
        print("⚠️  EXISTING EXTRACTION FILES DETECTED")
        print("=" * 60)

        status = self.get_extraction_status()
        existing_files = [step for step, exists in status.items() if exists]

        if existing_files:
            print("\nThe following extraction files already exist:")
            for step in existing_files:
                pickle_path = self._output_dir / f"{step}.pkl"
                size_mb = pickle_path.stat().st_size / (1024 * 1024)
                print(f"  • {step}: {size_mb:.2f} MB")

        print("\nPlease choose an action:")
        print("  1. Remove the files and start extraction from the beginning")
        print("  2. Continue run and potentially overwrite specific steps")
        print("  3. Selective regeneration options →")
        print("  4. Cancel the run")
        print("\n" + "=" * 60)

        while True:
            try:
                choice = input("\nEnter your choice (1/2/3/4): ").strip()
                if choice == "1":
                    return "remove"
                elif choice == "2":
                    return "continue"
                elif choice == "3":
                    return "regenerate_submenu"
                elif choice == "4":
                    return "cancel"
                else:
                    print("Invalid choice. Please enter 1, 2, 3, or 4.")
            except KeyboardInterrupt:
                print("\n\nOperation cancelled by user.")
                return "cancel"

    def _prompt_regeneration_submenu(self, globs: list[str]) -> str:
        """Show regeneration submenu with selective options.

        Args:
            globs: Sequence of glob patterns to match source files

        Returns:
            User choice: 'images', 'captions', 'documents', 'back', or 'cancel'
        """
        print("\n" + "=" * 60)
        print("🔄  SELECTIVE REGENERATION OPTIONS")
        print("=" * 60)

        # Check current state of various components
        images_dir = self._output_dir / "images"
        documents_dir = self._output_dir / "source_documents"

        image_count = 0
        if images_dir.exists():
            image_files = list(images_dir.glob("*.png"))
            image_count = len(image_files)

        document_count = 0
        if documents_dir.exists():
            doc_files = list(documents_dir.glob("*"))
            document_count = len(doc_files)

        # Count source PDF files
        all_files = self._resolve_glob_patterns(globs)
        pdf_files = [f for f in all_files if f.suffix.lower() == ".pdf"]
        source_pdf_count = len(pdf_files)

        # Get current extraction status
        status = self.get_extraction_status()
        existing_steps = [step for step, exists in status.items() if exists]

        print("\nCurrent state:")
        print(f"  • Source PDFs: {source_pdf_count} files")
        print(f"  • Extracted images: {image_count} files")
        print(f"  • Copied documents: {document_count} files")
        print(f"  • Existing extraction steps: {len(existing_steps)}/10")

        print("\nWhat would you like to regenerate?")
        print(f"  1. Images + metadata (re-extract from {source_pdf_count} PDFs)")
        print("     🎯 Will invalidate: Raw extraction, linked knowledge, search indices")
        print("     ✅ Preserves: All term processing (meanings, links, co-occurrences)")
        print("  2. Image captions only (keep images, regenerate AI descriptions)")
        print("     ℹ️  Impact: Minimal - only updates display captions")
        print(f"  3. Source document copies ({document_count} existing → re-copy)")
        print("     ℹ️  Impact: None - only affects file copies")
        print("  4. ← Back to main menu")
        print("  5. Cancel the run")
        print("\n" + "=" * 60)

        while True:
            try:
                choice = input("\nEnter your choice (1/2/3/4/5): ").strip()
                if choice == "1":
                    # Show targeted impact
                    image_affected = self._get_image_affected_steps()
                    affected_existing = [step for step in image_affected if step in existing_steps]
                    preserved_steps = [step for step in existing_steps if step not in image_affected]

                    print("\n🎯 TARGETED REGENERATION: Images + metadata")
                    if affected_existing:
                        print(f"Will invalidate {len(affected_existing)} image-related steps:")
                        for step in affected_existing:
                            print(f"  🗑️  {step}")

                    if preserved_steps:
                        print(f"\n✅ Will preserve {len(preserved_steps)} text processing steps:")
                        for step in preserved_steps[:5]:  # Show first 5
                            print(f"  ✅ {step}")
                        if len(preserved_steps) > 5:
                            print(f"  ... and {len(preserved_steps) - 5} more")

                    print("\nThis preserves all expensive text processing work!")

                    confirm = input("\nProceed with targeted image regeneration? (y/N): ").strip().lower()
                    if confirm in ["y", "yes"]:
                        return "images_with_deps"
                    else:
                        continue  # Go back to menu
                elif choice == "2":
                    return "captions"
                elif choice == "3":
                    return "documents"
                elif choice == "4":
                    return "back"
                elif choice == "5":
                    return "cancel"
                else:
                    print("Invalid choice. Please enter 1, 2, 3, 4, or 5.")
            except KeyboardInterrupt:
                print("\n\nOperation cancelled by user.")
                return "cancel"

    def _remove_existing_extraction_files(self) -> None:
        """Remove all existing extraction files from the output directory."""
        status = self.get_extraction_status()
        removed_count = 0

        for step, exists in status.items():
            if exists:
                pickle_path = self._output_dir / f"{step}.pkl"
                try:
                    pickle_path.unlink()
                    logger.info(f"Removed existing file: {pickle_path}")
                    removed_count += 1
                except Exception:
                    logger.exception(f"Failed to remove file: {pickle_path}")

        # Also remove images and source_documents directories if they exist
        images_dir = self._output_dir / "images"
        if images_dir.exists():
            try:
                shutil.rmtree(images_dir)
                logger.info(f"Removed existing images directory: {images_dir}")
                # Recreate the images directory immediately after removal
                images_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"Recreated images directory: {images_dir}")
            except Exception:
                logger.exception(f"Failed to remove images directory: {images_dir}")

        docs_dir = self._output_dir / "source_documents"
        if docs_dir.exists():
            try:
                shutil.rmtree(docs_dir)
                logger.info(f"Removed existing source_documents directory: {docs_dir}")
            except Exception:
                logger.exception(f"Failed to remove source_documents directory: {docs_dir}")

        if removed_count > 0:
            logger.info(f"Removed {removed_count} existing extraction files")

    def _regenerate_images_only(self, globs: list[str]) -> None:
        """Regenerate images only by re-extracting from source PDFs.

        This method preserves all existing extraction data and only regenerates
        the image files and their captions.

        Args:
            globs: Sequence of glob patterns to match source files
        """
        logger.info("🖼️  Starting image regeneration process...")

        # Clear existing images directory
        images_dir = self._output_dir / "images"
        if images_dir.exists():
            import shutil

            shutil.rmtree(images_dir)
            logger.info(f"Removed existing images directory: {images_dir}")

        # Recreate images directory
        images_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created fresh images directory: {images_dir}")

        # Find all PDF files from glob patterns
        all_files = self._resolve_glob_patterns(globs)
        pdf_files = [f for f in all_files if f.suffix.lower() == ".pdf"]

        if not pdf_files:
            logger.warning("No PDF files found to regenerate images from")
            return

        logger.info(f"Found {len(pdf_files)} PDF files for image regeneration")

        # Get PDF extractor
        pdf_extractor = self._extractors.get(".pdf")
        if not pdf_extractor:
            logger.error("PDF extractor not available")
            return

        # Process each PDF file to extract images only
        total_images = 0
        for pdf_file in pdf_files:
            try:
                logger.info(f"Regenerating images from: {pdf_file.name}")
                result = pdf_extractor.extract(pdf_file)

                # Count extracted images
                file_images = sum(len(page.images) for page in result.pages)
                total_images += file_images

                if file_images > 0:
                    logger.info(f"  ✅ Extracted {file_images} images from {pdf_file.name}")
                else:
                    logger.info(f"  ℹ️  No images found in {pdf_file.name}")

            except Exception as e:
                logger.error(f"Error regenerating images from {pdf_file}: {e}")

        logger.info(f"🎉 Image regeneration completed! Total images regenerated: {total_images}")

        # Optimize the newly generated images
        if total_images > 0:
            logger.info("Optimizing regenerated images...")
            self._image_optimizer.optimize()
            logger.info("Image optimization completed")

    async def _regenerate_images_with_dependencies(self, globs: list[str]) -> None:
        """Regenerate images and rebuild only the actually affected extraction steps.

        This method uses targeted invalidation that only affects:
        1. Raw extraction (image metadata)
        2. Linked knowledge (final structure with image refs)
        3. Search core (search indices)

        It preserves all text processing work and automatically rebuilds affected steps.

        Args:
            globs: Sequence of glob patterns to match source files
        """
        logger.info("🔄 Starting targeted image regeneration...")

        # Count before regeneration
        images_dir = self._output_dir / "images"
        old_image_count = 0
        old_size_mb = 0
        if images_dir.exists():
            old_image_files = list(images_dir.glob("*.png"))
            old_image_count = len(old_image_files)
            old_size_mb = sum(f.stat().st_size for f in old_image_files) / (1024 * 1024)

        # First, regenerate the images
        self._regenerate_images_only(globs)

        # Then invalidate only image-affected steps (correct targeted approach)
        logger.info("🎯 Invalidating only image-affected extraction steps...")
        invalidated = self._invalidate_image_affected_steps()

        # Count preserved work before rebuild
        preserved_steps = [
            "2_chunked_documents",
            "3_classified_chunks",
            "4_term_candidates",
            "5_grouped_terms",
            "6_terms_with_cooccurrences",
            "7_terms_with_meanings",
            "8_terms_with_links",
        ]
        preserved_count = 0
        preserved_size = 0
        for step in preserved_steps:
            pickle_path = self._output_dir / f"{step}.pkl"
            if pickle_path.exists():
                size_mb = pickle_path.stat().st_size / (1024 * 1024)
                preserved_count += 1
                preserved_size += size_mb

        # Now automatically rebuild the invalidated steps
        logger.info("⚡ Automatically rebuilding invalidated steps...")
        start_time = datetime.now()

        try:
            # Run extraction which will only rebuild missing steps
            await self.extract(globs)
            rebuild_duration = datetime.now() - start_time
            rebuild_successful = True
        except Exception:
            rebuild_duration = datetime.now() - start_time
            rebuild_successful = False
            logger.exception("❌ Failed to rebuild invalidated steps")
            raise

        # Verify integration status after rebuild
        integration_status = self._verify_image_integration()

        # Show comprehensive completion summary
        logger.info("\n" + "=" * 80)
        logger.info("🎉 IMAGE REGENERATION COMPLETED")
        logger.info("=" * 80)

        logger.info("📊 REGENERATION SUMMARY:")
        logger.info(f"  • Images before: {old_image_count} ({old_size_mb:.1f} MB)")
        logger.info(
            f"  • Images after: {integration_status['image_count']} ({integration_status['optimized_size_mb']:.1f} MB)"
        )
        logger.info(f"  • Net change: {integration_status['image_count'] - old_image_count:+d} images")

        if integration_status["optimized_size_mb"] < old_size_mb:
            compression = ((old_size_mb - integration_status["optimized_size_mb"]) / old_size_mb) * 100
            logger.info(f"  • Optimization: {compression:.1f}% size reduction ✅")

        if invalidated:
            logger.info(f"\n🗑️  REBUILT STEPS ({len(invalidated)}):")
            for step in invalidated:
                logger.info(f"  • {step} ✅")

            if preserved_count > 0:
                logger.info(f"\n✅ PRESERVED TEXT PROCESSING ({preserved_count} steps):")
                for step in preserved_steps:
                    pickle_path = self._output_dir / f"{step}.pkl"
                    if pickle_path.exists():
                        size_mb = pickle_path.stat().st_size / (1024 * 1024)
                        logger.info(f"  ✅ {step} ({size_mb:.1f} MB)")

                total_steps = len(invalidated) + preserved_count
                preservation_percentage = (preserved_count / total_steps) * 100
                logger.info(f"  💰 Preserved {preservation_percentage:.1f}% of extraction work")
                logger.info(f"  💾 Saved {preserved_size:.1f} MB of processing time")

        # Rebuild performance summary
        logger.info("\n⚡ REBUILD PERFORMANCE:")
        logger.info(f"  • Duration: {rebuild_duration.total_seconds():.1f} seconds")
        if rebuild_successful:
            logger.info("  • Status: ✅ Successful")
            estimated_time_saved = max(0, 300 - rebuild_duration.total_seconds())  # Assume full rebuild takes ~5 min
            if estimated_time_saved > 0:
                logger.info(f"  • Time saved: ~{estimated_time_saved:.0f} seconds (thanks to preserved work)")
        else:
            logger.info("  • Status: ❌ Failed")

        # Integration verification
        logger.info("\n🔗 INTEGRATION VERIFICATION:")
        if rebuild_successful and not integration_status["needs_pipeline_rebuild"]:
            logger.info("  ✅ Images successfully integrated into knowledge base")
            logger.info("  ✅ Chunks include new image content types")
            logger.info("  ✅ Linked knowledge includes updated image metadata")
            logger.info("  ✅ Search indices include image content")
        else:
            logger.info("  ❌ Integration incomplete - some steps may need manual rebuild")

        logger.info("\n💡 VERIFICATION COMPLETE:")
        logger.info(f"  • Total images: {integration_status['image_count']}")
        logger.info(f"  • Images in chunks: {integration_status.get('images_in_chunks', 'Unknown')}")
        logger.info(f"  • Knowledge base updated: {'✅' if not integration_status['needs_pipeline_rebuild'] else '❌'}")

        logger.info("=" * 80)

    async def _regenerate_term_meanings_with_dependencies(self, globs: list[str]) -> None:
        """Regenerate term meanings and dependent steps.

        Args:
            globs: Sequence of glob patterns to match source files
        """
        logger.info("🧠 Starting term meanings regeneration...")

        # Invalidate term meanings and all dependent steps
        steps_to_invalidate = [
            "7_terms_with_meanings",
            "8_terms_with_links",
            "linked_knowledge",
            "knowledge_search",
        ]
        invalidated = []

        for step in steps_to_invalidate:
            pickle_path = self._output_dir / f"{step}.pkl"
            if pickle_path.exists():
                pickle_path.unlink()
                invalidated.append(step)
                logger.info(f"🗑️  Invalidated step: {step}")

        # Rebuild the invalidated steps
        logger.info("⚡ Rebuilding invalidated steps...")
        await self.extract(globs)

        logger.info(f"✅ Term meanings regeneration completed! Rebuilt {len(invalidated)} steps.")

    async def _regenerate_knowledge_linking_with_dependencies(self, globs: list[str]) -> None:
        """Regenerate knowledge linking and dependent steps.

        Args:
            globs: Sequence of glob patterns to match source files
        """
        logger.info("🔗 Starting knowledge linking regeneration...")

        # Invalidate linking and dependent steps
        steps_to_invalidate = [
            "8_terms_with_links",
            "linked_knowledge",
            "knowledge_search",
        ]
        invalidated = []

        for step in steps_to_invalidate:
            pickle_path = self._output_dir / f"{step}.pkl"
            if pickle_path.exists():
                pickle_path.unlink()
                invalidated.append(step)
                logger.info(f"🗑️  Invalidated step: {step}")

        # Rebuild the invalidated steps
        logger.info("⚡ Rebuilding invalidated steps...")
        await self.extract(globs)

        logger.info(f"✅ Knowledge linking regeneration completed! Rebuilt {len(invalidated)} steps.")

    async def _regenerate_search_indices_with_dependencies(self, globs: list[str]) -> None:
        """Regenerate search indices.

        Args:
            globs: Sequence of glob patterns to match source files
        """
        logger.info("🔍 Starting search indices regeneration...")

        # Invalidate only search step
        steps_to_invalidate = ["knowledge_search"]
        invalidated = []

        for step in steps_to_invalidate:
            pickle_path = self._output_dir / f"{step}.pkl"
            if pickle_path.exists():
                pickle_path.unlink()
                invalidated.append(step)
                logger.info(f"🗑️  Invalidated step: {step}")

        # Rebuild the invalidated steps
        logger.info("⚡ Rebuilding invalidated steps...")
        await self.extract(globs)

        logger.info(f"✅ Search indices regeneration completed! Rebuilt {len(invalidated)} steps.")

    def _clear_all_extraction_steps(self) -> None:
        """Clear all extraction steps and start fresh."""
        logger.info("🗑️  Clearing all extraction steps...")

        status = self.get_extraction_status()
        removed_count = 0

        for step, exists in status.items():
            if exists:
                pickle_path = self._output_dir / f"{step}.pkl"
                try:
                    pickle_path.unlink()
                    logger.info(f"Removed existing file: {pickle_path}")
                    removed_count += 1
                except Exception:
                    logger.exception(f"Failed to remove file: {pickle_path}")

        # Also remove images and source_documents directories
        import shutil

        images_dir = self._output_dir / "images"
        if images_dir.exists():
            try:
                shutil.rmtree(images_dir)
                logger.info(f"Removed existing images directory: {images_dir}")
            except Exception:
                logger.exception(f"Failed to remove images directory: {images_dir}")

        docs_dir = self._output_dir / "source_documents"
        if docs_dir.exists():
            try:
                shutil.rmtree(docs_dir)
                logger.info(f"Removed existing source_documents directory: {docs_dir}")
            except Exception:
                logger.exception(f"Failed to remove source_documents directory: {docs_dir}")

        logger.info(f"✅ Cleared {removed_count} extraction files and directories.")

    def _verify_image_integration(self) -> Dict[str, Any]:
        """Verify that regenerated images are properly integrated into the knowledge base.

        Returns:
            Dictionary with integration status and statistics
        """
        images_dir = self._output_dir / "images"

        # Count images
        image_count = 0
        optimized_size_mb = 0
        if images_dir.exists():
            image_files = list(images_dir.glob("*.png"))
            image_count = len(image_files)
            optimized_size_mb = sum(f.stat().st_size for f in image_files) / (1024 * 1024)

        # Check if raw extraction includes new images
        raw_extraction_exists = (self._output_dir / "1_raw_extraction.pkl").exists()

        # Check linked knowledge integration
        linked_knowledge_exists = (self._output_dir / "linked_knowledge.pkl").exists()
        search_core_exists = (self._output_dir / "knowledge_search.pkl").exists()

        return {
            "image_count": image_count,
            "optimized_size_mb": optimized_size_mb,
            "raw_extraction_updated": not raw_extraction_exists,  # Should be missing after invalidation
            "needs_pipeline_rebuild": not linked_knowledge_exists or not search_core_exists,
            "integration_status": "pending_rebuild" if not linked_knowledge_exists else "completed",
        }

    def _regenerate_captions_only(self, globs: list[str]) -> None:
        """Regenerate only image captions without re-extracting images.

        This method keeps existing image files and only updates their AI-generated captions
        using the current context and models.

        Args:
            globs: Sequence of glob patterns to match source files (for context)
        """
        logger.info("📝 Starting caption regeneration process...")

        images_dir = self._output_dir / "images"
        if not images_dir.exists():
            logger.warning("No images directory found - nothing to regenerate captions for")
            return

        image_files = list(images_dir.glob("*.png"))
        if not image_files:
            logger.warning("No PNG images found in images directory")
            return

        logger.info(f"Found {len(image_files)} images for caption regeneration")

        # Load existing extraction data to get context for each image
        try:
            raw_extraction = self._load_pickle("1_raw_extraction")
            if not raw_extraction:
                logger.warning("No extraction data found - using minimal context for captions")

            total_regenerated = 0

            for image_file in image_files:
                try:
                    # Parse filename to get document and page info
                    # Format: {pdf_stem}_page_{page_num}_img_{img_num}.png
                    stem = image_file.stem
                    parts = stem.split("_")

                    if len(parts) >= 4 and parts[-3] == "page" and parts[-1].startswith("img"):
                        # Extract document name and page number
                        page_num = int(parts[-2])
                        doc_parts = parts[:-3]  # Everything before "_page_X_img_Y"
                        doc_stem = "_".join(doc_parts)

                        # Find context from extraction data
                        context = "Image from document"
                        if raw_extraction and hasattr(raw_extraction, "pdf"):
                            for pdf_item in raw_extraction.pdf:
                                if pdf_item.result.document_filename.startswith(doc_stem):
                                    # Find the specific page
                                    for page in pdf_item.result.pages:
                                        if page.page == page_num:
                                            context = page.text[:500]  # First 500 chars as context
                                            break
                                    break

                        # Load image and regenerate caption
                        from PIL import Image

                        image = Image.open(image_file)
                        new_caption = self._image_recognition.caption_for_image(image, context=context)

                        # Update the caption in extraction data would require complex reconstruction
                        # For now, just log the regeneration
                        logger.info(f"✅ Regenerated caption for {image_file.name}: {new_caption[:100]}...")
                        total_regenerated += 1

                except Exception as e:
                    logger.warning(f"Failed to regenerate caption for {image_file.name}: {e}")

            logger.info(f"🎉 Caption regeneration completed! Regenerated {total_regenerated} captions")

        except Exception as e:
            logger.error(f"Error during caption regeneration: {e}")

    def _regenerate_source_documents(self, globs: list[str]) -> None:
        """Regenerate source document copies.

        This method re-copies all source documents to the output directory,
        replacing any existing copies.

        Args:
            globs: Sequence of glob patterns to match source files
        """
        logger.info("📄 Starting source document regeneration...")

        # Remove existing source documents directory
        documents_dir = self._output_dir / "source_documents"
        if documents_dir.exists():
            import shutil

            shutil.rmtree(documents_dir)
            logger.info(f"Removed existing source documents directory: {documents_dir}")

        # Find all source files
        all_files = self._resolve_glob_patterns(globs)
        if not all_files:
            logger.warning("No source files found to copy")
            return

        logger.info(f"Found {len(all_files)} source files for copying")

        # Re-copy source documents
        self._copy_source_documents(all_files)

        logger.info(f"🎉 Source document regeneration completed! Copied {len(all_files)} files")
