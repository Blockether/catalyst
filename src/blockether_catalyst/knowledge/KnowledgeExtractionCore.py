"""
Simplified Document Terms Processing System
Self-contained implementation with standard logging
"""

import logging
import math
import os
import pickle
import re
import shutil
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps
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

import anyio
import numpy as np
import trio
from pydantic import BaseModel, RootModel
from rapidfuzz import fuzz

from blockether_catalyst.consensus.ConsensusTypes import ConsensusResult

from ..utils import ConcurrentProcessor
from .ImageOptimizer import ImageOptimizer
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
        self.calls = calls
        self._settings = settings
        self._output_dir = settings.extraction_output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"KnowledgeExtractionCore initialized with settings: {self._settings.model_dump()}")
        logger.info(f"KnowledgeExtractionCore initialized with output_dir: {self._output_dir}")

        # Validate that ALL typed calls are provided - they are MANDATORY
        if not self.calls.term_extraction_call:
            raise ValueError("term_extraction_call is mandatory in settings")

        if not self.calls.document_chunking_call:
            raise ValueError("document_chunking_call is mandatory in settings")

        image_output_dir = self._output_dir / "images"
        image_output_dir.mkdir(parents=True, exist_ok=True)

        self._image_optimizer = ImageOptimizer(image_output_dir, level=settings.image_optimization_level)

        # Define extractors for each supported extension
        self.extractors = {".pdf": PDFKnowledgeExtractor(image_output_dir, self._settings)}

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
            globs: List of glob patterns (e.g., '*.pdf', 'docs/**/*.txt')

        Returns:
            List of resolved file paths matching the patterns
        """
        all_files: Set[Path] = set()

        for pattern in globs:
            logger.info(f"Resolving glob pattern: {pattern}")
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
            if extension not in self.extractors:
                logger.warning(f"Skipping unsupported extension: {extension}")
                continue

            logger.info(f"Processing {len(file_list)} {extension} files")

            extractor = self.extractors[extension]
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
    ) -> Dict[str, Sequence[KnowledgeChunk]]:
        """Build document-to-chunks index for O(1) chunk lookups.

        Args:
            results_with_chunks: Chunked extraction results

        Returns:
            Dictionary mapping document IDs to their chunks for efficient access
        """
        chunks_index: Dict[str, Sequence[KnowledgeChunk]] = {}

        for item in results_with_chunks:
            chunks_index[item.id] = item.chunks

        return chunks_index

    def _build_chunk_term_index(
        self,
        grouped_terms: Dict[str, TermCandidateGrouped],
        chunks_index: Dict[str, Sequence[KnowledgeChunk]],
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
    async def extract(self, globs: list[str]) -> LinkedKnowledge:
        """
        Extract knowledge from files matching the provided glob patterns.

        Args:
            globs: Sequence of glob patterns to match files

        Returns:
            LinkedKnowledge containing all extracted knowledge and relationships
        """
        logger.info(f"Starting with {len(globs)} glob patterns")

        all_files = self._resolve_glob_patterns(globs)
        logger.info(f"Found {len(all_files)} files from glob patterns")

        files_by_extension = self._group_files_by_extension(all_files)
        logger.info(f"Grouped files by extension: {dict((k, len(v)) for k, v in files_by_extension.items())}")

        raw_extraction = self._process_files_by_extension(files_by_extension)
        logger.info(f"Raw extraction completed: {raw_extraction.model_dump().keys()} documents")

        self._save_pickle("1_raw_extraction", raw_extraction)

        results_with_chunks: Sequence[KnowledgeExtractionResultWithChunks] = await self._chunk_extraction(
            raw_extraction
        )
        total_chunks = self._count_total_chunks(results_with_chunks)
        logger.info(f"Created {total_chunks} chunks from {len(results_with_chunks)} documents")

        self._save_pickle("2_chunked_documents", results_with_chunks)

        document_to_chunks_index: Dict[str, Sequence[KnowledgeChunk]] = self._build_document_chunk_index(
            results_with_chunks
        )
        logger.info(f"Built chunks index for {len(document_to_chunks_index)} documents")

        terms: Sequence[TermCandidate] = await self._extract_terms_candidates_from_documents(results_with_chunks)
        logger.info(f"Found {len(terms)} term candidates")
        self._save_pickle("3_term_candidates", terms)

        grouped_terms = self._group_term_candidates(terms)
        self._save_pickle("4_grouped_terms", grouped_terms)

        terms_with_cooccurrences: Dict[str, TermCandidateGrouped] = self._enrich_with_cooccurrences(
            grouped_terms, document_to_chunks_index
        )
        logger.info(f"Added co-occurrences to {len(terms_with_cooccurrences)} terms")
        self._save_pickle("5_terms_with_cooccurrences", terms_with_cooccurrences)

        consolidated_terms = await self._enrich_with_terms_meanings(terms_with_cooccurrences, document_to_chunks_index)
        self._save_pickle("6_terms_with_meanings", consolidated_terms)

        consolidated_terms_with_links = self._link_terms(consolidated_terms)
        self._save_pickle("7_terms_with_links", consolidated_terms_with_links)

        logger.info("Step 8/12: Building LinkedKnowledge from extraction data")
        raw_data = RawExtractionData(
            results_with_chunks=results_with_chunks,
            terms=consolidated_terms_with_links,
            document_to_chunks_index=document_to_chunks_index,
        )
        linked_knowledge = LinkedKnowledge.from_extraction_data(raw_data)

        self._save_pickle("linked_knowledge", linked_knowledge)

        logger.info("Step 9/12: Optimizing extracted images")
        self._image_optimizer.optimize()

        self._copy_source_documents(all_files)

        logger.info("Step 10/12: Creating KnowledgeSearchCore with vector indices")
        search_core = KnowledgeSearchCore(
            linked_knowledge=linked_knowledge,
            pickle_path=self._output_dir / "knowledge_search.pkl",
        )

        logger.info("Step 11/12: Persisting KnowledgeSearchCore to pickle")
        search_core.persist()

        pickle_size_mb = (self._output_dir / "knowledge_search.pkl").stat().st_size / (1024 * 1024)
        logger.info(f"KnowledgeSearchCore saved ({pickle_size_mb:.2f} MB)")

        logger.info("Step 12/12: Knowledge extraction pipeline completed successfully")
        logger.info(
            f"Processed {len(linked_knowledge.documents)} documents with {linked_knowledge.total_chunks} chunks"
        )
        logger.info(
            f"Extracted {linked_knowledge.total_acronyms} acronyms and {linked_knowledge.total_keywords} keywords"
        )

        return linked_knowledge

    @timed_operation("Step 9/12: Copy source documents")
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
                                total=term_count,  # Use actual count instead of TF-IDF score
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
                                total=term_count,  # Use actual count instead of TF-IDF score
                                page=chunk.page,
                                chunk=chunk.index,
                                type="acronym",
                            )
                        )

        # Combine and sort all candidates by their TF-IDF scores (not by count)
        # We still want to preserve the TF-IDF ordering for importance
        all_candidates = keyword_candidates + acronyms_candidates
        # Since we no longer have scores, sort by total count instead
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
        document_to_chunks_index: Dict[str, Sequence[KnowledgeChunk]],
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

            response: ConsensusResult[TermMeaningExtractionResponse] = await self.calls.term_extraction_call.execute(
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
            chunk = KnowledgeChunk(
                document_id=document_id,
                document_name=document_name,
                doc_id="",
                index=0,
                text=chunk_decision.text.strip(),
                page=page.page,
            )
            chunks.append(chunk)
        return chunks

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

    async def _chunk_document_data(self, item):
        result = cast(KnowledgeExtractionResult, item.result)

        # Use the _chunk_extraction_pages method to chunk all pages
        chunks = await self._chunk_extraction_pages(
            pages=result.pages,
            document_name=result.document_filename,
            document_id=result.id,
            metadata=result.document_metadata,
        )

        # Create chunked result for this document
        return KnowledgeExtractionResultWithChunks(
            document_filename=result.document_filename,
            id=result.id,
            source_type=result.source_type,
            document_metadata=result.document_metadata,
            pages=result.pages,
            total_pages=result.total_pages,
            raw=result.raw,
            chunks=chunks,
            total_chunks=len(chunks),
        )

    @timed_operation("Step 5/12: Co-occurrence extraction")
    def _enrich_with_cooccurrences(
        self,
        grouped_terms: Dict[str, TermCandidateGrouped],
        chunks_index: Dict[str, Sequence[KnowledgeChunk]],
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
