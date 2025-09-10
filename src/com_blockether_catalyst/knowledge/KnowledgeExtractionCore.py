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

import numpy as np
from pydantic import BaseModel, RootModel
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import CountVectorizer

from ..utils import ConcurrentProcessor
from .ImageOptimizer import ImageOptimizer
from .KnowledgeExtractionCallBase import ExtractionCallsSettings
from .KnowledgeExtractionTypes import (
    DocumentMetadata,
    KnowledgeChunk,
    KnowledgeChunkWithTerms,
    KnowledgeExtractionItem,
    KnowledgeExtractionOutput,
    KnowledgeExtractionResult,
    KnowledgeExtractionResultWithChunks,
    KnowledgeMetadata,
    KnowledgePageData,
    KnowledgeProcessorSettings,
    LinkedKnowledge,
    Term,
    TermCandidate,
    TermCandidateGrouped,
    TermCooccurrence,
    TermLink,
    TermOccurrence,
    TermWithLinks,
)
from .KnowledgeSearchCore import KnowledgeSearchCore
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
    """Core knowledge extraction system"""

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

        # Future: Add more extractors here
        # ".docx": self.docx_extractor
        # ".txt": self.txt_extractor

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

    @staticmethod
    def normalize_term(term: str) -> str:
        """
        Normalize a term by lowercasing and removing unwanted characters.

        Args:
            term: The term to normalize

        Returns:
            Normalized term text
        """
        # Convert to lowercase
        normalized = term.lower()

        # Remove emojis using pre-compiled pattern
        normalized = KnowledgeExtractionCore.EMOJI_PATTERN.sub("", normalized)

        # Remove parenthetical content (e.g., "ROI (Return on Investment)" -> "ROI")
        normalized = re.sub(r"\s*\([^)]*\)", "", normalized)

        # Remove bullets and list markers - keep applying until no more markers found
        # This handles cases like "1. • text" where multiple markers are present
        previous = ""
        while previous != normalized:
            previous = normalized
            normalized = re.sub(r"^[\s]*[-•·*▪▸◦‣⁃]\s*", "", normalized)  # Unordered list markers
            normalized = re.sub(r"^[\s]*\d+[.)]\s*", "", normalized)  # Ordered list markers (1. or 1))
            normalized = re.sub(r"^[\s]*[a-z][.)]\s*", "", normalized)  # Lettered lists (a. or a))
            normalized = re.sub(r"^[\s]*[ivxlcdm]+[.)]\s*", "", normalized)  # Roman numerals

        # Remove multiple spaces and newlines
        normalized = re.sub(r"\s+", " ", normalized)

        # Remove trailing and leading hyphens (but keep internal hyphens like in "API-KEY")
        normalized = normalized.strip("-")

        # Strip leading/trailing whitespace
        normalized = normalized.strip()

        return normalized

    def _resolve_glob_patterns(self, globs: list[str]) -> list[Path]:
        all_files: Set[Path] = set()

        for pattern in globs:
            logger.info(f"Resolving glob pattern: {pattern}")
            matched_files = list(Path().rglob(pattern))
            all_files.update(matched_files)

        return list(all_files)

    def _group_files_by_extension(self, files: list[Path]) -> dict[str, list[Path]]:
        """
        Group files by their extension.

        Args:
            files: Sequence of file paths

        Returns:
            Dictionary mapping extension to list of file paths
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
        """
        Process files grouped by extension and return extraction output.

        Args:
            files_by_extension: Dictionary mapping file extensions to lists of file paths

        Returns:
            KnowledgeExtractionOutput with extraction results
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

            for file_path in file_list:
                logger.info(f"Processing: {file_path}")
                single_result = extractor.extract(file_path)
                extraction_item = KnowledgeExtractionItem(result=single_result)
                extraction_results.append(extraction_item)

            # Store results based on extension
            if extension == ".pdf":
                extraction_output.pdf = extraction_results
            # Future: Add more result storage here
            # elif extension == ".docx":
            #     extraction_output.docx = extraction_results

        return extraction_output

    def _count_total_chunks(self, results_with_chunks: Sequence[KnowledgeExtractionResultWithChunks]) -> int:
        """
        Count the total number of chunks in the extraction output.

        Args:
            extraction_output_with_chunks: The chunked extraction output

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
        """
        Build index for efficient chunk lookup: document_id -> chunks.

        Args:
            extraction_output_with_chunks: The chunked extraction output

        Returns:
            Dictionary mapping document IDs to their chunks
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
        """
        Build an index of which keywords appear in which chunks with their positions.

        Args:
            grouped_keywords: Dictionary of grouped keywords
            chunks_index: Pre-built index mapping document IDs to chunks

        Returns:
            Dictionary mapping (document_id, chunk_index) to dict of keywords with their positions
        """
        chunk_terms_index: Dict[Tuple[str, int], Dict[str, Sequence[int]]] = defaultdict(lambda: defaultdict(list))

        for term, group in grouped_terms.items():
            for occurrence in group.occurrences:
                # Use the pre-built chunks_index for efficient lookup
                if occurrence.document_id in chunks_index:
                    for chunk in chunks_index[occurrence.document_id]:
                        positions = self._find_term_positions_in_text(term, chunk.text)
                        if positions:
                            chunk_terms_index[(occurrence.document_id, chunk.index)][term] = positions

        return chunk_terms_index

    def _find_term_positions_in_text(self, term: str, text: str) -> Sequence[int]:
        """
        Find all positions where a term appears as a whole word in the text.

        Args:
            term: The term to search for
            text: The text to search in

        Returns:
            Sequence of character positions where the term starts
        """
        positions = []
        # Use word boundary regex to match exact terms only
        pattern = r"\b" + re.escape(term.lower()) + r"\b"
        text_lower = text.lower()

        for match in re.finditer(pattern, text_lower):
            positions.append(match.start())

        return positions

    def _calculate_cooccurrence_weight(self, positions1: Sequence[int], positions2: Sequence[int]) -> float:
        """
        Calculate the weighted cooccurrence score based on minimum distance between term positions.

        Args:
            positions1: Sequence of positions for first term
            positions2: Sequence of positions for second term

        Returns:
            Weighted cooccurrence score (higher = closer terms)
        """
        if not positions1 or not positions2:
            return 0.0

        # Find the minimum distance between any pair of positions
        min_distance = float("inf")
        for pos1 in positions1:
            for pos2 in positions2:
                distance = abs(pos1 - pos2)
                min_distance = min(min_distance, distance)

        # Convert distance to weight using exponential decay
        # Weight = e^(-distance/100) where distance is in characters
        # This gives high weight to close terms, low weight to distant terms
        weight = math.exp(-min_distance / 100.0)

        # Ensure minimum weight for same-chunk cooccurrence
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

        # Resolve all glob patterns to actual file paths
        all_files = self._resolve_glob_patterns(globs)
        logger.info(f"Found {len(all_files)} files from glob patterns")

        # Group files by extension
        files_by_extension = self._group_files_by_extension(all_files)
        logger.info(f"Grouped files by extension: {dict((k, len(v)) for k, v in files_by_extension.items())}")

        # Step 1: Process files and get extraction output
        raw_extraction = self._process_files_by_extension(files_by_extension)
        logger.info(f"Raw extraction completed: {raw_extraction.model_dump().keys()} documents")

        # Persist raw extraction results
        self._save_pickle("1_raw_extraction", raw_extraction)

        # Step 2: Chunk the documents and persist the results
        results_with_chunks: Sequence[KnowledgeExtractionResultWithChunks] = await self._chunk_extraction(
            raw_extraction
        )
        total_chunks = self._count_total_chunks(results_with_chunks)
        logger.info(f"Created {total_chunks} chunks from {len(results_with_chunks)} documents")

        self._save_pickle("2_chunked_documents", results_with_chunks)

        # Build chunks index for efficient lookup across all steps
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

        # Now validate and extract full forms for terms
        consolidated_terms = await self._enrich_with_terms_meanings(terms_with_cooccurrences, document_to_chunks_index)
        self._save_pickle("6_terms_with_meanings", consolidated_terms)

        consolidated_terms_with_links = self._link_terms(consolidated_terms)
        self._save_pickle("7_terms_with_links", consolidated_terms_with_links)

        # Build final LinkedKnowledge object
        linked_knowledge = self._build_linked_knowledge(
            results_with_chunks=results_with_chunks,
            terms=consolidated_terms_with_links,
            document_to_chunks_index=document_to_chunks_index,
        )

        # Serialize to pickle file
        self._save_pickle("linked_knowledge", linked_knowledge)

        # Step 10: Optimize extracted images
        logger.info("Step 10/12: Optimizing extracted images")
        self._image_optimizer.optimize()

        # Copy source documents to output directory
        self._copy_source_documents(all_files)

        # Step 11: Create and persist KnowledgeSearchCore
        logger.info("Step 12/12: Creating KnowledgeSearchCore with vector indices")
        search_core = KnowledgeSearchCore(
            linked_knowledge=linked_knowledge,
            pickle_path=self._output_dir / "knowledge_search.pkl",
        )

        # Step 11: Persist the search core
        logger.info("Step 11/12: Persisting KnowledgeSearchCore to pickle")
        search_core.persist()

        pickle_size_mb = (self._output_dir / "knowledge_search.pkl").stat().st_size / (1024 * 1024)
        logger.info(f"KnowledgeSearchCore saved ({pickle_size_mb:.2f} MB)")

        # Step 12: Final summary
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

    @timed_operation("Step 8/12: Build LinkedKnowledge")
    def _build_linked_knowledge(
        self,
        results_with_chunks: Sequence[KnowledgeExtractionResultWithChunks],
        terms: Dict[str, TermWithLinks],
        document_to_chunks_index: Dict[str, Sequence[KnowledgeChunk]],
    ) -> LinkedKnowledge:
        documents: Dict[str, DocumentMetadata] = {}
        pages_index: Dict[Tuple[str, int], KnowledgePageData] = {}

        # Build search indices
        (
            term_to_documents_index,
            document_to_terms_index,
            chunks,
            document_to_chunk_ids_index,
            document_page_to_chunks_index,
        ) = self._build_search_indices(terms, document_to_chunks_index)

        for result in results_with_chunks:
            # Build pages index
            for page in result.pages:
                # Create page key: (document_id, page_number) tuple
                page_key = (result.id, page.page)

                # Store page without raw_text in the pages index
                page_data = KnowledgePageData(
                    page=page.page,
                    text=page.text,
                    tables=page.tables,
                    images=page.images,
                )
                pages_index[page_key] = page_data

            # Calculate total tables from pages
            total_tables = sum(len(page.tables) for page in result.pages)

            # We'll calculate term counts later after building indices
            documents[result.id] = DocumentMetadata(
                document_id=result.id,
                filename=result.filename,
                total_pages=len(result.pages),
                total_chunks=result.total_chunks,
                total_terms=0,  # Will be updated below
                total_acronyms=0,  # Will be updated below
                total_keywords=0,  # Will be updated below
                total_tables=total_tables,
            )

        # Build inverted indices for term-to-chunk lookups
        term_to_chunks_index, term_to_document_with_page_index = self._build_inverted_indices(terms)

        # Update document metadata with actual term counts
        for doc_id in documents:
            # Get all terms for this document
            doc_terms = document_to_terms_index.get(doc_id, set())

            # Count acronyms and keywords for this document
            acronyms_count = 0
            keywords_count = 0

            for term_key in doc_terms:
                if term_key in terms:
                    term = terms[term_key]
                    if term.type == "acronym":
                        acronyms_count += 1
                    elif term.type == "keyword":
                        keywords_count += 1

            # Update the document metadata
            documents[doc_id].total_terms = len(doc_terms)
            documents[doc_id].total_acronyms = acronyms_count
            documents[doc_id].total_keywords = keywords_count

        # Calculate total statistics across all documents
        total_acronyms_count = sum(1 for term in terms.values() if term.type == "acronym")
        total_keywords_count = sum(1 for term in terms.values() if term.type == "keyword")
        total_chunks_count = sum(len(chunk_ids) for chunk_ids in document_to_chunk_ids_index.values())

        # Create and return LinkedKnowledge object with all indices
        return LinkedKnowledge(
            documents=documents,
            pages=pages_index,
            terms=terms,
            chunks=chunks,
            document_to_chunk_ids_index=document_to_chunk_ids_index,
            document_page_to_chunks_index=document_page_to_chunks_index,
            term_to_chunks_index=term_to_chunks_index,
            term_to_document_with_page_index=term_to_document_with_page_index,
            term_to_documents_index=term_to_documents_index,
            document_to_terms_index=document_to_terms_index,
            total_acronyms=total_acronyms_count,
            total_keywords=total_keywords_count,
            total_chunks=total_chunks_count,
        )

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

        # Create vectorizer for this document
        tfidf_counter_keywords = CountVectorizer(
            stop_words="english",
            strip_accents="ascii",
            ngram_range=(2, 3),
            min_df=keywords_min_df,
            analyzer="word",
            dtype=np.int64,
        )

        # Create vectorizer for this document - using simple pattern for acronyms (2+ consecutive capitals)
        tfidf_counter_acronyms = CountVectorizer(
            stop_words=None,
            strip_accents="ascii",
            ngram_range=(1, 1),
            min_df=acronyms_min_df,
            token_pattern=r"\b[A-Z]{2,}[_/-]?[A-Z]*\b",
            lowercase=False,
            dtype=np.int64,
        )

        # Check if chunk_texts are empty or only contain stop words
        if not chunk_texts or all(not text.strip() for text in chunk_texts):
            logger.warning(f"Document {document_result.filename} has no valid chunk texts")
            return []

        # Extract keywords and acronyms from chunks
        tfidf_keywords_matrix = tfidf_counter_keywords.fit_transform(chunk_texts)
        keywords = tfidf_counter_keywords.get_feature_names_out()
        keywords_scores_matrix = tfidf_keywords_matrix.toarray()  # type: ignore

        tfidf_acronyms_matrix = tfidf_counter_acronyms.fit_transform(chunk_texts)
        acronyms = tfidf_counter_acronyms.get_feature_names_out()
        acronyms_scores_matrix = tfidf_acronyms_matrix.toarray()  # type: ignore

        # Check if no features were extracted at all
        if keywords_scores_matrix.shape[1] == 0 and acronyms_scores_matrix.shape[1] == 0:
            logger.warning(f"No keywords or acronyms found in document {document_result.filename}")
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
                        keyword_candidates.append(
                            TermCandidate(
                                term=self.normalize_term(keywords[keyword_idx]),
                                document_name=document_result.filename,
                                document_id=document_result.id,
                                total=score,
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
                    if score > 0:  # Only add non-zero scores
                        acronyms_candidates.append(
                            TermCandidate(
                                term=self.normalize_term(acronyms[acronym_idx]),
                                document_name=document_result.filename,
                                document_id=document_result.id,
                                total=score,
                                page=chunk.page,
                                chunk=chunk.index,
                                type="acronym",
                            )
                        )

        # Combine and sort all candidates
        all_candidates = keyword_candidates + acronyms_candidates
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
            concurrency=5,
            max_retries=3,
        )

        # Create processor function for document extraction
        async def extract_terms_from_document(
            document_result: KnowledgeExtractionResultWithChunks,
        ) -> list[TermCandidate]:
            logger.info(f"Extracting terms from document: {document_result.filename}")
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
                document_name=candidate.document_name,
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
            concurrency=5,
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

            response = await self.calls.term_extraction_call.execute(
                term=item.term,
                type=item.type,
                occurrences_contexts=occurrences_contexts,
                cooccurring_terms=cooccurring_terms,
            )

            unwrapped_result = response.final_response
            if not unwrapped_result or unwrapped_result.type == "unimportant":
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
        metadata: KnowledgeMetadata,
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
            concurrency=5,  # Process 5 pages concurrently
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
        metadata: KnowledgeMetadata,
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
                text=chunk_decision.root.strip(),
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
        """
        chunked_results: list[KnowledgeExtractionResultWithChunks] = []

        # Iterate over all document types in the extraction output
        for attr_name, field_info in raw_extraction.model_fields.items():
            try:
                extraction_items = getattr(raw_extraction, attr_name)
            except AttributeError:
                continue

            if extraction_items is None or not isinstance(extraction_items, list):
                continue

            for item in extraction_items:
                if item.result is None:
                    continue

                result = cast(KnowledgeExtractionResult, item.result)

                # Use the _chunk_extraction_pages method to chunk all pages
                chunks = await self._chunk_extraction_pages(
                    pages=result.pages,
                    document_name=result.filename,
                    document_id=result.id,
                    metadata=result.metadata,
                )

                # Create chunked result for this document
                chunked_result = KnowledgeExtractionResultWithChunks(
                    filename=result.filename,
                    id=result.id,
                    source_type=result.source_type,
                    metadata=result.metadata,
                    pages=result.pages,
                    total_pages=result.total_pages,
                    raw=result.raw,
                    chunks=chunks,
                    total_chunks=len(chunks),
                )

                chunked_results.append(chunked_result)

                logger.info(
                    f"[{result.filename}] Created {len(chunks)} efficient semantic chunks using 2-page batching"
                )

        return chunked_results

    def _build_inverted_indices(
        self, terms: Dict[str, TermWithLinks]
    ) -> Tuple[Dict[str, Set[Tuple[str, int]]], Dict[str, Set[Tuple[str, int]]]]:
        """
        Build inverted indices for O(1) term lookups.

        Args:
            terms: Dictionary of all terms (keywords and acronyms)

        Returns:
            Tuple of (term_to_chunks_index, term_to_document_with_page_index)
        """
        from collections import defaultdict

        term_to_chunks: Dict[str, Set[Tuple[str, int]]] = defaultdict(set)
        term_to_document_with_page: Dict[str, Set[Tuple[str, int]]] = defaultdict(set)

        # Build inverted indices from term occurrences
        for term_name, term_data in terms.items():
            # Normalize the term name for consistent lookups
            normalized_term = self.normalize_term(term_name)

            # Add all occurrences for this term
            for occurrence in term_data.occurrences:
                # Add to chunks index
                term_to_chunks[normalized_term].add((occurrence.document_id, occurrence.chunk_index))
                # Add to document-page index
                term_to_document_with_page[normalized_term].add((occurrence.document_id, occurrence.page))

        # Convert defaultdicts to regular dicts for serialization
        return dict(term_to_chunks), dict(term_to_document_with_page)

    def _build_search_indices(
        self,
        terms: Dict[str, TermWithLinks],
        document_chunks: Dict[str, Sequence[KnowledgeChunk]],
    ) -> Tuple[
        Dict[str, Set[str]],
        Dict[str, Set[str]],
        Dict[str, KnowledgeChunkWithTerms],
        Dict[str, Set[str]],
        Dict[Tuple[str, int], Set[str]],
    ]:
        """
        Build search indices for fast lookup operations.

        Args:
            terms: Dictionary of all terms (keywords and acronyms)
            document_chunks: Dictionary mapping document IDs to chunks

        Returns:
            Tuple of (term_to_documents_index, document_to_terms_index,
                     chunks, document_to_chunk_ids_index, document_page_to_chunks_index)
        """
        from collections import defaultdict

        term_to_documents_index: Dict[str, Set[str]] = defaultdict(set)
        document_to_terms_index: Dict[str, Set[str]] = defaultdict(set)

        # Build flattened chunks structure with metadata
        chunks_dict: Dict[str, KnowledgeChunkWithTerms] = {}
        document_to_chunk_ids: Dict[str, Set[str]] = defaultdict(set)
        document_page_to_chunks: Dict[Tuple[str, int], Set[str]] = defaultdict(set)

        for doc_id, chunk_list in document_chunks.items():
            for chunk in chunk_list:
                # Find terms that appear in this chunk
                chunk_terms = []
                for term_name, term_data in terms.items():
                    # Check if this term appears in this specific chunk
                    for occurrence in term_data.occurrences:
                        if (
                            occurrence.document_id == doc_id
                            and occurrence.page == chunk.page
                            and occurrence.chunk_index == chunk.index
                        ):
                            chunk_terms.append(term_name)
                            break

                # Create KnowledgeChunkWithTerms with all fields from base chunk
                chunk_with_meta = KnowledgeChunkWithTerms(
                    document_id=chunk.document_id,
                    document_name=chunk.document_name,
                    doc_id=chunk.doc_id,
                    index=chunk.index,
                    text=chunk.text,
                    page=chunk.page,
                    terms=chunk_terms,
                )

                # Add to flattened chunks dict
                chunks_dict[chunk.doc_id] = chunk_with_meta

                # Add to lookup indices
                document_to_chunk_ids[chunk.document_id].add(chunk.doc_id)
                document_page_to_chunks[(chunk.document_id, chunk.page)].add(chunk.doc_id)

        # Build indices from term occurrences
        for term_name, term_data in terms.items():
            # Track which documents contain this term
            doc_ids = {occ.document_id for occ in term_data.occurrences}
            term_to_documents_index[term_name] = doc_ids

            # Add this term to each document's term list
            for doc_id in doc_ids:
                document_to_terms_index[doc_id].add(term_name)

        # Convert defaultdicts to regular dicts for serialization
        return (
            dict(term_to_documents_index),
            dict(document_to_terms_index),
            chunks_dict,
            dict(document_to_chunk_ids),
            dict(document_page_to_chunks),
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

            for term2_key, term2_data in terms_list[i + 1 :]:  # Avoid duplicate comparisons
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
            pickle.dump(model, f)
        file_size_mb = pickle_path.stat().st_size / (1024 * 1024)
        logger.info(f"Saved to {pickle_path} ({file_size_mb:.2f} MB)")
