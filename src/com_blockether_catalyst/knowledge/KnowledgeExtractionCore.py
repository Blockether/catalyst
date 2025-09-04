"""
Simplified Document Terms Processing System
Self-contained implementation with standard logging
"""

import json
import logging
import math
import os
import pickle
import re
import shutil
import time
from collections import Counter, defaultdict
from datetime import datetime
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Set, Tuple, TypeVar, Union, cast

import anyio
import tiktoken
from rapidfuzz import fuzz

from com_blockether_catalyst.consensus.internal.ConsensusTypes import (
    ConsensusResult,
)
from com_blockether_catalyst.knowledge.internal import PDFKnowledgeExtractor

from ..consensus.internal.Consensus import Consensus
from ..utils.BatchProcessor import BatchProcessor
from ..utils.TypedCalls import ArityOneTypedCall
from .internal import (
    KnowledgeExtractionItem,
    KnowledgeExtractionOutput,
    KnowledgeProcessorSettings,
)

# Import from base types
from .internal.KnowledgeExtractionBaseTypes import (
    AcronymMeaningExtractionResponse,
    ChunkingDecision,
    KeywordMeaningExtractionResponse,
)
from .internal.KnowledgeExtractionCallBase import (
    BaseAcronymExtractionCall,
    BaseChunkingCall,
    BaseKeywordExtractionCall,
)

# Import from main types
from .internal.KnowledgeExtractionTypes import (
    AcronymCandidate,
    DocumentMetadata,
    GroupedAcronym,
    GroupedKeyword,
    KeywordCandidate,
    KnowledgeChunk,
    KnowledgeChunkWithTerms,
    KnowledgeExtractionResult,
    KnowledgeExtractionResultWithChunks,
    KnowledgeMetadata,
    KnowledgePageData,
    LinkedKnowledge,
    Term,
    TermCooccurrence,
    TermLink,
    TermOccurrence,
)
from .KnowledgeSearchCore import KnowledgeSearchCore

logger = logging.getLogger(__name__)

T = TypeVar("T")


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

    def __init__(self, settings: KnowledgeProcessorSettings):
        self._settings = settings
        self._output_dir = settings.extraction_output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"KnowledgeExtractionCore initialized with settings: {self._settings.model_dump()}")
        logger.info(f"KnowledgeExtractionCore initialized with output_dir: {self._output_dir}")

        # Validate that ALL typed calls are provided - they are MANDATORY
        if not self._settings.acronym_extraction_call:
            raise ValueError("acronym_extraction_call is mandatory in settings")
        if not self._settings.keyword_extraction_call:
            raise ValueError("keyword_extraction_call is mandatory in settings")
        if not self._settings.chunking_call:
            raise ValueError("chunking_call is mandatory in settings")
        if not self._settings.chunk_acronym_extraction_call:
            raise ValueError("chunk_acronym_extraction_call is mandatory in settings - needed to find acronyms")
        if not self._settings.chunk_keyword_extraction_call:
            raise ValueError("chunk_keyword_extraction_call is mandatory in settings - needed to find keywords")

        # Define extractors for each supported extension
        self.extractors = {".pdf": PDFKnowledgeExtractor(self._settings)}

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
        """
        Resolve glob patterns to actual file paths.

        Args:
            globs: Sequence of glob patterns or file paths

        Returns:
            Sequence of resolved file paths
        """
        all_files = []
        for glob_pattern in globs:
            path = Path(glob_pattern)
            if path.is_file():
                # Direct file path
                all_files.append(path)
            else:
                # Glob pattern
                if "*" in glob_pattern or "?" in glob_pattern or "[" in glob_pattern:
                    # It's a glob pattern
                    parent = Path(glob_pattern).parent if "/" in glob_pattern else Path(".")
                    pattern = Path(glob_pattern).name
                    matches = list(parent.glob(pattern))
                    all_files.extend(matches)
                else:
                    # It might be a directory or non-existent path
                    path_obj = Path(glob_pattern)
                    if path_obj.is_dir():
                        # Get all files in directory
                        all_files.extend(path_obj.iterdir())
        return all_files

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
                files_by_extension[extension].append(file_path)
        return files_by_extension

    @timed_operation("Step 1/11: Raw file extraction")
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
        grouped_terms: Dict[str, Any],
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
                        positions = self._find_term_positions(term, chunk.text)
                        if positions:
                            chunk_terms_index[(occurrence.document_id, chunk.index)][term] = positions

        return chunk_terms_index

    def _find_term_positions(self, term: str, text: str) -> Sequence[int]:
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
        self._persist(
            "1_raw_extraction",
            {
                "timestamp": datetime.now().isoformat(),
                "raw_extraction": raw_extraction.model_dump(),
            },
        )

        # Step 2: Chunk the documents and persist the results
        results_with_chunks: Sequence[KnowledgeExtractionResultWithChunks] = await self._chunk_extraction(
            raw_extraction
        )
        total_chunks = self._count_total_chunks(results_with_chunks)
        logger.info(f"Created {total_chunks} chunks from {len(results_with_chunks)} documents")
        self._persist(
            "2_chunked_extraction",
            {
                "timestamp": datetime.now().isoformat(),
                "chunked_documents": [r.model_dump() for r in results_with_chunks],
                "total_chunks": total_chunks,
            },
        )

        # Build chunks index for efficient lookup across all steps
        document_to_chunks_index: Dict[str, Sequence[KnowledgeChunk]] = self._build_document_chunk_index(
            results_with_chunks
        )
        logger.info(f"Built chunks index for {len(document_to_chunks_index)} documents")

        # Step 3: Extract acronym candidates
        acronyms: Sequence[AcronymCandidate] = await self._extract_all_acronyms_candidates(results_with_chunks)
        logger.info(f"Found {len(acronyms)} total acronym candidates")
        self._persist(
            "3_acronym_candidates",
            {
                "timestamp": datetime.now().isoformat(),
                "total_acronym_candidates": len(acronyms),
                "acronyms": [a.model_dump() for a in acronyms],
            },
        )

        # Step 4: Group acronyms across documents/pages
        grouped_acronyms: Dict[str, GroupedAcronym] = self._group_acronym_candidates(acronyms)
        logger.info(f"{len(grouped_acronyms)} unique acronyms identified")

        self._persist(
            "4_acronyms_groupped",
            {
                "timestamp": datetime.now().isoformat(),
                "total_unique_acronyms": len(grouped_acronyms),
                "acronyms": {k: v.model_dump() for k, v in grouped_acronyms.items()},
            },
        )

        # Step 5: Extract keyword candidates
        keywords: Sequence[KeywordCandidate] = await self._extract_keywords_candidates_from_documents(
            results_with_chunks
        )
        logger.info(f"Found {len(keywords)} keyword candidates")
        self._persist(
            "5_keyword_candidates",
            {
                "timestamp": datetime.now().isoformat(),
                "total_keyword_candidates": len(keywords),
                "keywords_candidates": [k.model_dump() for k in keywords],
            },
        )

        # Step 6: Group keyword candidates
        grouped_keywords: Dict[str, GroupedKeyword] = self._group_keyword_candidates(keywords)
        logger.info(f"{len(grouped_keywords)} unique keywords identified")

        self._persist(
            "6_keywords_grouped",
            {
                "timestamp": datetime.now().isoformat(),
                "total_unique_keywords": len(grouped_keywords),
                "keywords": {key: value.model_dump() for key, value in grouped_keywords.items()},
            },
        )

        # Step 7: Extract co-occurrences for keywords and acronyms separately
        # Extract co-occurrences for keywords
        keywords_with_cooccurrences: Dict[str, GroupedKeyword] = self._extract_cooccurrences(
            grouped_keywords, document_to_chunks_index
        )
        logger.info(f"Added co-occurrences to {len(keywords_with_cooccurrences)} keywords")

        # Extract co-occurrences for acronyms
        acronyms_with_cooccurrences: Dict[str, GroupedAcronym] = self._extract_cooccurrences(
            grouped_acronyms, document_to_chunks_index
        )
        logger.info(f"Added co-occurrences to {len(acronyms_with_cooccurrences)} acronyms")

        # Persist co-occurrence results
        self._persist(
            "7_keywords_with_cooccurrences",
            {
                "timestamp": datetime.now().isoformat(),
                "total_keywords": len(keywords_with_cooccurrences),
                "keywords": {key: value.model_dump() for key, value in keywords_with_cooccurrences.items()},
            },
        )
        self._persist(
            "7_acronyms_with_cooccurrences",
            {
                "timestamp": datetime.now().isoformat(),
                "total_acronyms": len(acronyms_with_cooccurrences),
                "acronyms": {key: value.model_dump() for key, value in acronyms_with_cooccurrences.items()},
            },
        )

        # Now validate and extract full forms for acronyms
        (
            consolidated_acronyms,
            keywords_from_acronyms_proposals,
            rejected_acronym_metadata,
        ) = await self._consolidate_and_validate_acronyms_meanings(
            acronyms_with_cooccurrences, document_to_chunks_index
        )
        self._persist(
            "8_rejected_acronyms_as_keywords",
            {
                "timestamp": datetime.now().isoformat(),
                "rejected_acronyms": {
                    key: [kw.model_dump() for kw in value] if isinstance(value, Sequence) else value.model_dump()
                    for key, value in keywords_from_acronyms_proposals.items()
                },
                "rejected_acronym_metadata": rejected_acronym_metadata,  # Include all validation metadata
            },
        )

        valid_acronym_count = len(consolidated_acronyms)
        rejected_acronym_count = len(acronyms_with_cooccurrences) - valid_acronym_count
        logger.info(f"{valid_acronym_count} validated acronyms, {rejected_acronym_count} rejected")
        # Persist validated acronyms
        self._persist(
            "8_acronyms_consolidated",
            {
                "timestamp": datetime.now().isoformat(),
                "total_validated": valid_acronym_count,
                "total_rejected": rejected_acronym_count,
                "acronyms": {key: value.model_dump() for key, value in consolidated_acronyms.items()},
            },
        )

        # Merge rejected acronyms (now as keywords) with existing keywords
        merged_keywords = self._merge_rejected_acronyms_with_keywords(
            keywords_with_cooccurrences, keywords_from_acronyms_proposals
        )

        # Step 8: Validate keywords and extract meanings (WITH co-occurrence context)
        consolidated_keywords = await self._consolidate_and_extract_keyword_meanings(
            merged_keywords, document_to_chunks_index
        )
        valid_keyword_count = len(consolidated_keywords)
        rejected_keyword_count = len(merged_keywords) - valid_keyword_count
        logger.info(f"{valid_keyword_count} validated keywords, {rejected_keyword_count} rejected")

        # Persist validated keywords
        self._persist(
            "9_keywords_consolidated",
            {
                "timestamp": datetime.now().isoformat(),
                "total_validated": valid_keyword_count,
                "total_rejected": rejected_keyword_count,
                "keywords": {key: value.model_dump() for key, value in consolidated_keywords.items()},
            },
        )

        # Step 9: Validate acronyms and extract full forms (WITH co-occurrence context)
        logger.info("Step 9/11: Starting acronym validation and full form extraction with co-occurrence context")

        # Step 10: Link acronyms with keywords based on similarity
        links = self._link_acronyms_with_keywords(consolidated_keywords, consolidated_acronyms)
        logger.info(f"Found {len(links)} acronym-keyword links")

        # Persist link between acronyms and keywords
        self._persist(
            "10_term_links",
            {
                "timestamp": datetime.now().isoformat(),
                "links": [link.model_dump() for link in links],
            },
        )

        # Build final LinkedKnowledge object
        linked_knowledge = self._build_linked_knowledge(
            results_with_chunks=results_with_chunks,
            consolidated_keywords=consolidated_keywords,
            consolidated_acronyms=consolidated_acronyms,
            links=links,
            document_to_chunks_index=document_to_chunks_index,
        )

        self._persist(
            "11_linked_knowledge",
            {
                "timestamp": datetime.now().isoformat(),
                "linked_knowledge": linked_knowledge.model_dump(),
            },
        )

        # Serialize to pickle file
        self._save_linked_knowledge_pickle(linked_knowledge)

        # Copy source documents to output directory
        self._copy_source_documents(all_files)

        logger.info(
            f"Performance summary: {len(linked_knowledge.documents)} documents, {len(linked_knowledge.terms)} terms, {len(linked_knowledge.links)} links"
        )

        # Step 12: Create and persist KnowledgeSearchCore
        logger.info("Step 12/13: Creating KnowledgeSearchCore with vector indices")
        search_core = KnowledgeSearchCore(
            linked_knowledge=linked_knowledge,
            pickle_path=self._output_dir / "knowledge_search.pkl",
        )

        # Step 13: Persist the search core
        logger.info("Step 13/13: Persisting KnowledgeSearchCore to pickle")
        search_core.persist()

        pickle_size_mb = (self._output_dir / "knowledge_search.pkl").stat().st_size / (1024 * 1024)
        logger.info(f"KnowledgeSearchCore saved ({pickle_size_mb:.2f} MB)")

        return linked_knowledge

    @timed_operation("Step 15/15: LinkedKnowledge pickle serialization")
    def _save_linked_knowledge_pickle(self, linked_knowledge: LinkedKnowledge) -> None:
        """Save LinkedKnowledge to pickle file.

        Args:
            linked_knowledge: The LinkedKnowledge object to serialize
        """
        pickle_path = self._output_dir / "linked_knowledge.pkl"
        with open(pickle_path, "wb") as f:
            pickle.dump(linked_knowledge, f)
        file_size_mb = pickle_path.stat().st_size / (1024 * 1024)
        logger.info(f"Saved to {pickle_path} ({file_size_mb:.2f} MB)")

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
                # Overwrite if collision occurs
                dest_filename = f"{file_path.stem}_{hash(str(file_path))}{file_path.suffix}"
                dest_path = docs_dir / dest_filename

                if dest_path.exists():
                    logger.info(f"Overwriting existing file: {dest_path}")

                shutil.copy2(file_path, dest_path)
                copied_count += 1
                logger.debug(f"Copied: {file_path.name} -> {dest_path}")

        # Log summary
        logger.info(f"Document copying completed: {copied_count} files copied to {docs_dir}")

    def _build_linked_knowledge(
        self,
        results_with_chunks: Sequence[KnowledgeExtractionResultWithChunks],
        consolidated_keywords: Dict[str, Term],
        consolidated_acronyms: Dict[str, Term],
        links: Sequence[TermLink],
        document_to_chunks_index: Dict[str, Sequence[KnowledgeChunk]],
    ) -> LinkedKnowledge:
        """Build the final LinkedKnowledge object with minimized documents.

        Args:
            results_with_chunks: Sequence of extraction results with chunks
            consolidated_keywords: Dictionary of validated keywords
            consolidated_acronyms: Dictionary of validated acronyms
            links: Sequence of term links between acronyms and keywords
            document_to_chunks_index: Index mapping document IDs to chunks

        Returns:
            LinkedKnowledge object containing all extracted knowledge
        """
        # Build normalized indices: documents metadata and pages separately
        documents: Dict[str, DocumentMetadata] = {}
        pages_index: Dict[Tuple[str, int], KnowledgePageData] = {}

        # Combine all terms (keywords and acronyms) with co-occurrences
        final_terms: Dict[str, Term] = {}
        final_terms.update(consolidated_keywords)
        final_terms.update(consolidated_acronyms)

        # Build search indices
        (
            term_to_documents_index,
            document_to_terms_index,
            chunks,
            document_to_chunk_ids_index,
            document_page_to_chunks_index,
        ) = self._build_search_indices(final_terms, document_to_chunks_index)

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
                    lines=page.lines,
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
        term_to_chunks_index, term_to_document_with_page_index = self._build_inverted_indices(final_terms)

        # Update document metadata with actual term counts
        for doc_id in documents:
            # Get all terms for this document
            doc_terms = document_to_terms_index.get(doc_id, set())

            # Count acronyms and keywords for this document
            acronyms_count = 0
            keywords_count = 0

            for term_key in doc_terms:
                if term_key in final_terms:
                    term = final_terms[term_key]
                    if term.term_type == "acronym":
                        acronyms_count += 1
                    elif term.term_type == "keyword":
                        keywords_count += 1

            # Update the document metadata
            documents[doc_id].total_terms = len(doc_terms)
            documents[doc_id].total_acronyms = acronyms_count
            documents[doc_id].total_keywords = keywords_count

        # Calculate total statistics across all documents
        total_acronyms_count = sum(1 for term in final_terms.values() if term.term_type == "acronym")
        total_keywords_count = sum(1 for term in final_terms.values() if term.term_type == "keyword")
        total_chunks_count = sum(len(chunk_ids) for chunk_ids in document_to_chunk_ids_index.values())

        # Create and return LinkedKnowledge object with all indices
        return LinkedKnowledge(
            documents=documents,
            pages=pages_index,
            terms=final_terms,
            links=links,
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

    async def _extract_keywords_from_document(
        self, document_result: KnowledgeExtractionResultWithChunks
    ) -> Sequence[KeywordCandidate]:
        """
        Extract keyword candidates from a single document using LLM.

        Args:
            document_result: The parent document result for metadata

        Returns:
            Sequence of keyword candidates from this document
        """
        # First, count total words in document for normalization
        total_words = 0
        for chunk in document_result.chunks:
            total_words += len(chunk.text.split())

        # Track term occurrences for scoring
        term_occurrences: dict[str, int] = {}
        keywords_candidates: list[KeywordCandidate] = []

        # Create BatchProcessor for concurrent chunk processing
        processor = BatchProcessor[KnowledgeChunk, Tuple[KeywordCandidate, int]](
            batch_size=8,  # Process 8 chunks concurrently
            max_retries=2,  # Retry failed chunks up to 2 times
            retry_min_wait=1,  # Min 1 second between retries
            retry_max_wait=5,  # Max 5 seconds between retries
        )

        # Create processor function for chunk extraction
        async def extract_from_chunk(
            chunk: KnowledgeChunk,
        ) -> list[Tuple[KeywordCandidate, int]]:
            # Call the LLM to extract keywords from this chunk
            result = await self._settings.chunk_keyword_extraction_call.execute(
                chunk_text=chunk.text,
                document_name=document_result.filename,
                page_number=chunk.page,
                chunk_index=chunk.index,
            )

            # Convert extracted keywords to KeywordCandidate objects with term counts
            chunk_results = []
            for keyword_data in result.final_response.keywords:
                normalized_term = self.normalize_term(keyword_data.term)

                # Count occurrences of this term in the chunk
                term_count = chunk.text.lower().count(normalized_term.lower())

                candidate = KeywordCandidate(
                    term=normalized_term,
                    document_id=document_result.id,
                    document_name=document_result.filename,
                    page=chunk.page,
                    chunk=chunk.index,
                    score=0.0,  # Will calculate after processing all chunks
                )
                chunk_results.append((candidate, term_count))

                logger.debug(f"Found keyword '{keyword_data.term}' in chunk {chunk.index} " f"(page {chunk.page})")

            return chunk_results

        # Process all chunks concurrently
        chunk_results = await processor.process_batch(
            items=document_result.chunks,
            processor_func=extract_from_chunk,
            flatten_results=True,
        )

        # Process results and build term occurrences
        for candidate, term_count in chunk_results:
            term_occurrences[candidate.term] = term_occurrences.get(candidate.term, 0) + term_count
            keywords_candidates.append(candidate)

        # Calculate TF scores for all candidates
        for candidate in keywords_candidates:
            if total_words > 0:
                # Term frequency normalized by total words in document
                tf_score = term_occurrences.get(candidate.term, 1) / total_words
                # Apply log scaling to avoid very small numbers
                candidate.score = min(1.0, tf_score * 1000)  # Scale up and cap at 1.0
            else:
                candidate.score = 0.0

        logger.info(f"Extracted {len(keywords_candidates)} keyword candidates from {document_result.filename}")
        return keywords_candidates

    @async_timed_operation("Step 5/11: Keyword extraction")
    async def _extract_keywords_candidates_from_documents(
        self, results_with_chunks: Sequence[KnowledgeExtractionResultWithChunks]
    ) -> Sequence[KeywordCandidate]:
        """
        Extract keyword candidates from the chunked knowledge extraction output using BatchProcessor.

        Args:
            results_with_chunks: The chunked extraction output to process.

        Returns:
            Sequence of keyword candidates.
        """
        if not results_with_chunks:
            return []

        # Create BatchProcessor for concurrent document processing
        processor = BatchProcessor[KnowledgeExtractionResultWithChunks, KeywordCandidate](
            batch_size=5,  # Process 5 documents concurrently
            max_retries=2,  # Retry failed documents up to 2 times
            retry_min_wait=1,  # Min 1 second between retries
            retry_max_wait=5,  # Max 5 seconds between retries
        )

        # Create processor function for document extraction
        async def extract_from_document(
            document_result: KnowledgeExtractionResultWithChunks,
        ) -> list[KeywordCandidate]:
            logger.info(f"Extracting keywords from document: {document_result.filename}")
            doc_keywords = await self._extract_keywords_from_document(document_result)
            return list(doc_keywords)

        # Process all documents concurrently
        keywords = await processor.process_batch(
            items=results_with_chunks,
            processor_func=extract_from_document,
            flatten_results=True,
        )

        logger.info(f"Extracted {len(keywords)} keyword candidates from {len(results_with_chunks)} documents")
        return keywords

    @timed_operation("Step 4/11: Acronym grouping")
    def _group_acronym_candidates(self, acronym_candidates: List[AcronymCandidate]) -> Dict[str, GroupedAcronym]:
        """
        Group acronym candidates across all documents and pages.

        Args:
            acronym_candidates: List of all acronym candidates from extraction

        Returns:
            Dictionary mapping acronym to GroupedAcronym object
        """
        groupped: Dict[str, GroupedAcronym] = {}

        for candidate in acronym_candidates:
            term = candidate.term

            if term not in groupped:
                groupped[term] = GroupedAcronym(term=term)

            occurrence = TermOccurrence(
                document_id=candidate.document_id,
                document_name=candidate.document_name,
                page=candidate.page,
                chunk_index=candidate.chunk,
            )

            groupped[term].occurrences.append(occurrence)
            groupped[term].total_count += 1

        return groupped

    @async_timed_operation("Step 9/11: Acronym validation")
    async def _consolidate_and_validate_acronyms_meanings(
        self,
        consolidated_acronyms: Dict[str, GroupedAcronym],
        chunks_index: Dict[str, List[KnowledgeChunk]],
    ) -> Tuple[Dict[str, Term], Dict[str, List[GroupedKeyword]], Dict[str, Dict[str, Any]]]:
        """
        Validate consolidated acronyms and extract their meanings using BatchProcessor.

        Args:
            extraction_output_with_chunks: The chunked extraction output for context
            consolidated_acronyms: Dictionary of grouped acronym candidates
            chunks_index: Pre-built index mapping document IDs to chunks

        Returns:
            Tuple of:
            - Dictionary of validated acronyms with meanings
            - Dictionary of rejected acronyms converted to keyword proposals
            - Dictionary of rejected acronym metadata
        """

        # Create BatchProcessor for concurrent acronym validation
        processor = BatchProcessor[Tuple[str, GroupedAcronym], Tuple[str, GroupedAcronym, Any]](
            batch_size=5,  # Process 5 acronyms concurrently
            max_retries=3,  # Retry failed acronyms up to 3 times
            retry_min_wait=1,  # Min 1 second between retries
            retry_max_wait=10,  # Max 10 seconds between retries
        )

        # Prepare acronym items for batch processing
        acronym_items = [(acronym, consolidated) for acronym, consolidated in consolidated_acronyms.items()]

        # Create processor function for validation
        async def validate_acronym(
            item: Tuple[str, GroupedAcronym],
        ) -> list[Tuple[str, GroupedAcronym, Any]]:
            acronym, consolidated = item
            # Gather contexts for this acronym from chunks
            acronym = KnowledgeExtractionCore.normalize_term(acronym)

            contexts = self._find_all_contexts_for_term(consolidated, chunks_index)

            # Gather contexts for each co-occurring term as a dictionary
            cooccurrences_with_contexts = []
            for cooccurrence in consolidated.cooccurrences:
                cooc_contexts = self._find_all_contexts_for_term(
                    consolidated_acronyms.get(cooccurrence.term, GroupedAcronym(term=cooccurrence.term)),
                    chunks_index,
                )
                cooccurrences_with_contexts.append((cooccurrence, cooc_contexts))

            # Validate the acronym using the validator
            validation_result = await self._extract_single_acronym(acronym, contexts, cooccurrences_with_contexts)

            # Return tuple with validation results
            return [(acronym, consolidated, validation_result)]

        # Process all acronyms concurrently
        validation_results = await processor.process_batch(
            items=acronym_items,
            processor_func=validate_acronym,
            flatten_results=True,
        )

        # Process validation results
        validated_acronyms: Dict[str, Term] = {}
        keywords_proposals: Dict[str, List[GroupedKeyword]] = {}
        rejected_acronym_metadata: Dict[str, Dict[str, Any]] = {}

        for acronym, consolidated, validation_result in validation_results:
            if not validation_result.is_valid or not validation_result.full_form:
                if not validation_result.is_valid:
                    logger.warning(f"📝 Acronym '{acronym}' REJECTED as acronym: {validation_result.reasoning}")

                if not validation_result.full_form:
                    logger.warning(
                        f"📝 Acronym '{acronym}' REJECTED as acronym. Received VALID acronym without the full form."
                    )

                logger.info(
                    f"➡️  Transferring '{acronym}' to keywords for re-evaluation "
                    f"(occurrences: {consolidated.total_count})"
                )

                # Store the rejected acronym metadata including validation results
                rejected_acronym_metadata[acronym] = {
                    "term": acronym,
                    "is_valid": validation_result.is_valid,
                    "full_form": validation_result.full_form,
                    "meaning": validation_result.meaning,
                    "reasoning": validation_result.reasoning,
                    "total_count": consolidated.total_count,
                    "mean_score": consolidated.mean_score,
                }

                # Initialize the list if not present
                if acronym not in keywords_proposals:
                    keywords_proposals[acronym] = []

                keywords_proposals[acronym].append(
                    GroupedKeyword(
                        term=acronym,
                        occurrences=consolidated.occurrences,
                        cooccurrences=consolidated.cooccurrences,
                        total_count=consolidated.total_count,
                        mean_score=consolidated.mean_score,
                    )
                )
                continue

            full_form = KnowledgeExtractionCore.normalize_term(validation_result.full_form)

            validated_acronym = Term(
                term=acronym,
                term_type="acronym",
                full_form=full_form,
                occurrences=consolidated.occurrences,
                total_count=consolidated.total_count,
                cooccurrences=consolidated.cooccurrences,
                mean_score=consolidated.mean_score,
                meaning=validation_result.meaning,
                reasoning=validation_result.reasoning,
            )

            validated_acronyms[acronym] = validated_acronym

        # Log summary of validation results
        if keywords_proposals:
            total_transferred = sum(len(keywords) for keywords in keywords_proposals.values())
            logger.info(
                f"📊 Acronym validation summary: "
                f"{len(validated_acronyms)} validated, "
                f"{len(keywords_proposals)} rejected → "
                f"{total_transferred} transferred to keywords"
            )

        return validated_acronyms, keywords_proposals, rejected_acronym_metadata

    async def _extract_single_acronym(
        self,
        acronym: str,
        contexts: List[str],
        cooccurrences_with_contexts: List[Tuple[TermCooccurrence, List[str]]],
    ) -> AcronymMeaningExtractionResponse:
        """
        Extract and validate a single acronym using the configured validator.

        Args:
            acronym: The acronym to validate/extract
            contexts: Sequence of contexts where the acronym appears
            cooccurrences_with_contexts: Sequence of tuples (TermCooccurrence, Sequence of contexts for that term)

        Returns:
            AcronymMeaningExtractionResponse with validation results
        """
        # Acronym extraction is mandatory - no fallback

        # Execute the user-implemented extraction call
        result: ConsensusResult[AcronymMeaningExtractionResponse] = (
            await self._settings.acronym_extraction_call.execute(
                acronym=acronym,
                contexts=contexts,
                cooccurrences_with_contexts=cooccurrences_with_contexts,
                max_display_occurrences=self._settings.max_display_occurrences,
                max_display_cooccurrences=self._settings.max_display_cooccurrences,
            )
        )

        return result.final_response

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for a text string using tiktoken for accuracy.

        Args:
            text: The text to estimate tokens for

        Returns:
            Estimated token count
        """
        if not text:
            return 0

        encoding = tiktoken.get_encoding(self._settings.encoding_model)

        return len(encoding.encode(text))

    def _truncate_to_token_limit(self, text: str, max_tokens: int) -> str:
        """
        Truncate text to fit within token limit.

        Args:
            text: The text to truncate
            max_tokens: Maximum number of tokens

        Returns:
            Truncated text
        """
        estimated_chars = max_tokens * 4
        if len(text) <= estimated_chars:
            return text
        # Truncate with ellipsis indicator
        return text[: estimated_chars - 20] + "\n\n[... truncated to fit token limit ...]"

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
        # Create BatchProcessor with retry logic
        processor = BatchProcessor[KnowledgePageData, KnowledgeChunk](
            batch_size=5,  # Process 5 pages concurrently
            max_retries=3,  # Retry failed pages up to 3 times
            retry_min_wait=1000,  # Min 1 second between retries
            retry_max_wait=10000,  # Max 10 seconds between retries
        )

        # Create a wrapper function that captures the context
        async def process_page(page: KnowledgePageData) -> list[KnowledgeChunk]:
            """Process a single page and return its chunks."""
            chunks = await self._process_page_wrapper(page, document_name, document_id, metadata)
            return list(chunks)

        # Process all pages using BatchProcessor with automatic flattening
        all_chunks = await processor.process_batch(
            items=pages,
            processor_func=process_page,
            flatten_results=True,
        )

        # Re-index all chunks with global indices after collection
        # This ensures proper sequential indexing regardless of processing order
        for idx, chunk in enumerate(all_chunks):
            chunk.index = idx

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

        result = await self._settings.chunking_call.execute(page=page, document_name=document_name, metadata=metadata)

        # Convert ChunkingDecision to KnowledgeChunks
        chunks = []
        for i, chunk_output in enumerate(result.final_response.chunks):
            chunk = KnowledgeChunk(
                document_id=document_id,
                document_name=document_name,
                doc_id=f"{document_id}_p{page.page}_c{i}",
                index=0,
                text=chunk_output.text.strip(),
                page=page.page,
            )
            chunks.append(chunk)

        return chunks

    async def _chunk_extraction(
        self,
        raw_extraction: KnowledgeExtractionOutput,
    ) -> Sequence[KnowledgeExtractionResultWithChunks]:
        """
        Efficient chunking using the DocumentChunkingStrategy that processes 2 pages at once with AI.
        """
        chunked_results: list[KnowledgeExtractionResultWithChunks] = []

        # Iterate over all document types in the extraction output
        for attr_name in dir(raw_extraction):
            if attr_name.startswith("_") or callable(getattr(raw_extraction, attr_name)):
                continue

            extraction_items = getattr(raw_extraction, attr_name)
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

    def _extract_toc_summary(self, text: str) -> str:
        """Extract a summary of document structure from headers."""
        import re

        # Find markdown headers
        headers = []
        for match in re.finditer(r"^(#{1,3})\s+(.+)$", text, re.MULTILINE):
            level = len(match.group(1))
            title = match.group(2).strip()
            if len(title) < 100:  # Skip overly long headers
                headers.append((level, title))

        if not headers:
            return ""

        # Build condensed TOC (limit to first 10 major sections)
        toc_parts = []
        for level, title in headers[:10]:
            if level == 1:
                toc_parts.append(title)
            elif level == 2 and len(toc_parts) < 8:
                toc_parts.append(f"  - {title}")

        if toc_parts:
            return "Sections: " + "; ".join(toc_parts[:8])
        return ""

    def _clean_markdown_whitespace(self, text: str) -> str:
        """Clean up whitespace while preserving markdown structure."""
        lines = text.split("\n")
        cleaned_lines = []

        prev_line_empty = False

        for line in lines:
            # Keep table rows as-is
            if "|" in line:
                cleaned_lines.append(line)
                prev_line_empty = False
                continue

            # Keep code blocks as-is
            if line.strip().startswith("```"):
                cleaned_lines.append(line)
                prev_line_empty = False
                continue

            # Handle empty lines
            if not line.strip():
                if not prev_line_empty:
                    cleaned_lines.append("")
                    prev_line_empty = True
                continue

            # Regular line
            cleaned_lines.append(line.rstrip())
            prev_line_empty = False

        # Remove trailing empty lines
        while cleaned_lines and not cleaned_lines[-1].strip():
            cleaned_lines.pop()

        return "\n".join(cleaned_lines)

    @async_timed_operation("Step 3/11: Acronym extraction")
    async def _extract_all_acronyms_candidates(
        self, results_with_chunks: Sequence[KnowledgeExtractionResultWithChunks]
    ) -> Sequence[AcronymCandidate]:
        """
        Extract all acronym candidates from the given text using BatchProcessor.

        Args:
            results_with_chunks: Chunked extraction output

        Returns:
            Sequence of acronym candidates
        """
        if not results_with_chunks:
            return []

        # Create BatchProcessor for concurrent document processing
        processor = BatchProcessor[KnowledgeExtractionResultWithChunks, AcronymCandidate](
            batch_size=5,  # Process 5 documents concurrently
            max_retries=2,  # Retry failed documents up to 2 times
            retry_min_wait=1,  # Min 1 second between retries
            retry_max_wait=5,  # Max 5 seconds between retries
        )

        # Create processor function for document extraction
        async def extract_from_document(
            item: KnowledgeExtractionResultWithChunks,
        ) -> list[AcronymCandidate]:
            acronyms = await self._extract_acronyms_from_chunks(item)
            return list(acronyms)

        # Process all documents concurrently
        acronyms_candidates = await processor.process_batch(
            items=results_with_chunks,
            processor_func=extract_from_document,
            flatten_results=True,
        )

        logger.info(
            f"Extracted {len(acronyms_candidates)} acronym candidates from {len(results_with_chunks)} documents"
        )
        return acronyms_candidates

    async def _extract_acronyms_from_chunks(
        self, chunked_result: KnowledgeExtractionResultWithChunks
    ) -> Sequence[AcronymCandidate]:
        """
        Extract acronym candidates from chunks using LLM with BatchProcessor.

        Args:
            chunked_result: Result with chunks to extract acronyms from

        Returns:
            Sequence of acronym candidates found in chunks
        """
        if not chunked_result.chunks:
            return []

        # Create BatchProcessor for concurrent chunk processing
        processor = BatchProcessor[KnowledgeChunk, AcronymCandidate](
            batch_size=8,  # Process 8 chunks concurrently
            max_retries=2,  # Retry failed chunks up to 2 times
            retry_min_wait=1,  # Min 1 second between retries
            retry_max_wait=5,  # Max 5 seconds between retries
        )

        # Create processor function for chunk extraction
        async def extract_from_chunk(chunk: KnowledgeChunk) -> list[AcronymCandidate]:
            # Call the LLM to extract acronyms from this chunk
            result = await self._settings.chunk_acronym_extraction_call.execute(
                chunk_text=chunk.text,
                document_name=chunked_result.filename,
                page_number=chunk.page,
                chunk_index=chunk.index,
            )

            # Convert extracted acronyms to AcronymCandidate objects
            candidates = []
            for acronym_data in result.final_response.acronyms:
                candidate = AcronymCandidate(
                    term=acronym_data.term.upper(),  # Ensure uppercase for acronyms
                    document_id=chunked_result.id,
                    document_name=chunked_result.filename,
                    page=chunk.page,
                    chunk=chunk.index,
                )
                candidates.append(candidate)

                logger.debug(
                    f"Found acronym '{acronym_data.term}' ({acronym_data.full_form}) in chunk {chunk.index} "
                    f"(page {chunk.page})"
                )

            return candidates

        # Process all chunks concurrently
        acronym_candidates = await processor.process_batch(
            items=chunked_result.chunks,
            processor_func=extract_from_chunk,
            flatten_results=True,
        )

        logger.info(
            f"Extracted {len(acronym_candidates)} acronym candidates from {chunked_result.filename} ({len(chunked_result.chunks)} chunks)"
        )
        return acronym_candidates

    def _merge_rejected_acronyms_with_keywords(
        self,
        existing_keywords: Dict[str, GroupedKeyword],
        rejected_acronyms_dict: Dict[str, Any],
    ) -> Dict[str, GroupedKeyword]:
        """
        Merge rejected acronym candidates (now treated as keywords) with existing keywords.

        When an acronym candidate is rejected during validation (because it's not a valid acronym),
        it should be reconsidered as a potential keyword. This function merges these rejected
        acronym candidates into the existing keywords dictionary, handling any conflicts.

        Args:
            existing_keywords: Keywords already extracted through the normal keyword extraction process
            rejected_acronyms_dict: Dictionary of rejected acronyms (could be various formats)

        Returns:
            Merged dictionary containing both existing keywords and rejected acronyms as keywords
        """
        merged = existing_keywords.copy()

        # Handle the case where rejected_acronyms_dict contains lists of GroupedKeywords
        for keyword_data in rejected_acronyms_dict.values():
            # Handle if it's a list of GroupedKeywords
            if isinstance(keyword_data, list):
                keyword_candidates = keyword_data
            else:
                # Handle if it's a single item
                keyword_candidates = [keyword_data]

            for keyword_candidate in keyword_candidates:
                term = keyword_candidate.term

                if term not in merged:
                    # No conflict - directly add the rejected acronym as a keyword
                    merged[term] = keyword_candidate
                    logger.info(
                        f"✅ Successfully transferred '{term}' from rejected acronyms to keywords "
                        f"(occurrences: {keyword_candidate.total_count})"
                    )
                else:
                    # Conflict detected - merge the occurrences and co-occurrences
                    existing = merged[term]

                    logger.warning(
                        f"⚠️  CONFLICT: Term '{term}' exists both as keyword AND rejected acronym! "
                        f"Merging occurrences (keyword: {existing.total_count}, rejected acronym: {keyword_candidate.total_count})"
                    )

                    # Merge occurrences
                    merged_occurrences = existing.occurrences.copy()
                    for occurrence in keyword_candidate.occurrences:
                        # Check if this occurrence already exists
                        if not any(
                            occ.document_id == occurrence.document_id and occ.chunk_index == occurrence.chunk_index
                            for occ in merged_occurrences
                        ):
                            merged_occurrences.append(occurrence)

                    # Merge co-occurrences
                    merged_cooccurrences = existing.cooccurrences.copy()
                    cooc_dict = {cooc.term: cooc for cooc in merged_cooccurrences}

                    for cooc in keyword_candidate.cooccurrences:
                        if cooc.term in cooc_dict:
                            # Update existing co-occurrence with higher frequency and confidence
                            existing_cooc = cooc_dict[cooc.term]
                            existing_cooc.frequency += cooc.frequency
                            existing_cooc.confidence = max(existing_cooc.confidence, cooc.confidence)
                        else:
                            merged_cooccurrences.append(cooc)

                    # Update total count and mean score
                    new_total_count = len(merged_occurrences)

                    # Recalculate mean score if both have scores
                    if existing.mean_score > 0 and keyword_candidate.mean_score > 0:
                        new_mean_score = (
                            existing.mean_score * existing.total_count
                            + keyword_candidate.mean_score * keyword_candidate.total_count
                        ) / (existing.total_count + keyword_candidate.total_count)
                    else:
                        new_mean_score = existing.mean_score or keyword_candidate.mean_score

                    # Create merged keyword
                    merged[term] = GroupedKeyword(
                        term=term,
                        occurrences=merged_occurrences,
                        cooccurrences=sorted(
                            merged_cooccurrences,
                            key=lambda x: x.confidence,
                            reverse=True,
                        )[
                            :10
                        ],  # Keep top 10 co-occurrences
                        total_count=new_total_count,
                        mean_score=new_mean_score,
                    )

                    logger.info(
                        f"✔️  Merged '{term}': occurrences {existing.total_count} + {keyword_candidate.total_count} = {new_total_count}, "
                        f"co-occurrences: {len(merged[term].cooccurrences)} unique terms"
                    )

        return merged

    def _build_inverted_indices(
        self, terms: Dict[str, Term]
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
        terms: Dict[str, Term],
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

    @timed_operation("Step 6/11: Keyword grouping")
    def _group_keyword_candidates(self, keyword_candidates: Sequence[KeywordCandidate]) -> Dict[str, GroupedKeyword]:
        """
        Group keyword candidates by their term text.

        Args:
            keyword_candidates: Sequence of all keyword candidates from extraction

        Returns:
            Dictionary mapping term to GroupedKeyword object
        """
        grouped: Dict[str, GroupedKeyword] = {}

        for candidate in keyword_candidates:
            term = KnowledgeExtractionCore.normalize_term(candidate.term)

            if term not in grouped:
                grouped[term] = GroupedKeyword(
                    term=term,
                    occurrences=[],
                    cooccurrences=[],
                    total_count=0,
                    mean_score=0.0,
                )

            occurrence = TermOccurrence(
                document_id=candidate.document_id,
                document_name=candidate.document_name,
                page=candidate.page,
                chunk_index=candidate.chunk,
            )

            grouped[term].occurrences.append(occurrence)
            grouped[term].total_count += 1

            # Update mean score
            current_mean = grouped[term].mean_score
            grouped[term].mean_score = (current_mean * (grouped[term].total_count - 1) + candidate.score) / grouped[
                term
            ].total_count

        return grouped

    def _find_all_contexts_for_term(
        self,
        grouped: GroupedAcronym | GroupedKeyword,
        chunks_index: Dict[str, List[KnowledgeChunk]],
    ) -> List[str]:
        """
        Find all contexts for a given term across document chunks.

        Args:
            term: The term to find contexts for
            chunks_index: Pre-built index mapping document IDs to chunks

        Returns:
            Sequence of context strings where the term appears
        """
        contexts = []
        term = grouped.term

        for item in grouped.occurrences:
            if item.document_id in chunks_index:
                for chunk in chunks_index[item.document_id]:
                    if self._find_term_positions(term, chunk.text):
                        contexts.append(chunk.text)
                        break
        return contexts

    @async_timed_operation("Step 8/11: Keyword validation")
    async def _consolidate_and_extract_keyword_meanings(
        self,
        grouped_keywords: Dict[str, GroupedKeyword],
        chunks_index: Dict[str, List[KnowledgeChunk]],
    ) -> Dict[str, Term]:
        """
        Validate grouped keywords and extract their meanings using BatchProcessor.

        Args:
            grouped_keywords: Dictionary of grouped keyword candidates
            chunks_index: Pre-built index mapping document IDs to chunks

        Returns:
            Dictionary of validated keywords with meanings
        """

        # Filter keywords by minimum score first
        valid_keywords = {
            term: grouped
            for term, grouped in grouped_keywords.items()
            if grouped.mean_score >= self._settings.min_term_score
        }

        skipped_count = len(grouped_keywords) - len(valid_keywords)
        if skipped_count > 0:
            logger.info(f"Step 8/11: Skipped {skipped_count} keywords due to low mean score")

        if not valid_keywords:
            logger.info("Step 8/11: No valid keywords to process")
            return {}

        # Create BatchProcessor for concurrent keyword validation
        processor = BatchProcessor[Tuple[str, GroupedKeyword], Tuple[str, GroupedKeyword, Any]](
            batch_size=5,  # Process 5 keywords concurrently
            max_retries=3,  # Retry failed keywords up to 3 times
            retry_min_wait=1,  # Min 1 second between retries
            retry_max_wait=10,  # Max 10 seconds between retries
        )

        # Prepare keyword items for batch processing
        keyword_items = [(term, grouped) for term, grouped in valid_keywords.items()]

        # Create processor function for validation
        async def validate_keyword(
            item: Tuple[str, GroupedKeyword],
        ) -> list[Tuple[str, GroupedKeyword, Any]]:
            term, grouped = item
            contexts = self._find_all_contexts_for_term(grouped, chunks_index)

            # Gather contexts for each co-occurring term as a dictionary (same as acronyms)
            cooccurrences_with_contexts = []
            for cooccurrence in grouped.cooccurrences:
                cooc_contexts = self._find_all_contexts_for_term(
                    grouped_keywords.get(cooccurrence.term, GroupedKeyword(term=cooccurrence.term)),
                    chunks_index,
                )
                cooccurrences_with_contexts.append((cooccurrence, cooc_contexts))

            # Validate the term using the validator
            validation_result = await self._extract_keyword_meaning(term, contexts, cooccurrences_with_contexts)

            # Return tuple with validation results
            return [(term, grouped, validation_result)]

        # Process all keywords concurrently
        validation_results = await processor.process_batch(
            items=keyword_items,
            processor_func=validate_keyword,
            flatten_results=True,
        )

        # Process validation results
        consolidated_keywords: Dict[str, Term] = {}

        for term, grouped, validation_result in validation_results:
            if not validation_result.is_valid:
                logger.info(f"Step 8/11: Skipping term '{term}' due to non valid keyword failure")
                continue

            # Create Term for keyword
            # Use the full_form from validation if provided, otherwise use the term itself
            full_form = validation_result.full_form if validation_result.full_form else grouped.term

            consolidated_keyword = Term(
                term=grouped.term,
                term_type="keyword",
                full_form=full_form,
                occurrences=grouped.occurrences,
                cooccurrences=grouped.cooccurrences,
                total_count=grouped.total_count,
                mean_score=grouped.mean_score,
                meaning=validation_result.meaning,
                reasoning=validation_result.reasoning,
            )

            consolidated_keywords[term] = consolidated_keyword

        return consolidated_keywords

    async def _extract_keyword_meaning(
        self,
        term: str,
        contexts: List[str],
        cooccurrences_with_contexts: List[Tuple[TermCooccurrence, List[str]]],
    ) -> Any:  # Returns KeywordMeaningExtractionResponseComputed but typing is complex with ConsensusResult
        """
        Extract meaning/definition for a keyword using LLM.

        Args:
            term: The keyword to extract meaning for
            contexts: Sequence of contexts where the keyword appears
            cooccurrences_with_contexts: Sequence of tuples (TermCooccurrence, Sequence of contexts for that term)

        Returns:
            TermMeaningExtractionResponse with meaning extraction results
        """
        # Keyword extraction is mandatory - no fallback

        # Execute the user-implemented extraction call
        # This will return KeywordMeaningExtractionResponseComputed due to post-processing
        result = await self._settings.keyword_extraction_call.execute(
            term=term,
            contexts=contexts,
            cooccurrences_with_contexts=cooccurrences_with_contexts,
            max_display_occurrences=self._settings.max_display_occurrences,
            max_display_cooccurrences=self._settings.max_display_cooccurrences,
        )

        return result.final_response

    @timed_operation("Step 7/11: Co-occurrence extraction")
    def _extract_cooccurrences(
        self,
        grouped_terms: Dict[str, Any],
        chunks_index: Dict[str, Sequence[KnowledgeChunk]],
    ) -> Dict[str, Any]:
        """
        Extract and calculate co-occurrences between terms in the same chunks.

        This method:
        1. Builds an index of term positions in each chunk
        2. Calculates weighted cooccurrence scores based on term proximity
        3. Adds the top cooccurrences to each term

        Args:
            grouped_terms: Dictionary of grouped terms (keywords/acronyms)
            chunks_index: Dictionary mapping document IDs to chunks

        Returns:
            Updated terms dictionary with cooccurrence data
        """
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
                    # Calculate confidence based on normalized score
                    confidence = weighted_score / term.total_count if term.total_count > 0 else 0.0
                    # Convert weighted score to frequency
                    frequency = max(1, int(weighted_score + 0.5))

                    cooccurrences.append(
                        TermCooccurrence(
                            term=cooccurring_term,
                            frequency=frequency,
                            confidence=min(confidence, 1.0),
                        )
                    )

                # Sort by confidence and take top 10
                cooccurrences.sort(key=lambda x: x.confidence, reverse=True)
                cooccurrences = cooccurrences[:10]

            # Update term with cooccurrences (empty list if none found)
            updated_terms[term_key] = term.model_copy(update={"cooccurrences": cooccurrences})

        return updated_terms

    def _calculate_acronym_keyword_match(self, full_form: str, keyword: str) -> float:
        """
        Match acronym full forms to keywords using token-based fuzzy matching.

        Args:
            full_form: The full form of the acronym (e.g., "Application Programming Interface")
            keyword: The keyword to match against (e.g., "Programming")

        Returns:
            Float between 0 and 1 representing match strength
        """
        # Handle edge cases
        if not keyword or not keyword.strip():
            return 0.0

        if len(keyword) == 1:
            # Single character matching is unreliable
            return 0.0

        full_lower = full_form.lower()
        keyword_lower = keyword.lower()

        # Direct substring match (highest confidence)
        if keyword_lower in full_lower:
            # Scale by relative length - longer matches are better
            length_ratio = len(keyword_lower) / len(full_lower)
            return 0.7 + (0.3 * length_ratio)

        # Token set ratio - handles partial word matches perfectly
        # This ignores word order and duplicates
        token_set_score = fuzz.token_set_ratio(full_lower, keyword_lower) / 100.0

        # Partial ratio - finds best matching substring
        partial_score = fuzz.partial_ratio(full_lower, keyword_lower) / 100.0

        # Weight token_set higher since it's better for this use case
        return (token_set_score * 0.7) + (partial_score * 0.3)

    @timed_operation("Step 10/11: Acronym-keyword linking")
    def _link_acronyms_with_keywords(
        self,
        consolidated_keywords: Dict[str, Term],
        consolidated_acronyms: Dict[str, Term],
    ) -> Sequence[TermLink]:
        """
        Link acronyms with keywords based on similarity between acronym full forms and keyword text.

        Args:
            consolidated_keywords: Dictionary containing validated keywords
            consolidated_acronyms: Dictionary containing validated acronyms

        Returns:
            Sequence of TermLink objects representing linked acronyms and keywords
        """

        links = []

        # For each acronym, find matching keywords based on similarity
        for _, acronym_data in consolidated_acronyms.items():
            if not acronym_data.full_form:
                logger.warning(f"Acronym '{acronym_data.term}' has no full form, skipping linking")
                continue

            acronym_full_form = acronym_data.full_form

            for _, keyword_data in consolidated_keywords.items():
                # Calculate semantic similarity between acronym full form and keyword
                match_score = self._calculate_acronym_keyword_match(acronym_full_form, keyword_data.term)

                # Create link if similarity is above threshold
                if match_score >= self._settings.linking_threshold:
                    link = TermLink(
                        acronym=acronym_data.term,
                        keyword=keyword_data.term,
                        match_score=match_score,
                    )
                    links.append(link)

                    logger.info(
                        f"Linked acronym '{acronym_data.term}' ({acronym_full_form}) with keyword '{keyword_data.term}' (match_score: {match_score:.2f})"
                    )

        return links

    def _persist(self, subname: str, results: Dict[str, Any]) -> Path:
        """
        Save extraction results to JSON file.

        Args:
            subname: Prefix for the filename
            results: Extraction results (will be converted to dict if needed)

        Returns:
            Path to saved JSON file
        """
        timestamp = getattr(results, "timestamp", datetime.now().strftime("%Y%m%d_%H%M%S"))
        output_file = self._output_dir / f"{subname}_{timestamp}.json"

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False)

        logger.info(f"Saved {subname} results to {output_file}")

        return output_file
