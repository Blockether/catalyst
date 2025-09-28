"""
Knowledge-based search system that integrates vector search with extracted terms and relationships.

This module provides sophisticated search functionality that combines semantic search
with knowledge about terms, their meanings, co-occurrences, and relationships.
"""

import logging
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from urllib.parse import quote

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore
from pydantic import BaseModel, Field

from blockether_catalyst.encoder.PotionEightEncoder import PotionEightEncoder
from blockether_catalyst.knowledge.extraction.internal.KnowledgeVectorizers import (
    KnowledgeVectorizers,
)
from blockether_catalyst.knowledge.KnowledgeTypes import (
    CompactSearchResult,
    ImageInfo,
    KnowledgeSearchResult,
    LinkedKnowledge,
    OptimizedSearchResponse,
    SearchResult,
    SearchResultMetadata,
    TableInfo,
    TermInfo,
    TermWithLinks,
    TopTermsResult,
)

logger = logging.getLogger(__name__)


class EncoderEmbeddings(Embeddings):
    """Custom embeddings class that uses our PotionEightEncoder."""

    def embed_documents(self, texts: List[str]) -> Any:
        """
        Embed a list of documents.

        Args:
            texts: List of texts to embed

        Returns:
            List of embeddings as lists of floats
        """
        if not texts:
            return []
        embeddings = PotionEightEncoder.encode(texts)
        return embeddings.tolist()  # type: ignore[no-any-return]

    def embed_query(self, text: str) -> Any:
        """
        Embed a query text.

        Args:
            text: Query text to embed

        Returns:
            Embedding as list of floats
        """
        embedding = PotionEightEncoder.encode_single(text)
        return embedding.tolist()  # type: ignore[no-any-return]


class KnowledgeSearchCore:
    """
    Knowledge-enhanced search system that combines vector search with extracted knowledge.

    Provides intelligent search that understands:
    - Term meanings and definitions
    - Relationships between acronyms and keywords
    - Co-occurrence patterns
    - Document structure and context
    """

    # Search configuration
    DEFAULT_K_RESULTS: int = 5
    DEFAULT_THRESHOLD: float = 0.1
    DEFAULT_MAX_DEPTH: int = 2
    DEFAULT_MAX_COOCCURRENCES: int = 5

    # Scoring weights
    SIMILARITY_WEIGHT: float = 0.6
    TERM_RELEVANCE_WEIGHT: float = 0.4
    TERM_FREQUENCY_WEIGHT: float = 0.7
    TERM_DIVERSITY_WEIGHT: float = 0.3
    KEYWORD_BOOST_WEIGHT: float = 0.3
    ACRONYM_BOOST_WEIGHT: float = 0.4
    NGRAM_SIZE_BOOST: float = 0.3

    # Quick boost values for term presence
    QUICK_ACRONYM_BOOST: float = 0.2
    QUICK_KEYWORD_BOOST: float = 0.1
    TERM_INDEX_BOOST: float = 0.3
    FINAL_TERM_INDEX_BOOST: float = 0.2

    # Query term extraction limits
    MAX_TOP_KEYWORDS_FROM_QUERY: int = 5
    MAX_TOP_ACRONYMS_FROM_QUERY: int = 5

    # Processing limits
    MAX_KEYWORDS_FOR_MATCHING: int = 3
    MAX_ACRONYMS_FOR_MATCHING: int = 3
    MAX_TERMS_PER_CHUNK: int = 20
    MAX_PRIMARY_TERMS: int = 5
    MAX_RELATED_TERMS: int = 5
    MAX_LINKS_PER_TERM: int = 2
    MAX_TOTAL_RELATED_TERMS: int = 8

    # Search expansion factor
    SEARCH_EXPANSION_FACTOR: float = 1.2
    # TF-IDF extraction configuration
    DEFAULT_TFIDF_MIN_DF: int = 1
    DEFAULT_MAX_KEYWORD_FEATURES: int = 10
    DEFAULT_MAX_ACRONYM_FEATURES: int = 10

    # Logging and debugging configuration
    DEFAULT_DEBUG_TOP_RESULTS_LIMIT: int = 5

    @staticmethod
    def _clean_image_markers_from_text(text: str) -> str:
        """
        Remove image caption markers from text for display purposes.

        Args:
            text: Text potentially containing image caption markers

        Returns:
            Clean text without image caption markers
        """
        from blockether_catalyst.knowledge.KnowledgeProcessingUtils import (
            KnowledgeProcessingUtils,
        )

        return KnowledgeProcessingUtils.clean_image_markers_from_text(text)

    # Term relevance scoring weights
    TERM_FREQUENCY_SCORE_WEIGHT: float = 0.5
    PRIMARY_TERM_COUNT_WEIGHT: float = 0.3

    # Related terms multiplier
    MAX_RELATED_TERMS_MULTIPLIER: int = 2

    @staticmethod
    def create_chunk_id(doc_id: str, page_num: int, chunk_idx: int) -> str:
        """Create a standardized chunk identifier.

        Args:
            doc_id: Document identifier
            page_num: Page number (0-based)
            chunk_idx: Chunk index within the document

        Returns:
            Formatted chunk identifier like 'doc123_p0_c5'
        """
        return f"{doc_id}_p{page_num}_c{chunk_idx}"

    @staticmethod
    def parse_chunk_id(chunk_id: str) -> Optional[Tuple[str, int, int]]:
        """Parse a chunk identifier into its components.

        Args:
            chunk_id: Chunk identifier to parse

        Returns:
            Tuple of (doc_id, page_num, chunk_idx) or None if invalid
        """
        import re

        match = re.match(r"^(.+)_p(\d+)_c(\d+)$", chunk_id)
        if match:
            doc_id, page_num, chunk_idx = match.groups()
            return (doc_id, int(page_num), int(chunk_idx))
        return None

    @staticmethod
    def create_chunk_id_pattern(doc_id: str, page_num: Optional[int] = None) -> str:
        """Create a chunk ID pattern for matching.

        Args:
            doc_id: Document identifier
            page_num: Optional page number. If None, creates pattern for any page

        Returns:
            Pattern string that can be used with format() for page number
        """
        if page_num is not None:
            return f"{doc_id}_p{page_num}_c{{0}}"
        return f"{doc_id}_p{{0}}_c{{1}}"

    _is_initialized_from_state: bool = False

    def __init__(
        self,
        resources_base_url: Optional[str] = None,
        linked_knowledge: Optional[LinkedKnowledge] = None,
        pickle_path: Optional[Union[str, Path]] = None,
        auto_load: bool = True,
    ):
        """
        Initialize the knowledge search core.

        Args:
            linked_knowledge: Complete knowledge structure with documents, terms, and links
            pickle_path: Path to pickle file for persistence/loading
            auto_load: If True and pickle_path exists, automatically load from pickle

        During initialization, all terms (acronyms, keywords, full forms) are indexed
        for semantic search with their associated metadata.
        """

        start_time = time.time()

        self._resources_base_url = resources_base_url
        self.pickle_path = Path(pickle_path) if pickle_path else None

        # Debug logging
        logger.debug(
            f"KnowledgeSearchCore init called with linked_knowledge={type(linked_knowledge)}, is None: {linked_knowledge is None}"
        )

        if linked_knowledge is None and auto_load and self.pickle_path and self.pickle_path.exists():
            logger.info(f"Loading KnowledgeSearchCore from pickle: {self.pickle_path}")
            self._load(self.pickle_path)
            self._is_initialized_from_state = True
            return
        elif linked_knowledge is None:
            # If no pickle file and no linked_knowledge, raise error
            raise ValueError("linked_knowledge is required when not loading from pickle")

        self._linked_knowledge = linked_knowledge
        self._vector_store = InMemoryVectorStore(embedding=EncoderEmbeddings())
        self._is_initialized_from_state = False

        # Populate vector store with all chunks
        self._populate_vector_store()

        init_time = time.time() - start_time
        logger.info(f"KnowledgeSearchCore initialization took {init_time:.3f}s")

    @property
    def linked_knowledge(self) -> LinkedKnowledge:
        """Get the underlying linked knowledge structure."""
        return self._linked_knowledge

    def _resolve_term(self, term_key: str) -> Optional[TermWithLinks]:
        """
        Resolve a term key to its TermWithLinks object.

        Args:
            term_key: The term string to resolve

        Returns:
            The TermWithLinks object if found, else None
        """
        return self._linked_knowledge.terms.get(term_key)

    def _populate_vector_store(self) -> None:
        """
        Populate the vector store with all chunks from LinkedKnowledge.

        Each chunk becomes a Document in the vector store with metadata for retrieval.
        """
        logger.info("Populating vector store with chunks...")

        documents = []
        for chunk_id, chunk in self._linked_knowledge.chunks.items():
            document_metadata = self._linked_knowledge.documents.get(chunk.document_id)

            if not document_metadata:
                logger.warning(f"Chunk {chunk_id} references unknown document ID {chunk.document_id}")
                continue

            # Create SearchResultMetadata and convert to dict for LangChain
            metadata = SearchResultMetadata(
                chunk_id=chunk_id,
                document_id=chunk.document_id,
                document_path=document_metadata.document_path,
                document_name=chunk.document_name,
                page=chunk.page,
                chunk_index=chunk.index,
                terms=chunk.terms,
                author=document_metadata.author if document_metadata else None,
                publication_date=document_metadata.publication_date if document_metadata else None,
                modified_date=document_metadata.modification_date if document_metadata else None,
                document_title=document_metadata.title,
                document_subject=document_metadata.subject if document_metadata else None,
            )

            # LangChain Document expects metadata as a dictionary
            doc = Document(
                page_content=chunk.text,
                metadata=metadata.model_dump(),
            )
            documents.append(doc)

        # Add all documents to the vector store
        if documents:
            self._vector_store.add_documents(documents)
            logger.info(f"Added {len(documents)} chunks to vector store")
        else:
            logger.warning("No chunks found to add to vector store")

    def _similarity_search(
        self,
        query: str,
        k: int = DEFAULT_K_RESULTS,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> List[SearchResult]:
        """
        Perform basic vector similarity search.

        Args:
            query: Search query text
            k: Number of results to return
            threshold: Minimum similarity threshold

        Returns:
            List of search results
        """
        # Handle empty query
        if not query or not query.strip():
            return []

        # Perform similarity search with scores
        results_with_scores = self._vector_store.similarity_search_with_score(query, k=k)

        # Convert to SearchResult objects and filter by threshold
        search_results = []
        for doc, score in results_with_scores:
            if score >= threshold:
                result = SearchResult(
                    text=doc.page_content,
                    score=score,
                    doc_id=doc.metadata.get("chunk_id", ""),
                    metadata=SearchResultMetadata(**doc.metadata),
                )
                search_results.append(result)

        return search_results

    def search(
        self,
        query: str,
        k: int = DEFAULT_K_RESULTS,
        threshold: float = DEFAULT_THRESHOLD,
        max_depth: int = DEFAULT_MAX_DEPTH,
        max_cooccurrences: int = DEFAULT_MAX_COOCCURRENCES,
        max_keywords: Optional[int] = None,
        max_acronyms: Optional[int] = None,
        max_terms_chunk: Optional[int] = None,
        max_primary_terms: Optional[int] = None,
        max_related_terms: Optional[int] = None,
    ) -> OptimizedSearchResponse:
        """
        Perform enhanced search with term analysis.

        Args:
            query: Search query text
            k: Number of results to return
            threshold: Minimum similarity threshold
            max_depth: Maximum depth for exploring related terms
            max_cooccurrences: Maximum number of co-occurring terms to include
            max_keywords: Maximum keywords to extract from query
            max_acronyms: Maximum acronyms to extract from query
            max_terms_chunk: Maximum terms to process per chunk
            max_primary_terms: Maximum primary terms per result
            max_related_terms: Maximum related terms per result

        Returns:
            OptimizedSearchResponse with search results and terms
        """
        # Validate inputs
        if not query or not query.strip():
            return OptimizedSearchResponse(results=[], terms={}, total_results=0)

        if k <= 0:
            raise ValueError("k must be positive")

        if not (0.0 <= threshold <= 1.0):
            raise ValueError("threshold must be between 0 and 1")

        # Use provided values or defaults
        config = {
            "max_keywords": max_keywords or self.MAX_KEYWORDS_FOR_MATCHING,
            "max_acronyms": max_acronyms or self.MAX_ACRONYMS_FOR_MATCHING,
            "max_terms_chunk": max_terms_chunk or self.MAX_TERMS_PER_CHUNK,
            "max_primary_terms": max_primary_terms or self.MAX_PRIMARY_TERMS,
            "max_related_terms": max_related_terms or self.MAX_RELATED_TERMS,
        }

        start_time = time.time()
        logger.info(f"Performing search for query: '{query}'")

        # Extract terms from query
        top_terms = self._extract_query_terms(query)

        # Find relevant chunks
        term_based_chunks = self._find_term_based_chunks(top_terms)

        # Perform vector search
        search_results = self._perform_vector_search(query, k, threshold, top_terms)

        # Score and rank results (single boost application)
        ranked_results = self._score_and_rank_results(search_results, query, top_terms, term_based_chunks, k, config)

        # Build final response with intelligent term classification
        response = self._build_search_response(ranked_results, max_depth, max_cooccurrences, config, query, top_terms)

        search_time = time.time() - start_time
        logger.info(f"Search completed in {search_time:.3f}s, returned {len(response.results)} results")

        return response

    def _extract_query_terms(self, query: str) -> TopTermsResult:
        """Extract keywords and acronyms from query using TF-IDF."""
        vectorizers = KnowledgeVectorizers(
            keywords_min_df=self.DEFAULT_TFIDF_MIN_DF,
            acronyms_min_df=self.DEFAULT_TFIDF_MIN_DF,
        )
        top_terms = self._get_top_keywords_and_acronyms(
            query,
            vectorizers,
            max_keywords=self.MAX_TOP_KEYWORDS_FROM_QUERY,
            max_acronyms=self.MAX_TOP_ACRONYMS_FROM_QUERY,
        )

        if top_terms.keywords:
            logger.info(f"Top keywords: {[kw[0] for kw in top_terms.keywords]}")
        if top_terms.acronyms:
            logger.info(f"Top acronyms: {[ac[0] for ac in top_terms.acronyms]}")

        return top_terms

    def _find_term_based_chunks(self, top_terms: TopTermsResult) -> Set[Tuple[str, int]]:
        """Find chunks containing query terms using term index."""
        term_based_chunks: Set[Tuple[str, int]] = set()

        if not (self._linked_knowledge and self._linked_knowledge.term_to_chunks_index):
            return term_based_chunks

        for term_tuple in top_terms.acronyms + top_terms.keywords:
            term = term_tuple[0]
            normalized_term = self._linked_knowledge._normalize_term(term)

            if normalized_term in self._linked_knowledge.term_to_chunks_index:
                chunk_refs = self._linked_knowledge.term_to_chunks_index[normalized_term]
                term_based_chunks.update(chunk_refs)
                logger.debug(f"Found {len(chunk_refs)} chunks for term: {term}")

        return term_based_chunks

    def _perform_vector_search(
        self, query: str, k: int, threshold: float, top_terms: TopTermsResult
    ) -> List[SearchResult]:
        """Perform vector similarity search with appropriate threshold."""
        # Expand search if we have query terms to boost
        expanded_k = int(k * self.SEARCH_EXPANSION_FACTOR) if (top_terms.keywords or top_terms.acronyms) else k
        return self._similarity_search(query, k=expanded_k, threshold=threshold)

    def _score_and_rank_results(
        self,
        search_results: List[SearchResult],
        query: str,
        top_terms: TopTermsResult,
        term_based_chunks: Set[Tuple[str, int]],
        k: int,
        config: dict,
    ) -> List[SearchResult]:
        """Score and rank results with single boost application."""
        # Pre-compute term sets for efficient matching
        keyword_terms = {kw[0].lower() for kw in top_terms.keywords[: config["max_keywords"]]}
        acronym_terms = {ac[0].lower() for ac in top_terms.acronyms[: config["max_acronyms"]]}

        scored_results = []
        for result in search_results:
            text_lower = result.text.lower()

            # Calculate single comprehensive boost
            boost = 0.0

            # Use set intersection for efficient term matching
            text_words = set(text_lower.split())

            if text_words & acronym_terms:  # Set intersection - O(1) average case
                boost += self.QUICK_ACRONYM_BOOST

            if text_words & keyword_terms:
                boost += self.QUICK_KEYWORD_BOOST

            # Term index boost
            if (
                result.metadata.document_id,
                result.metadata.chunk_index,
            ) in term_based_chunks:
                boost += self.TERM_INDEX_BOOST

            # Apply boost once to get final score
            final_score = result.score + boost
            scored_results.append((final_score, result))

        # Sort and return top k
        scored_results.sort(key=lambda x: x[0], reverse=True)
        return [result for _, result in scored_results[:k]]

    def _build_search_response(
        self,
        ranked_results: List[SearchResult],
        max_depth: int,
        max_cooccurrences: int,
        config: dict,
        query: Optional[str] = None,
        top_terms: Optional[TopTermsResult] = None,
    ) -> OptimizedSearchResponse:
        """Build final search response with intelligent term classification and advanced scoring."""
        terms_dict: Dict[str, TermInfo] = {}
        results_with_scores = []

        # Process all results efficiently
        for result in ranked_results:
            # Extract terms with intelligent classification
            primary_terms, related_terms = self._extract_result_terms(
                result, max_depth, max_cooccurrences, config, query, top_terms
            )

            # Calculate advanced scoring
            term_boost = 0.0
            term_relevance_score = 0.0
            final_score = result.score  # Default to original score

            if query and top_terms:
                # Calculate term boost from text analysis
                term_boost = self._calculate_term_boost_score(result.text, top_terms)

                # Calculate term relevance from primary terms
                term_relevance_score, _, _ = self._calculate_term_relevance_score(primary_terms, query, top_terms)

                # Calculate composite final score
                final_score = (
                    result.score + term_boost
                ) * self.SIMILARITY_WEIGHT + term_relevance_score * self.TERM_RELEVANCE_WEIGHT

            # Build compact result directly (no intermediate objects)
            primary_term_keys = []
            for term in primary_terms:
                if term.term not in terms_dict:
                    terms_dict[term.term] = self._convert_to_term_info(term)
                primary_term_keys.append(term.term)

            related_term_keys = []
            for term in related_terms:
                if term.term not in terms_dict:
                    terms_dict[term.term] = self._convert_to_term_info(term)
                related_term_keys.append(term.term)

            # Build images and tables directly
            images = self._build_image_infos(result)
            tables = self._build_table_infos(result)

            # Build document href
            href = self._build_document_href(result.metadata.document_path if result.metadata else None)

            # Replace markers with markdown for display
            from blockether_catalyst.knowledge.KnowledgeProcessingUtils import (
                KnowledgeProcessingUtils,
            )

            # First replace image markers with markdown images
            content_with_images = KnowledgeProcessingUtils.replace_image_markers_with_markdown(result.text, images)
            # Then replace table markers with markdown tables
            clean_content = KnowledgeProcessingUtils.replace_table_markers_with_markdown(content_with_images, tables)

            compact_result = CompactSearchResult(
                score=final_score,  # Use final composite score
                content=clean_content,
                document_name=result.metadata.document_name if result.metadata else "",
                page=result.metadata.page,
                author=result.metadata.author,
                publication_date=result.metadata.publication_date,
                modified_date=result.metadata.modified_date,
                href=href,
                primary_term_keys=primary_term_keys,
                related_term_keys=related_term_keys,
                images=images,
                tables=tables,
            )

            # Store with final score for re-ranking
            results_with_scores.append((final_score, compact_result))

        # Final re-ranking by composite score
        results_with_scores.sort(key=lambda x: x[0], reverse=True)
        compact_results = [result for _, result in results_with_scores]

        return OptimizedSearchResponse(
            results=compact_results,
            terms=terms_dict,
            total_results=len(compact_results),
        )

    def _extract_result_terms(
        self,
        result: SearchResult,
        max_depth: int,
        max_cooccurrences: int,
        config: dict,
        query: Optional[str] = None,
        top_terms: Optional[TopTermsResult] = None,
    ) -> Tuple[List, List]:
        """Extract primary and related terms for a result with intelligent classification."""
        if not result.metadata or not result.metadata.terms:
            return [], []

        # Get limited terms from chunk
        chunk_terms = list(result.metadata.terms)[: config["max_terms_chunk"]]

        # Batch resolve terms
        resolved_terms = {
            term_key: self._linked_knowledge.terms[term_key]
            for term_key in chunk_terms
            if term_key in self._linked_knowledge.terms
        }

        primary_terms: List[TermWithLinks] = []
        related_terms: List[TermWithLinks] = []

        # Prepare query-based classification if query info is provided
        query_lower = query.lower() if query else ""
        top_keyword_set = set()
        top_acronym_set = set()

        if top_terms:
            max_keywords = config.get("max_keywords", self.MAX_KEYWORDS_FOR_MATCHING)
            max_acronyms = config.get("max_acronyms", self.MAX_ACRONYMS_FOR_MATCHING)
            top_keyword_set = {kw[0].lower() for kw in top_terms.keywords[:max_keywords]}
            top_acronym_set = {ac[0].lower() for ac in top_terms.acronyms[:max_acronyms]}

        # Intelligent term classification
        for term in resolved_terms.values():
            term_lower = term.term.lower()

            # Priority 1: Terms that appear in the original query
            is_query_relevant = (
                term_lower in query_lower or term_lower in top_keyword_set or term_lower in top_acronym_set
            )

            if len(primary_terms) < config["max_primary_terms"]:
                if is_query_relevant:
                    # High priority - directly related to query
                    primary_terms.append(term)
                elif not primary_terms:
                    # If no query-relevant terms, take first available terms as primary
                    primary_terms.append(term)
            elif len(related_terms) < config["max_related_terms"]:
                related_terms.append(term)
            else:
                break

        # Add linked terms if depth > 0 and we have primary terms
        if max_depth > 0 and primary_terms:
            self._add_linked_terms(primary_terms, related_terms, resolved_terms, config, max_depth)

        return primary_terms, related_terms

    def _add_linked_terms(
        self,
        primary_terms: List[TermWithLinks],
        related_terms: List[TermWithLinks],
        resolved_terms: Dict[str, TermWithLinks],
        config: dict,
        max_depth: int,
    ) -> None:
        """Add linked terms from primary terms if depth > 0."""
        if not primary_terms or max_depth <= 0:
            return

        max_total_related = min(
            config["max_related_terms"] * self.MAX_RELATED_TERMS_MULTIPLIER,
            self.MAX_TOTAL_RELATED_TERMS,
        )

        # Add linked terms from the first primary term
        first_term = primary_terms[0]
        if first_term.links:
            for link in first_term.links[: self.MAX_LINKS_PER_TERM]:
                if (
                    link.link_to in resolved_terms
                    and len(related_terms) < max_total_related
                    and link.link_to not in [t.term for t in related_terms]
                ):
                    related_terms.append(resolved_terms[link.link_to])

    def _calculate_term_relevance_score(
        self,
        primary_terms: List[TermWithLinks],
        query: str,
        top_terms: TopTermsResult,
    ) -> Tuple[float, int, int]:
        """
        Calculate term relevance score based on primary terms and query matching.

        Args:
            primary_terms: List of primary terms for this result
            query: Original search query
            top_terms: Top keywords and acronyms from query

        Returns:
            Tuple of (term_relevance_score, primary_count, term_freq_sum)
        """
        if not primary_terms:
            return 0.0, 0, 0

        query_lower = query.lower()
        primary_count = len(primary_terms)
        term_freq_sum = 0

        # Count how many primary terms match the query
        for term in primary_terms:
            term_lower = term.term.lower()
            if term_lower in query_lower:
                term_freq_sum += 1

        # Calculate composite term relevance score
        term_relevance_score = (
            term_freq_sum * self.TERM_FREQUENCY_SCORE_WEIGHT + primary_count * self.PRIMARY_TERM_COUNT_WEIGHT
        )

        return term_relevance_score, primary_count, term_freq_sum

    def _build_image_infos(self, result: SearchResult) -> List[ImageInfo]:
        """Build image info list for result by looking up page data."""
        if not result.metadata or not result.metadata.document_id or result.metadata.page is None:
            return []

        page_key = (result.metadata.document_id, result.metadata.page)
        if page_key not in self._linked_knowledge.pages:
            return []

        page_data = self._linked_knowledge.pages[page_key]
        return [
            ImageInfo(
                caption=img.caption,
                href=f"{self._resources_base_url}/{quote(img.path, safe='/')}" if self._resources_base_url else "",
                page=img.page,
                document_name=result.metadata.document_name,
            )
            for img in page_data.images
        ]

    def _build_table_infos(self, result: SearchResult) -> List[TableInfo]:
        """Build table info list for result by looking up page data."""
        if not result.metadata or not result.metadata.document_id or result.metadata.page is None:
            return []

        page_key = (result.metadata.document_id, result.metadata.page)
        if page_key not in self._linked_knowledge.pages:
            return []

        page_data = self._linked_knowledge.pages[page_key]

        tables = []
        for table in page_data.tables:
            # Get table content
            content = table.to_markdown()

            # Use the caption from extraction phase (mandatory field)
            caption = table.caption

            tables.append(
                TableInfo(
                    content=content,
                    page=table.page,
                    caption=caption,
                )
            )
        return tables

    def _build_document_href(self, document_path: Optional[str]) -> str:
        """Build document href from path."""
        if not document_path or not self._resources_base_url:
            return ""
        return f"{self._resources_base_url}/{quote(document_path, safe='/')}"

    def _get_top_keywords_and_acronyms(
        self,
        query: str,
        vectorizers: KnowledgeVectorizers,
        max_keywords: int = DEFAULT_MAX_KEYWORD_FEATURES,
        max_acronyms: int = DEFAULT_MAX_ACRONYM_FEATURES,
    ) -> TopTermsResult:
        """
        Extract and rank keywords and acronyms from a query using TF-IDF.

        Keywords are sorted by n-gram size (descending) then by TF-IDF score (descending).
        Acronyms are sorted by TF-IDF score (descending).

        Args:
            query: The search query text
            vectorizers: KnowledgeVectorizers instance for TF-IDF extraction
            max_keywords: Maximum number of keywords to return
            max_acronyms: Maximum number of acronyms to return

        Returns:
            TopTermsResult containing ranked keywords and acronyms
        """
        keyword_vectorizer = vectorizers.keywords_vectorizer()
        acronyms_vectorizer = vectorizers.acronyms_vectorizer()

        try:
            # Extract keywords using TF-IDF
            top_keywords: List[Tuple[str, float]] = []
            tfidf_keywords_matrix = keyword_vectorizer.fit_transform([query])
            keywords = keyword_vectorizer.get_feature_names_out()
            if len(keywords) > 0:  # Check if we have any features
                keywords_scores = tfidf_keywords_matrix.toarray()[0]  # type: ignore

                # Create list of (keyword, score) for non-zero scores
                keyword_pairs = [
                    (keywords[idx], keywords_scores[idx]) for idx in range(len(keywords)) if keywords_scores[idx] > 0
                ]

                # Sort by n-gram size (word count) descending, then by score descending
                keyword_pairs.sort(key=lambda x: (len(x[0].split()), x[1]), reverse=True)
                top_keywords = keyword_pairs[:max_keywords]

                if top_keywords:
                    logger.debug(
                        f"Top keywords from query (by n-gram size): {top_keywords[: self.DEFAULT_DEBUG_TOP_RESULTS_LIMIT]}"
                    )

        except ValueError as e:
            logger.debug(f"No multi-word keywords found in query '{query}': {e}")
            top_keywords = []

        # Extract acronyms using TF-IDF
        top_acronyms: List[Tuple[str, float]] = []
        try:
            tfidf_acronyms_matrix = acronyms_vectorizer.fit_transform([query])
            acronyms = acronyms_vectorizer.get_feature_names_out()
            if len(acronyms) > 0:  # Check if we have any features
                acronyms_scores = tfidf_acronyms_matrix.toarray()[0]  # type: ignore

                # Create list of (acronym, score) for non-zero scores, sorted by score
                acronym_pairs = [
                    (acronyms[idx], acronyms_scores[idx]) for idx in range(len(acronyms)) if acronyms_scores[idx] > 0
                ]

                # Sort by score descending
                acronym_pairs.sort(key=lambda x: x[1], reverse=True)
                top_acronyms = acronym_pairs[:max_acronyms]

                if top_acronyms:
                    logger.debug(f"Top acronyms from query: {top_acronyms[: self.DEFAULT_DEBUG_TOP_RESULTS_LIMIT]}")
        except ValueError as e:
            logger.debug(f"No acronyms found in query '{query}': {e}")
            top_acronyms = []

        return TopTermsResult(keywords=top_keywords, acronyms=top_acronyms)

    def _calculate_term_boost_score(
        self,
        text: str,
        top_terms: TopTermsResult,
    ) -> float:
        """
        Calculate total boost score based on presence of top keywords and acronyms in text.

        Args:
            text: The text to search for keywords and acronyms in
            top_terms: The top keywords and acronyms extracted from the query

        Returns:
            Total boost score
        """
        keyword_boost = 0.0
        acronym_boost = 0.0
        text_lower = text.lower()

        # Create lookup dictionaries for faster matching (case-insensitive)
        top_keyword_dict = {kw[0].lower(): (kw[0], kw[1], len(kw[0].split())) for kw in top_terms.keywords}
        top_acronym_dict = {ac[0].lower(): (ac[0], ac[1]) for ac in top_terms.acronyms}

        # Check for top keywords in the text
        for keyword_lower, (
            keyword_orig,
            score,
            ngram_size,
        ) in top_keyword_dict.items():
            if keyword_lower in text_lower:
                # Apply boost based on TF-IDF score and n-gram size
                keyword_boost += score * self.KEYWORD_BOOST_WEIGHT
                # Additional boost for larger n-grams (more specific terms)
                keyword_boost += (ngram_size - 1) * self.NGRAM_SIZE_BOOST * score
                logger.debug(f"Found keyword '{keyword_orig}' in text, boost: {score * self.KEYWORD_BOOST_WEIGHT}")

        # Check for top acronyms in the text
        for acronym_lower, (acronym_orig, score) in top_acronym_dict.items():
            if acronym_lower in text_lower:
                acronym_boost += score * self.ACRONYM_BOOST_WEIGHT
                logger.debug(f"Found acronym '{acronym_orig}' in text, boost: {score * self.ACRONYM_BOOST_WEIGHT}")

        total_boost = keyword_boost + acronym_boost

        # Log significant boosts
        if total_boost > 0:
            logger.debug(
                f"Calculated boost scores - Total: {total_boost:.3f} "
                f"(keyword: {keyword_boost:.3f}, acronym: {acronym_boost:.3f})"
            )

        return total_boost

    def _convert_to_term_info(self, term: TermWithLinks, link_limit: int = DEFAULT_MAX_COOCCURRENCES) -> TermInfo:
        """
        Convert a TermWithLinks to a TermInfo Pydantic model.

        Args:
            term: The term to convert
            link_limit: Maximum number of linked terms to include

        Returns:
            TermInfo model
        """
        # Get linked term keys
        linked_term_keys = []
        link_scores = {}

        for link in term.links[:link_limit]:
            linked_term_keys.append(link.link_to)
            link_scores[link.link_to] = link.score

        return TermInfo(
            term=term.term,
            meaning=term.meaning,
            term_type=term.type,
            total_times_occurred_in_knowledgebase=term.total,
            linked_terms=linked_term_keys,
            link_scores=link_scores,
        )

    def get_extraction_summary(self) -> str:
        """Get extraction summary using the configured resources base URL.

        Returns:
            Dictionary containing all extraction summary with proper URLs
        """
        if not self._linked_knowledge:
            return "No knowledge base available."

        if not self._resources_base_url:
            raise ValueError("Resources base URL is not set")

        return self._linked_knowledge.get_extraction_summary(base_url=self._resources_base_url, detailed=True)

    def persist(self, path: Optional[Union[str, Path]] = None) -> None:
        """
        Persist the complete KnowledgeSearchCore instance to a pickle file.

        This saves:
        - LinkedKnowledge structure with all documents, terms, chunks, and indices
        - Vector store with embeddings using LangChain's dump method

        Args:
            path: Path to save the pickle file. If None, uses self.pickle_path

        Raises:
            ValueError: If no path is provided and self.pickle_path is not set
        """

        save_path = Path(path) if path else self.pickle_path
        if not save_path:
            raise ValueError("No path provided for persistence")

        # Create parent directory if it doesn't exist
        save_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Persisting KnowledgeSearchCore to {save_path}")

        # Create a serializable state object with the vector store directly
        state = {
            "linked_knowledge": self._linked_knowledge,
            "vector_store": self._vector_store.store,
        }

        # Save to pickle
        with open(save_path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info(f"Successfully persisted KnowledgeSearchCore to {save_path}")

        # Update internal pickle_path
        self.pickle_path = save_path

    def _load(self, path: Union[str, Path]) -> None:
        """
        Load a KnowledgeSearchCore instance from a pickle file.

        This restores:
        - LinkedKnowledge structure with all documents, terms, chunks, and indices
        - Vector store with embeddings using LangChain's load method

        Args:
            path: Path to load the pickle file from

        Raises:
            FileNotFoundError: If the pickle file doesn't exist
            ValueError: If the pickle file is corrupted or incompatible
        """

        load_path = Path(path)
        if not load_path.exists():
            raise FileNotFoundError(f"Pickle file not found: {load_path}")

        logger.info(f"Loading KnowledgeSearchCore from {load_path}")

        try:
            with open(load_path, "rb") as f:
                state = pickle.load(f)

            # Restore all state
            self._linked_knowledge = state["linked_knowledge"]

            # Load the vector store directly from state
            self._vector_store = InMemoryVectorStore(embedding=EncoderEmbeddings())
            self._vector_store.store = state["vector_store"]

            logger.info(f"Successfully loaded KnowledgeSearchCore from {load_path}")
            logger.info(f"  - Documents: {len(self._linked_knowledge.documents)}")
            logger.info(f"  - Terms: {len(self._linked_knowledge.terms)}")
            logger.info(f"  - Chunks: {len(self._linked_knowledge.chunks)}")

        except Exception as e:
            raise ValueError(f"Failed to load pickle file: {e}") from e

        # Update internal pickle_path
        self.pickle_path = load_path

    @classmethod
    def from_pickle(cls, path: Union[str, Path], resources_base_url: str) -> "KnowledgeSearchCore":
        """
        Create a KnowledgeSearchCore instance by loading from a pickle file.

        This is a convenience class method that creates a new instance and loads
        the state from a pickle file.

        Args:
            path: Path to the pickle file

        Returns:
            Loaded KnowledgeSearchCore instance

        Raises:
            FileNotFoundError: If the pickle file doesn't exist
        """
        pickle_path = Path(path)
        if not pickle_path.exists():
            raise FileNotFoundError(f"Pickle file not found: {pickle_path}")

        instance = cls(
            linked_knowledge=None,
            pickle_path=path,
            auto_load=True,
            resources_base_url=resources_base_url,
        )

        return instance
