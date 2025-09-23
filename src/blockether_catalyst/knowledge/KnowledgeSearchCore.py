"""
Knowledge-based search system that integrates vector search with extracted terms and relationships.

This module provides sophisticated search functionality that combines semantic search
with knowledge about terms, their meanings, co-occurrences, and relationships.
"""

import heapq
import logging
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from urllib.parse import quote

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore

from blockether_catalyst.knowledge.KnowledgeVectorizers import KnowledgeVectorizers

from ..encoder.EncoderCore import EncoderCore
from .KnowledgeTypes import (
    CompactSearchResult,
    ImageInfo,
    KnowledgeSearchResult,
    LinkedKnowledge,
    LinkedTermInfo,
    NormalizedSearchResult,
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
    """Custom embeddings class that uses our EncoderCore."""

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
        embeddings = EncoderCore.encode(texts)
        return embeddings.tolist()  # type: ignore[no-any-return]

    def embed_query(self, text: str) -> Any:
        """
        Embed a query text.

        Args:
            text: Query text to embed

        Returns:
            Embedding as list of floats
        """
        embedding = EncoderCore.encode_single(text)
        return embedding.tolist()  # type: ignore[no-any-return]


# All search-related types are now imported from KnowledgeTypes
# This provides a single source of truth for all type definitions


class KnowledgeSearchCore:
    """
    Knowledge-enhanced search system that combines vector search with extracted knowledge.

    Provides intelligent search that understands:
    - Term meanings and definitions
    - Relationships between acronyms and keywords
    - Co-occurrence patterns
    - Document structure and context
    """

    # Class constants
    DEFAULT_K_RESULTS: int = 5
    DEFAULT_THRESHOLD: float = 0.1
    DEFAULT_MAX_DEPTH: int = 2
    DEFAULT_MAX_COOCCURRENCES: int = 5
    SIMILARITY_WEIGHT: float = 0.6
    TERM_RELEVANCE_WEIGHT: float = 0.4
    TERM_FREQUENCY_WEIGHT: float = 0.7
    TERM_DIVERSITY_WEIGHT: float = 0.3
    KEYWORD_BOOST_WEIGHT: float = 0.3  # Boost for results containing top keywords
    ACRONYM_BOOST_WEIGHT: float = 0.4  # Boost for results containing top acronyms
    NGRAM_SIZE_BOOST: float = 0.3  # Additional boost per n-gram size for keywords
    MAX_TOP_KEYWORDS_FROM_QUERY: int = 5  # Maximum number of top keywords to extract from query
    MAX_TOP_ACRONYMS_FROM_QUERY: int = 5  # Maximum number of top acronyms to extract from query

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
    ) -> OptimizedSearchResponse:
        """
        Perform enhanced search and return normalized results ready for consumption.

        Acronyms:
            TF: Term Frequency - count of how often a term appears
            VSS: Vector Similarity Search - finding semantically similar content using vector embeddings

        This method:
        1. Performs vector similarity search
        2. Extracts terms from results using efficient indices
        3. Analyzes term frequencies in the query
        4. Calculates relevance scores based on term statistics
        5. Sorts results using both similarity and term relevance
        6. Returns normalized results as Pydantic models

        Args:
            query: Search query text
            k: Number of results to return
            threshold: Minimum similarity threshold
            max_depth: Maximum depth for exploring related terms
            max_cooccurrences: Maximum number of co-occurring terms to include

        Returns:
            List of NormalizedSearchResult Pydantic models ready for consumption
        """
        start_time = time.time()
        logger.info(f"Performing enhanced search for query: '{query}'")

        vectorizers = KnowledgeVectorizers(keywords_min_df=1, acronyms_min_df=1)

        # Extract top keywords and acronyms from the query FIRST
        top_terms = self._get_top_keywords_and_acronyms(
            query,
            vectorizers,
            max_keywords=self.MAX_TOP_KEYWORDS_FROM_QUERY,
            max_acronyms=self.MAX_TOP_ACRONYMS_FROM_QUERY,
        )

        # Log the extracted terms if present
        if top_terms.keywords:
            logger.info(
                f"Top keywords from query (by n-gram size): {top_terms.keywords[: self.MAX_TOP_KEYWORDS_FROM_QUERY]}"
            )
        if top_terms.acronyms:
            logger.info(f"Top acronyms from query: {top_terms.acronyms[: self.MAX_TOP_ACRONYMS_FROM_QUERY]}")

        # Phase 1: Direct term-based retrieval using term_to_chunks_index
        term_based_chunks = set()
        if self._linked_knowledge and self._linked_knowledge.term_to_chunks_index:
            for term_tuple in top_terms.acronyms + top_terms.keywords:
                term = term_tuple[0]  # Get the term string from tuple
                normalized_term = self._linked_knowledge._normalize_term(term)

                # Look up chunks that contain this term
                if normalized_term in self._linked_knowledge.term_to_chunks_index:
                    chunk_refs = self._linked_knowledge.term_to_chunks_index[normalized_term]
                    for doc_id, chunk_idx in chunk_refs:
                        # Store doc_id and chunk_idx for later matching
                        # We'll match against actual chunk IDs from search results
                        term_based_chunks.add((doc_id, chunk_idx))
                        logger.debug(
                            f"Found chunk reference doc={doc_id}, idx={chunk_idx} via term index for term: {term}"
                        )

        # Phase 2: Vector similarity search
        # Use a lower threshold (min 0.1) to get more candidates for term matching
        effective_threshold = min(threshold, 0.1) if (top_terms.acronyms or top_terms.keywords) else threshold
        search_results = self._similarity_search(query, k=int(k*1.2), threshold=effective_threshold)

        # Enhance search results with term analysis and boosting
        # The function now handles efficient top-k selection internally
        enhanced_results = self._enhance_search_results(
            search_results,
            query,
            top_terms,
            max_depth,
            max_cooccurrences,
            int(k * 1.2),  # Get more for filtering
        )

        # Filter enhanced results based on acronym/keyword matches
        final_results = []

        # Create sets for fast lookup
        query_acronyms = {ac[0].lower() for ac in top_terms.acronyms}
        query_keywords = {kw[0].lower() for kw in top_terms.keywords}

        for result in enhanced_results:
            # Check if this result contains any of the query acronyms/keywords
            contains_query_terms = False

            if query_acronyms or query_keywords:
                # Check primary terms for matches
                for term in result.primary_terms:
                    term_lower = term.term.lower()
                    if term_lower in query_acronyms or term_lower in query_keywords:
                        contains_query_terms = True
                        break

                # If not found in primary, check the text content
                if not contains_query_terms:
                    text_lower = result.text.lower()
                    for acronym in query_acronyms:
                        if acronym in text_lower:
                            contains_query_terms = True
                            break
                    if not contains_query_terms:
                        for keyword in query_keywords:
                            if keyword in text_lower:
                                contains_query_terms = True
                                break

            # Check if this chunk was found via term index
            from_term_index = False
            # Check if this (doc_id, chunk_index) pair is in our term-based chunks
            for doc_id, chunk_idx in term_based_chunks:
                if result.document_id == doc_id and result.chunk_index == chunk_idx:
                    from_term_index = True
                    break

            # Boost the score if contains query terms
            if contains_query_terms or from_term_index:
                # Apply boost to final score for sorting
                boost = (
                    0.2
                    if any(
                        term[0].lower() in query_acronyms
                        for term in top_terms.acronyms
                        if result.text and term[0].lower() in result.text.lower()
                    )
                    else 0.1
                )
                result.final_score = min(1.0, result.final_score + boost)

            # Include result if:
            # 1. It was found via term index (direct term match)
            # 2. It contains query acronyms/keywords (regardless of original score)
            # 3. Its similarity score meets the threshold
            if from_term_index:
                logger.debug(f"Including result from term index with score {result.score:.3f}")
                final_results.append(result)
            elif contains_query_terms:
                logger.debug(f"Including result with score {result.score:.3f} - contains query terms")
                final_results.append(result)
            elif result.score >= threshold:
                final_results.append(result)
            else:
                logger.debug(
                    f"Filtering out result with score {result.score:.3f} - below threshold and no term matches"
                )

        # Sort by final score and take top k
        final_results.sort(key=lambda r: r.final_score, reverse=True)
        final_results = final_results[:k]

        # Convert to NormalizedSearchResult Pydantic models
        normalized_results: List[NormalizedSearchResult] = []

        for result in final_results:
            # Build document reference with href (properly encoded)
            doc_href = ""
            if self._resources_base_url and result.document_path:
                # Encode the path to handle spaces and special characters
                encoded_path = quote(result.document_path, safe="/")
                doc_href = f"{self._resources_base_url}/{encoded_path}"

            # Convert images to ImageInfo models
            images = []
            for img in result.images:
                images.append(
                    ImageInfo(
                        caption=img.caption,
                        href=f"{self._resources_base_url}/{quote(img.path, safe='/')}",
                        page=img.page,
                        document_name=result.document_name,
                    )
                )

            # Convert tables to TableInfo models
            tables = []
            for table in result.tables:
                tables.append(
                    TableInfo(
                        markdown_content=table.to_markdown(),
                        page=table.page,
                    )
                )

            # Convert terms to TermInfo models
            primary_terms = [self._convert_to_term_info(term, k) for term in result.primary_terms[:k]]
            related_terms = [self._convert_to_term_info(term, int(k / 2)) for term in result.related_terms[:3]]

            # Create NormalizedSearchResult
            normalized_result = NormalizedSearchResult(
                score=result.score,
                content=result.text,
                document_name=result.document_name,
                page=result.page if result.page else None,
                author=result.metadata.author if result.metadata else None,
                publication_date=result.metadata.publication_date if result.metadata else None,
                modified_date=result.metadata.modified_date if result.metadata else None,
                href=doc_href if doc_href else None,
                images=images,
                tables=tables,
                primary_terms=[term for term in primary_terms if term],  # Filter out None values
                related_terms=[term for term in related_terms if term],  # Filter out None values
            )

            normalized_results.append(normalized_result)

        # Build deduplicated terms dictionary and compact results
        terms_dict: Dict[str, TermInfo] = {}
        compact_results = []

        for result in normalized_results:
            # Process primary terms
            primary_term_keys = []
            for term in result.primary_terms:
                term_key = term.term  # Use term name as key
                if term_key not in terms_dict:
                    terms_dict[term_key] = term
                primary_term_keys.append(term_key)

            # Process related terms
            related_term_keys = []
            for term in result.related_terms:
                term_key = term.term  # Use term name as key
                if term_key not in terms_dict:
                    terms_dict[term_key] = term
                related_term_keys.append(term_key)

            # Create compact result
            compact_result = CompactSearchResult(
                score=result.score,
                content=result.content,
                document_name=result.document_name,
                page=result.page,
                author=result.author,
                publication_date=result.publication_date,
                modified_date=result.modified_date,
                href=result.href,
                primary_term_keys=primary_term_keys,
                related_term_keys=related_term_keys,
                images=result.images,
                tables=result.tables,
            )
            compact_results.append(compact_result)

        search_time = time.time() - start_time
        logger.info(
            f"🔍 Enhanced search took {search_time:.3f}s, returned {len(normalized_results)} results for query: '{query}'"
        )

        return OptimizedSearchResponse(
            results=compact_results,
            terms=terms_dict,
            total_results=len(normalized_results),
            query=query,
            search_type="hybrid",
        )

    def _get_top_keywords_and_acronyms(
        self,
        query: str,
        vectorizers: KnowledgeVectorizers,
        max_keywords: int = 10,
        max_acronyms: int = 10,
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
                    logger.debug(f"Top keywords from query (by n-gram size): {top_keywords[:5]}")

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
                    logger.debug(f"Top acronyms from query: {top_acronyms[:5]}")
        except ValueError as e:
            logger.debug(f"No acronyms found in query '{query}': {e}")
            top_acronyms = []

        return TopTermsResult(keywords=top_keywords, acronyms=top_acronyms)

    def _calculate_term_boost_score(
        self,
        text: str,
        top_terms: TopTermsResult,
    ) -> Tuple[float, float, float]:
        """
        Calculate boost scores based on presence of top keywords and acronyms in text.

        Args:
            text: The text to search for keywords and acronyms in
            top_terms: The top keywords and acronyms extracted from the query

        Returns:
            Tuple of (total_boost, keyword_boost, acronym_boost)
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

        return total_boost, keyword_boost, acronym_boost

    def _enhance_search_results(
        self,
        search_results: List[SearchResult],
        query: str,
        top_terms: TopTermsResult,
        max_depth: int,
        max_cooccurrences: int,
        k: int,
    ) -> List[KnowledgeSearchResult]:
        """
        Convert basic search results to enhanced results with term analysis and boosting.

        This method:
        1. Applies keyword/acronym boost scores
        2. Prioritizes terms based on top keywords/acronyms
        3. Extracts and resolves primary and secondary terms
        4. Calculates term frequencies and relevance scores
        5. Adds page content (images/tables)
        6. Efficiently selects top k results

        Args:
            search_results: Basic search results from vector search
            query: The original search query
            top_terms: Top keywords and acronyms extracted from query
            max_depth: Maximum depth for exploring related terms
            max_cooccurrences: Maximum number of co-occurring terms
            k: Number of results to return

        Returns:
            List of top k enhanced search results with full term analysis
        """
        # Create sets of top terms for fast lookup (case-insensitive)
        top_keyword_set = {kw[0].lower() for kw in top_terms.keywords}
        top_acronym_set = {ac[0].lower() for ac in top_terms.acronyms}

        enhanced_results: List[KnowledgeSearchResult] = []

        for result in search_results:
            enh_result = KnowledgeSearchResult(
                text=result.text,
                score=result.score,
                document_id=result.metadata.document_id or "",
                document_name=result.metadata.document_name or "",
                page=result.metadata.page or 0,
                chunk_index=result.metadata.chunk_index or 0,
                metadata=result.metadata,
                document_path=result.metadata.document_path,
            )

            # Calculate and apply boost score based on presence of top keywords and acronyms
            total_boost, keyword_boost, acronym_boost = self._calculate_term_boost_score(result.text, top_terms)

            # Extract and categorize terms based on top keywords/acronyms
            chunk_terms = result.metadata.terms
            query_lower = query.lower()

            for term_key in chunk_terms:
                if term_key in self._linked_knowledge.terms:
                    term = self._linked_knowledge.terms[term_key]
                    term_lower = term.term.lower()

                    # Check if this term matches any top keywords or acronyms from query
                    is_top_term = (
                        term_lower in top_keyword_set
                        or term_lower in top_acronym_set
                        or any(keyword in term_lower for keyword in top_keyword_set)
                        or any(acronym in term_lower for acronym in top_acronym_set)
                    )

                    # Also check if term appears in the query
                    is_in_query = term_lower in query_lower

                    # Primary terms: top terms or terms that appear in query
                    if is_top_term or is_in_query:
                        enh_result.primary_terms.append(term)
                        enh_result.all_terms.add(term.term)
                        # Also count frequency while we're at it
                        count = query_lower.count(term_lower)
                        if count > 0:
                            enh_result.term_frequencies[term.term] = count
                    else:
                        # Secondary (related) terms
                        if term not in enh_result.related_terms:
                            enh_result.related_terms.append(term)
                            enh_result.all_terms.add(term.term)

            # Resolve co-occurrences and links for primary terms only
            visited_terms = set()
            for term in enh_result.primary_terms:
                visited_terms.add(term.term)

                # Add linked terms as secondary
                term_visited = visited_terms.copy()
                linked_terms = self._resolve_linked_terms(term, max_depth, visited=term_visited)
                for linked_term in linked_terms:
                    if linked_term not in enh_result.related_terms and linked_term not in enh_result.primary_terms:
                        enh_result.related_terms.append(linked_term)
                        enh_result.all_terms.add(linked_term.term)

                # Add co-occurring terms as secondary
                if term.cooccurrences and max_cooccurrences > 0:
                    for cooccurrence in term.cooccurrences[:max_cooccurrences]:
                        if cooccurrence.term in self._linked_knowledge.terms:
                            cooccurring_term = self._linked_knowledge.terms[cooccurrence.term]
                            if (
                                cooccurring_term not in enh_result.related_terms
                                and cooccurring_term not in enh_result.primary_terms
                            ):
                                enh_result.related_terms.append(cooccurring_term)
                                enh_result.all_terms.add(cooccurring_term.term)

            # Calculate term relevance score
            if enh_result.term_frequencies:
                total_freq = sum(enh_result.term_frequencies.values())
                unique_terms = len(enh_result.term_frequencies)
                enh_result.term_relevance_score = (
                    total_freq * self.TERM_FREQUENCY_WEIGHT + unique_terms * self.TERM_DIVERSITY_WEIGHT
                )

            # Sort terms by their importance
            enh_result.primary_terms.sort(key=lambda t: t.total, reverse=True)
            enh_result.related_terms.sort(key=lambda t: t.total, reverse=True)

            # Add images and tables from the page if available
            if enh_result.page and enh_result.document_id:
                self._add_page_content(enh_result, enh_result.document_id, enh_result.page)

            # Calculate final composite score (boost is already applied to score)
            # Use the boosted score directly in the composite calculation
            enh_result.final_score = (
                enh_result.score + total_boost
            ) * self.SIMILARITY_WEIGHT + enh_result.term_relevance_score * self.TERM_RELEVANCE_WEIGHT

            # Log significant boosts
            if total_boost > 0:
                logger.debug(
                    f"Boosted result from {result.score:.3f} to {enh_result.score + total_boost:.3f} "
                    f"(final composite: {enh_result.final_score:.3f})"
                )

            enhanced_results.append(enh_result)

        # Efficiently get top k results using heapq
        # Use negative score for max heap behavior
        top_k_results = heapq.nlargest(k, enhanced_results, key=lambda r: r.final_score)

        return top_k_results

    def _clean_empty_values(self, data: Any) -> Any:
        """
        Recursively remove empty values from data structures.

        Removes:
        - None values
        - Empty strings
        - Empty lists/vectors
        - Empty dicts
        - Dict keys with empty values

        Args:
            data: The data to clean (can be dict, list, or any value)

        Returns:
            Cleaned data with empty values removed
        """
        if data is None:
            return None

        if isinstance(data, dict):
            cleaned = {}
            for key, value in data.items():
                cleaned_value = self._clean_empty_values(value)
                # Only include if value is not empty
                if cleaned_value is not None and cleaned_value != "" and cleaned_value != [] and cleaned_value != {}:
                    cleaned[key] = cleaned_value
            return cleaned if cleaned else None

        if isinstance(data, list):
            cleaned_list: list = []
            for item in data:
                cleaned_item = self._clean_empty_values(item)
                # Only include non-empty items
                if cleaned_item is not None and cleaned_item != "" and cleaned_item != [] and cleaned_item != {}:
                    cleaned_list.append(cleaned_item)
            return cleaned_list if cleaned_list else None

        # For strings, check if empty
        if isinstance(data, str):
            return data if data.strip() else None

        # For other types, return as is
        return data

    def _convert_to_term_info(self, term: TermWithLinks, link_limit: int) -> Optional[TermInfo]:
        """
        Convert a TermWithLinks to a TermInfo Pydantic model.

        Args:
            term: The term to convert
            link_limit: Maximum number of linked terms to include

        Returns:
            TermInfo model or None if conversion fails
        """
        # Resolve linked terms
        linked_terms = []

        for link in term.links[:link_limit]:
            # Resolve the link_to string to get the actual term object
            linked_term = self._resolve_term(link.link_to)
            if linked_term:
                linked_terms.append(
                    LinkedTermInfo(
                        term=linked_term.term,
                        meaning=linked_term.meaning[:600] if linked_term.meaning else None,
                        term_type=linked_term.type,
                        link_score=link.score,
                        total_times_occurred_in_knowledgebase=linked_term.total,
                        linked_terms=[],  # Usually empty to prevent deep nesting
                    )
                )

        # Sort linked terms by score
        linked_terms.sort(key=lambda x: x.link_score if x.link_score else 0, reverse=True)

        return TermInfo(
            term=term.term,
            meaning=term.meaning,
            term_type=term.type,
            total_times_occurred_in_knowledgebase=term.total,
            linked_terms=linked_terms,
        )

    def _resolve_linked_terms(
        self,
        term: TermWithLinks,
        max_depth: int,
        current_depth: int = 0,
        visited: Optional[Set[str]] = None,
    ) -> List[TermWithLinks]:
        """
        Recursively resolve linked terms up to max_depth.

        Args:
            term: The term to resolve links for
            max_depth: Maximum depth to traverse
            current_depth: Current recursion depth
            visited: Set of already visited term keys to prevent cycles

        Returns:
            List of linked terms resolved up to max_depth
        """
        if current_depth >= max_depth:
            return []

        if visited is None:
            visited = set()

        linked_terms = []

        for link in term.links:
            linked_term_key = link.link_to

            # Skip if already visited (prevent cycles)
            if linked_term_key in visited:
                continue

            if linked_term_key and linked_term_key in self._linked_knowledge.terms:
                linked_term = self._linked_knowledge.terms[linked_term_key]
                visited.add(linked_term_key)
                linked_terms.append(linked_term)

                # Recursively resolve deeper links (pass the same visited set)
                deeper_terms = self._resolve_linked_terms(
                    linked_term,
                    max_depth,
                    current_depth + 1,
                    visited,
                )
                linked_terms.extend(deeper_terms)

        return linked_terms

    def _add_page_content(
        self,
        result: KnowledgeSearchResult,
        document_id: str,
        page: int,
    ) -> None:
        """
        Add images and tables from the page to the search result.

        Args:
            result: The result to enhance with page content
            document_id: Document identifier
            page: Page number
        """
        page_key = (document_id, page)
        if page_key in self._linked_knowledge.pages:
            page_data = self._linked_knowledge.pages[page_key]

            result.images.extend(page_data.images)
            result.tables.extend(page_data.tables)

    def get_extraction_details(self) -> Dict[str, Any]:
        """Get extraction details using the configured resources base URL.

        Returns:
            Dictionary containing all extraction details with proper URLs
        """
        if not self._linked_knowledge:
            return {}

        return self._linked_knowledge.get_extraction_details(base_url=self._resources_base_url or "")

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
