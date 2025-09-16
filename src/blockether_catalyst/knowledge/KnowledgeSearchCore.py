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

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore
from pydantic import BaseModel, Field

from blockether_catalyst.knowledge.KnowledgeVectorizers import KnowledgeVectorizers

from ..encoder.EncoderCore import EncoderCore
from .KnowledgeTypes import (
    ImageMetadata,
    KnowledgeSearchResult,
    KnowledgeTableData,
    LinkedKnowledge,
    SearchResult,
    SearchResultMetadata,
    Term,
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
    DEFAULT_K_RESULTS: int = 10
    DEFAULT_THRESHOLD: float = 0.1
    DEFAULT_MAX_DEPTH: int = 2
    DEFAULT_MAX_COOCCURRENCES: int = 5
    SIMILARITY_WEIGHT: float = 0.6
    TERM_RELEVANCE_WEIGHT: float = 0.4
    TERM_FREQUENCY_WEIGHT: float = 0.7
    TERM_DIVERSITY_WEIGHT: float = 0.3
    KEYWORD_BOOST_WEIGHT: float = 0.3  # Boost for results containing top keywords
    ACRONYM_BOOST_WEIGHT: float = 0.2  # Boost for results containing top acronyms
    NGRAM_SIZE_BOOST: float = 0.1  # Additional boost per n-gram size for keywords
    _is_initialized_from_state: bool = False

    def __init__(
        self,
        base_url: Optional[str] = None,
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

        self._base_url = base_url

        self.pickle_path = Path(pickle_path) if pickle_path else None
        if not linked_knowledge and auto_load and self.pickle_path and self.pickle_path.exists():
            logger.info(f"Loading KnowledgeSearchCore from pickle: {self.pickle_path}")
            self._load(self.pickle_path)
            self._is_initialized_from_state = True
            return
        else:
            # Otherwise, initialize from linked_knowledge
            if linked_knowledge is None:
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

    def resolve_term(self, term_key: str) -> Optional[TermWithLinks]:
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
    ) -> List[Dict[str, Any]]:
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
        6. Returns normalized results in TypedDict format

        Args:
            query: Search query text
            k: Number of results to return
            threshold: Minimum similarity threshold
            max_depth: Maximum depth for exploring related terms
            max_cooccurrences: Maximum number of co-occurring terms to include

        Returns:
            List of normalized search results ready for consumption
        """
        start_time = time.time()
        logger.info(f"Performing enhanced search for query: '{query}'")

        vectorizers = KnowledgeVectorizers(keywords_min_df=1, acronyms_min_df=1)

        # Perform vector search
        search_results = self._similarity_search(query, k=k, threshold=threshold)

        max_keywords_from_query = 5
        max_acronyms_from_query = 5
        top_terms = self._get_top_keywords_and_acronyms(query, vectorizers, max_keywords=max_keywords_from_query, max_acronyms=max_acronyms_from_query)

        # Log the extracted terms if present
        if top_terms.keywords:
            logger.info(f"Top keywords from query (by n-gram size): {top_terms.keywords[:max_keywords_from_query]}")
        if top_terms.acronyms:
            logger.info(f"Top acronyms from query: {top_terms.acronyms[:max_acronyms_from_query]}")

        # Convert to enhanced search results with term analysis
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

            # Extract primary terms efficiently using chunk metadata
            chunk_terms = result.metadata.terms
            for term_key in chunk_terms:
                if term_key in self._linked_knowledge.terms:
                    term = self._linked_knowledge.terms[term_key]
                    enh_result.primary_terms.append(term)
                    enh_result.all_terms.add(term.term)

            # Resolve co-occurrences and links for all primary terms
            visited_terms = set()
            for term in enh_result.primary_terms:
                # Add the primary term key to visited set
                visited_terms.add(term.term)

                # Recursively resolve linked terms up to max_depth
                # Create a new visited set for each primary term to allow cross-references
                term_visited = visited_terms.copy()
                linked_terms = self._resolve_linked_terms(term, max_depth, visited=term_visited)
                for linked_term in linked_terms:
                    if linked_term not in enh_result.related_terms:
                        enh_result.related_terms.append(linked_term)
                        enh_result.all_terms.add(linked_term.term)

                # Add co-occurring terms (only at first level, not recursive)
                if term.cooccurrences and max_cooccurrences > 0:
                    for cooccurrence in term.cooccurrences[:max_cooccurrences]:
                        if cooccurrence.term in self._linked_knowledge.terms:
                            cooccurring_term = self._linked_knowledge.terms[cooccurrence.term]
                            if cooccurring_term not in enh_result.related_terms:
                                enh_result.related_terms.append(cooccurring_term)
                                enh_result.all_terms.add(cooccurring_term.term)

            # Calculate term frequencies in the query
            for term_str in enh_result.all_terms:
                term_lower = term_str.lower()
                # Count occurrences of this term in the query
                count = query.lower().count(term_lower)
                if count > 0:
                    enh_result.term_frequencies[term_str] = count

            # Calculate term relevance score
            # Higher score for results with more query terms and higher frequencies
            if enh_result.term_frequencies:
                # Sum of frequencies weighted by term importance
                total_freq = sum(enh_result.term_frequencies.values())
                unique_terms = len(enh_result.term_frequencies)
                # Combine frequency and diversity
                enh_result.term_relevance_score = (
                    total_freq * self.TERM_FREQUENCY_WEIGHT + unique_terms * self.TERM_DIVERSITY_WEIGHT
                )

            enh_result.primary_terms.sort(key=lambda t: t.total)
            enh_result.related_terms.sort(key=lambda t: t.total)

            # Add images and tables from the page if available
            if enh_result.page and enh_result.document_id:
                self._add_page_content(enh_result, enh_result.document_id, enh_result.page)

            enhanced_results.append(enh_result)

        # Sort results by combined score (similarity + term relevance)
        enhanced_results.sort(
            key=lambda r: (r.score * self.SIMILARITY_WEIGHT + r.term_relevance_score * self.TERM_RELEVANCE_WEIGHT),
            reverse=True,
        )

        # Convert to normalized format
        normalized_results: List[Dict[str, Any]] = []

        for result in enhanced_results:
            # Build document reference with href
            doc_href = ""
            if self._base_url and result.document_path:
                doc_href = f"{self._base_url}/{result.document_path}"

            # Simplified normalized result
            normalized_result = {
                "score": result.score,
                "content": result.text,
                "document_name": result.document_name,
                "document_path": result.document_path,
                "page": result.page,
                "author": result.metadata.author if result.metadata else None,
                "publication_date": result.metadata.publication_date if result.metadata else None,
                "modified_date": result.metadata.modified_date if result.metadata else None,
                "href": doc_href,
                "images": [
                    {
                        "caption": img.caption,
                        "href": f"{self._base_url}/{img.path}",
                        "page": img.page,
                    }
                    for img in result.images
                ],
                "tables": [
                    {
                        "markdown": table.to_markdown(),
                        "page": table.page,
                        "number_of_rows": table.rows,
                        "number_of_columns": table.columns,
                    }
                    for table in result.tables
                ],
                "primary_terms": [self._normalize_term_info(term, k) for term in result.primary_terms],
                "related_terms": [self._normalize_term_info(term, max(3, k)) for term in result.related_terms],
            }

            normalized_results.append(normalized_result)

        search_time = time.time() - start_time
        logger.info(
            f"🔍 Enhanced search took {search_time:.3f}s, returned {len(normalized_results)} results for query: '{query}'"
        )

        return normalized_results

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

        # Extract acronyms using TF-IDF
        top_acronyms: List[Tuple[str, float]] = []
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

        return TopTermsResult(keywords=top_keywords, acronyms=top_acronyms)

    def _normalize_term_info(self, term: TermWithLinks, link_limit: int) -> Dict[str, Any]:
        """
        Normalize a term with its linked terms into a simple dictionary format.

        Args:
            term: The term to normalize
            link_limit: Maximum number of linked terms to include

        Returns:
            Normalized term information as dictionary
        """
        # Resolve linked terms
        linked_terms = []

        for link in term.links[:link_limit]:
            # Resolve the link_to string to get the actual term object
            linked_term = self.resolve_term(link.link_to)
            if linked_term:
                linked_terms.append(
                    {
                        "term": linked_term.term,
                        "meaning": linked_term.meaning or "N/A",
                        "term_type": linked_term.type,
                        "link_score": link.score,
                        "total_times_occurred_in_knowledgebase": linked_term.total,
                    }
                )

        # Sort linked terms by score
        linked_terms.sort(key=lambda x: x["link_score"], reverse=True)

        return {
            "term": term.term,
            "meaning": term.meaning,
            "term_type": term.type,
            "total_times_occurred_in_knowledgebase": term.total,
            "linked_terms": linked_terms,
        }

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
    def from_pickle(cls, path: Union[str, Path], base_url: str) -> "KnowledgeSearchCore":
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
            base_url=base_url,
        )

        return instance
