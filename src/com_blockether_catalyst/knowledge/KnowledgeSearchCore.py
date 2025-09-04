"""
Knowledge-based search system that integrates vector search with extracted terms and relationships.

This module provides sophisticated search functionality that combines semantic search
with knowledge about terms, their meanings, co-occurrences, and relationships.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore

from com_blockether_catalyst.encoder.EncoderCore import EncoderCore

from .internal.KnowledgeExtractionBaseTypes import (
    KnowledgeMetadata,
    KnowledgeTableData,
)
from .internal.KnowledgeExtractionTypes import (
    LinkedKnowledge,
    Term,
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


class SearchResult:
    """Simple search result matching the expected interface."""

    def __init__(
        self,
        text: str,
        score: float,
        doc_id: str,
        metadata: Dict[str, Any],
    ):
        """Initialize search result."""
        self.text = text
        self.score = score
        self.doc_id = doc_id
        self.metadata = metadata


class SimilaritySearchResult:
    """Represents a basic similarity search result with media content."""

    def __init__(
        self,
        text: str,
        score: float,
        document_id: str,
        document_name: str,
        page: Optional[int] = None,
        chunk_index: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize similarity search result.

        Args:
            text: Matching text content
            score: Relevance score
            document_id: Document identifier
            document_name: Document filename
            page: Page number where content appears
            chunk_index: Chunk index where content appears
            metadata: Additional metadata
        """
        self.text = text
        self.score = score
        self.document_id = document_id
        self.document_name = document_name
        self.page = page
        self.chunk_index = chunk_index
        self.metadata = metadata or {}

        # Empty term information (similarity search doesn't include terms)
        self.primary_terms: List[Term] = []
        self.related_terms: List[Term] = []

        # Page content (corrected types)
        self.images: List[str] = []  # Base64 strings
        self.tables: List[KnowledgeTableData] = []


class KnowledgeSearchResult:
    """Represents an enhanced search result with term analysis and statistics."""

    def __init__(
        self,
        text: str,
        score: float,
        document_id: str,
        document_name: str,
        page: Optional[int] = None,
        chunk_index: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize enhanced search result.

        Args:
            text: Matching text content
            score: Relevance score
            document_id: Document identifier
            document_name: Document filename
            page: Page number where content appears
            chunk_index: Chunk index where content appears
            metadata: Additional metadata
        """
        self.text = text
        self.score = score
        self.document_id = document_id
        self.document_name = document_name
        self.page = page
        self.chunk_index = chunk_index
        self.metadata = metadata or {}

        # Term analysis
        self.primary_terms: List[Term] = []
        self.related_terms: List[Term] = []  # Includes co-occurrences and linked terms
        self.all_terms: Set[str] = set()  # All unique terms (primary + related)

        # Term frequency statistics
        self.term_frequencies: Dict[str, int] = {}  # Term -> frequency in query
        self.term_relevance_score: float = 0.0  # Combined relevance based on term frequencies

        # Page content (corrected types)
        self.images: List[str] = []  # Base64 strings
        self.tables: List[KnowledgeTableData] = []


class KnowledgeSearchSnapshot:
    """Serializable snapshot of the knowledge search state."""

    def __init__(
        self,
        linked_knowledge: LinkedKnowledge,
        vector_store_dump: str,
        term_mappings: Dict[str, str],
    ):
        """
        Initialize snapshot.

        Args:
            linked_knowledge: Complete knowledge structure
            vector_store_dump: Serialized vector store as JSON string
            term_mappings: Mappings between terms and their variants
        """
        self.linked_knowledge = linked_knowledge
        self.vector_store_dump = vector_store_dump
        self.term_mappings = term_mappings


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
    DEFAULT_K_RESULTS = 10
    DEFAULT_THRESHOLD = 0.1
    DEFAULT_MAX_DEPTH = 2
    DEFAULT_MAX_COOCCURRENCES = 5
    SIMILARITY_WEIGHT = 0.6
    TERM_RELEVANCE_WEIGHT = 0.4
    TERM_FREQUENCY_WEIGHT = 0.7
    TERM_DIVERSITY_WEIGHT = 0.3

    def __init__(
        self,
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

        self.pickle_path = Path(pickle_path) if pickle_path else None

        # Try to load from pickle if auto_load is enabled
        if auto_load and self.pickle_path and self.pickle_path.exists():
            logger.info(f"Loading KnowledgeSearchCore from pickle: {self.pickle_path}")
            self.load(self.pickle_path)
            init_time = time.time() - start_time
            logger.info(f"KnowledgeSearchCore initialization from pickle took {init_time:.3f}s")
            return

        # Otherwise, initialize from linked_knowledge
        if linked_knowledge is None:
            raise ValueError("linked_knowledge is required when not loading from pickle")

        self.linked_knowledge = linked_knowledge

        # Initialize embeddings and vector store
        self._embeddings = EncoderEmbeddings()
        self._vector_store = InMemoryVectorStore(embedding=self._embeddings)

        # Mappings for efficient lookups
        self._term_to_embedding_id: Dict[str, str] = {}
        self._embedding_id_to_term: Dict[str, str] = {}
        self._acronym_to_full_form: Dict[str, str] = {}
        self._full_form_to_acronym: Dict[str, str] = {}

        self._term_to_documents_index = linked_knowledge.term_to_documents_index
        self._document_to_terms_index = linked_knowledge.document_to_terms_index

        # Build acronym mappings from terms
        for term_key, term in linked_knowledge.terms.items():
            if term.term_type == "acronym" and term.full_form:
                self._acronym_to_full_form[term.term] = term.full_form
                self._full_form_to_acronym[term.full_form] = term.term

        # Initialize search index with chunks and terms
        self._initialize_search_index()

        init_time = time.time() - start_time
        logger.info(f"KnowledgeSearchCore initialization took {init_time:.3f}s")

    def _initialize_search_index(self) -> None:
        """Initialize the search index with chunks."""
        documents = []

        # Index all chunks
        for (
            doc_id,
            chunk_ids,
        ) in self.linked_knowledge.document_to_chunk_ids_index.items():
            # Get chunks from the flattened chunks dict
            for chunk_id in chunk_ids:
                chunk = self.linked_knowledge.chunks[chunk_id]
                metadata = {
                    "document_id": doc_id,
                    "document_name": chunk.document_name,
                    "page": chunk.page,
                    "chunk_index": chunk.index,
                    "chunk_id": chunk_id,
                }
                doc = Document(page_content=chunk.text, metadata=metadata)
                documents.append(doc)

        # Add all documents to vector store at once
        if documents:
            self._vector_store.add_documents(documents)

    def _search(
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
            # LangChain returns distance, we want similarity (1 - distance for cosine)
            similarity_score = 1 - score
            if similarity_score >= threshold:
                result = SearchResult(
                    text=doc.page_content,
                    score=similarity_score,
                    doc_id=doc.metadata.get("chunk_id", ""),
                    metadata=doc.metadata,
                )
                search_results.append(result)

        return search_results

    def search_similarity(
        self,
        query: str,
        k: int = DEFAULT_K_RESULTS,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> List[SimilaritySearchResult]:
        """
        Perform similarity-based search with media content.

        This method performs vector similarity search and returns results with
        associated images and tables from the pages.

        Args:
            query: Search query text
            k: Number of results to return
            threshold: Minimum similarity threshold

        Returns:
            List of similarity search results with media content
        """
        start_time = time.time()
        logger.info(f"Performing similarity search for query: '{query}'")

        try:
            # Perform vector search
            search_results = self._search(query, k=k, threshold=threshold)

            # Convert to similarity search results
            similarity_results = []
            for result in search_results:
                sim_result = SimilaritySearchResult(
                    text=result.text,
                    score=result.score,
                    document_id=result.metadata.get("document_id", ""),
                    document_name=result.metadata.get("document_name", ""),
                    page=result.metadata.get("page"),
                    chunk_index=result.metadata.get("chunk_index"),
                    metadata=result.metadata,
                )

                # Add images and tables from the page if available
                if sim_result.page and sim_result.document_id:
                    self._add_page_content(sim_result, sim_result.document_id, sim_result.page)

                similarity_results.append(sim_result)

            search_time = time.time() - start_time
            logger.info(f"Similarity search took {search_time:.3f}s, returned {len(similarity_results)} results")
            return similarity_results

        except Exception:
            search_time = time.time() - start_time
            logger.exception(f"Similarity search failed after {search_time:.3f}s")
            raise

    def search_enhanced(
        self,
        query: str,
        k: int = DEFAULT_K_RESULTS,
        threshold: float = DEFAULT_THRESHOLD,
        max_depth: int = DEFAULT_MAX_DEPTH,
        max_cooccurrences: int = DEFAULT_MAX_COOCCURRENCES,
    ) -> List[KnowledgeSearchResult]:
        """
        Perform enhanced search with term frequency analysis and statistics.

        This method:
        1. Performs vector similarity search
        2. Extracts terms from results and resolves relationships
        3. Analyzes term frequencies in the query
        4. Calculates relevance scores based on term statistics
        5. Sorts results using both similarity and term relevance

        Args:
            query: Search query text
            k: Number of results to return
            threshold: Minimum similarity threshold
            max_depth: Maximum depth for exploring related terms
            max_cooccurrences: Maximum number of co-occurring terms to include

        Returns:
            List of enhanced search results with term statistics and relevance scores
        """
        start_time = time.time()
        logger.info(f"Performing enhanced search for query: '{query}'")

        try:
            # Perform vector search
            search_results = self._search(query, k=k, threshold=threshold)

            # Convert to enhanced search results with term analysis
            enhanced_results = []
            query_lower = query.lower()

            for result in search_results:
                enh_result = KnowledgeSearchResult(
                    text=result.text,
                    score=result.score,
                    document_id=result.metadata.get("document_id", ""),
                    document_name=result.metadata.get("document_name", ""),
                    page=result.metadata.get("page"),
                    chunk_index=result.metadata.get("chunk_index"),
                    metadata=result.metadata,
                )

                # Extract primary terms from the result text
                result_text_lower = enh_result.text.lower()
                for term_key, term in self.linked_knowledge.terms.items():
                    if term_key.lower() in result_text_lower:
                        enh_result.primary_terms.append(term)
                        enh_result.all_terms.add(term.term)

                # Resolve co-occurrences and links for all primary terms
                for term in enh_result.primary_terms:
                    # Add linked terms (acronym-keyword relationships)
                    for link in self.linked_knowledge.links:
                        linked_term_key = None
                        if term.term_type == "acronym" and link.acronym == term.term:
                            linked_term_key = link.keyword
                        elif term.term_type == "keyword" and link.keyword == term.term:
                            linked_term_key = link.acronym

                        if linked_term_key and linked_term_key in self.linked_knowledge.terms:
                            linked_term = self.linked_knowledge.terms[linked_term_key]
                            if linked_term not in enh_result.related_terms:
                                enh_result.related_terms.append(linked_term)
                                enh_result.all_terms.add(linked_term.term)

                    # Add co-occurring terms
                    if term.cooccurrences and max_cooccurrences > 0:
                        for cooccurrence in term.cooccurrences[:max_cooccurrences]:
                            if cooccurrence.term in self.linked_knowledge.terms:
                                cooccurring_term = self.linked_knowledge.terms[cooccurrence.term]
                                if cooccurring_term not in enh_result.related_terms:
                                    enh_result.related_terms.append(cooccurring_term)
                                    enh_result.all_terms.add(cooccurring_term.term)

                # Calculate term frequencies in the query
                for term_str in enh_result.all_terms:
                    term_lower = term_str.lower()
                    # Count occurrences of this term in the query
                    count = query_lower.count(term_lower)
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

                # Add images and tables from the page if available
                if enh_result.page and enh_result.document_id:
                    self._add_page_content(enh_result, enh_result.document_id, enh_result.page)

                enhanced_results.append(enh_result)

            # Sort results by combined score (similarity + term relevance)
            enhanced_results.sort(
                key=lambda r: (
                    r.score * self.SIMILARITY_WEIGHT + min(r.term_relevance_score / 10, self.TERM_RELEVANCE_WEIGHT)
                ),
                reverse=True,
            )

            search_time = time.time() - start_time
            logger.info(f"Enhanced search took {search_time:.3f}s, returned {len(enhanced_results)} results")
            return enhanced_results

        except Exception:
            search_time = time.time() - start_time
            logger.exception(f"Enhanced search failed after {search_time:.3f}s")
            raise

    def _create_simple_result(self, search_result: SearchResult) -> Optional[KnowledgeSearchResult]:
        """
        Create a simple result with just media (images/tables), no term enrichment.

        Args:
            search_result: SearchResult from vector search

        Returns:
            Simple search result with media or None if invalid
        """
        text = search_result.text
        score = search_result.score
        metadata = search_result.metadata

        result = KnowledgeSearchResult(
            text=text,
            score=score,
            document_id=metadata.get("document_id", ""),
            document_name=metadata.get("document_name", ""),
            page=metadata.get("page"),
            chunk_index=metadata.get("chunk_index"),
            metadata=metadata,
        )

        # Add images and tables from the page if available
        if result.page and result.document_id:
            self._add_page_content(result, result.document_id, result.page)

        return result

    def _create_knowledge_result_from_search(
        self,
        search_result: SearchResult,
        original_query: str,
        max_depth: int,
        max_cooccurrences: int,
    ) -> Optional[KnowledgeSearchResult]:
        """
        Convert a SearchResult into a knowledge-enhanced result.

        Args:
            search_result: SearchResult from vector search
            original_query: Original search query
            max_depth: Maximum relationship depth to explore
            max_cooccurrences: Maximum co-occurrences to include

        Returns:
            Enhanced knowledge search result or None if invalid
        """
        # Convert SearchResult to dict format for backward compatibility
        raw_result = {
            "text": search_result.text,
            "score": search_result.score,
            "doc_id": search_result.doc_id,
            "metadata": search_result.metadata,
        }
        return self._create_chunk_result(raw_result, max_depth, max_cooccurrences)

    def _create_chunk_result(
        self,
        raw_result: Dict[str, Any],
        max_depth: int,
        max_cooccurrences: int,
    ) -> KnowledgeSearchResult:
        """Create a knowledge result from a document chunk."""
        metadata = raw_result["metadata"]

        result = KnowledgeSearchResult(
            text=raw_result["text"],
            score=raw_result["score"],
            document_id=metadata["document_id"],
            document_name=metadata["document_name"],
            page=metadata.get("page"),
            chunk_index=metadata.get("chunk_index"),
            metadata=metadata,
        )

        # Find terms that appear in this chunk
        self._enhance_result_with_terms(result, max_depth, max_cooccurrences)

        return result

    def _enhance_result_with_terms(
        self,
        result: KnowledgeSearchResult,
        max_depth: int,
        max_cooccurrences: int,
    ) -> None:
        """
        Enhance a chunk result with relevant terms found in the text.

        Args:
            result: The result to enhance
            max_depth: Maximum relationship depth
            max_cooccurrences: Maximum co-occurrences to include
        """
        result_text = result.text.lower()

        # Find terms that appear in the result text
        for term_key, term in self.linked_knowledge.terms.items():
            if term_key.lower() in result_text:
                result.primary_terms.append(term)

                # Add co-occurring terms to related_terms
                if max_cooccurrences > 0 and term.cooccurrences:
                    for cooccurrence in term.cooccurrences[:max_cooccurrences]:
                        if cooccurrence.term in self.linked_knowledge.terms:
                            cooccurring_term = self.linked_knowledge.terms[cooccurrence.term]
                            if cooccurring_term not in result.related_terms:
                                result.related_terms.append(cooccurring_term)

        # Find related terms through links if max_depth > 1
        if max_depth > 1:
            self._add_related_terms(result, max_depth - 1)

        # Add images and tables from the page if available
        if result.page and result.document_id:
            self._add_page_content(result, result.document_id, result.page)

    def _enhance_result_with_relationships(
        self,
        result: KnowledgeSearchResult,
        term_key: str,
        max_cooccurrences: int,
    ) -> None:
        """
        Enhance a term result with its relationships and co-occurrences.

        Args:
            result: The result to enhance
            term_key: The primary term key
            max_cooccurrences: Maximum co-occurrences to include
        """
        term = self.linked_knowledge.terms.get(term_key)
        if not term:
            return

        # Add linked terms (acronym-keyword relationships)
        for link in self.linked_knowledge.links:
            if link.acronym == term_key or link.keyword == term_key:
                linked_term_key = link.keyword if link.acronym == term_key else link.acronym
                if linked_term_key in self.linked_knowledge.terms:
                    result.related_terms.append(self.linked_knowledge.terms[linked_term_key])

        # Add co-occurring terms to the end of related_terms
        if term.cooccurrences and max_cooccurrences > 0:
            # Convert TermCooccurrence to Term objects and add to related_terms
            for cooccurrence in term.cooccurrences[:max_cooccurrences]:
                if cooccurrence.term in self.linked_knowledge.terms:
                    cooccurring_term = self.linked_knowledge.terms[cooccurrence.term]
                    if cooccurring_term not in result.related_terms:
                        result.related_terms.append(cooccurring_term)

    def _add_related_terms(self, result: KnowledgeSearchResult, remaining_depth: int) -> None:
        """
        Recursively add related terms up to the specified depth.

        Args:
            result: The result to enhance
            remaining_depth: Remaining depth for recursion
        """
        if remaining_depth <= 0:
            return

        current_terms = [term for term in result.primary_terms]

        for term in current_terms:
            # Find linked terms
            for link in self.linked_knowledge.links:
                linked_term_key = None
                if term.term_type == "acronym" and link.acronym == term.term:
                    linked_term_key = link.keyword
                elif term.term_type == "keyword" and link.keyword == term.term:
                    linked_term_key = link.acronym

                if linked_term_key and linked_term_key in self.linked_knowledge.terms:
                    linked_term = self.linked_knowledge.terms[linked_term_key]
                    if linked_term not in result.related_terms:
                        result.related_terms.append(linked_term)

    def _add_page_content(
        self,
        result: Union[KnowledgeSearchResult, SimilaritySearchResult, KnowledgeSearchResult],
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
        if page_key in self.linked_knowledge.pages:
            page_data = self.linked_knowledge.pages[page_key]

            # Images are already base64 strings in page_data
            result.images.extend(page_data.images)

            # Tables are already KnowledgeTableData objects
            result.tables.extend(page_data.tables)

    def persist(self, path: Optional[Union[str, Path]] = None) -> None:
        """
        Persist the complete KnowledgeSearchCore instance to a pickle file.

        This saves:
        - LinkedKnowledge structure with all documents, terms, chunks, and indices
        - Vector store with embeddings using LangChain's dump method
        - All internal mappings and indices

        Args:
            path: Path to save the pickle file. If None, uses self.pickle_path

        Raises:
            ValueError: If no path is provided and self.pickle_path is not set
        """
        import pickle

        save_path = Path(path) if path else self.pickle_path
        if not save_path:
            raise ValueError("No path provided for persistence")

        # Create parent directory if it doesn't exist
        save_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Persisting KnowledgeSearchCore to {save_path}")

        # Create a serializable state object with the vector store directly
        state = {
            "linked_knowledge": self.linked_knowledge,
            "vector_store": self._vector_store.store,  # Store the vector store data directly
            "term_to_embedding_id": self._term_to_embedding_id,
            "embedding_id_to_term": self._embedding_id_to_term,
            "acronym_to_full_form": self._acronym_to_full_form,
            "full_form_to_acronym": self._full_form_to_acronym,
            "term_to_documents_index": self._term_to_documents_index,
            "document_to_terms_index": self._document_to_terms_index,
        }

        # Save to pickle
        with open(save_path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info(f"Successfully persisted KnowledgeSearchCore to {save_path}")

        # Update internal pickle_path
        self.pickle_path = save_path

    def load(self, path: Union[str, Path]) -> None:
        """
        Load a KnowledgeSearchCore instance from a pickle file.

        This restores:
        - LinkedKnowledge structure with all documents, terms, chunks, and indices
        - Vector store with embeddings using LangChain's load method
        - All internal mappings and indices

        Args:
            path: Path to load the pickle file from

        Raises:
            FileNotFoundError: If the pickle file doesn't exist
            ValueError: If the pickle file is corrupted or incompatible
        """
        import pickle

        load_path = Path(path)
        if not load_path.exists():
            raise FileNotFoundError(f"Pickle file not found: {load_path}")

        logger.info(f"Loading KnowledgeSearchCore from {load_path}")

        try:
            with open(load_path, "rb") as f:
                state = pickle.load(f)

            # Restore all state
            self.linked_knowledge = state["linked_knowledge"]

            # Initialize embeddings
            self._embeddings = EncoderEmbeddings()

            # Check for vector store in state
            if "vector_store" in state:
                # Load the vector store directly from state
                self._vector_store = InMemoryVectorStore(embedding=self._embeddings)
                self._vector_store.store = state["vector_store"]
            else:
                # Fallback: reinitialize vector store (for backward compatibility)
                logger.warning("Old pickle format detected, reinitializing vector store")
                self._vector_store = InMemoryVectorStore(embedding=self._embeddings)
                self._initialize_search_index()

            self._term_to_embedding_id = state["term_to_embedding_id"]
            self._embedding_id_to_term = state["embedding_id_to_term"]
            self._acronym_to_full_form = state["acronym_to_full_form"]
            self._full_form_to_acronym = state["full_form_to_acronym"]
            self._term_to_documents_index = state["term_to_documents_index"]
            self._document_to_terms_index = state["document_to_terms_index"]

            logger.info(f"Successfully loaded KnowledgeSearchCore from {load_path}")
            logger.info(f"  - Documents: {len(self.linked_knowledge.documents)}")
            logger.info(f"  - Terms: {len(self.linked_knowledge.terms)}")
            logger.info(f"  - Chunks: {len(self.linked_knowledge.chunks)}")

        except Exception as e:
            raise ValueError(f"Failed to load pickle file: {e}") from e

        # Update internal pickle_path
        self.pickle_path = load_path

    @classmethod
    def from_pickle(cls, path: Union[str, Path]) -> "KnowledgeSearchCore":
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

        instance = cls(linked_knowledge=None, pickle_path=path, auto_load=True)
        return instance
