"""
Internal components for the Knowledge Search system.

This module contains supporting classes and protocols for the KnowledgeSearchCore,
including term management, scoring strategies, and value objects.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Set, Tuple

from pydantic import BaseModel, Field

from blockether_catalyst.knowledge.KnowledgeTypes import (
    CompactSearchResult,
    KnowledgeSearchResult,
    OptimizedSearchResponse,
    TermInfo,
    TermWithLinks,
)


@dataclass(frozen=True)
class ChunkReference:
    """Represents a reference to a document chunk."""

    document_id: str
    chunk_index: int

    def __hash__(self) -> int:
        """Make ChunkReference hashable for use in sets."""
        return hash((self.document_id, self.chunk_index))

    def __eq__(self, other: object) -> bool:
        """Check equality based on document_id and chunk_index."""
        if not isinstance(other, ChunkReference):
            return False
        return self.document_id == other.document_id and self.chunk_index == other.chunk_index


@dataclass(frozen=True)
class TermReference:
    """Reference to a term in the registry."""

    term_key: str
    score: float = 0.0
    is_primary: bool = False


class SearchContext(BaseModel):
    """Context object containing all search-related data."""

    query: str
    query_lower: str
    query_words: Set[str]
    query_acronyms: Set[str]
    query_keywords: Set[str]
    top_keywords: List[Tuple[str, float]]
    top_acronyms: List[Tuple[str, float]]
    k: int
    threshold: float
    max_depth: int
    max_cooccurrences: int


class StoredTerm(BaseModel):
    """
    The ONLY place where full term data lives.
    This is the single source of truth for term information.
    """

    term_key: str = Field(description="Unique identifier for the term")
    term: str = Field(description="The actual term text")
    meaning: Optional[str] = Field(default=None, description="Term meaning/definition")
    term_type: Optional[str] = Field(default=None, description="Type: acronym, keyword, etc")
    total_occurrences: int = Field(default=0, description="Total times term appears in knowledge base")
    linked_term_keys: List[str] = Field(default_factory=list, description="Keys of linked terms (NOT objects!)")
    link_scores: Dict[str, float] = Field(default_factory=dict, description="Mapping of linked term key to link score")


class TermRegistry:
    """
    Centralized registry for managing terms without duplication.

    This class ensures that each unique term is stored ONLY ONCE and
    provides efficient lookups via term keys. No nested objects, only references!
    """

    def __init__(self):
        """Initialize the term registry."""
        self._stored_terms: Dict[str, StoredTerm] = {}
        self._term_to_results: Dict[str, Set[str]] = {}  # term_key -> result_ids
        self._raw_term_cache: Dict[str, TermWithLinks] = {}  # Optional cache for raw terms

    def register_term(
        self,
        term: TermWithLinks,
        linked_knowledge_terms: Optional[Dict[str, TermWithLinks]] = None,
    ) -> str:
        """
        Register a term in the registry, storing it ONLY ONCE.

        Args:
            term: The term to register
            linked_knowledge_terms: Optional dict of all terms (for resolving links)

        Returns:
            The term key for reference
        """
        term_key = term.term

        # If already registered, return the key
        if term_key in self._stored_terms:
            return term_key

        # Extract ONLY the keys of linked terms (NOT objects!)
        linked_keys = []
        link_scores = {}

        for link in term.links:
            if link.link_to:  # Only add valid links
                linked_keys.append(link.link_to)
                link_scores[link.link_to] = link.score

        # Store the term ONCE with only references
        self._stored_terms[term_key] = StoredTerm(
            term_key=term_key,
            term=term.term,
            meaning=term.meaning,
            term_type=term.type if hasattr(term, "type") else None,
            total_occurrences=term.total if hasattr(term, "total") else 0,
            linked_term_keys=linked_keys,  # KEYS ONLY!
            link_scores=link_scores,
        )

        # Optionally cache the raw term for special cases
        if linked_knowledge_terms:
            self._raw_term_cache[term_key] = term

        return term_key

    def add_result_reference(self, term_key: str, result_id: str) -> None:
        """
        Track which results reference which terms.

        Args:
            term_key: The term key
            result_id: Identifier for the search result
        """
        if term_key not in self._term_to_results:
            self._term_to_results[term_key] = set()
        self._term_to_results[term_key].add(result_id)

    def get_stored_term(self, term_key: str) -> Optional[StoredTerm]:
        """
        Get a stored term by its key.

        Args:
            term_key: The term key to look up

        Returns:
            StoredTerm if found, None otherwise
        """
        return self._stored_terms.get(term_key)

    def get_terms_for_response(self) -> Dict[str, TermInfo]:
        """
        Convert stored terms to TermInfo format for API response.
        This maintains backward compatibility while using our optimized storage.

        Returns:
            Dictionary mapping term keys to TermInfo objects
        """
        result = {}
        for term_key, stored_term in self._stored_terms.items():
            # Convert StoredTerm to TermInfo (with linked term KEYS only)
            result[term_key] = TermInfo(
                term=stored_term.term,
                meaning=stored_term.meaning,
                term_type=stored_term.term_type,
                total_times_occurred_in_knowledgebase=stored_term.total_occurrences,
                linked_terms=stored_term.linked_term_keys,  # Just keys, not objects!
                link_scores=stored_term.link_scores,  # Include the link scores dictionary
            )
        return result

    def get_referenced_terms(self, result_id: str) -> List[str]:
        """
        Get all term keys referenced by a specific result.

        Args:
            result_id: The result identifier

        Returns:
            List of term keys referenced by this result
        """
        return [term_key for term_key, result_ids in self._term_to_results.items() if result_id in result_ids]


class ScoringStrategy(Protocol):
    """Protocol for scoring strategies."""

    def calculate_score(
        self,
        base_score: float,
        term_boost: float,
        term_relevance: float,
    ) -> float:
        """
        Calculate the final score for a search result.

        Args:
            base_score: Base similarity score
            term_boost: Boost from term presence
            term_relevance: Relevance based on term frequency/diversity

        Returns:
            Final calculated score
        """
        ...


class WeightedScoringStrategy:
    """Default weighted scoring strategy."""

    def __init__(
        self,
        similarity_weight: float = 0.6,
        term_relevance_weight: float = 0.4,
    ):
        """
        Initialize the scoring strategy.

        Args:
            similarity_weight: Weight for base similarity score
            term_relevance_weight: Weight for term relevance score
        """
        self._similarity_weight = similarity_weight
        self._term_relevance_weight = term_relevance_weight

    def calculate_score(
        self,
        base_score: float,
        term_boost: float,
        term_relevance: float,
    ) -> float:
        """
        Calculate weighted score.

        Args:
            base_score: Base similarity score
            term_boost: Boost from term presence
            term_relevance: Relevance based on term frequency/diversity

        Returns:
            Final weighted score
        """
        boosted_score = base_score + term_boost
        return boosted_score * self._similarity_weight + term_relevance * self._term_relevance_weight


class SearchResultBuilder:
    """Builder for constructing search results efficiently."""

    def __init__(
        self,
        term_registry: TermRegistry,
        resources_base_url: Optional[str] = None,
    ):
        """
        Initialize the result builder.

        Args:
            term_registry: Registry for term management
            resources_base_url: Base URL for resources
        """
        self._term_registry = term_registry
        self._resources_base_url = resources_base_url
        self._results: List[CompactSearchResult] = []

    def add_result(
        self,
        result: KnowledgeSearchResult,
        max_primary_terms: int = 5,
        max_related_terms: int = 3,
    ) -> "SearchResultBuilder":
        """
        Add a processed result to the builder.

        Args:
            result: The KnowledgeSearchResult to add
            max_primary_terms: Maximum primary terms to include
            max_related_terms: Maximum related terms to include

        Returns:
            Self for chaining
        """
        from urllib.parse import quote

        from blockether_catalyst.knowledge.KnowledgeTypes import ImageInfo, TableInfo

        # Build document href
        doc_href = ""
        if self._resources_base_url and result.document_path:
            encoded_path = quote(result.document_path, safe="/")
            doc_href = f"{self._resources_base_url}/{encoded_path}"

        # Convert images
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

        # Convert tables with intelligent caption generation
        tables = []
        for table in result.tables:
            content = table.to_markdown()

            # Use table's caption from extraction phase (mandatory field)
            caption = table.caption

            tables.append(
                TableInfo(
                    content=content,
                    page=table.page,
                    caption=caption,
                )
            )

        # Extract term keys
        primary_term_keys = [term.term for term in result.primary_terms[:max_primary_terms]]
        related_term_keys = [term.term for term in result.related_terms[:max_related_terms]]

        # Track references
        result_id = f"{result.document_id}_p{result.page}_c{result.chunk_index}"
        for term_key in primary_term_keys + related_term_keys:
            self._term_registry.add_result_reference(term_key, result_id)

        # Create compact result
        compact_result = CompactSearchResult(
            score=result.score,
            content=result.text,
            document_name=result.document_name,
            page=result.page if result.page else None,
            author=result.metadata.author if result.metadata else None,
            publication_date=result.metadata.publication_date if result.metadata else None,
            modified_date=result.metadata.modified_date if result.metadata else None,
            href=doc_href if doc_href else None,
            primary_term_keys=primary_term_keys,
            related_term_keys=related_term_keys,
            images=images,
            tables=tables,
        )

        self._results.append(compact_result)
        return self

    def build(self, query: str) -> OptimizedSearchResponse:
        """
        Build the final OptimizedSearchResponse.

        Args:
            query: The original search query

        Returns:
            Complete OptimizedSearchResponse
        """
        return OptimizedSearchResponse(
            results=self._results,
            terms=self._term_registry.get_terms_for_response(),
            total_results=len(self._results),
        )
