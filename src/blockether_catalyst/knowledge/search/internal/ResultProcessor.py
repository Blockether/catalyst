"""
Result processing component for the Knowledge Search system.

This module handles the processing of raw search results into optimized formats,
using term references instead of duplicating term data.
"""

from typing import Dict, List, Set

from blockether_catalyst.knowledge.KnowledgeTypes import (
    SearchResult,
    SearchResultMetadata,
    TermWithLinks,
)

from .SearchComponents import (
    ChunkReference,
    SearchContext,
    StoredTerm,
    TermReference,
    TermRegistry,
)


class ResultProcessor:
    """
    Process raw search results into optimized format using term references.

    This class is responsible for:
    - Converting raw results to use term references instead of full objects
    - Classifying terms as primary or related
    - Managing term registration in the registry
    - Avoiding any duplication of term data
    """

    MAX_PRIMARY_TERMS_PER_RESULT = 10
    MAX_RELATED_TERMS_PER_RESULT = 5

    def __init__(
        self,
        term_registry: TermRegistry,
        linked_knowledge_terms: Dict[str, TermWithLinks],
    ):
        """
        Initialize the result processor.

        Args:
            term_registry: Registry for term deduplication
            linked_knowledge_terms: Dictionary of all terms in linked knowledge
        """
        self._term_registry = term_registry
        self._linked_knowledge_terms = linked_knowledge_terms

    def process_single_result(
        self, result: SearchResult, context: SearchContext, result_id: str
    ) -> List[TermReference]:
        """
        Process a single search result and extract term references.

        Args:
            result: The raw search result to process
            context: Search context with query information
            result_id: Unique identifier for this result

        Returns:
            List of TermReference objects (NOT full terms!)
        """
        term_references = []
        processed_terms = set()  # Track processed terms to avoid duplicates

        # Get terms from the chunk metadata
        chunk_terms: List[str] = list(result.metadata.terms.keys()) if result.metadata and result.metadata.terms else []

        primary_count = 0
        related_count = 0

        for term_key in chunk_terms:
            # Skip if already processed
            if term_key in processed_terms:
                continue

            # Skip if term doesn't exist in knowledge base
            if term_key not in self._linked_knowledge_terms:
                continue

            term = self._linked_knowledge_terms[term_key]
            term_lower = term.term.lower()

            # Classify term as primary or related
            is_primary = self._is_primary_term(term_lower, context)

            if is_primary and primary_count < self.MAX_PRIMARY_TERMS_PER_RESULT:
                # Register the term ONCE in the registry
                registered_key = self._term_registry.register_term(term, self._linked_knowledge_terms)

                # Create a lightweight reference
                term_ref = TermReference(
                    term_key=registered_key,
                    score=self._calculate_term_score(term_lower, context),
                    is_primary=True,
                )
                term_references.append(term_ref)
                primary_count += 1

            elif not is_primary and related_count < self.MAX_RELATED_TERMS_PER_RESULT:
                # Register the term ONCE in the registry
                registered_key = self._term_registry.register_term(term, self._linked_knowledge_terms)

                # Create a lightweight reference
                term_ref = TermReference(
                    term_key=registered_key,
                    score=self._calculate_term_score(term_lower, context),
                    is_primary=False,
                )
                term_references.append(term_ref)
                related_count += 1

            processed_terms.add(term_key)

            # Track that this result references this term
            self._term_registry.add_result_reference(term_key, result_id)

        # Sort by score and primary status
        term_references.sort(key=lambda ref: (ref.is_primary, ref.score), reverse=True)

        return term_references

    def _is_primary_term(self, term_lower: str, context: SearchContext) -> bool:
        """
        Determine if a term is primary based on the search context.

        Args:
            term_lower: Lowercase version of the term
            context: Search context with query information

        Returns:
            True if the term is primary, False if it's related
        """
        # Check if term appears in query
        if term_lower in context.query_lower:
            return True

        # Check if term is in top keywords or acronyms
        if term_lower in context.query_keywords or term_lower in context.query_acronyms:
            return True

        # Check if term is in the top extracted terms
        for keyword, _ in context.top_keywords:
            if keyword.lower() == term_lower:
                return True

        for acronym, _ in context.top_acronyms:
            if acronym.lower() == term_lower:
                return True

        return False

    def _calculate_term_score(self, term_lower: str, context: SearchContext) -> float:
        """
        Calculate a relevance score for a term based on the search context.

        Args:
            term_lower: Lowercase version of the term
            context: Search context with query information

        Returns:
            Relevance score for the term
        """
        score = 0.0

        # Base score for appearing in query
        if term_lower in context.query_lower:
            score += 1.0
            # Additional score based on frequency
            score += context.query_lower.count(term_lower) * 0.1

        # Score for being a top keyword
        for keyword, keyword_score in context.top_keywords:
            if keyword.lower() == term_lower:
                score += keyword_score * 0.5
                break

        # Score for being a top acronym
        for acronym, acronym_score in context.top_acronyms:
            if acronym.lower() == term_lower:
                score += acronym_score * 0.7  # Acronyms get higher weight
                break

        return score

    def process_batch(self, results: List[SearchResult], context: SearchContext) -> Dict[str, List[TermReference]]:
        """
        Process a batch of search results.

        Args:
            results: List of raw search results
            context: Search context with query information

        Returns:
            Dictionary mapping result IDs to lists of term references
        """
        result_term_refs = {}

        for idx, result in enumerate(results):
            # Generate unique result ID
            result_id = f"{result.metadata.document_id}_p{result.metadata.page}_c{result.metadata.chunk_index}"

            # Process the result
            term_refs = self.process_single_result(result, context, result_id)
            result_term_refs[result_id] = term_refs

        return result_term_refs
