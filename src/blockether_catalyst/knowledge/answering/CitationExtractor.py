"""
Citation extraction utilities to ensure citations use actual search result metadata.

This module provides programmatic extraction of citations from search results,
ensuring that only real metadata is used and preventing AI hallucination of
citation information.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

from blockether_catalyst.knowledge.KnowledgeTypes import CompactSearchResult, OptimizedSearchResponse
from blockether_catalyst.knowledge.answering.AnswerProviderAgent import Citation, ImageAttachment, TableAttachment


class CitationExtractor:
    """Extracts citations from search results with guaranteed metadata integrity."""

    @staticmethod
    def extract_citation_from_result(
        result: CompactSearchResult,
        quote: Optional[str] = None
    ) -> Citation:
        """
        Convert a search result into a citation, using ONLY actual metadata.

        Args:
            result: The search result containing actual document metadata
            quote: Optional quote extracted from the result's content

        Returns:
            Citation with actual metadata from search result
        """
        # Extract images if available
        images = None
        if result.images:
            images = [
                ImageAttachment(
                    caption=img.caption,
                    href=img.href,
                    page=img.page,
                    document_name=result.document_name,
                    score=result.score
                )
                for img in result.images
            ]

        # Extract tables if available
        tables = None
        if result.tables:
            tables = [
                TableAttachment(
                    content=table.content,
                    caption=f"Table from page {table.page or 'unknown'}",  # Default caption since TableInfo doesn't have one
                    page=table.page or 0,
                    document_name=result.document_name,
                    score=result.score
                )
                for table in result.tables
            ]

        # Build href with page anchor if both href and page are available
        href_with_page = result.href
        if result.href and result.page:
            # Check if href already has a page anchor
            if '#page=' not in result.href:
                href_with_page = f"{result.href}#page={result.page}"
            # Otherwise use the href as-is (it already has page info)

        return Citation(
            title=result.document_name,  # Use actual document name
            author=result.author,  # Will be None if not in search results
            publication_date=result.publication_date,  # Will be None if not in search results
            page=result.page or 0,  # Use actual page or default to 0
            href=href_with_page,  # Include page anchor in URL if available
            quote=quote,  # Optional quote from content
            images=images,
            tables=tables
        )

    @staticmethod
    def extract_all_citations(
        search_response: OptimizedSearchResponse,
        max_citations: int = 10
    ) -> List[Citation]:
        """
        Extract all citations from search results.

        Args:
            search_response: The optimized search response
            max_citations: Maximum number of citations to extract

        Returns:
            List of citations with actual metadata
        """
        citations = []

        for result in search_response.results[:max_citations]:
            citation = CitationExtractor.extract_citation_from_result(result)
            citations.append(citation)

        return citations

    @staticmethod
    def create_citation_context(
        search_response: OptimizedSearchResponse,
        max_citations: int = 10
    ) -> str:
        """
        Create a formatted string of available citations for the AI to reference.

        This provides the AI with a clear list of available citations that it
        can reference by index, ensuring it only uses real citations.

        Args:
            search_response: The optimized search response
            max_citations: Maximum number of citations to include

        Returns:
            Formatted string of available citations
        """
        citations = CitationExtractor.extract_all_citations(search_response, max_citations)

        if not citations:
            return "No citations available from search results."

        context_lines = ["Available citations from search results:"]

        for i, citation in enumerate(citations, 1):
            # Build citation info
            cite_info = f"[{i}] {citation.title}"

            if citation.page:
                cite_info += f" (page {citation.page})"

            if citation.author:
                cite_info += f" by {citation.author}"

            if citation.publication_date:
                cite_info += f" ({citation.publication_date})"

            if citation.href:
                cite_info += f"\n    URL: {citation.href}"

            context_lines.append(cite_info)

        return "\n".join(context_lines)

    @staticmethod
    def validate_citation(citation: Citation, search_response: OptimizedSearchResponse) -> bool:
        """
        Validate that a citation actually comes from search results.

        Args:
            citation: The citation to validate
            search_response: The search response to validate against

        Returns:
            True if citation matches a search result, False otherwise
        """
        for result in search_response.results:
            # Build the expected href with page anchor for comparison
            expected_href = result.href
            if result.href and result.page and '#page=' not in result.href:
                expected_href = f"{result.href}#page={result.page}"

            # Check if citation matches, accounting for page-anchored URLs
            href_matches = (citation.href == result.href or
                           citation.href == expected_href or
                           (citation.href and result.href and
                            citation.href.split('#')[0] == result.href.split('#')[0]))

            if (citation.title == result.document_name and
                citation.author == result.author and
                citation.publication_date == result.publication_date and
                href_matches and
                citation.page == (result.page or 0)):
                return True

        return False

    @staticmethod
    def sanitize_citations(
        citations: List[Citation],
        search_response: OptimizedSearchResponse
    ) -> List[Citation]:
        """
        Filter out any citations that don't match actual search results.

        This is a safety mechanism to catch any hallucinated citations
        that might slip through.

        Args:
            citations: List of citations to sanitize
            search_response: The search response to validate against

        Returns:
            List of validated citations only
        """
        valid_citations = []

        for citation in citations:
            if CitationExtractor.validate_citation(citation, search_response):
                valid_citations.append(citation)

        return valid_citations


class CitationMapper:
    """Maps between search results and citations with guaranteed consistency."""

    def __init__(self, search_response: OptimizedSearchResponse):
        """
        Initialize the citation mapper with search results.

        Args:
            search_response: The search response to map citations from
        """
        self.search_response = search_response
        self._citation_cache: Dict[str, Citation] = {}
        self._build_citation_cache()

    def _build_citation_cache(self) -> None:
        """Build a cache of citations from search results."""
        for result in self.search_response.results:
            key = self._get_result_key(result)
            self._citation_cache[key] = CitationExtractor.extract_citation_from_result(result)

    def _get_result_key(self, result: CompactSearchResult) -> str:
        """Generate a unique key for a search result."""
        return f"{result.document_name}_{result.page or 0}_{hash(result.content[:100])}"

    def get_citation_for_content(
        self,
        content_snippet: str,
        quote: Optional[str] = None
    ) -> Optional[Citation]:
        """
        Get the appropriate citation for a content snippet.

        Args:
            content_snippet: A snippet of content to find the citation for
            quote: Optional quote to include in the citation

        Returns:
            Citation if found, None otherwise
        """
        # Find the matching search result
        for result in self.search_response.results:
            if content_snippet in result.content:
                citation = CitationExtractor.extract_citation_from_result(result, quote)
                return citation

        return None

    def get_all_available_citations(self) -> List[Citation]:
        """Get all available citations from the search results."""
        return list(self._citation_cache.values())

    def format_for_agent_input(self) -> str:
        """
        Format citations for inclusion in agent input.

        Returns:
            Formatted string of citations for the agent to use
        """
        return CitationExtractor.create_citation_context(self.search_response)