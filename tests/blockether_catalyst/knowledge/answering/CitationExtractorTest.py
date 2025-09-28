"""
Comprehensive tests for CitationExtractor to ensure citation integrity.

Tests cover:
- Citation extraction from search results
- Page anchor URL generation
- Citation validation and sanitization
- Integration with real agents
"""

import pytest
from typing import List
from unittest.mock import Mock, patch, AsyncMock

from blockether_catalyst.knowledge.KnowledgeTypes import (
    CompactSearchResult,
    OptimizedSearchResponse,
    ImageInfo,
    TableInfo,
)
from blockether_catalyst.knowledge.answering.CitationExtractor import (
    CitationExtractor,
    CitationMapper,
)
from blockether_catalyst.knowledge.answering.AnswerProviderAgent import (
    Citation,
    ImageAttachment,
    TableAttachment,
)


class TestCitationExtractor:
    """Test citation extraction from search results."""

    def test_extract_citation_with_full_metadata(self):
        """Test extracting citation when all metadata is present."""
        # Create a search result with full metadata
        result = CompactSearchResult(
            score=0.95,
            content="The risk limit for corporate clients is 1M EUR as per policy.",
            document_name="Risk_Management_Policy.pdf",
            page=45,
            author="RBI Risk Department",
            publication_date="2023-11-15",
            href="https://example.com/docs/Risk_Management_Policy.pdf",
            primary_term_keys=["risk", "limit"],
            related_term_keys=["corporate", "policy"],
            images=[],
            tables=[],
            modified_date=None,
        )

        # Extract citation
        citation = CitationExtractor.extract_citation_from_result(
            result,
            quote="The risk limit for corporate clients is 1M EUR"
        )

        # Verify all fields are correctly extracted
        assert citation.title == "Risk_Management_Policy.pdf"
        assert citation.author == "RBI Risk Department"
        assert citation.publication_date == "2023-11-15"
        assert citation.page == 45
        assert citation.href == "https://example.com/docs/Risk_Management_Policy.pdf#page=45"
        assert citation.quote == "The risk limit for corporate clients is 1M EUR"

    def test_extract_citation_with_missing_metadata(self):
        """Test extracting citation when some metadata is missing."""
        result = CompactSearchResult(
            score=0.85,
            content="LMS system handles limit management for the bank.",
            document_name="LMS_User_Manual.pdf",
            page=12,
            author=None,  # No author
            publication_date=None,  # No publication date
            href=None,  # No URL
            primary_term_keys=["lms", "limit"],
            related_term_keys=[],
            images=[],
            tables=[],
            modified_date=None,
        )

        citation = CitationExtractor.extract_citation_from_result(result)

        # Verify missing fields are None
        assert citation.title == "LMS_User_Manual.pdf"
        assert citation.author is None
        assert citation.publication_date is None
        assert citation.page == 12
        assert citation.href is None  # No URL means no page anchor
        assert citation.quote is None

    def test_page_anchor_generation(self):
        """Test that page anchors are correctly added to URLs."""
        # Test with URL and page
        result = CompactSearchResult(
            score=0.9,
            content="Content",
            document_name="Document.pdf",
            page=23,
            href="https://example.com/Document.pdf",
            author=None,
            publication_date=None,
            primary_term_keys=[],
            related_term_keys=[],
            images=[],
            tables=[],
            modified_date=None,
        )

        citation = CitationExtractor.extract_citation_from_result(result)
        assert citation.href == "https://example.com/Document.pdf#page=23"

    def test_page_anchor_not_duplicated(self):
        """Test that page anchors are not duplicated if already present."""
        result = CompactSearchResult(
            score=0.9,
            content="Content",
            document_name="Document.pdf",
            page=23,
            href="https://example.com/Document.pdf#page=23",  # Already has page anchor
            author=None,
            publication_date=None,
            primary_term_keys=[],
            related_term_keys=[],
            images=[],
            tables=[],
            modified_date=None,
        )

        citation = CitationExtractor.extract_citation_from_result(result)
        # Should not duplicate the page anchor
        assert citation.href == "https://example.com/Document.pdf#page=23"

    def test_no_page_anchor_without_page(self):
        """Test that page anchor is not added when page is None."""
        result = CompactSearchResult(
            score=0.9,
            content="Content",
            document_name="Document.pdf",
            page=None,  # No page number
            href="https://example.com/Document.pdf",
            author=None,
            publication_date=None,
            primary_term_keys=[],
            related_term_keys=[],
            images=[],
            tables=[],
            modified_date=None,
        )

        citation = CitationExtractor.extract_citation_from_result(result)
        # Should not add page anchor when page is None
        assert citation.href == "https://example.com/Document.pdf"

    def test_extract_citation_with_images(self):
        """Test extracting citation with images."""
        result = CompactSearchResult(
            score=0.95,
            content="Content with images",
            document_name="Document_With_Images.pdf",
            page=10,
            href="https://example.com/doc.pdf",
            author=None,
            publication_date=None,
            primary_term_keys=[],
            related_term_keys=[],
            images=[
                ImageInfo(
                    caption="Risk Matrix Diagram",
                    href="https://example.com/images/risk_matrix.png",
                    page=10,
                    document_name="Document_With_Images.pdf",
                    score=0.9,
                ),
                ImageInfo(
                    caption="Process Flow",
                    href="https://example.com/images/process.png",
                    page=11,
                    document_name="Document_With_Images.pdf",
                    score=0.85,
                ),
            ],
            tables=[],
            modified_date=None,
        )

        citation = CitationExtractor.extract_citation_from_result(result)

        # Verify images are extracted
        assert citation.images is not None
        assert len(citation.images) == 2
        assert citation.images[0].caption == "Risk Matrix Diagram"
        assert citation.images[1].caption == "Process Flow"

    def test_extract_citation_with_tables(self):
        """Test extracting citation with tables."""
        result = CompactSearchResult(
            score=0.95,
            content="Content with tables",
            document_name="Document_With_Tables.pdf",
            page=20,
            href="https://example.com/doc.pdf",
            author=None,
            publication_date=None,
            primary_term_keys=[],
            related_term_keys=[],
            images=[],
            tables=[
                TableInfo(
                    content="| Limit | Amount |\n|-------|--------|\n| Daily | 1M EUR |",
                    page=20,
                ),
            ],
            modified_date=None,
        )

        citation = CitationExtractor.extract_citation_from_result(result)

        # Verify tables are extracted
        assert citation.tables is not None
        assert len(citation.tables) == 1
        assert citation.tables[0].caption == "Table from page 20"  # Default caption generated
        assert "1M EUR" in citation.tables[0].content

    def test_extract_all_citations(self):
        """Test extracting multiple citations from search response."""
        results = [
            CompactSearchResult(
                score=0.95,
                content="First document content",
                document_name="Doc1.pdf",
                page=1,
                href="https://example.com/doc1.pdf",
                author="Author 1",
                publication_date="2023-01-01",
                primary_term_keys=[],
                related_term_keys=[],
                images=[],
                tables=[],
                modified_date=None,
            ),
            CompactSearchResult(
                score=0.90,
                content="Second document content",
                document_name="Doc2.pdf",
                page=2,
                href="https://example.com/doc2.pdf",
                author="Author 2",
                publication_date="2023-02-01",
                primary_term_keys=[],
                related_term_keys=[],
                images=[],
                tables=[],
                modified_date=None,
            ),
            CompactSearchResult(
                score=0.85,
                content="Third document content",
                document_name="Doc3.pdf",
                page=3,
                href="https://example.com/doc3.pdf",
                author="Author 3",
                publication_date="2023-03-01",
                primary_term_keys=[],
                related_term_keys=[],
                images=[],
                tables=[],
                modified_date=None,
            ),
        ]

        search_response = OptimizedSearchResponse(
            results=results,
            terms={},
            total_results=3,
        )

        # Extract with limit
        citations = CitationExtractor.extract_all_citations(search_response, max_citations=2)

        assert len(citations) == 2
        assert citations[0].title == "Doc1.pdf"
        assert citations[1].title == "Doc2.pdf"
        # Third citation should be excluded due to limit

    def test_create_citation_context(self):
        """Test creating formatted citation context for AI."""
        results = [
            CompactSearchResult(
                score=0.95,
                content="Risk management content",
                document_name="Risk_Policy.pdf",
                page=45,
                href="https://example.com/risk.pdf",
                author="Risk Department",
                publication_date="2023-11-15",
                primary_term_keys=[],
                related_term_keys=[],
                images=[],
                tables=[],
                modified_date=None,
            ),
            CompactSearchResult(
                score=0.90,
                content="LMS documentation",
                document_name="LMS_Manual.pdf",
                page=12,
                href=None,  # No URL
                author=None,  # No author
                publication_date=None,  # No date
                primary_term_keys=[],
                related_term_keys=[],
                images=[],
                tables=[],
                modified_date=None,
            ),
        ]

        search_response = OptimizedSearchResponse(
            results=results,
            terms={},
            total_results=2,
        )

        context = CitationExtractor.create_citation_context(search_response)

        # Verify context format
        assert "Available citations from search results:" in context
        assert "[1] Risk_Policy.pdf (page 45) by Risk Department (2023-11-15)" in context
        assert "URL: https://example.com/risk.pdf#page=45" in context
        assert "[2] LMS_Manual.pdf (page 12)" in context
        # Should not have URL line for second citation

    def test_validate_citation_valid(self):
        """Test validation of a valid citation."""
        result = CompactSearchResult(
            score=0.95,
            content="Content",
            document_name="Valid_Doc.pdf",
            page=10,
            href="https://example.com/valid.pdf",
            author="Test Author",
            publication_date="2023-01-01",
            primary_term_keys=[],
            related_term_keys=[],
            images=[],
            tables=[],
            modified_date=None,
        )

        search_response = OptimizedSearchResponse(
            results=[result],
            terms={},
            total_results=1,
        )

        # Create citation from the result
        citation = CitationExtractor.extract_citation_from_result(result)

        # Should be valid
        assert CitationExtractor.validate_citation(citation, search_response) is True

    def test_validate_citation_invalid(self):
        """Test validation of an invalid (hallucinated) citation."""
        result = CompactSearchResult(
            score=0.95,
            content="Content",
            document_name="Real_Doc.pdf",
            page=10,
            href="https://example.com/real.pdf",
            author="Real Author",
            publication_date="2023-01-01",
            primary_term_keys=[],
            related_term_keys=[],
            images=[],
            tables=[],
            modified_date=None,
        )

        search_response = OptimizedSearchResponse(
            results=[result],
            terms={},
            total_results=1,
        )

        # Create a fake citation
        fake_citation = Citation(
            title="Fake_Doc.pdf",  # Wrong title
            author="Dr. Fake Author",  # Hallucinated author
            publication_date="2023-06-20",  # Wrong date
            page=99,  # Wrong page
            href="https://example.com/fake.pdf",  # Wrong URL
            quote="This is a hallucinated quote",
            images=None,
            tables=None,
        )

        # Should be invalid
        assert CitationExtractor.validate_citation(fake_citation, search_response) is False

    def test_validate_citation_with_page_anchor(self):
        """Test validation handles page anchors correctly."""
        result = CompactSearchResult(
            score=0.95,
            content="Content",
            document_name="Doc.pdf",
            page=10,
            href="https://example.com/doc.pdf",  # No page anchor
            author=None,
            publication_date=None,
            primary_term_keys=[],
            related_term_keys=[],
            images=[],
            tables=[],
            modified_date=None,
        )

        search_response = OptimizedSearchResponse(
            results=[result],
            terms={},
            total_results=1,
        )

        # Citation with page anchor added
        citation = Citation(
            title="Doc.pdf",
            author=None,
            publication_date=None,
            page=10,
            href="https://example.com/doc.pdf#page=10",  # Has page anchor
            quote=None,
            images=None,
            tables=None,
        )

        # Should still be valid despite URL difference
        assert CitationExtractor.validate_citation(citation, search_response) is True

    def test_sanitize_citations(self):
        """Test sanitization removes invalid citations."""
        results = [
            CompactSearchResult(
                score=0.95,
                content="Real content 1",
                document_name="Real1.pdf",
                page=1,
                href="https://example.com/real1.pdf",
                author="Author 1",
                publication_date="2023-01-01",
                primary_term_keys=[],
                related_term_keys=[],
                images=[],
                tables=[],
                modified_date=None,
            ),
            CompactSearchResult(
                score=0.90,
                content="Real content 2",
                document_name="Real2.pdf",
                page=2,
                href="https://example.com/real2.pdf",
                author="Author 2",
                publication_date="2023-02-01",
                primary_term_keys=[],
                related_term_keys=[],
                images=[],
                tables=[],
                modified_date=None,
            ),
        ]

        search_response = OptimizedSearchResponse(
            results=results,
            terms={},
            total_results=2,
        )

        # Mix of valid and fake citations
        citations_to_check = [
            # Valid citation 1
            CitationExtractor.extract_citation_from_result(results[0]),
            # Fake citation
            Citation(
                title="Hallucinated_Doc.pdf",
                author="Dr. Fake Person",
                publication_date="2023-99-99",
                page=999,
                href="https://fake.com/fake.pdf",
                quote="This doesn't exist",
                images=None,
                tables=None,
            ),
            # Valid citation 2
            CitationExtractor.extract_citation_from_result(results[1]),
            # Another fake citation
            Citation(
                title="Another_Fake.pdf",
                author="Made Up Author",
                publication_date=None,
                page=0,
                href=None,
                quote=None,
                images=None,
                tables=None,
            ),
        ]

        # Sanitize
        valid_citations = CitationExtractor.sanitize_citations(
            citations_to_check,
            search_response
        )

        # Should only have 2 valid citations
        assert len(valid_citations) == 2
        assert valid_citations[0].title == "Real1.pdf"
        assert valid_citations[1].title == "Real2.pdf"


class TestCitationMapper:
    """Test citation mapping functionality."""

    def test_citation_mapper_initialization(self):
        """Test CitationMapper initialization and cache building."""
        results = [
            CompactSearchResult(
                score=0.95,
                content="Content 1",
                document_name="Doc1.pdf",
                page=1,
                href="https://example.com/doc1.pdf",
                author="Author 1",
                publication_date="2023-01-01",
                primary_term_keys=[],
                related_term_keys=[],
                images=[],
                tables=[],
                modified_date=None,
            ),
            CompactSearchResult(
                score=0.90,
                content="Content 2",
                document_name="Doc2.pdf",
                page=2,
                href="https://example.com/doc2.pdf",
                author="Author 2",
                publication_date="2023-02-01",
                primary_term_keys=[],
                related_term_keys=[],
                images=[],
                tables=[],
                modified_date=None,
            ),
        ]

        search_response = OptimizedSearchResponse(
            results=results,
            terms={},
            total_results=2,
        )

        mapper = CitationMapper(search_response)

        # Check cache is built
        assert len(mapper._citation_cache) == 2
        all_citations = mapper.get_all_available_citations()
        assert len(all_citations) == 2

    def test_get_citation_for_content(self):
        """Test finding the right citation for content snippets."""
        results = [
            CompactSearchResult(
                score=0.95,
                content="The risk limit for corporate clients is 1M EUR according to policy.",
                document_name="Risk_Policy.pdf",
                page=45,
                href="https://example.com/risk.pdf",
                author="Risk Dept",
                publication_date="2023-11-15",
                primary_term_keys=[],
                related_term_keys=[],
                images=[],
                tables=[],
                modified_date=None,
            ),
            CompactSearchResult(
                score=0.90,
                content="LMS handles all limit management functions for the bank.",
                document_name="LMS_Guide.pdf",
                page=12,
                href="https://example.com/lms.pdf",
                author=None,
                publication_date=None,
                primary_term_keys=[],
                related_term_keys=[],
                images=[],
                tables=[],
                modified_date=None,
            ),
        ]

        search_response = OptimizedSearchResponse(
            results=results,
            terms={},
            total_results=2,
        )

        mapper = CitationMapper(search_response)

        # Find citation for content snippet
        citation = mapper.get_citation_for_content(
            "risk limit for corporate clients",
            quote="The risk limit is 1M EUR"
        )

        assert citation is not None
        assert citation.title == "Risk_Policy.pdf"
        assert citation.quote == "The risk limit is 1M EUR"
        assert citation.page == 45

        # Find citation for different content
        citation2 = mapper.get_citation_for_content(
            "LMS handles all limit"
        )

        assert citation2 is not None
        assert citation2.title == "LMS_Guide.pdf"
        assert citation2.page == 12

        # Non-existent content
        citation3 = mapper.get_citation_for_content(
            "This content doesn't exist anywhere"
        )

        assert citation3 is None

    def test_format_for_agent_input(self):
        """Test formatting citations for agent input."""
        results = [
            CompactSearchResult(
                score=0.95,
                content="Content",
                document_name="Doc.pdf",
                page=10,
                href="https://example.com/doc.pdf",
                author="Test Author",
                publication_date="2023-01-01",
                primary_term_keys=[],
                related_term_keys=[],
                images=[],
                tables=[],
                modified_date=None,
            ),
        ]

        search_response = OptimizedSearchResponse(
            results=results,
            terms={},
            total_results=1,
        )

        mapper = CitationMapper(search_response)
        formatted = mapper.format_for_agent_input()

        assert "Available citations from search results:" in formatted
        assert "[1] Doc.pdf (page 10) by Test Author (2023-01-01)" in formatted
        assert "URL: https://example.com/doc.pdf#page=10" in formatted


class TestCitationIntegrity:
    """Integration tests to ensure citation integrity through the full pipeline."""

    def test_no_hallucination_possible(self):
        """Test that hallucinated citations cannot pass through the system."""
        # Create search results with specific documents
        real_results = [
            CompactSearchResult(
                score=0.95,
                content="Real content from actual search",
                document_name="Real_Document.pdf",
                page=15,
                href="https://realsite.com/real.pdf",
                author="Actual Author",
                publication_date="2023-01-15",
                primary_term_keys=["real", "content"],
                related_term_keys=[],
                images=[],
                tables=[],
                modified_date=None,
            ),
        ]

        search_response = OptimizedSearchResponse(
            results=real_results,
            terms={},
            total_results=1,
        )

        # Simulate AI trying to create hallucinated citations
        ai_generated_citations = [
            # One valid citation
            CitationExtractor.extract_citation_from_result(real_results[0]),
            # Multiple hallucinated citations
            Citation(
                title="Completely Fake Document.pdf",
                author="Dr. Imaginary Person",
                publication_date="2023-12-32",  # Invalid date
                page=999,
                href="https://fakesite.com/fake.pdf",
                quote="This is completely made up information",
                images=None,
                tables=None,
            ),
            Citation(
                title="Another Hallucination.pdf",
                author="Prof. Doesn't Exist",
                publication_date="2024-13-45",  # Impossible date
                page=12345,
                href="https://madeup.com/nothing.pdf",
                quote="More fake information",
                images=None,
                tables=None,
            ),
        ]

        # Sanitize citations
        valid_citations = CitationExtractor.sanitize_citations(
            ai_generated_citations,
            search_response
        )

        # Only the real citation should remain
        assert len(valid_citations) == 1
        assert valid_citations[0].title == "Real_Document.pdf"
        assert valid_citations[0].author == "Actual Author"
        # Hallucinated citations are completely filtered out

    def test_edge_cases_in_citation_extraction(self):
        """Test various edge cases in citation extraction."""
        # Edge case 1: Empty search results
        empty_response = OptimizedSearchResponse(
            results=[],
            terms={},
            total_results=0,
        )

        citations = CitationExtractor.extract_all_citations(empty_response)
        assert len(citations) == 0

        context = CitationExtractor.create_citation_context(empty_response)
        assert context == "No citations available from search results."

        # Edge case 2: Result with no metadata at all
        minimal_result = CompactSearchResult(
            score=0.5,
            content="Some content",
            document_name="Unnamed.txt",
            page=None,
            author=None,
            publication_date=None,
            href=None,
            primary_term_keys=[],
            related_term_keys=[],
            images=[],
            tables=[],
            modified_date=None,
        )

        citation = CitationExtractor.extract_citation_from_result(minimal_result)
        assert citation.title == "Unnamed.txt"
        assert citation.page == 0  # Defaults to 0 when None
        assert citation.author is None
        assert citation.publication_date is None
        assert citation.href is None

        # Edge case 3: Very long content
        long_content_result = CompactSearchResult(
            score=0.9,
            content="x" * 10000,  # Very long content
            document_name="Long.pdf",
            page=1,
            author=None,
            publication_date=None,
            href="https://example.com/long.pdf",
            primary_term_keys=[],
            related_term_keys=[],
            images=[],
            tables=[],
            modified_date=None,
        )

        citation = CitationExtractor.extract_citation_from_result(
            long_content_result,
            quote="x" * 500  # Long quote
        )
        assert len(citation.quote) == 500  # Quote is preserved