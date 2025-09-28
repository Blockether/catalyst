"""
Comprehensive tests for the improved citation system.

Tests the pre-extraction, selection, and formatting of citations
to ensure integrity and prevent AI hallucination.
"""

import pytest
from unittest.mock import Mock, patch
from typing import List

from blockether_catalyst.knowledge.KnowledgeTypes import (
    CompactSearchResult,
    OptimizedSearchResponse,
    ImageInfo,
    TableInfo,
)
from blockether_catalyst.knowledge.answering.AnswerProviderAgent import (
    Citation,
    ImageAttachment,
    TableAttachment,
    AnswerProviderInput,
)
from blockether_catalyst.knowledge.answering.CitationExtractor import CitationExtractor
from blockether_catalyst.knowledge.answering.CitationFormatter import CitationFormatter


class TestCitationExtraction:
    """Test citation extraction from search results."""

    def test_extract_citation_with_full_metadata(self):
        """Test extracting citation with all metadata fields."""
        search_result = CompactSearchResult(
            score=0.95,
            document_name="Risk Management Policy v2.1",
            content="The daily limit is 1M EUR for corporate accounts.",
            page=45,
            href="https://bank.com/docs/risk-policy.pdf",
            author="RBI Risk Department",
            publication_date="2024-03-15",
        )

        citation = CitationExtractor.extract_citation_from_result(search_result)

        assert citation.title == "Risk Management Policy v2.1"
        assert citation.author == "RBI Risk Department"
        assert citation.publication_date == "2024-03-15"
        assert citation.page == 45
        assert citation.href == "https://bank.com/docs/risk-policy.pdf#page=45"
        assert citation.quote is None

    def test_extract_citation_with_missing_metadata(self):
        """Test extracting citation when some metadata is missing."""
        search_result = CompactSearchResult(
            score=0.85,
            document_name="Internal Procedures",
            content="Follow the standard approval process.",
            page=12,
            # No href, author, or publication_date
        )

        citation = CitationExtractor.extract_citation_from_result(search_result)

        assert citation.title == "Internal Procedures"
        assert citation.author is None
        assert citation.publication_date is None
        assert citation.page == 12
        assert citation.href is None

    def test_extract_citation_with_images(self):
        """Test extracting citation with image attachments."""
        search_result = CompactSearchResult(
            score=0.90,
            document_name="LMS User Manual",
            content="System architecture overview.",
            page=8,
            images=[
                ImageInfo(
                    caption="System Architecture Diagram",
                    href="https://bank.com/images/arch.png",
                    page=8,
                )
            ],
        )

        citation = CitationExtractor.extract_citation_from_result(search_result)

        assert citation.images is not None
        assert len(citation.images) == 1
        assert citation.images[0].caption == "System Architecture Diagram"
        assert citation.images[0].href == "https://bank.com/images/arch.png"
        assert citation.images[0].page == 8
        assert citation.images[0].document_name == "LMS User Manual"

    def test_extract_citation_with_tables(self):
        """Test extracting citation with table attachments."""
        search_result = CompactSearchResult(
            score=0.88,
            document_name="Limit Configuration Guide",
            content="Limit thresholds by category.",
            page=23,
            tables=[
                TableInfo(
                    content="| Category | Limit |\n|----------|-------|\n| Corp | 1M EUR |",
                    page=23,
                )
            ],
        )

        citation = CitationExtractor.extract_citation_from_result(search_result)

        assert citation.tables is not None
        assert len(citation.tables) == 1
        assert citation.tables[0].content == "| Category | Limit |\n|----------|-------|\n| Corp | 1M EUR |"
        assert citation.tables[0].page == 23
        assert citation.tables[0].caption == "Table from page 23"

    def test_page_anchor_generation(self):
        """Test that page anchors are correctly added to URLs."""
        search_result = CompactSearchResult(
            score=0.92,
            document_name="Compliance Manual",
            content="Regulatory requirements.",
            page=67,
            href="https://bank.com/docs/compliance.pdf",
        )

        citation = CitationExtractor.extract_citation_from_result(search_result)
        assert citation.href == "https://bank.com/docs/compliance.pdf#page=67"

    def test_no_duplicate_page_anchor(self):
        """Test that existing page anchors aren't duplicated."""
        search_result = CompactSearchResult(
            score=0.91,
            document_name="Policy Document",
            content="Policy details.",
            page=34,
            href="https://bank.com/docs/policy.pdf#page=34",
        )

        citation = CitationExtractor.extract_citation_from_result(search_result)
        assert citation.href == "https://bank.com/docs/policy.pdf#page=34"
        assert citation.href.count("#page=") == 1

    def test_create_citation_context(self):
        """Test creating formatted citation context for agents."""
        search_response = OptimizedSearchResponse(
            results=[
                CompactSearchResult(
                    score=0.95,
                    document_name="Risk Policy",
                    content="Content 1",
                    page=10,
                    href="https://bank.com/risk.pdf",
                    author="Risk Team",
                    publication_date="2024-01",
                ),
                CompactSearchResult(
                    score=0.85,
                    document_name="Procedures",
                    content="Content 2",
                    page=5,
                ),
            ],
            total_results=2,
        )

        context = CitationExtractor.create_citation_context(search_response, max_citations=5)

        assert "Available citations from search results:" in context
        assert "[1] Risk Policy (page 10) by Risk Team (2024-01)" in context
        assert "URL: https://bank.com/risk.pdf#page=10" in context
        assert "[2] Procedures (page 5)" in context

    def test_extract_all_citations_respects_limit(self):
        """Test that citation extraction respects the max_citations limit."""
        results = [
            CompactSearchResult(
                score=0.9 - i * 0.1,
                document_name=f"Document {i}",
                content=f"Content {i}",
                page=i,
            )
            for i in range(10)
        ]

        search_response = OptimizedSearchResponse(results=results, total_results=10)

        citations = CitationExtractor.extract_all_citations(search_response, max_citations=3)

        assert len(citations) == 3
        assert citations[0].title == "Document 0"
        assert citations[2].title == "Document 2"


class TestCitationFormatting:
    """Test programmatic citation style transformation."""

    def test_inline_numeric_to_footnote(self):
        """Test transforming inline numeric to footnote style."""
        text = "The limit is 1M EUR [1] for corporate clients [2]."
        citations = [
            Citation(title="Risk Policy", page=10),
            Citation(title="Corporate Guide", page=20),
        ]

        transformed, _ = CitationFormatter.transform_citations_in_text(
            text, citations, from_style="inline_numeric", to_style="footnote"
        )

        assert transformed == "The limit is 1M EUR [^1] for corporate clients [^2]."

    def test_inline_numeric_to_superscript(self):
        """Test transforming inline numeric to superscript style."""
        text = "Information from source [1]."
        citations = [Citation(title="Source Document", page=5)]

        transformed, _ = CitationFormatter.transform_citations_in_text(
            text, citations, from_style="inline_numeric", to_style="superscript"
        )

        assert transformed == "Information from source <sup>1</sup>."

    def test_inline_numeric_to_author_date(self):
        """Test transforming inline numeric to author-date style."""
        text = "The policy states [1] that limits apply [2]."
        citations = [
            Citation(title="Risk Policy", author="Smith, J.", publication_date="2024-03-15", page=10),
            Citation(title="Guidelines", author="Johnson", publication_date="2023", page=5),
        ]

        transformed, _ = CitationFormatter.transform_citations_in_text(
            text, citations, from_style="inline_numeric", to_style="author_date"
        )

        assert "(Smith, 2024)" in transformed
        assert "(Johnson, 2023)" in transformed

    def test_no_transformation_same_style(self):
        """Test that no transformation occurs when styles are the same."""
        text = "Original text [1] with citations [2]."
        citations = [
            Citation(title="Doc 1", page=1),
            Citation(title="Doc 2", page=2),
        ]

        transformed, returned_citations = CitationFormatter.transform_citations_in_text(
            text, citations, from_style="inline_numeric", to_style="inline_numeric"
        )

        assert transformed == text
        assert returned_citations == citations

    def test_get_style_instructions(self):
        """Test retrieving style-specific instructions."""
        instructions = CitationFormatter.get_style_instructions("inline_numeric")
        assert "square brackets [1]" in instructions

        instructions = CitationFormatter.get_style_instructions("footnote")
        assert "[^1]" in instructions

        instructions = CitationFormatter.get_style_instructions("superscript")
        assert "<sup>" in instructions

        instructions = CitationFormatter.get_style_instructions("author_date")
        assert "(Author, Year)" in instructions

    def test_validate_citation_consistency(self):
        """Test validation of citation style consistency."""
        # Valid inline_numeric text
        text = "Information [1] from source [2]."
        result = CitationFormatter.validate_citation_consistency(text, "inline_numeric")
        assert result["valid"] is True
        assert result["citations_found"] == 2

        # Invalid - mixed styles
        text = "Information [1] from source<sup>2</sup>."
        result = CitationFormatter.validate_citation_consistency(text, "inline_numeric")
        assert result["valid"] is False
        assert "superscript" in str(result["issues"])

    def test_extract_year_from_dates(self):
        """Test year extraction from various date formats."""
        assert CitationFormatter._extract_year("2024-03-15") == "2024"
        assert CitationFormatter._extract_year("March 2024") == "2024"
        assert CitationFormatter._extract_year("2024") == "2024"
        assert CitationFormatter._extract_year("15/03/2024") == "2024"
        assert CitationFormatter._extract_year(None) == "n.d."
        assert CitationFormatter._extract_year("") == "n.d."
        assert CitationFormatter._extract_year("no date") == "n.d."


class TestAnswerProviderInput:
    """Test the AnswerProviderInput with pre-extracted citations."""

    def test_input_with_available_citations(self):
        """Test that AnswerProviderInput accepts available_citations."""
        citations = [
            Citation(title="Doc 1", page=10, author="Author 1"),
            Citation(title="Doc 2", page=20, author="Author 2"),
        ]

        input_data = AnswerProviderInput(
            user_prompt="What are the limits?",
            knowledge_base="KB content here",
            intent="User wants to know limits",
            available_citations=citations,
            citation_style="inline_numeric",
            reasoning="Testing the AnswerProviderInput with available citations to ensure proper handling",
        )

        assert input_data.available_citations == citations
        assert input_data.citation_style == "inline_numeric"
        assert len(input_data.available_citations) == 2

    def test_input_without_citations(self):
        """Test that AnswerProviderInput works without citations."""
        input_data = AnswerProviderInput(
            user_prompt="Hello",
            knowledge_base="No KB needed",
            intent="Greeting",
            reasoning="Testing AnswerProviderInput without citations for greeting responses",
        )

        assert input_data.available_citations is None
        assert input_data.citation_style == "inline_numeric"  # Default  # Default


class TestCitationSanitization:
    """Test citation sanitization against search results."""

    def test_sanitize_citations_removes_invalid(self):
        """Test that sanitization removes citations not in search results."""
        search_response = OptimizedSearchResponse(
            results=[
                CompactSearchResult(
                    score=0.95,
                    document_name="Valid Document",
                    content="Valid content",
                    page=10,
                )
            ],
            total_results=1,
        )

        citations = [
            Citation(title="Valid Document", page=10),  # Valid
            Citation(title="Fabricated Document", page=99),  # Invalid
        ]

        sanitized = CitationExtractor.sanitize_citations(citations, search_response)

        assert len(sanitized) == 1
        assert sanitized[0].title == "Valid Document"

    def test_sanitize_citations_preserves_valid(self):
        """Test that sanitization preserves all valid citations."""
        search_response = OptimizedSearchResponse(
            results=[
                CompactSearchResult(
                    score=0.95,
                    document_name="Doc 1",
                    content="Content 1",
                    page=10,
                ),
                CompactSearchResult(
                    score=0.85,
                    document_name="Doc 2",
                    content="Content 2",
                    page=20,
                ),
            ],
            total_results=2,
        )

        citations = [
            Citation(title="Doc 1", page=10),
            Citation(title="Doc 2", page=20),
        ]

        sanitized = CitationExtractor.sanitize_citations(citations, search_response)

        assert len(sanitized) == 2
        assert sanitized[0].title == "Doc 1"
        assert sanitized[1].title == "Doc 2"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])