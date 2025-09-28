"""
Test citation workflow determinism and consistency across multiple runs.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from typing import List, Optional

from blockether_catalyst.knowledge.KnowledgeTypes import (
    CompactSearchResult,
    OptimizedSearchResponse,
    ImageInfo,
    TableInfo
)
from blockether_catalyst.knowledge.answering.CitationExtractor import (
    CitationExtractor,
    CitationMapper
)
from blockether_catalyst.knowledge.answering.AnswerProviderAgent import (
    Citation,
    ImageAttachment,
    TableAttachment,
    AnswerOutput
)
# Note: Integration with workflow removed as StepsWorkflowCore is not a class
# These tests focus on the CitationExtractor determinism and functionality


class TestCitationWorkflowDeterminism:
    """Test that citation workflow produces consistent results across multiple runs."""

    def _create_consistent_search_results(self) -> OptimizedSearchResponse:
        """Create consistent search results for determinism testing."""
        return OptimizedSearchResponse(
            results=[
                CompactSearchResult(
                    document_name="Risk Management Policy v1.0",
                    content="This policy defines risk assessment procedures. Key risk indicators include credit exposure, market volatility, and operational risk factors.",
                    score=0.95,
                    page=15,
                    href="https://docs.example.com/risk_policy.pdf",
                    author="Risk Committee",
                    publication_date="2024-01-15",
                    primary_term_keys=[],
                    related_term_keys=[],
                    modified_date=None,
                    images=[
                        ImageInfo(
                            caption="Risk Matrix Diagram",
                            href="https://docs.example.com/risk_matrix.png",
                            page=15,
                            document_name="Risk Management Policy v1.0"
                        )
                    ],
                    tables=[
                        TableInfo(
                            content="Risk Level|Impact|Probability\nHigh|>$10M|>50%\nMedium|$1M-$10M|20-50%\nLow|<$1M|<20%",
                            page=15
                        )
                    ]
                ),
                CompactSearchResult(
                    document_name="Compliance Guidelines 2024",
                    content="Regulatory compliance requirements for financial institutions. Includes GDPR, SOX, and Basel III compliance.",
                    score=0.88,
                    page=42,
                    href="https://docs.example.com/compliance.pdf",
                    author="Legal Department",
                    publication_date="2024-02-01",
                    primary_term_keys=[],
                    related_term_keys=[],
                    images=[],
                    tables=[],
                    modified_date=None
                ),
                CompactSearchResult(
                    document_name="Internal Audit Report Q3",
                    content="Quarterly audit findings and recommendations. No critical issues identified.",
                    score=0.82,
                    page=7,
                    href=None,  # Test handling of missing href
                    author=None,  # Test handling of missing author
                    publication_date=None,  # Test handling of missing date
                    primary_term_keys=[],
                    related_term_keys=[],
                    images=[],
                    tables=[],
                    modified_date=None
                )
            ],
            total_results=3,
            query="risk management compliance audit"
        )

    def test_citation_extraction_determinism(self):
        """Test that citation extraction produces identical results on multiple runs."""
        search_results = self._create_consistent_search_results()

        # Run extraction multiple times
        runs = []
        for _ in range(5):
            citations = CitationExtractor.extract_all_citations(search_results, max_citations=10)
            runs.append(citations)

        # All runs should produce identical citations
        first_run = runs[0]
        for i, run in enumerate(runs[1:], 1):
            assert len(run) == len(first_run), f"Run {i+1} has different number of citations"

            for j, (citation1, citation2) in enumerate(zip(first_run, run)):
                assert citation1.title == citation2.title, f"Citation {j} title mismatch in run {i+1}"
                assert citation1.href == citation2.href, f"Citation {j} href mismatch in run {i+1}"
                assert citation1.page == citation2.page, f"Citation {j} page mismatch in run {i+1}"
                assert citation1.author == citation2.author, f"Citation {j} author mismatch in run {i+1}"
                assert citation1.publication_date == citation2.publication_date, f"Citation {j} date mismatch in run {i+1}"

    def test_citation_context_determinism(self):
        """Test that citation context generation is deterministic."""
        search_results = self._create_consistent_search_results()

        # Generate context multiple times
        contexts = []
        for _ in range(5):
            context = CitationExtractor.create_citation_context(search_results, max_citations=10)
            contexts.append(context)

        # All contexts should be identical
        first_context = contexts[0]
        for i, context in enumerate(contexts[1:], 1):
            assert context == first_context, f"Context in run {i+1} differs from first run"

    def test_citation_mapper_determinism(self):
        """Test that CitationMapper produces consistent results."""
        search_results = self._create_consistent_search_results()

        # Create multiple mappers
        mappers = [CitationMapper(search_results) for _ in range(5)]

        # All mappers should produce identical citations
        first_citations = mappers[0].get_all_available_citations()

        for i, mapper in enumerate(mappers[1:], 1):
            citations = mapper.get_all_available_citations()
            assert len(citations) == len(first_citations), f"Mapper {i+1} has different number of citations"

            for j, (c1, c2) in enumerate(zip(first_citations, citations)):
                assert c1.title == c2.title, f"Citation {j} title mismatch in mapper {i+1}"
                assert c1.href == c2.href, f"Citation {j} href mismatch in mapper {i+1}"

    def test_citation_validation_determinism(self):
        """Test that citation validation is deterministic."""
        search_results = self._create_consistent_search_results()

        # Create test citations
        valid_citation = Citation(
            title="Risk Management Policy v1.0",
            author="Risk Committee",
            publication_date="2024-01-15",
            page=15,
            href="https://docs.example.com/risk_policy.pdf#page=15",
            quote="Key risk indicators",
            images=None,
            tables=None
        )

        invalid_citation = Citation(
            title="Fake Document",
            author="Fake Author",
            publication_date="2024-12-31",
            page=99,
            href="https://fake.example.com/fake.pdf",
            quote="Fake quote",
            images=None,
            tables=None
        )

        # Validate multiple times
        valid_results = []
        invalid_results = []

        for _ in range(5):
            valid_results.append(CitationExtractor.validate_citation(valid_citation, search_results))
            invalid_results.append(CitationExtractor.validate_citation(invalid_citation, search_results))

        # Results should be consistent
        assert all(result is True for result in valid_results), "Valid citation validation inconsistent"
        assert all(result is False for result in invalid_results), "Invalid citation validation inconsistent"

    def test_citation_sanitization_determinism(self):
        """Test that citation sanitization produces consistent results."""
        search_results = self._create_consistent_search_results()

        # Create mix of valid and invalid citations
        citations = [
            Citation(
                title="Risk Management Policy v1.0",
                author="Risk Committee",
                publication_date="2024-01-15",
                page=15,
                href="https://docs.example.com/risk_policy.pdf#page=15",
                quote="Key risk indicators",
                images=None,
                tables=None
            ),
            Citation(
                title="Fake Document",  # Invalid - not in search results
                author="Fake Author",
                publication_date="2024-12-31",
                page=99,
                href="https://fake.example.com/fake.pdf",
                quote="Fake quote",
                images=None,
                tables=None
            ),
            Citation(
                title="Compliance Guidelines 2024",
                author="Legal Department",
                publication_date="2024-02-01",
                page=42,
                href="https://docs.example.com/compliance.pdf#page=42",
                quote="GDPR compliance",
                images=None,
                tables=None
            )
        ]

        # Sanitize multiple times
        sanitized_runs = []
        for _ in range(5):
            sanitized = CitationExtractor.sanitize_citations(citations, search_results)
            sanitized_runs.append(sanitized)

        # All runs should produce identical results
        first_run = sanitized_runs[0]
        assert len(first_run) == 2, "Should have filtered out the fake citation"

        for i, run in enumerate(sanitized_runs[1:], 1):
            assert len(run) == len(first_run), f"Run {i+1} has different number of sanitized citations"

            for j, (c1, c2) in enumerate(zip(first_run, run)):
                assert c1.title == c2.title, f"Citation {j} title mismatch in run {i+1}"
                assert c1.href == c2.href, f"Citation {j} href mismatch in run {i+1}"

    def test_page_anchor_determinism(self):
        """Test that page anchor generation is deterministic."""
        results_with_pages = [
            CompactSearchResult(
                document_name="Document A",
                content="Content A",
                score=0.9,
                page=10,
                href="https://example.com/doc_a.pdf",
                author="Author A",
                publication_date="2024-01-01"
            ),
            CompactSearchResult(
                document_name="Document B",
                content="Content B",
                score=0.8,
                page=25,
                href="https://example.com/doc_b.pdf#page=25",  # Already has anchor
                author="Author B",
                publication_date="2024-01-02"
            ),
            CompactSearchResult(
                document_name="Document C",
                content="Content C",
                score=0.7,
                page=None,  # No page number
                href="https://example.com/doc_c.pdf",
                author="Author C",
                publication_date="2024-01-03"
            )
        ]

        # Extract citations multiple times
        runs = []
        for _ in range(5):
            citations = []
            for result in results_with_pages:
                citation = CitationExtractor.extract_citation_from_result(result)
                citations.append(citation)
            runs.append(citations)

        # Check that page anchors are consistent
        expected_hrefs = [
            "https://example.com/doc_a.pdf#page=10",  # Should add anchor
            "https://example.com/doc_b.pdf#page=25",  # Should keep existing
            "https://example.com/doc_c.pdf"  # No anchor (no page)
        ]

        for run in runs:
            actual_hrefs = [c.href for c in run]
            assert actual_hrefs == expected_hrefs, "Page anchor generation not deterministic"

    def test_citation_extraction_with_workflow_data(self):
        """Test citation extraction with typical workflow data."""
        search_results = self._create_consistent_search_results()

        # Test that citations can be extracted consistently from workflow data
        for _ in range(5):
            citations = CitationExtractor.extract_all_citations(search_results)

            # Verify all citations have page anchors where appropriate
            for citation in citations:
                if citation.href and citation.page:
                    assert "#page=" in citation.href, f"Missing page anchor in {citation.href}"

            # Verify citation integrity
            for citation in citations:
                is_valid = CitationExtractor.validate_citation(citation, search_results)
                assert is_valid, f"Invalid citation: {citation.title}"

    def test_edge_case_determinism(self):
        """Test determinism with edge cases like empty results, None values, etc."""
        edge_cases = [
            # Empty search results
            OptimizedSearchResponse(
                results=[],
                total_results=0,
                query="test query"
            ),
            # Results with all None metadata
            OptimizedSearchResponse(
                results=[
                    CompactSearchResult(
                        document_name="Unnamed Document",
                        content="Some content",
                        score=0.5,
                        page=None,
                        href=None,
                        author=None,
                        publication_date=None,
                        images=[],
                        tables=[],
                        primary_term_keys=[],
                        related_term_keys=[],
                        modified_date=None
                    )
                ],
                total_results=1,
                query="test query"
            ),
            # Results with special characters
            OptimizedSearchResponse(
                results=[
                    CompactSearchResult(
                        document_name="Document with 特殊 characters & symbols!",
                        content="Content with unicode: 中文内容",
                        score=0.9,
                        page=100,
                        href="https://example.com/doc%20with%20spaces.pdf",
                        author="Author (with) [brackets]",
                        publication_date="2024/01/01",
                        images=[],
                        tables=[],
                        primary_term_keys=[],
                        related_term_keys=[],
                        modified_date=None
                    )
                ],
                total_results=1,
                query="test query"
            )
        ]

        for search_results in edge_cases:
            # Run extraction multiple times for each edge case
            runs = []
            for _ in range(3):
                citations = CitationExtractor.extract_all_citations(search_results)
                context = CitationExtractor.create_citation_context(search_results)
                runs.append((citations, context))

            # Verify determinism for each edge case
            first_citations, first_context = runs[0]
            for i, (citations, context) in enumerate(runs[1:], 1):
                assert len(citations) == len(first_citations), f"Edge case run {i+1} citation count mismatch"
                assert context == first_context, f"Edge case run {i+1} context mismatch"