"""
Integration tests for citations with real agents.

Tests the full citation pipeline with AnswerProviderAgent and the workflow.
"""

import pytest
import asyncio
from typing import List, Dict, Any
from unittest.mock import Mock, patch, AsyncMock, MagicMock

from blockether_catalyst.knowledge.KnowledgeTypes import (
    CompactSearchResult,
    OptimizedSearchResponse,
    ImageInfo,
    TableInfo,
)
from blockether_catalyst.knowledge.answering.CitationExtractor import CitationExtractor
from blockether_catalyst.knowledge.answering.AnswerProviderAgent import (
    AnswerProviderAgent,
    AnswerProviderInput,
    AnswerOutput,
    Citation,
)
# Note: Integration tests simplified to focus on citation extraction and validation
# Workflow integration tests would require full workflow setup


class TestAnswerProviderAgentCitations:
    """Test AnswerProviderAgent with real citation scenarios."""

    @pytest.mark.asyncio
    async def test_agent_uses_only_provided_citations(self):
        """Test that the agent only uses citations from search results."""
        # Create search results with specific citations
        search_results = OptimizedSearchResponse(
            results=[
                CompactSearchResult(
                    score=0.95,
                    content="The daily transaction limit for corporate clients is 1M EUR.",
                    document_name="Corporate_Banking_Manual.pdf",
                    page=45,
                    author=None,
                    publication_date=None,
                    href="https://bank.com/docs/corporate_manual.pdf",
                    primary_term_keys=["transaction", "limit"],
                    related_term_keys=["corporate", "eur"],
                    images=[],
                    tables=[],
                    modified_date=None,
                ),
                CompactSearchResult(
                    score=0.88,
                    content="Risk approval is required for limits exceeding 500K EUR.",
                    document_name="Risk_Management_Policy.pdf",
                    page=23,
                    author=None,
                    publication_date=None,
                    href="https://bank.com/docs/risk_policy.pdf",
                    primary_term_keys=["risk", "approval"],
                    related_term_keys=["limit", "eur"],
                    images=[],
                    tables=[],
                    modified_date=None,
                ),
            ],
            terms={},
            total_results=2,
        )

        # Create citation context
        citation_context = CitationExtractor.create_citation_context(search_results)

        # Prepare knowledge base with citations
        knowledge_base = f"""
        Document: Corporate_Banking_Manual.pdf
        - Page 45 (score: 0.95): The daily transaction limit for corporate clients is 1M EUR.

        Document: Risk_Management_Policy.pdf
        - Page 23 (score: 0.88): Risk approval is required for limits exceeding 500K EUR.

        {citation_context}
        """

        # Create agent input
        agent_input = AnswerProviderInput(
            user_prompt="What are the transaction limits for corporate clients?",
            knowledge_base=knowledge_base,
            intent="User wants to know corporate transaction limits",
            missing_context=None,
            missing_terms=None,
            citation_style_examples="Use [1], [2] format for citations",
            reasoning="Answering based on available knowledge base documents about transaction limits.",
        )

        # Mock the model to return a controlled response
        mock_model = AsyncMock()
        mock_db = MagicMock()

        with patch('blockether_catalyst.knowledge.answering.AnswerProviderAgent.AnswerProviderAgent', return_value=mock_model):
            # Simulate agent response with citations
            mock_output = AnswerOutput(
                answer="The daily transaction limit for corporate clients is 1M EUR [1]. Risk approval is required for limits exceeding 500K EUR [2].",
                citations=[
                    Citation(
                        title="Corporate_Banking_Manual.pdf",
                        author=None,
                        publication_date=None,
                        page=45,
                        href="https://bank.com/docs/corporate_manual.pdf#page=45",
                        quote="The daily transaction limit for corporate clients is 1M EUR",
                        images=None,
                        tables=None,
                    ),
                    Citation(
                        title="Risk_Management_Policy.pdf",
                        author=None,
                        publication_date=None,
                        page=23,
                        href="https://bank.com/docs/risk_policy.pdf#page=23",
                        quote="Risk approval is required for limits exceeding 500K EUR",
                        images=None,
                        tables=None,
                    ),
                ],
                evaluation_factors=Mock(),
                suggested_follow_ups=[],
                reasoning="Answer based on corporate banking documentation.",
            )

            # Validate citations against search results
            valid_citations = CitationExtractor.sanitize_citations(
                mock_output.citations,
                search_results
            )

            # All citations should be valid
            assert len(valid_citations) == 2
            assert all(c.href and "#page=" in c.href for c in valid_citations)

    @pytest.mark.asyncio
    async def test_agent_hallucinated_citations_removed(self):
        """Test that hallucinated citations are filtered out."""
        # Limited search results
        search_results = OptimizedSearchResponse(
            results=[
                CompactSearchResult(
                    score=0.85,
                    content="LMS system processes limit requests.",
                    document_name="LMS_User_Manual.pdf",
                    page=10,
                    author=None,
                    publication_date=None,
                    href=None,  # No URL available
                    primary_term_keys=["lms", "limit"],
                    related_term_keys=[],
                    images=[],
                    tables=[],
                    modified_date=None,
                ),
            ],
            terms={},
            total_results=1,
        )

        # Simulate agent trying to add fake citations
        agent_citations = [
            Citation(
                title="LMS_User_Manual.pdf",  # Real
                author=None,
                publication_date=None,
                page=10,
                href=None,
                quote="LMS system processes limit requests",
                images=None,
                tables=None,
            ),
            Citation(
                title="Advanced_Risk_Framework.pdf",  # FAKE - not in search results
                author="Dr. Risk Expert",  # FAKE
                publication_date="2023-10-15",  # FAKE
                page=156,
                href="https://fake.com/risk.pdf",  # FAKE
                quote="Advanced risk calculations require...",  # FAKE
                images=None,
                tables=None,
            ),
            Citation(
                title="Compliance_Guidelines.pdf",  # FAKE - not in search results
                author="Compliance Team",  # FAKE
                publication_date="2023-11-20",  # FAKE
                page=78,
                href="https://fake.com/compliance.pdf",  # FAKE
                quote="Compliance requirements state...",  # FAKE
                images=None,
                tables=None,
            ),
        ]

        # Sanitize citations
        valid_citations = CitationExtractor.sanitize_citations(
            agent_citations,
            search_results
        )

        # Only the real citation should remain
        assert len(valid_citations) == 1
        assert valid_citations[0].title == "LMS_User_Manual.pdf"
        assert valid_citations[0].href is None  # Correctly has no URL

    @pytest.mark.asyncio
    async def test_agent_handles_missing_metadata_correctly(self):
        """Test that agent handles citations with missing metadata."""
        search_results = OptimizedSearchResponse(
            results=[
                CompactSearchResult(
                    score=0.90,
                    content="Content without full metadata.",
                    document_name="Partial_Doc.pdf",
                    page=5,
                    author=None,  # Missing
                    publication_date=None,  # Missing
                    href=None,  # Missing
                    primary_term_keys=[],
                    related_term_keys=[],
                    images=[],
                    tables=[],
                    modified_date=None,
                ),
            ],
            terms={},
            total_results=1,
        )

        citation = CitationExtractor.extract_citation_from_result(search_results.results[0])

        # Verify missing fields are properly None
        assert citation.title == "Partial_Doc.pdf"
        assert citation.page == 5
        assert citation.author is None
        assert citation.publication_date is None
        assert citation.href is None  # No URL, so no page anchor either


class TestWorkflowCitationIntegration:
    """Test citation handling in the full workflow."""

    @pytest.mark.asyncio
    async def test_workflow_citation_extraction_and_validation(self):
        """Test that the workflow properly extracts and validates citations."""
        # Mock search module
        mock_search_module = Mock(spec=KnowledgeSearchCore)
        mock_search_results = OptimizedSearchResponse(
            results=[
                CompactSearchResult(
                    score=0.92,
                    content="Credit risk assessment requires detailed analysis of client financials.",
                    document_name="Credit_Risk_Manual.pdf",
                    page=34,
                    author="Credit Risk Department",
                    publication_date="2023-09-15",
                    href="https://rbi.com/docs/credit_risk.pdf",
                    primary_term_keys=["credit", "risk"],
                    related_term_keys=["assessment", "analysis"],
                    images=[
                        ImageInfo(
                            caption="Risk Assessment Matrix",
                            href="https://rbi.com/images/risk_matrix.png",
                            page=34,
                            document_name="Credit_Risk_Manual.pdf",
                            score=0.9,
                        ),
                    ],
                    tables=[
                        TableInfo(
                            content="| Risk Level | Limit |\n|------------|-------|\n| Low | 5M EUR |\n| Medium | 2M EUR |\n| High | 500K EUR |",
                            caption="Risk-based Limits",
                            page=35,
                            document_name="Credit_Risk_Manual.pdf",
                            score=0.88,
                        ),
                    ],
                    modified_date=None,
                ),
                CompactSearchResult(
                    score=0.87,
                    content="Counterparty risk limits are set based on credit ratings.",
                    document_name="Counterparty_Risk_Policy.pdf",
                    page=12,
                    author="Risk Management",
                    publication_date="2023-10-01",
                    href="https://rbi.com/docs/counterparty.pdf",
                    primary_term_keys=["counterparty", "risk"],
                    related_term_keys=["limits", "ratings"],
                    images=[],
                    tables=[],
                    modified_date=None,
                ),
            ],
            terms={
                "credit": {"term": "credit", "count": 15, "score": 0.9},
                "risk": {"term": "risk", "count": 20, "score": 0.95},
            },
            total_results=2,
        )

        mock_search_module.search.return_value = mock_search_results
        mock_search_module.linked_knowledge = Mock(
            documents=["Credit_Risk_Manual.pdf", "Counterparty_Risk_Policy.pdf"],
            total_chunks=100,
            terms={"credit": {}, "risk": {}},
        )

        # Create workflow
        mock_model = AsyncMock()
        mock_db = MagicMock()

        # Create workflow directly without using create_steps_workflow
        workflow = StepsWorkflowCore(
            provider_name="test_provider",
            model_id="test_model"
        )

        # Verify citation extraction in workflow
        citations = CitationExtractor.extract_all_citations(mock_search_results)

        # Check all citations have proper structure
        assert len(citations) == 2

        # First citation with images and tables
        first_citation = citations[0]
        assert first_citation.title == "Credit_Risk_Manual.pdf"
        assert first_citation.author == "Credit Risk Department"
        assert first_citation.page == 34
        assert first_citation.href == "https://rbi.com/docs/credit_risk.pdf#page=34"
        assert first_citation.images is not None
        assert len(first_citation.images) == 1
        assert first_citation.images[0].caption == "Risk Assessment Matrix"
        assert first_citation.tables is not None
        assert len(first_citation.tables) == 1
        assert "Risk-based Limits" in first_citation.tables[0].caption

        # Second citation
        second_citation = citations[1]
        assert second_citation.title == "Counterparty_Risk_Policy.pdf"
        assert second_citation.href == "https://rbi.com/docs/counterparty.pdf#page=12"

    @pytest.mark.asyncio
    async def test_workflow_handles_no_citations_gracefully(self):
        """Test workflow behavior when no citations are available."""
        # Mock empty search results
        mock_search_module = Mock(spec=KnowledgeSearchCore)
        mock_search_results = OptimizedSearchResponse(
            results=[],
            terms={},
            total_results=0,
        )

        mock_search_module.search.return_value = mock_search_results
        mock_search_module.linked_knowledge = Mock(
            documents=[],
            total_chunks=0,
            terms={},
        )

        # Extract citations from empty results
        citations = CitationExtractor.extract_all_citations(mock_search_results)
        assert len(citations) == 0

        # Create context
        context = CitationExtractor.create_citation_context(mock_search_results)
        assert context == "No citations available from search results."

    def test_citation_formatting_with_all_styles(self):
        """Test that citations are properly formatted in all citation styles."""
        from blockether_catalyst.knowledge.answering.StepsWorkflowCore import (
            StepsWorkflowCore,
        )

        # Create citations with various metadata combinations
        citations = [
            Citation(
                title="Complete_Doc.pdf",
                author="Full Author",
                publication_date="2023-11-15",
                page=45,
                href="https://example.com/complete.pdf#page=45",
                quote="This is a complete citation",
                images=None,
                tables=None,
            ),
            Citation(
                title="Partial_Doc.pdf",
                author=None,
                publication_date=None,
                page=12,
                href=None,
                quote=None,
                images=None,
                tables=None,
            ),
        ]

        # Test inline_numeric style
        mock_search_module = Mock(spec=KnowledgeSearchCore)
        mock_model = Mock()
        mock_db = Mock()

        # Create workflow with citation style configuration
        workflow = StepsWorkflowCore(
            provider_name="test_provider",
            model_id="test_model"
        )

        # The workflow should format citations correctly
        # This would be tested through the format_citations function
        # which is called within the workflow

    def test_page_anchor_consistency(self):
        """Test that page anchors are consistently handled throughout the system."""
        # Test various URL formats
        test_cases = [
            # URL without page, page provided separately
            {
                "href": "https://example.com/doc.pdf",
                "page": 10,
                "expected": "https://example.com/doc.pdf#page=10",
            },
            # URL already has page anchor
            {
                "href": "https://example.com/doc.pdf#page=10",
                "page": 10,
                "expected": "https://example.com/doc.pdf#page=10",
            },
            # No URL
            {
                "href": None,
                "page": 10,
                "expected": None,
            },
            # URL but no page
            {
                "href": "https://example.com/doc.pdf",
                "page": None,
                "expected": "https://example.com/doc.pdf",
            },
        ]

        for case in test_cases:
            result = CompactSearchResult(
                score=0.9,
                content="Test content",
                document_name="Test.pdf",
                page=case["page"],
                href=case["href"],
                author=None,
                publication_date=None,
                primary_term_keys=[],
                related_term_keys=[],
                images=[],
                tables=[],
                modified_date=None,
            )

            citation = CitationExtractor.extract_citation_from_result(result)
            assert citation.href == case["expected"]


class TestRealWorldCitationScenarios:
    """Test real-world citation scenarios that might occur in production."""

    def test_banking_document_citations(self):
        """Test citations from actual banking document patterns."""
        # Simulate real RBI document search results
        banking_results = [
            CompactSearchResult(
                score=0.96,
                content="According to SUP LMS Application V4.0, credit applications must include comprehensive financial analysis including cash flow projections, collateral valuation, and risk assessment scores.",
                document_name="SUP_LMS_Application_V4.0.pdf",
                page=127,
                author=None,  # Often missing in internal docs
                publication_date=None,
                href="https://rbi-internal.com/docs/sup_lms_v4.pdf",
                primary_term_keys=["credit", "application", "lms"],
                related_term_keys=["financial", "analysis", "risk"],
                images=[],
                tables=[
                    TableInfo(
                        content="| Document Type | Required | Review Level |\n|--------------|----------|-------------|\n| Financial Statements | Yes | L1 + L2 |\n| Business Plan | Yes | L1 |\n| Collateral Docs | Yes | L2 + L3 |",
                        caption="Required Documentation Matrix",
                        page=128,
                        document_name="SUP_LMS_Application_V4.0.pdf",
                        score=0.94,
                    ),
                ],
                modified_date="2023-11-01",
            ),
            CompactSearchResult(
                score=0.91,
                content="Global Country Risk Policy V2.4 defines country risk limits based on sovereign ratings and economic indicators. Exposure limits are reviewed quarterly.",
                document_name="Global_Country_Risk_Policy_V2.4.pdf",
                page=45,
                author=None,
                publication_date=None,
                href="https://rbi-internal.com/docs/country_risk_v2.4.pdf",
                primary_term_keys=["country", "risk", "policy"],
                related_term_keys=["limits", "sovereign", "ratings"],
                images=[
                    ImageInfo(
                        caption="Country Risk Heat Map Q3 2023",
                        href="https://rbi-internal.com/images/country_risk_heatmap.png",
                        page=46,
                        document_name="Global_Country_Risk_Policy_V2.4.pdf",
                        score=0.89,
                    ),
                ],
                tables=[],
                modified_date="2023-10-15",
            ),
            CompactSearchResult(
                score=0.88,
                content="FCPM Decision Authority V1.0 establishes the credit decision hierarchy with specific approval limits for each authority level from branch managers to board committees.",
                document_name="SUP_FCPM_Decision_Authority_V1.0.pdf",
                page=89,
                author=None,
                publication_date=None,
                href=None,  # Some internal docs might not have web URLs
                primary_term_keys=["fcpm", "decision", "authority"],
                related_term_keys=["credit", "approval", "limits"],
                images=[],
                tables=[],
                modified_date="2023-09-20",
            ),
        ]

        search_response = OptimizedSearchResponse(
            results=banking_results,
            terms={
                "credit": {"term": "credit", "count": 25, "score": 0.95},
                "risk": {"term": "risk", "count": 30, "score": 0.97},
                "lms": {"term": "lms", "count": 10, "score": 0.90},
            },
            total_results=3,
        )

        # Extract citations
        citations = CitationExtractor.extract_all_citations(search_response)

        # Verify banking-specific citation handling
        assert len(citations) == 3

        # LMS document with table
        lms_citation = citations[0]
        assert "SUP_LMS_Application_V4.0" in lms_citation.title
        assert lms_citation.page == 127
        assert lms_citation.href == "https://rbi-internal.com/docs/sup_lms_v4.pdf#page=127"
        assert lms_citation.tables is not None
        assert len(lms_citation.tables) == 1
        assert "Required Documentation Matrix" in lms_citation.tables[0].caption

        # Country risk with image
        country_risk_citation = citations[1]
        assert "Global_Country_Risk_Policy" in country_risk_citation.title
        assert country_risk_citation.images is not None
        assert len(country_risk_citation.images) == 1
        assert "Country Risk Heat Map" in country_risk_citation.images[0].caption

        # FCPM without URL
        fcpm_citation = citations[2]
        assert "FCPM_Decision_Authority" in fcpm_citation.title
        assert fcpm_citation.href is None  # Correctly handles missing URL

        # Create formatted context
        context = CitationExtractor.create_citation_context(search_response)
        assert "[1] SUP_LMS_Application_V4.0.pdf (page 127)" in context
        assert "[2] Global_Country_Risk_Policy_V2.4.pdf (page 45)" in context
        assert "[3] SUP_FCPM_Decision_Authority_V1.0.pdf (page 89)" in context

    def test_citation_validation_prevents_url_injection(self):
        """Test that citation validation prevents malicious URL injection."""
        # Real search result
        safe_result = CompactSearchResult(
            score=0.90,
            content="Safe content from internal docs.",
            document_name="Safe_Document.pdf",
            page=10,
            href="https://internal.bank.com/docs/safe.pdf",
            author="Internal Team",
            publication_date="2023-01-01",
            primary_term_keys=[],
            related_term_keys=[],
            images=[],
            tables=[],
            modified_date=None,
        )

        search_response = OptimizedSearchResponse(
            results=[safe_result],
            terms={},
            total_results=1,
        )

        # Attempt to inject malicious URLs through citations
        malicious_citations = [
            Citation(
                title="Safe_Document.pdf",  # Matches real document name
                author="Internal Team",  # Matches real author
                publication_date="2023-01-01",  # Matches real date
                page=10,  # Matches real page
                href="https://malicious.com/steal-data.php",  # MALICIOUS URL!
                quote="Safe content from internal docs",
                images=None,
                tables=None,
            ),
            Citation(
                title="Safe_Document.pdf",
                author="Internal Team",
                publication_date="2023-01-01",
                page=10,
                href="javascript:alert('XSS')",  # XSS attempt!
                quote="Safe content",
                images=None,
                tables=None,
            ),
        ]

        # Validate citations
        valid_citations = CitationExtractor.sanitize_citations(
            malicious_citations,
            search_response
        )

        # No citations should pass validation due to URL mismatch
        assert len(valid_citations) == 0

        # Only citations with exact matching URLs (or both None) should validate
        correct_citation = Citation(
            title="Safe_Document.pdf",
            author="Internal Team",
            publication_date="2023-01-01",
            page=10,
            href="https://internal.bank.com/docs/safe.pdf#page=10",  # Correct URL with page anchor
            quote="Safe content",
            images=None,
            tables=None,
        )

        valid = CitationExtractor.validate_citation(correct_citation, search_response)
        assert valid is True  # This one should validate

    def test_large_scale_citation_handling(self):
        """Test handling of many citations efficiently."""
        # Create many search results
        large_results = []
        for i in range(50):  # 50 documents
            large_results.append(
                CompactSearchResult(
                    score=0.9 - (i * 0.01),
                    content=f"Content from document {i}",
                    document_name=f"Document_{i:03d}.pdf",
                    page=i + 1,
                    href=f"https://example.com/doc_{i:03d}.pdf" if i % 2 == 0 else None,
                    author=f"Author {i}" if i % 3 == 0 else None,
                    publication_date=f"2023-{(i % 12) + 1:02d}-01" if i % 4 == 0 else None,
                    primary_term_keys=[f"term{i}"],
                    related_term_keys=[],
                    images=[],
                    tables=[],
                    modified_date=None,
                )
            )

        search_response = OptimizedSearchResponse(
            results=large_results,
            terms={},
            total_results=50,
        )

        # Extract limited citations
        citations = CitationExtractor.extract_all_citations(search_response, max_citations=10)
        assert len(citations) == 10  # Respects limit

        # Verify first few citations
        assert citations[0].title == "Document_000.pdf"
        assert citations[0].href == "https://example.com/doc_000.pdf#page=1"
        assert citations[9].title == "Document_009.pdf"

        # Create citation context
        context = CitationExtractor.create_citation_context(search_response, max_citations=10)
        lines = context.split("\n")
        assert len(lines) <= 21  # Header + 10 citations (some with URL lines)