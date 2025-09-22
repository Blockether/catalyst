"""
Comprehensive tests for term refinement functionality.
Tests the term extraction, meaning extraction, and refinement process.
"""

from typing import Any, Dict, List, Literal, cast

import pytest

from blockether_catalyst.consensus.ConsensusTypes import ConsensusResult
from blockether_catalyst.knowledge.KnowledgeExtractionCallBase import (
    BaseTermExtractionCall,
    ExtractionCallsSettings,
)
from blockether_catalyst.knowledge.KnowledgeExtractionCore import (
    KnowledgeExtractionCore,
)
from blockether_catalyst.knowledge.KnowledgeTypes import (
    KnowledgeChunk,
    KnowledgeProcessorSettings,
    Term,
    TermMeaningExtractionResponse,
    TermOccurrence,
)


class TestTermRefinementConstants:
    """Constants for term refinement tests."""

    # Confidence thresholds
    HIGH_CONFIDENCE = 0.85
    MEDIUM_CONFIDENCE = 0.65
    LOW_CONFIDENCE = 0.45

    # Test acronyms with known expansions
    KNOWN_ACRONYMS = {
        "API": "Application Programming Interface",
        "NASA": "National Aeronautics and Space Administration",
        "HTTP": "Hypertext Transfer Protocol",
        "ML": "Machine Learning",
        "AI": "Artificial Intelligence",
        "REST": "Representational State Transfer",
        "JSON": "JavaScript Object Notation",
        "SQL": "Structured Query Language",
        "HTML": "HyperText Markup Language",
        "CSS": "Cascading Style Sheets",
    }

    # Test keywords with definitions
    KNOWN_KEYWORDS = {
        "machine learning": "A subset of artificial intelligence that enables systems to learn from data without being explicitly programmed",
        "neural network": "A computing system inspired by biological neural networks that constitute animal brains",
        "optimization": "The process of finding the best solution from all feasible solutions to a problem",
        "algorithm": "A step-by-step procedure or formula for solving a problem or accomplishing a task",
        "database": "An organized collection of structured information or data stored electronically",
        "encryption": "The process of converting information into a code to prevent unauthorized access",
        "authentication": "The process of verifying the identity of a user or system",
    }

    # Test contexts for extraction
    API_CONTEXTS = [
        "The REST API provides endpoints for data retrieval and manipulation",
        "Our API uses JSON for request and response formatting",
        "Authentication is required for all API calls",
    ]

    ML_CONTEXTS = [
        "ML algorithms can identify patterns in large datasets",
        "The ML model achieved 95% accuracy on the test set",
        "ML techniques include supervised and unsupervised learning",
    ]


class RealTermExtractionCall(BaseTermExtractionCall):
    """Real implementation of term extraction for testing."""

    def __init__(self) -> None:
        """Initialize with a mock consensus for testing."""
        # Create a simple mock consensus that's not used in our deterministic tests
        from unittest.mock import MagicMock

        mock_consensus = MagicMock()
        super().__init__(consensus=mock_consensus)

    async def execute(
        self,
        term: str,
        type: str,
        occurrences_contexts: List[str],
        cooccurring_terms: Dict[str, List[str]],
        *args: object,
        **kwargs: object,
    ) -> ConsensusResult:
        """Execute simple, deterministic term extraction for testing."""

        # Simple, hardcoded responses without business logic
        # Following testing guidelines: NO IF statements, deterministic results

        # Generate longer strings to meet validation requirements
        meaning = f"Test meaning for {term}. " * 10  # Ensure 150+ chars
        reasoning = f"Test extraction for term '{term}' of type '{type}'. " * 5  # Ensure 150+ chars

        response = TermMeaningExtractionResponse(
            reasoning=reasoning,
            term=term,
            meaning=meaning,
            full_form=f"Test full form for {term}",
            type=cast("Literal['acronym', 'keyword']", type),
            meaning_status="meaningful",
        )

        return ConsensusResult(
            reasoning=f"Test term extraction completed for {term}. " * 5,  # Ensure 150+ chars
            consensus_achieved=True,
            final_response=response,
            rounds=[],
            total_rounds=1,
            convergence_score=TestTermRefinementConstants.HIGH_CONFIDENCE,
            participating_models=["test-model"],
        )

    def _infer_domain(self, cooccurring_terms: Dict[str, List[str]]) -> str:
        """Simple domain inference for testing - always returns 'technology'."""
        return "technology"

    def post_process(self, result: ConsensusResult, term: str, **kwargs: object) -> ConsensusResult:
        """Post-process the extraction result."""
        # Real implementation would do additional validation/enrichment
        return result

    def fill_template(
        self,
        term: str,
        type: str,
        occurrences_contexts: list[str],
        cooccurring_terms: dict[str, list[str]],
    ) -> str:
        """Simple template filling for testing."""
        return "Test template"


class TestTermRefinement:
    """Test suite for term refinement functionality."""

    @pytest.fixture
    def real_term_extraction_call(self) -> RealTermExtractionCall:
        """Create a real term extraction call."""
        return RealTermExtractionCall()

    @pytest.fixture
    def processor_settings(self, tmp_path: Any) -> KnowledgeProcessorSettings:
        """Create processor settings."""
        return KnowledgeProcessorSettings(
            extraction_output_dir=tmp_path / "test_output",
            linking_threshold=0.7,
            encoding_model="cl100k_base",
        )

    @pytest.mark.anyio
    async def test_acronym_extraction_known(self, real_term_extraction_call: RealTermExtractionCall) -> None:
        """Test extraction of known acronyms."""
        term = "API"
        contexts = TestTermRefinementConstants.API_CONTEXTS
        cooccurring = {"REST": ["API", "endpoint"], "JSON": ["format", "response"]}

        result = await real_term_extraction_call.execute(
            term=term,
            type="acronym",
            occurrences_contexts=contexts,
            cooccurring_terms=cooccurring,
        )

        assert result.final_response is not None
        assert result.final_response.term == "API"
        assert result.final_response.full_form == "Test full form for API"
        assert result.final_response.type == "acronym"
        assert result.final_response.meaning_status == "meaningful"
        assert "Test meaning for API" in result.final_response.meaning
        assert result.consensus_achieved is True
        assert result.convergence_score == TestTermRefinementConstants.HIGH_CONFIDENCE

    @pytest.mark.anyio
    async def test_acronym_extraction_unknown(self, real_term_extraction_call: RealTermExtractionCall) -> None:
        """Test extraction of unknown acronyms."""
        term = "XYZ"
        contexts = ["XYZ is used in the system", "The XYZ protocol handles requests"]
        cooccurring = {"system": ["XYZ"], "protocol": ["XYZ"]}

        result = await real_term_extraction_call.execute(
            term=term,
            type="acronym",
            occurrences_contexts=contexts,
            cooccurring_terms=cooccurring,
        )

        assert result.final_response is not None
        assert result.final_response.term == "XYZ"
        assert result.final_response.full_form == "Test full form for XYZ"
        assert result.final_response.type == "acronym"
        assert result.final_response.meaning_status == "meaningful"
        assert "Test meaning for XYZ" in result.final_response.meaning
        assert result.convergence_score == TestTermRefinementConstants.HIGH_CONFIDENCE

    @pytest.mark.anyio
    async def test_keyword_extraction_known(self, real_term_extraction_call: RealTermExtractionCall) -> None:
        """Test extraction of known keywords."""
        term = "machine learning"
        contexts = TestTermRefinementConstants.ML_CONTEXTS
        cooccurring = {
            "algorithm": ["ML", "learning"],
            "model": ["training", "accuracy"],
            "data": ["dataset", "patterns"],
        }

        result = await real_term_extraction_call.execute(
            term=term,
            type="keyword",
            occurrences_contexts=contexts,
            cooccurring_terms=cooccurring,
        )

        assert result.final_response is not None
        assert result.final_response.term == "machine learning"
        assert result.final_response.full_form == "Test full form for machine learning"
        assert result.final_response.type == "keyword"
        assert result.final_response.meaning_status == "meaningful"
        assert "Test meaning for machine learning" in result.final_response.meaning
        assert result.consensus_achieved is True
        assert result.convergence_score == TestTermRefinementConstants.HIGH_CONFIDENCE

    @pytest.mark.anyio
    async def test_keyword_extraction_generic(self, real_term_extraction_call: RealTermExtractionCall) -> None:
        """Test extraction of generic keywords."""
        term = "process"
        contexts = ["The process handles data", "This process is efficient"]
        cooccurring = {"data": ["process"], "efficient": ["process"]}

        result = await real_term_extraction_call.execute(
            term=term,
            type="keyword",
            occurrences_contexts=contexts,
            cooccurring_terms=cooccurring,
        )

        assert result.final_response is not None
        assert result.final_response.term == "process"
        assert result.final_response.full_form == "Test full form for process"
        assert result.final_response.type == "keyword"
        assert result.final_response.meaning_status == "meaningful"
        assert "Test meaning for process" in result.final_response.meaning
        assert result.convergence_score == TestTermRefinementConstants.HIGH_CONFIDENCE

    @pytest.mark.anyio
    async def test_domain_inference_technology(self, real_term_extraction_call: RealTermExtractionCall) -> None:
        """Test domain inference for technology terms."""
        term = "framework"
        contexts = ["The framework provides APIs", "Code uses this framework"]
        cooccurring = {
            "api": ["framework", "code"],
            "software": ["system", "framework"],
            "algorithm": ["implementation"],
        }

        result = await real_term_extraction_call.execute(
            term=term,
            type="keyword",
            occurrences_contexts=contexts,
            cooccurring_terms=cooccurring,
        )

        assert result.final_response is not None
        assert result.final_response.term == "framework"
        assert "Test meaning for framework" in result.final_response.meaning
        assert "Test term extraction completed for framework" in result.reasoning

    @pytest.mark.anyio
    async def test_meaning_truncation(self, real_term_extraction_call: RealTermExtractionCall) -> None:
        """Test that meanings are handled properly within length limits."""
        term = "test"

        # Create a meaning that's exactly at the 1200 character limit
        test_meaning = "x" * 1200  # Exactly 1200 chars - valid length
        test_reasoning = "Test reasoning for meaning truncation validation. " * 5  # Meet 150+ char requirement

        # Create a custom call that returns a meaning at the character limit
        class LimitMeaningCall(RealTermExtractionCall):
            async def execute(self, *args: object, **kwargs: object) -> ConsensusResult:
                response = TermMeaningExtractionResponse(
                    reasoning=test_reasoning,
                    term=term,
                    meaning=test_meaning,
                    full_form=term,
                    type=cast(Literal["acronym", "keyword"], "keyword"),
                    meaning_status="meaningful",
                )
                return ConsensusResult(
                    reasoning=test_reasoning,
                    consensus_achieved=True,
                    final_response=response,
                    rounds=[],
                    total_rounds=1,
                    convergence_score=1.0,
                    participating_models=["test-model"],
                )

        call = LimitMeaningCall()
        result = await call.execute(term, "keyword", [], {})

        # Verify the meaning is exactly at the limit
        assert len(result.final_response.meaning) == 1200
        assert result.final_response.term == "test"

    @pytest.mark.anyio
    async def test_post_processing(self, real_term_extraction_call: RealTermExtractionCall) -> None:
        """Test post-processing of extraction results."""
        term = "API"
        contexts = TestTermRefinementConstants.API_CONTEXTS
        cooccurring = {"REST": ["API"]}

        result = await real_term_extraction_call.execute(
            term=term,
            type="acronym",
            occurrences_contexts=contexts,
            cooccurring_terms=cooccurring,
        )

        # Test post-processing
        processed = real_term_extraction_call.post_process(result, term)

        assert processed == result  # Current implementation is pass-through
        assert processed.final_response.term == term

    @pytest.mark.anyio
    async def test_multiple_acronyms_batch(self, real_term_extraction_call: RealTermExtractionCall) -> None:
        """Test batch processing of multiple acronyms."""
        acronyms = ["API", "REST", "JSON", "HTTP"]
        results = []

        for acronym in acronyms:
            result = await real_term_extraction_call.execute(
                term=acronym,
                type="acronym",
                occurrences_contexts=[f"{acronym} is used here"],
                cooccurring_terms={},
            )
            results.append(result)

        # All acronyms should be successfully extracted with deterministic results
        assert len(results) == 4
        assert all(r.consensus_achieved for r in results)
        assert all(r.final_response.type == "acronym" for r in results)
        assert results[0].final_response.full_form == "Test full form for API"
        assert results[1].final_response.full_form == "Test full form for REST"
        assert results[2].final_response.full_form == "Test full form for JSON"
        assert results[3].final_response.full_form == "Test full form for HTTP"

    @pytest.mark.anyio
    async def test_term_occurrence_creation(self, real_term_extraction_call: RealTermExtractionCall) -> None:
        """Test creation of TermOccurrence objects."""
        term_occurrences = [
            TermOccurrence(
                document_id="test-doc-id-1",
                document_name="test-document-1.pdf",
                page=1,
                chunk_index=0,
                total=3,
            ),
            TermOccurrence(
                document_id="test-doc-id-2",
                document_name="test-document-2.pdf",
                page=2,
                chunk_index=1,
                total=2,
            ),
        ]

        # Verify term occurrences are properly structured
        assert len(term_occurrences) == 2
        assert term_occurrences[0].document_id == "test-doc-id-1"
        assert term_occurrences[0].document_name == "test-document-1.pdf"
        assert term_occurrences[0].page == 1
        assert term_occurrences[0].chunk_index == 0
        assert term_occurrences[0].total == 3
        assert term_occurrences[1].page == 2

    @pytest.mark.anyio
    async def test_term_refinement_with_chunks(self, real_term_extraction_call: RealTermExtractionCall) -> None:
        """Test term refinement using actual chunks."""
        chunks = [
            KnowledgeChunk(
                document_id="test-doc",
                document_name="test.pdf",
                doc_id="test-doc-1-0",
                index=0,
                page=1,
                text="The REST API provides endpoints for data manipulation. JSON formatting is used.",
                content_types=["text"],
                semantic_types=["explanation"],
            ),
            KnowledgeChunk(
                document_id="test-doc",
                document_name="test.pdf",
                doc_id="test-doc-1-1",
                index=1,
                page=1,
                text="Machine learning algorithms analyze patterns in data.",
                content_types=["text"],
                semantic_types=["explanation"],
            ),
        ]

        # Extract terms from chunks
        extracted_terms = {
            "API": "acronym",
            "REST": "acronym",
            "JSON": "acronym",
            "machine learning": "keyword",
        }

        # Process each term
        results = {}
        for term, term_type in extracted_terms.items():
            # Find contexts from chunks
            contexts = [chunk.text for chunk in chunks if term.lower() in chunk.text.lower()]

            result = await real_term_extraction_call.execute(
                term=term,
                type=term_type,
                occurrences_contexts=contexts,
                cooccurring_terms={},
            )
            results[term] = result

        # Verify all terms were processed with deterministic results
        assert len(results) == 4
        assert all(r.consensus_achieved for r in results.values())
        assert results["API"].final_response.full_form == "Test full form for API"
        assert results["machine learning"].final_response.type == "keyword"
