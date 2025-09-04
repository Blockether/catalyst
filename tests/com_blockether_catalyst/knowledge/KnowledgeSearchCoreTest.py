"""
Comprehensive test suite for KnowledgeSearchCore functionality.

This module combines:
- Unit tests for search functionality and enrichment
- Integration tests for pickle persistence with real data
- Performance tests for initialization and search operations
"""

import json
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

from com_blockether_catalyst.knowledge.internal.KnowledgeExtractionBaseTypes import (
    KnowledgeTableData,
)
from com_blockether_catalyst.knowledge.internal.KnowledgeExtractionTypes import (
    DocumentMetadata,
    KnowledgeChunkWithTerms,
    KnowledgePageData,
    LinkedKnowledge,
    Term,
    TermCooccurrence,
    TermLink,
    TermOccurrence,
)
from com_blockether_catalyst.knowledge.KnowledgeSearchCore import (
    KnowledgeSearchCore,
    KnowledgeSearchResult,
    SimilaritySearchResult,
)

# ============================================================================
# Shared Fixtures and Helper Functions
# ============================================================================


def create_sample_dataset() -> LinkedKnowledge:
    """Create a comprehensive sample dataset for testing."""

    # Create sample terms
    terms = {
        "ml_term": Term(
            term="machine learning",
            term_type="keyword",
            full_form="machine learning",
            meaning="A method of data analysis that automates analytical model building",
            occurrences=[
                TermOccurrence(document_id="doc1", document_name="ML Guide", chunk_index=0, page=1),
                TermOccurrence(
                    document_id="doc2",
                    document_name="AI Overview",
                    chunk_index=1,
                    page=1,
                ),
            ],
            cooccurrences=[
                TermCooccurrence(term="artificial intelligence", frequency=5, confidence=0.8),
                TermCooccurrence(term="neural networks", frequency=3, confidence=0.7),
                TermCooccurrence(term="deep learning", frequency=4, confidence=0.9),
            ],
        ),
        "ai_term": Term(
            term="artificial intelligence",
            term_type="keyword",
            full_form="artificial intelligence",
            meaning="The simulation of human intelligence in machines",
            occurrences=[
                TermOccurrence(
                    document_id="doc2",
                    document_name="AI Overview",
                    chunk_index=0,
                    page=1,
                )
            ],
            cooccurrences=[
                TermCooccurrence(term="machine learning", frequency=5, confidence=0.8),
                TermCooccurrence(term="robotics", frequency=2, confidence=0.6),
            ],
        ),
        "api_acronym": Term(
            term="API",
            term_type="acronym",
            full_form="Application Programming Interface",
            meaning="A set of protocols and tools for building software applications",
            occurrences=[TermOccurrence(document_id="doc1", document_name="ML Guide", chunk_index=2, page=2)],
            cooccurrences=[
                TermCooccurrence(term="REST", frequency=8, confidence=0.9),
                TermCooccurrence(term="HTTP", frequency=6, confidence=0.8),
            ],
        ),
        "rest_acronym": Term(
            term="REST",
            term_type="acronym",
            full_form="Representational State Transfer",
            meaning="An architectural style for distributed hypermedia systems",
            occurrences=[TermOccurrence(document_id="doc1", document_name="ML Guide", chunk_index=2, page=2)],
            cooccurrences=[
                TermCooccurrence(term="API", frequency=8, confidence=0.9),
                TermCooccurrence(term="HTTP", frequency=7, confidence=0.85),
            ],
        ),
    }

    # Create sample documents
    documents = {
        "doc1": DocumentMetadata(
            document_id="doc1",
            filename="ml_guide.pdf",
            total_pages=3,
            total_chunks=3,
            total_terms=3,
            total_tables=0,
            total_acronyms=1,
            total_keywords=2,
        ),
        "doc2": DocumentMetadata(
            document_id="doc2",
            filename="ai_overview.pdf",
            total_pages=2,
            total_chunks=1,
            total_terms=2,
            total_tables=0,
            total_acronyms=0,
            total_keywords=2,
        ),
    }

    # Create sample chunks
    chunks = {
        "chunk_1": KnowledgeChunkWithTerms(
            document_id="doc1",
            document_name="ML Guide",
            doc_id="chunk_1",
            index=0,
            page=1,
            text="Machine learning is a subset of artificial intelligence that enables computers to learn without being explicitly programmed.",
            terms=["machine learning", "artificial intelligence"],
        ),
        "chunk_2": KnowledgeChunkWithTerms(
            document_id="doc2",
            document_name="AI Overview",
            doc_id="chunk_2",
            index=0,
            page=1,
            text="Artificial intelligence has revolutionized many industries through automation and data analysis.",
            terms=["artificial intelligence"],
        ),
        "chunk_3": KnowledgeChunkWithTerms(
            document_id="doc1",
            document_name="ML Guide",
            doc_id="chunk_3",
            index=1,
            page=1,
            text="Modern machine learning algorithms can process vast amounts of data to identify patterns and make predictions.",
            terms=["machine learning"],
        ),
        "chunk_4": KnowledgeChunkWithTerms(
            document_id="doc1",
            document_name="ML Guide",
            doc_id="chunk_4",
            index=2,
            page=2,
            text="RESTful APIs provide a standardized way to interact with web services using HTTP methods.",
            terms=["API", "REST"],
        ),
    }

    # Create sample tables
    sample_table = KnowledgeTableData(
        page=2,
        rows=3,
        columns=2,
        data=[
            ["Method", "Accuracy"],
            ["Linear Regression", "0.85"],
            ["Random Forest", "0.92"],
        ],
    )

    # Create sample pages
    pages = {
        ("doc1", 1): KnowledgePageData(
            page=1,
            text="Machine learning content from page 1",
            images=["/fake/path/image1.png"],
            tables=[],
        ),
        ("doc1", 2): KnowledgePageData(page=2, text="API content from page 2", images=[], tables=[sample_table]),
        ("doc2", 1): KnowledgePageData(page=1, text="AI overview content from page 1", images=[], tables=[]),
    }

    # Build indices
    term_to_documents_index = {
        "machine learning": {"doc1", "doc2"},
        "artificial intelligence": {"doc2"},
        "API": {"doc1"},
        "REST": {"doc1"},
    }

    document_to_terms_index = {
        "doc1": {"machine learning", "API", "REST"},
        "doc2": {"machine learning", "artificial intelligence"},
    }

    document_to_chunk_ids_index = {
        "doc1": {"chunk_1", "chunk_3", "chunk_4"},
        "doc2": {"chunk_2"},
    }

    return LinkedKnowledge(
        documents=documents,
        terms=terms,
        chunks=chunks,
        pages=pages,
        term_to_documents_index=term_to_documents_index,
        document_to_terms_index=document_to_terms_index,
        document_to_chunk_ids_index=document_to_chunk_ids_index,
        total_acronyms=2,  # API, ML
        total_keywords=2,  # machine learning, neural network
        total_chunks=3,  # 3 chunks total
    )


@pytest.fixture
def sample_document_chunks() -> List[KnowledgeChunkWithTerms]:
    """Create sample document chunks for testing."""
    return [
        KnowledgeChunkWithTerms(
            document_id="doc1",
            document_name="ml_guide.pdf",
            doc_id="doc1_p1_c0",
            index=0,
            text="Machine learning algorithms require significant computational power and data preprocessing.",
            page=1,
            terms=["machine learning", "algorithms"],
        ),
        KnowledgeChunkWithTerms(
            document_id="doc1",
            document_name="ml_guide.pdf",
            doc_id="doc1_p1_c1",
            index=1,
            text="API endpoints provide REST interface for accessing ML models and services.",
            page=1,
            terms=["API", "ML"],
        ),
        KnowledgeChunkWithTerms(
            document_id="doc1",
            document_name="ml_guide.pdf",
            doc_id="doc1_p2_c2",
            index=2,
            text="Deep learning neural networks process complex patterns in large datasets.",
            page=2,
            terms=["neural networks"],
        ),
    ]


@pytest.fixture
def sample_document() -> DocumentMetadata:
    """Create a sample document metadata."""
    return DocumentMetadata(
        document_id="doc1",
        filename="ml_guide.pdf",
        total_pages=2,
        total_chunks=3,
        total_terms=5,
        total_tables=0,
        total_acronyms=1,
        total_keywords=4,
    )


@pytest.fixture
def sample_terms() -> Dict[str, Term]:
    """Create sample terms with meanings and co-occurrences."""
    ml_keyword = Term(
        term="machine learning",
        term_type="keyword",
        full_form="machine learning",
        occurrences=[
            TermOccurrence(document_id="doc1", document_name="ml_guide.pdf", page=1, chunk_index=0),
            TermOccurrence(document_id="doc1", document_name="ml_guide.pdf", page=2, chunk_index=2),
        ],
        cooccurrences=[
            TermCooccurrence(term="algorithms", frequency=3, confidence=0.9),
            TermCooccurrence(term="API", frequency=2, confidence=0.7),
            TermCooccurrence(term="neural networks", frequency=2, confidence=0.8),
        ],
        total_count=2,
        mean_score=0.95,
        meaning="A method of data analysis that automates analytical model building",
        reasoning="Core AI concept with clear definition",
    )

    api_acronym = Term(
        term="API",
        term_type="acronym",
        full_form="Application Programming Interface",
        occurrences=[TermOccurrence(document_id="doc1", document_name="ml_guide.pdf", page=1, chunk_index=1)],
        cooccurrences=[
            TermCooccurrence(term="REST", frequency=3, confidence=0.95),
            TermCooccurrence(term="endpoints", frequency=2, confidence=0.85),
            TermCooccurrence(term="machine learning", frequency=1, confidence=0.6),
        ],
        total_count=1,
        mean_score=0.85,
        meaning="A set of protocols and tools for building software applications",
        reasoning="Standard technical acronym",
    )

    ml_acronym = Term(
        term="ML",
        term_type="acronym",
        full_form="Machine Learning",
        occurrences=[TermOccurrence(document_id="doc1", document_name="ml_guide.pdf", page=1, chunk_index=1)],
        cooccurrences=[
            TermCooccurrence(term="models", frequency=2, confidence=0.9),
            TermCooccurrence(term="API", frequency=1, confidence=0.7),
        ],
        total_count=1,
        mean_score=0.9,
        meaning="Machine Learning - automated learning from data",
        reasoning="Common ML abbreviation",
    )

    algorithms_keyword = Term(
        term="algorithms",
        term_type="keyword",
        full_form="algorithms",
        occurrences=[TermOccurrence(document_id="doc1", document_name="ml_guide.pdf", page=1, chunk_index=0)],
        cooccurrences=[
            TermCooccurrence(term="machine learning", frequency=3, confidence=0.9),
            TermCooccurrence(term="computational", frequency=2, confidence=0.8),
        ],
        total_count=1,
        mean_score=0.75,
        meaning="A set of rules or instructions for solving a problem",
        reasoning="Fundamental CS concept",
    )

    neural_networks_keyword = Term(
        term="neural networks",
        term_type="keyword",
        full_form="neural networks",
        occurrences=[TermOccurrence(document_id="doc1", document_name="ml_guide.pdf", page=2, chunk_index=2)],
        cooccurrences=[
            TermCooccurrence(term="deep learning", frequency=3, confidence=0.95),
            TermCooccurrence(term="machine learning", frequency=2, confidence=0.8),
        ],
        total_count=1,
        mean_score=0.88,
        meaning="Computing systems inspired by biological neural networks",
        reasoning="Key deep learning concept",
    )

    return {
        "machine learning": ml_keyword,
        "API": api_acronym,
        "ML": ml_acronym,
        "algorithms": algorithms_keyword,
        "neural networks": neural_networks_keyword,
    }


@pytest.fixture
def sample_links() -> List[TermLink]:
    """Create term links between acronyms and keywords."""
    return [
        TermLink(acronym="API", keyword="Application Programming Interface", match_score=0.98),
        TermLink(acronym="ML", keyword="machine learning", match_score=0.95),
    ]


@pytest.fixture
def sample_pages() -> Dict[Tuple[str, int], KnowledgePageData]:
    """Create sample page data with images and tables."""
    return {
        ("doc1", 1): KnowledgePageData(
            page=1,
            text="Machine learning algorithms require significant computational power.",
            tables=[
                KnowledgeTableData(
                    page=1,
                    rows=2,
                    columns=3,
                    data=[
                        ["Algorithm", "Complexity", "Accuracy"],
                        ["SVM", "O(n^2)", "95%"],
                    ],
                )
            ],
            images=["doc1_page1_image1.png"],
            lines=5,
        ),
        ("doc1", 2): KnowledgePageData(
            page=2,
            text="Deep learning neural networks process complex patterns.",
            tables=[],
            images=["doc1_page2_image1.png", "doc1_page2_image2.png"],
            lines=3,
        ),
    }


@pytest.fixture
def sample_linked_knowledge(
    sample_document: DocumentMetadata,
    sample_terms: Dict[str, Term],
    sample_links: List[TermLink],
    sample_document_chunks: List[KnowledgeChunkWithTerms],
    sample_pages: Dict[Tuple[str, int], KnowledgePageData],
) -> LinkedKnowledge:
    """Create complete LinkedKnowledge structure."""
    # Convert chunks to the new format
    chunks_dict = {chunk.doc_id: chunk for chunk in sample_document_chunks}

    # Build the document_to_chunk_ids_index
    document_to_chunk_ids_index = {"doc1": {"doc1_p1_c0", "doc1_p1_c1", "doc1_p2_c2"}}

    # Calculate totals from the sample data
    total_acronyms = sum(1 for term in sample_terms.values() if term.term_type == "acronym")
    total_keywords = sum(1 for term in sample_terms.values() if term.term_type == "keyword")
    total_chunks = len(chunks_dict)

    return LinkedKnowledge(
        documents={"doc1": sample_document},
        terms=sample_terms,
        links=sample_links,
        chunks=chunks_dict,
        pages=sample_pages,
        document_to_chunk_ids_index=document_to_chunk_ids_index,
        total_acronyms=total_acronyms,
        total_keywords=total_keywords,
        total_chunks=total_chunks,
    )


@pytest.fixture
def search_core(sample_linked_knowledge: LinkedKnowledge) -> KnowledgeSearchCore:
    """Create initialized KnowledgeSearchCore instance."""
    return KnowledgeSearchCore(sample_linked_knowledge)


# ============================================================================
# Unit Tests (from original KnowledgeSearchCoreTest.py)
# ============================================================================


class TestKnowledgeSearchCoreInitialization:
    """Test KnowledgeSearchCore initialization."""

    def test_initialization_with_linked_knowledge(self, sample_linked_knowledge: LinkedKnowledge) -> None:
        """Test proper initialization with LinkedKnowledge."""
        search_core = KnowledgeSearchCore(sample_linked_knowledge)

        assert search_core.linked_knowledge == sample_linked_knowledge
        assert search_core._vector_store is not None
        assert len(search_core._acronym_to_full_form) == 2  # API and ML

    def test_initialization_with_empty_knowledge(self) -> None:
        """Test initialization with empty LinkedKnowledge."""
        empty_knowledge = LinkedKnowledge(
            documents={},
            terms={},
            links=[],
            total_acronyms=0,
            total_keywords=0,
            total_chunks=0,
        )
        search_core = KnowledgeSearchCore(empty_knowledge)

        assert search_core.linked_knowledge == empty_knowledge
        assert len(search_core._acronym_to_full_form) == 0

    def test_initialization_builds_indices(self, sample_linked_knowledge: LinkedKnowledge) -> None:
        """Test that initialization properly builds search indices."""
        search_core = KnowledgeSearchCore(sample_linked_knowledge)

        # Check acronym mappings are built
        assert "API" in search_core._acronym_to_full_form
        assert "ML" in search_core._acronym_to_full_form
        assert search_core._acronym_to_full_form["API"] == "Application Programming Interface"
        assert search_core._acronym_to_full_form["ML"] == "Machine Learning"


class TestKnowledgeSearchCoreSearch:
    """Test search functionality."""

    def test_hybrid_search_returns_results(self, search_core: KnowledgeSearchCore) -> None:
        """Test that hybrid search returns appropriate results."""
        results = search_core.search_similarity("machine learning", k=5)

        assert len(results) <= 5
        assert all(isinstance(r, (SimilaritySearchResult, KnowledgeSearchResult)) for r in results)
        assert all(r.score >= 0 for r in results)

    def test_search_respects_k_parameter(self, search_core: KnowledgeSearchCore) -> None:
        """Test that search respects the k parameter."""
        results_k2 = search_core.search_similarity("API", k=2)
        results_k5 = search_core.search_similarity("API", k=5)

        assert len(results_k2) <= 2
        assert len(results_k5) <= 5

    def test_search_with_max_depth(self, search_core: KnowledgeSearchCore) -> None:
        """Test search with different max_depth values."""
        results_depth1 = search_core.search_enhanced("machine learning", max_depth=1, k=3)
        results_depth3 = search_core.search_enhanced("machine learning", max_depth=3, k=3)

        # Both should return valid results
        assert all(isinstance(r, (SimilaritySearchResult, KnowledgeSearchResult)) for r in results_depth1)
        assert all(isinstance(r, (SimilaritySearchResult, KnowledgeSearchResult)) for r in results_depth3)

    def test_search_with_threshold(self, search_core: KnowledgeSearchCore) -> None:
        """Test that search respects similarity threshold."""
        results_low_threshold = search_core.search_similarity("test", k=10, threshold=0.01)
        results_high_threshold = search_core.search_similarity("test", k=10, threshold=0.9)

        # High threshold should return fewer results
        assert len(results_high_threshold) <= len(results_low_threshold)

    def test_search_enriches_with_terms(self, search_core: KnowledgeSearchCore) -> None:
        """Test that enhanced search results are enriched with term information."""
        results = search_core.search_enhanced("machine learning algorithms", k=3)

        # At least one result should have primary terms
        results_with_terms = [r for r in results for _ in r.primary_terms]
        assert len(results_with_terms) > 0

    def test_search_includes_cooccurrences_in_related_terms(self, search_core: KnowledgeSearchCore) -> None:
        """Test that enhanced search results include co-occurring terms in related_terms."""
        results = search_core.search_enhanced("machine learning", max_cooccurrences=5, k=3)

        # Check that related_terms includes both linked terms and co-occurrences
        for result in results:
            assert isinstance(result.related_terms, list)
            # All items in related_terms should be Term objects
            assert all(isinstance(term, Term) for term in result.related_terms)

    def test_standard_search_mode(self, search_core: KnowledgeSearchCore) -> None:
        """Test standard search mode (enhanced=False) - no terms but includes media."""
        results = search_core.search_similarity("machine learning", k=5)

        assert len(results) <= 5
        assert all(isinstance(r, (SimilaritySearchResult, KnowledgeSearchResult)) for r in results)
        assert all(r.score >= 0 for r in results)

        # Standard search should NOT have terms
        for result in results:
            assert len(result.primary_terms) == 0
            assert len(result.related_terms) == 0
            # But should have images/tables fields
            assert hasattr(result, "images")
            assert hasattr(result, "tables")

    def test_enhanced_search_mode(self, search_core: KnowledgeSearchCore) -> None:
        """Test enhanced search mode (enhanced=True) - includes terms and media."""
        results = search_core.search_enhanced("machine learning", k=5)

        assert len(results) <= 5
        assert all(isinstance(r, (SimilaritySearchResult, KnowledgeSearchResult)) for r in results)
        assert all(r.score >= 0 for r in results)

        # Enhanced search should have terms for results that contain known terms
        ml_results = [r for r in results if "machine learning" in r.text.lower()]
        if len(ml_results) > 0:
            # At least the machine learning results should have primary terms
            ml_result = ml_results[0]
            ml_terms = [t for t in ml_result.primary_terms if t.term == "machine learning"]
            assert len(ml_terms) > 0, "Enhanced search should identify 'machine learning' as a primary term"

        # Should also have images/tables fields
        for result in results:
            assert hasattr(result, "images")
            assert hasattr(result, "tables")

    @pytest.mark.parametrize("enhanced", [True, False])
    def test_both_search_modes_return_valid_results(self, search_core: KnowledgeSearchCore, enhanced: bool) -> None:
        """Test that both search modes return valid results."""
        results = (
            search_core.search_enhanced("API algorithms", k=3)
            if enhanced
            else search_core.search_similarity("API algorithms", k=3)
        )

        assert len(results) <= 3
        assert all(isinstance(r, (SimilaritySearchResult, KnowledgeSearchResult)) for r in results)
        assert all(r.score >= 0 and r.score <= 1.0 for r in results)


class TestKnowledgeSearchResultEnrichment:
    """Test result enrichment with terms and relationships."""

    def test_result_contains_metadata(self, search_core: KnowledgeSearchCore) -> None:
        """Test that search results contain proper metadata."""
        results = search_core.search_similarity("API", k=1)
        result = results[0]

        assert result.text is not None
        assert result.score >= 0
        assert result.document_id is not None
        assert result.document_name is not None
        assert result.metadata is not None

    def test_result_primary_terms_populated(self, search_core: KnowledgeSearchCore) -> None:
        """Test that enhanced search results have primary terms properly populated with exact matches."""
        results = search_core.search_enhanced("machine learning", k=3)

        # Find the result that contains "machine learning" text
        ml_results = [r for r in results if "machine learning" in r.text.lower()]
        assert len(ml_results) > 0, "Should find results containing 'machine learning'"

        ml_result = ml_results[0]

        # Should have the "machine learning" term as a primary term
        ml_primary_terms = [t for t in ml_result.primary_terms if t.term == "machine learning"]
        assert (
            len(ml_primary_terms) == 1
        ), f"Should have exactly 1 'machine learning' primary term, got {len(ml_primary_terms)}"

        # Verify the term has the expected properties
        ml_term = ml_primary_terms[0]
        assert ml_term.term_type == "keyword"
        assert ml_term.total_count == 2
        assert ml_term.mean_score == 0.95

    def test_result_includes_images_and_tables(self, search_core: KnowledgeSearchCore) -> None:
        """Test that results include correct images and tables from their specific pages."""
        results = search_core.search_similarity("machine learning", k=3)

        # Find results from each page to verify specific content
        page1_results = [r for r in results if r.page == 1]
        page2_results = [r for r in results if r.page == 2]

        # Verify page 1 results have the expected image and table
        for result in page1_results:
            # Page 1 should have exactly 1 image and 1 table based on our test data
            assert len(result.images) == 1, f"Page 1 should have 1 image, got {len(result.images)}"
            assert len(result.tables) == 1, f"Page 1 should have 1 table, got {len(result.tables)}"

            # Verify specific image content (images are now strings)
            image = result.images[0]
            assert image == "doc1_page1_image1.png"

            # Verify specific table content
            table = result.tables[0]
            assert table.rows == 2
            assert table.columns == 3
            expected_data = [
                ["Algorithm", "Complexity", "Accuracy"],
                ["SVM", "O(n^2)", "95%"],
            ]
            assert table.data == expected_data

        # Verify page 2 results have the expected images
        for result in page2_results:
            # Page 2 should have exactly 2 images and 0 tables based on our test data
            assert len(result.images) == 2, f"Page 2 should have 2 images, got {len(result.images)}"
            assert len(result.tables) == 0, f"Page 2 should have 0 tables, got {len(result.tables)}"

            # Verify specific image paths (images are now strings)
            image_paths = set(result.images)
            expected_paths = {"doc1_page2_image1.png", "doc1_page2_image2.png"}
            assert image_paths == expected_paths


class TestKnowledgeSearchEdgeCases:
    """Test edge cases and error handling."""

    def test_search_with_empty_query(self, search_core: KnowledgeSearchCore) -> None:
        """Test search with empty query string."""
        results = search_core.search_similarity("", k=5)
        assert isinstance(results, list)

    def test_search_with_unknown_terms(self, search_core: KnowledgeSearchCore) -> None:
        """Test search with terms not in the knowledge base."""
        results = search_core.search_similarity("quantum computing blockchain", k=5)
        assert isinstance(results, list)
        # Should still return some results based on semantic similarity

    def test_search_with_zero_k(self, search_core: KnowledgeSearchCore) -> None:
        """Test search with k=0."""
        results = search_core.search_similarity("test", k=0)
        assert len(results) == 0

    def test_search_with_very_high_threshold(self, search_core: KnowledgeSearchCore) -> None:
        """Test search with very high similarity threshold."""
        results = search_core.search_similarity("test", threshold=0.99, k=10)
        # Should return few or no results due to high threshold
        assert len(results) <= 10

    def test_search_with_special_characters(self, search_core: KnowledgeSearchCore) -> None:
        """Test search with special characters in query."""
        results = search_core.search_similarity("machine-learning & API's", k=3)
        assert isinstance(results, list)

    def test_multiple_searches_consistency(self, search_core: KnowledgeSearchCore) -> None:
        """Test that multiple searches return consistent results."""
        results1 = search_core.search_similarity("API", k=3)
        results2 = search_core.search_similarity("API", k=3)

        # Same query should return same number of results
        assert len(results1) == len(results2)


# ============================================================================
# Integration Tests (from KnowledgePickleIntegrationTest.py)
# ============================================================================


class TestKnowledgeSearchCorePickleIntegration:
    """Integration tests for pickle functionality without mocking."""

    def test_full_pickle_roundtrip_with_real_data(self) -> None:
        """Test complete pickle save/load cycle with comprehensive data."""
        dataset = create_sample_dataset()

        with tempfile.TemporaryDirectory() as temp_dir:
            pickle_path = Path(temp_dir) / "knowledge_search_integration.pkl"

            # Create and initialize original instance
            original_core = KnowledgeSearchCore(linked_knowledge=dataset, pickle_path=pickle_path, auto_load=False)

            # Perform some searches on original to verify functionality
            original_ml_results = original_core.search_enhanced("machine learning", k=3)
            original_api_results = original_core.search_similarity("API REST", k=2)

            # Verify we got meaningful results
            assert len(original_ml_results) > 0
            assert len(original_api_results) > 0

            # Save to pickle
            original_core.persist()
            assert pickle_path.exists()

            # Create new instance and load from pickle
            loaded_core = KnowledgeSearchCore.from_pickle(pickle_path)

            # Verify loaded data structure integrity
            assert len(loaded_core.linked_knowledge.documents) == len(dataset.documents)
            assert len(loaded_core.linked_knowledge.terms) == len(dataset.terms)
            assert len(loaded_core.linked_knowledge.chunks) == len(dataset.chunks)
            assert len(loaded_core.linked_knowledge.pages) == len(dataset.pages)

            # Verify specific content
            assert "ml_term" in loaded_core.linked_knowledge.terms
            assert "doc1" in loaded_core.linked_knowledge.documents
            assert "chunk_1" in loaded_core.linked_knowledge.chunks

            # Perform identical searches on loaded instance
            loaded_ml_results = loaded_core.search_enhanced("machine learning", k=3)
            loaded_api_results = loaded_core.search_similarity("API REST", k=2)

            # Verify search results are consistent
            assert len(loaded_ml_results) == len(original_ml_results)
            assert len(loaded_api_results) == len(original_api_results)

            # Verify specific result content matches
            for orig, loaded in zip(original_ml_results, loaded_ml_results):
                assert orig.text == loaded.text
                assert orig.document_id == loaded.document_id
                # Enhanced search should have similar data
                if hasattr(orig, "primary_terms") and hasattr(loaded, "primary_terms"):
                    assert len(orig.primary_terms) == len(loaded.primary_terms)

    def test_pickle_preserves_search_performance(self) -> None:
        """Test that pickled data maintains search performance."""
        dataset = create_sample_dataset()

        with tempfile.TemporaryDirectory() as temp_dir:
            pickle_path = Path(temp_dir) / "performance_test.pkl"

            # Create original
            original_core = KnowledgeSearchCore(linked_knowledge=dataset, pickle_path=pickle_path, auto_load=False)

            # Time original search
            start_time = time.time()
            original_results = original_core.search_enhanced("machine learning artificial intelligence", k=5)
            original_search_time = time.time() - start_time

            # Save and load
            original_core.persist()
            loaded_core = KnowledgeSearchCore.from_pickle(pickle_path)

            # Time loaded search
            start_time = time.time()
            loaded_results = loaded_core.search_enhanced("machine learning artificial intelligence", k=5)
            loaded_search_time = time.time() - start_time

            # Performance should be similar (within reasonable variance)
            assert len(loaded_results) == len(original_results)
            # Allow for some variance in timing
            assert loaded_search_time < original_search_time * 2  # No more than 2x slower

    def test_pickle_preserves_term_relationships(self) -> None:
        """Test that term relationships and co-occurrences are preserved."""
        dataset = create_sample_dataset()

        with tempfile.TemporaryDirectory() as temp_dir:
            pickle_path = Path(temp_dir) / "relationships_test.pkl"

            # Create and save
            original_core = KnowledgeSearchCore(linked_knowledge=dataset, pickle_path=pickle_path, auto_load=False)
            original_core.persist()

            # Load and verify term relationships
            loaded_core = KnowledgeSearchCore.from_pickle(pickle_path)

            # Check acronym mappings
            assert len(loaded_core._acronym_to_full_form) == 2  # API and REST
            assert loaded_core._acronym_to_full_form["API"] == "Application Programming Interface"
            assert loaded_core._acronym_to_full_form["REST"] == "Representational State Transfer"

            # Check term co-occurrences
            ml_term = loaded_core.linked_knowledge.terms["ml_term"]
            assert len(ml_term.cooccurrences) > 0

            cooccurrence_terms = [co.term for co in ml_term.cooccurrences]
            assert "artificial intelligence" in cooccurrence_terms
            assert "neural networks" in cooccurrence_terms

    def test_pickle_preserves_media_content(self) -> None:
        """Test that images and tables are preserved correctly."""
        dataset = create_sample_dataset()

        with tempfile.TemporaryDirectory() as temp_dir:
            pickle_path = Path(temp_dir) / "media_test.pkl"

            # Create and save
            original_core = KnowledgeSearchCore(linked_knowledge=dataset, pickle_path=pickle_path, auto_load=False)
            original_core.persist()

            # Load and verify media content
            loaded_core = KnowledgeSearchCore.from_pickle(pickle_path)

            # Check images and tables exist
            doc1_page1 = loaded_core.linked_knowledge.pages[("doc1", 1)]
            doc1_page2 = loaded_core.linked_knowledge.pages[("doc1", 2)]

            assert len(doc1_page1.images) == 1
            assert len(doc1_page2.tables) == 1

            # Verify specific content
            image_path = doc1_page1.images[0]
            assert image_path == "/fake/path/image1.png"

            table = doc1_page2.tables[0]
            assert table.rows == 3
            assert table.columns == 2
            assert len(table.data) == 3

    def test_multiple_pickle_cycles(self) -> None:
        """Test multiple save/load cycles don't corrupt data."""
        dataset = create_sample_dataset()

        with tempfile.TemporaryDirectory() as temp_dir:
            pickle_path = Path(temp_dir) / "multi_cycle_test.pkl"

            # Initial creation
            core = KnowledgeSearchCore(linked_knowledge=dataset, pickle_path=pickle_path, auto_load=False)

            # Perform multiple save/load cycles
            for cycle in range(3):
                # Save current state
                core.persist()

                # Load fresh instance
                core = KnowledgeSearchCore.from_pickle(pickle_path)

                # Verify integrity each cycle
                assert len(core.linked_knowledge.documents) == 2
                assert len(core.linked_knowledge.terms) == 4
                assert len(core.linked_knowledge.chunks) == 4

                # Verify search still works
                results = core.search_enhanced("machine learning", k=2)
                assert len(results) > 0

                # Verify specific content hasn't been corrupted
                ml_term = core.linked_knowledge.terms["ml_term"]
                assert ml_term.term == "machine learning"
                assert ml_term.term_type == "keyword"
                assert len(ml_term.cooccurrences) >= 3

    def test_pickle_file_size_reasonable(self) -> None:
        """Test that pickle files are reasonably sized."""
        dataset = create_sample_dataset()

        with tempfile.TemporaryDirectory() as temp_dir:
            pickle_path = Path(temp_dir) / "size_test.pkl"

            core = KnowledgeSearchCore(linked_knowledge=dataset, pickle_path=pickle_path, auto_load=False)
            core.persist()

            # Check file size is reasonable (should be < 1MB for our small dataset)
            file_size_mb = pickle_path.stat().st_size / (1024 * 1024)
            assert file_size_mb < 1.0  # Less than 1MB
            assert file_size_mb > 0.001  # But not empty (at least 1KB)


class TestKnowledgeSearchPersistence:
    """Test pickle persistence functionality."""

    def test_persist_and_load(self, sample_linked_knowledge: LinkedKnowledge, tmp_path: Path) -> None:
        """Test saving and loading KnowledgeSearchCore with pickle."""
        pickle_path = tmp_path / "test_knowledge_search.pkl"

        # Create and configure search core
        search_core = KnowledgeSearchCore(
            linked_knowledge=sample_linked_knowledge,
            pickle_path=pickle_path,
            auto_load=False,
        )

        # Test search before saving
        results_before = search_core.search_similarity("machine learning", k=2)
        assert len(results_before) > 0

        # Save to pickle
        search_core.persist()
        assert pickle_path.exists()

        # Load from pickle using class method
        loaded_search_core = KnowledgeSearchCore.from_pickle(pickle_path)

        # Test search after loading
        results_after = loaded_search_core.search_similarity("machine learning", k=2)
        assert len(results_after) == len(results_before)

        # Verify loaded data
        assert len(loaded_search_core.linked_knowledge.documents) == 1
        assert len(loaded_search_core.linked_knowledge.terms) == 5
        assert len(loaded_search_core.linked_knowledge.chunks) == 3

    def test_auto_load_on_init(self, sample_linked_knowledge: LinkedKnowledge, tmp_path: Path) -> None:
        """Test auto-load functionality when pickle file exists."""
        pickle_path = tmp_path / "test_knowledge_search.pkl"

        # Create and save first instance
        search_core = KnowledgeSearchCore(
            linked_knowledge=sample_linked_knowledge,
            pickle_path=pickle_path,
            auto_load=False,
        )
        search_core.persist()

        # Create new instance with auto-load
        auto_loaded = KnowledgeSearchCore(linked_knowledge=None, pickle_path=pickle_path, auto_load=True)

        # Verify auto-loaded data
        assert len(auto_loaded.linked_knowledge.documents) == 1
        assert len(auto_loaded.linked_knowledge.terms) == 5
        assert len(auto_loaded.linked_knowledge.chunks) == 3

    def test_persist_without_path_raises_error(self, sample_linked_knowledge: LinkedKnowledge) -> None:
        """Test that persist without path raises ValueError."""
        search_core = KnowledgeSearchCore(linked_knowledge=sample_linked_knowledge)

        with pytest.raises(ValueError, match="No path provided for persistence"):
            search_core.persist()

    def test_load_nonexistent_file_raises_error(self, tmp_path: Path) -> None:
        """Test that loading non-existent file raises FileNotFoundError."""
        pickle_path = tmp_path / "nonexistent.pkl"

        with pytest.raises(FileNotFoundError):
            KnowledgeSearchCore.from_pickle(pickle_path)


# ============================================================================
# Performance Tests (from KnowledgePerformanceTest.py)
# ============================================================================


class TestKnowledgeSearchCorePerformance:
    """Performance tests for KnowledgeSearchCore operations."""

    def test_initialization_time_under_limit(self) -> None:
        """Test that initialization completes within 0.5 seconds."""
        dataset = create_sample_dataset()

        start_time = time.time()
        KnowledgeSearchCore(linked_knowledge=dataset)
        init_time = time.time() - start_time

        # Should initialize in under 0.5 seconds
        assert init_time < 0.5, f"Initialization took {init_time:.3f}s, expected < 0.5s"

    def test_pickle_load_time_under_limit(self) -> None:
        """Test that loading from pickle completes within 0.5 seconds."""
        dataset = create_sample_dataset()

        with tempfile.TemporaryDirectory() as temp_dir:
            pickle_path = Path(temp_dir) / "perf_test.pkl"

            # Create and save
            core = KnowledgeSearchCore(linked_knowledge=dataset, pickle_path=pickle_path, auto_load=False)
            core.persist()

            # Time the loading
            start_time = time.time()
            KnowledgeSearchCore.from_pickle(pickle_path)
            load_time = time.time() - start_time

            # Should load in under 0.5 seconds
            assert load_time < 0.5, f"Pickle loading took {load_time:.3f}s, expected < 0.5s"

    def test_standard_search_time_under_limit(self) -> None:
        """Test that standard search completes within 0.5 seconds."""
        dataset = create_sample_dataset()
        core = KnowledgeSearchCore(linked_knowledge=dataset)

        start_time = time.time()
        results = core.search_similarity("machine learning", k=5)
        search_time = time.time() - start_time

        # Should search in under 0.5 seconds
        assert search_time < 0.5, f"Standard search took {search_time:.3f}s, expected < 0.5s"
        assert len(results) > 0, "Search should return results"

    def test_enhanced_search_time_under_limit(self) -> None:
        """Test that enhanced search completes within 0.5 seconds."""
        dataset = create_sample_dataset()
        core = KnowledgeSearchCore(linked_knowledge=dataset)

        start_time = time.time()
        results = core.search_enhanced("machine learning artificial intelligence", k=5)
        search_time = time.time() - start_time

        # Should search in under 0.5 seconds
        assert search_time < 0.5, f"Enhanced search took {search_time:.3f}s, expected < 0.5s"
        assert len(results) > 0, "Search should return results"

    def test_multiple_searches_performance(self) -> None:
        """Test that multiple consecutive searches maintain performance."""
        dataset = create_sample_dataset()
        core = KnowledgeSearchCore(linked_knowledge=dataset)

        queries = [
            "machine learning",
            "artificial intelligence",
            "API REST",
            "neural networks",
            "data analysis",
        ]

        all_times = []
        for query in queries:
            start_time = time.time()
            core.search_enhanced(query, k=3)
            search_time = time.time() - start_time
            all_times.append(search_time)

            # Each search should be under limit
            assert search_time < 0.5, f"Search for '{query}' took {search_time:.3f}s, expected < 0.5s"

        # Average time should also be well under limit
        avg_time = sum(all_times) / len(all_times)
        assert avg_time < 0.3, f"Average search time {avg_time:.3f}s should be < 0.3s"

    def test_search_with_large_k_performance(self) -> None:
        """Test search performance with large k values."""
        dataset = create_sample_dataset()
        core = KnowledgeSearchCore(linked_knowledge=dataset)

        start_time = time.time()
        core.search_enhanced("machine learning", k=100)  # Request many results
        search_time = time.time() - start_time

        # Should still complete quickly even with large k
        assert search_time < 0.5, f"Large k search took {search_time:.3f}s, expected < 0.5s"

    def test_search_performance_after_pickle_load(self) -> None:
        """Test that search performance is maintained after pickle load."""
        dataset = create_sample_dataset()

        with tempfile.TemporaryDirectory() as temp_dir:
            pickle_path = Path(temp_dir) / "perf_after_load.pkl"

            # Create, save, and load
            original_core = KnowledgeSearchCore(linked_knowledge=dataset, pickle_path=pickle_path, auto_load=False)
            original_core.persist()

            loaded_core = KnowledgeSearchCore.from_pickle(pickle_path)

            # Test search performance on loaded instance
            start_time = time.time()
            results = loaded_core.search_enhanced("machine learning AI", k=5)
            search_time = time.time() - start_time

            assert search_time < 0.5, f"Post-pickle search took {search_time:.3f}s, expected < 0.5s"
            assert len(results) > 0, "Search should return results"

    def test_concurrent_search_performance(self) -> None:
        """Test performance when running multiple searches."""
        dataset = create_sample_dataset()
        core = KnowledgeSearchCore(linked_knowledge=dataset)

        # Run multiple searches in sequence (simulating concurrent usage)
        queries = ["machine learning"] * 10  # Same query multiple times

        start_time = time.time()
        for query in queries:
            results = core.search_similarity(query, k=3)
            assert len(results) >= 0  # Ensure we get results

        total_time = time.time() - start_time
        avg_time_per_search = total_time / len(queries)

        # Average time per search should still be under limit
        assert avg_time_per_search < 0.5, f"Average search time {avg_time_per_search:.3f}s, expected < 0.5s"

    @pytest.mark.parametrize("enhanced", [True, False])
    def test_search_modes_performance(self, enhanced: bool) -> None:
        """Test performance of both search modes."""
        dataset = create_sample_dataset()
        core = KnowledgeSearchCore(linked_knowledge=dataset)

        mode_name = "enhanced" if enhanced else "standard"

        start_time = time.time()
        if enhanced:
            results = core.search_enhanced("machine learning", k=5)
            assert len(results) > 0, f"{mode_name} search should return results"
        else:
            sim_results = core.search_similarity("machine learning", k=5)
            assert len(sim_results) > 0, f"{mode_name} search should return results"
        search_time = time.time() - start_time

        assert search_time < 0.5, f"{mode_name} search took {search_time:.3f}s, expected < 0.5s"

    def test_performance_with_complex_query(self) -> None:
        """Test performance with complex multi-term queries."""
        dataset = create_sample_dataset()
        core = KnowledgeSearchCore(linked_knowledge=dataset)

        complex_query = "machine learning artificial intelligence API REST neural networks deep learning"

        start_time = time.time()
        results = core.search_enhanced(complex_query, k=10)
        search_time = time.time() - start_time

        assert search_time < 0.5, f"Complex query search took {search_time:.3f}s, expected < 0.5s"
        assert len(results) >= 0, "Complex query should complete successfully"
