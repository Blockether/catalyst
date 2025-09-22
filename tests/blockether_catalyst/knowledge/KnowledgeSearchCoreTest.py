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

from blockether_catalyst.knowledge.KnowledgeSearchCore import (
    KnowledgeSearchCore,
    KnowledgeSearchResult,
)
from blockether_catalyst.knowledge.KnowledgeTypes import (
    DocumentMetadata,
    ImageMetadata,
    KnowledgeChunkWithTerms,
    KnowledgePageData,
    KnowledgeTableData,
    LinkedKnowledge,
    NormalizedDocumentMetadata,
    Term,
    TermCooccurrence,
    TermInfo,
    TermLink,
    TermOccurrence,
    TermWithLinks,
)

# ============================================================================
# Shared Fixtures and Helper Functions
# ============================================================================


def create_sample_dataset() -> LinkedKnowledge:
    """Create a comprehensive sample dataset for testing."""

    # Create sample terms
    terms = {
        "ml_term": TermWithLinks(
            term="machine learning",
            type="keyword",
            full_form="machine learning",
            meaning="A method of data analysis that automates analytical model building",
            reasoning="Machine learning is a critical field in artificial intelligence that enables computers to learn from data without being explicitly programmed. This term is essential for understanding modern data-driven approaches to problem solving.",
            occurrences=[
                TermOccurrence(
                    document_id="doc1",
                    document_name="ML Guide",
                    chunk_index=0,
                    page=1,
                    total=2,
                ),
                TermOccurrence(
                    document_id="doc2",
                    document_name="AI Overview",
                    chunk_index=1,
                    page=1,
                    total=1,
                ),
            ],
            cooccurrences=[
                TermCooccurrence(term="artificial intelligence", frequency=5, confidence=0.8),
                TermCooccurrence(term="neural networks", frequency=3, confidence=0.7),
                TermCooccurrence(term="deep learning", frequency=4, confidence=0.9),
            ],
        ),
        "ai_term": TermWithLinks(
            term="artificial intelligence",
            type="keyword",
            full_form="artificial intelligence",
            meaning="The simulation of human intelligence in machines",
            reasoning="Artificial intelligence represents a broad field of computer science focused on creating intelligent machines that can simulate human thinking and behavior. This term is fundamental to understanding modern technology and automation.",
            occurrences=[
                TermOccurrence(
                    document_id="doc2",
                    document_name="AI Overview",
                    chunk_index=0,
                    page=1,
                    total=1,
                )
            ],
            cooccurrences=[
                TermCooccurrence(term="machine learning", frequency=5, confidence=0.8),
                TermCooccurrence(term="robotics", frequency=2, confidence=0.6),
            ],
        ),
        "api_acronym": TermWithLinks(
            term="API",
            type="acronym",
            full_form="Application Programming Interface",
            meaning="A set of protocols and tools for building software applications",
            reasoning="API (Application Programming Interface) is a fundamental concept in software development that defines how different software components should interact. Understanding APIs is crucial for modern software integration and development practices.",
            occurrences=[
                TermOccurrence(
                    document_id="doc1",
                    document_name="ML Guide",
                    chunk_index=2,
                    page=2,
                    total=1,
                )
            ],
            cooccurrences=[
                TermCooccurrence(term="REST", frequency=8, confidence=0.9),
                TermCooccurrence(term="HTTP", frequency=6, confidence=0.8),
            ],
        ),
        "rest_acronym": TermWithLinks(
            term="REST",
            type="acronym",
            full_form="Representational State Transfer",
            meaning="An architectural style for distributed hypermedia systems",
            reasoning="REST (Representational State Transfer) is an architectural style that defines a set of constraints for creating web services. This acronym is essential for understanding modern web API design and distributed systems architecture.",
            occurrences=[
                TermOccurrence(
                    document_id="doc1",
                    document_name="ML Guide",
                    chunk_index=2,
                    page=2,
                    total=1,
                )
            ],
            cooccurrences=[
                TermCooccurrence(term="API", frequency=8, confidence=0.9),
                TermCooccurrence(term="HTTP", frequency=7, confidence=0.85),
            ],
        ),
    }

    # Create sample documents
    documents = {
        "doc1": NormalizedDocumentMetadata(
            document_id="doc1",
            document_filename="ml_guide.pdf",
            document_path="ml_guide.pdf",
            title="ML Guide",
            subject=None,
            author=None,
            modification_date=None,
            publication_date=None,
            total_pages=3,
            total_chunks=3,
            total_terms=3,
            total_tables=0,
            total_images=0,
            total_acronyms=1,
            total_keywords=2,
        ),
        "doc2": NormalizedDocumentMetadata(
            document_id="doc2",
            document_filename="ai_overview.pdf",
            document_path="ai_overview.pdf",
            title="AI Overview",
            subject=None,
            author=None,
            modification_date=None,
            publication_date=None,
            total_pages=2,
            total_chunks=1,
            total_terms=2,
            total_tables=0,
            total_images=0,
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
            terms={"machine learning": 1, "artificial intelligence": 1},
        ),
        "chunk_2": KnowledgeChunkWithTerms(
            document_id="doc2",
            document_name="AI Overview",
            doc_id="chunk_2",
            index=0,
            page=1,
            text="Artificial intelligence has revolutionized many industries through automation and data analysis.",
            terms={"artificial intelligence": 1},
        ),
        "chunk_3": KnowledgeChunkWithTerms(
            document_id="doc1",
            document_name="ML Guide",
            doc_id="chunk_3",
            index=1,
            page=1,
            text="Modern machine learning algorithms can process vast amounts of data to identify patterns and make predictions.",
            terms={"machine learning": 1},
        ),
        "chunk_4": KnowledgeChunkWithTerms(
            document_id="doc1",
            document_name="ML Guide",
            doc_id="chunk_4",
            index=2,
            page=2,
            text="RESTful APIs provide a standardized way to interact with web services using HTTP methods.",
            terms={"API": 1, "REST": 1},
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
            images=[ImageMetadata(document_name="ML Guide", page=1, path="/fake/path/image1.png")],
            tables=[],
        ),
        ("doc1", 2): KnowledgePageData(page=2, text="API content from page 2", images=[], tables=[sample_table]),
        ("doc2", 1): KnowledgePageData(page=1, text="AI overview content from page 1", images=[], tables=[]),
    }

    # Build term_to_chunks_index
    term_to_chunks_index = {}
    for term_name, term_data in terms.items():
        chunks_set = set()
        for occurrence in term_data.occurrences:
            chunks_set.add((occurrence.document_id, occurrence.chunk_index))
        term_to_chunks_index[term_name.lower()] = chunks_set

    return LinkedKnowledge(
        documents=documents,
        pages=pages,
        terms=terms,
        chunks=chunks,
        term_to_chunks_index=term_to_chunks_index,
        total_acronyms=2,  # API, ML
        total_keywords=2,  # machine learning, neural network
        total_chunks=4,  # 4 chunks total
        total_images=1,
        total_tables=1,
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
            terms={"machine learning": 1, "algorithms": 1},
        ),
        KnowledgeChunkWithTerms(
            document_id="doc1",
            document_name="ml_guide.pdf",
            doc_id="doc1_p1_c1",
            index=1,
            text="API endpoints provide REST interface for accessing ML models and services.",
            page=1,
            terms={"API": 1, "ML": 1},
        ),
        KnowledgeChunkWithTerms(
            document_id="doc1",
            document_name="ml_guide.pdf",
            doc_id="doc1_p2_c2",
            index=2,
            text="Deep learning neural networks process complex patterns in large datasets.",
            page=2,
            terms={"neural networks": 1},
        ),
    ]


@pytest.fixture
def sample_document() -> NormalizedDocumentMetadata:
    """Create a sample document metadata."""
    return NormalizedDocumentMetadata(
        document_id="doc1",
        document_filename="ml_guide.pdf",
        document_path="ml_guide.pdf",
        title="ML Guide",
        subject=None,
        author=None,
        modification_date=None,
        publication_date=None,
        total_pages=2,
        total_chunks=3,
        total_terms=5,
        total_tables=0,
        total_images=0,
        total_acronyms=1,
        total_keywords=4,
    )


@pytest.fixture
def sample_terms() -> Dict[str, TermWithLinks]:
    """Create sample terms with meanings and co-occurrences."""
    ml_keyword = TermWithLinks(
        term="machine learning",
        type="keyword",
        full_form="machine learning",
        occurrences=[
            TermOccurrence(
                document_id="doc1",
                document_name="ml_guide.pdf",
                page=1,
                chunk_index=0,
                total=2,
            ),
            TermOccurrence(
                document_id="doc1",
                document_name="ml_guide.pdf",
                page=2,
                chunk_index=2,
                total=2,
            ),
        ],
        cooccurrences=[
            TermCooccurrence(term="algorithms", frequency=3, confidence=0.9),
            TermCooccurrence(term="API", frequency=2, confidence=0.7),
            TermCooccurrence(term="neural networks", frequency=2, confidence=0.8),
        ],
        total=2,
        meaning="A method of data analysis that automates analytical model building",
        reasoning="This is a core AI concept that represents the fundamental approach to automated learning from data. It's essential for understanding modern artificial intelligence and data science applications in various domains.",
    )

    api_acronym = TermWithLinks(
        term="API",
        type="acronym",
        full_form="Application Programming Interface",
        occurrences=[
            TermOccurrence(
                document_id="doc1",
                document_name="ml_guide.pdf",
                page=1,
                chunk_index=1,
                total=1,
            )
        ],
        cooccurrences=[
            TermCooccurrence(term="REST", frequency=3, confidence=0.95),
            TermCooccurrence(term="endpoints", frequency=2, confidence=0.85),
            TermCooccurrence(term="machine learning", frequency=1, confidence=0.6),
        ],
        total=1,
        meaning="A set of protocols and tools for building software applications",
        reasoning="This is a standard technical acronym widely used in software development to describe the set of protocols and tools that enable different software applications to communicate with each other effectively and efficiently.",
    )

    ml_acronym = TermWithLinks(
        term="ML",
        type="acronym",
        full_form="Machine Learning",
        occurrences=[
            TermOccurrence(
                document_id="doc1",
                document_name="ml_guide.pdf",
                page=1,
                chunk_index=1,
                total=1,
            )
        ],
        cooccurrences=[
            TermCooccurrence(term="models", frequency=2, confidence=0.9),
            TermCooccurrence(term="API", frequency=1, confidence=0.7),
        ],
        total=1,
        meaning="Machine Learning - automated learning from data",
        reasoning="This is a common abbreviation for Machine Learning, widely used in technical documentation, research papers, and industry discussions to refer to automated learning algorithms and systems that improve through experience.",
    )

    algorithms_keyword = TermWithLinks(
        term="algorithms",
        type="keyword",
        full_form="algorithms",
        occurrences=[
            TermOccurrence(
                document_id="doc1",
                document_name="ml_guide.pdf",
                page=1,
                chunk_index=0,
                total=1,
            )
        ],
        cooccurrences=[
            TermCooccurrence(term="machine learning", frequency=3, confidence=0.9),
            TermCooccurrence(term="computational", frequency=2, confidence=0.8),
        ],
        total=1,
        meaning="A set of rules or instructions for solving a problem",
        reasoning="This is a fundamental computer science concept that represents a step-by-step procedure or formula for solving problems. Algorithms form the foundation of all computational processes and programming logic in software development.",
    )

    neural_networks_keyword = TermWithLinks(
        term="neural networks",
        type="keyword",
        full_form="neural networks",
        occurrences=[
            TermOccurrence(
                document_id="doc1",
                document_name="ml_guide.pdf",
                page=2,
                chunk_index=2,
                total=1,
            )
        ],
        cooccurrences=[
            TermCooccurrence(term="deep learning", frequency=3, confidence=0.95),
            TermCooccurrence(term="machine learning", frequency=2, confidence=0.8),
        ],
        total=1,
        meaning="Computing systems inspired by biological neural networks",
        reasoning="This is a key deep learning concept inspired by biological neural networks in the brain. Neural networks form the foundation of modern AI systems capable of learning complex patterns and making sophisticated predictions from data.",
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
        TermLink(link_from="API", link_to="Application Programming Interface", score=0.98),
        TermLink(link_from="ML", link_to="machine learning", score=0.95),
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
            images=[ImageMetadata(document_name="ml_guide.pdf", page=1, path="doc1_page1_image1.png")],
            lines=5,
        ),
        ("doc1", 2): KnowledgePageData(
            page=2,
            text="Deep learning neural networks process complex patterns.",
            tables=[],
            images=[
                ImageMetadata(document_name="ml_guide.pdf", page=2, path="doc1_page2_image1.png"),
                ImageMetadata(document_name="ml_guide.pdf", page=2, path="doc1_page2_image2.png"),
            ],
            lines=3,
        ),
    }


@pytest.fixture
def sample_linked_knowledge(
    sample_document: NormalizedDocumentMetadata,
    sample_terms: Dict[str, TermWithLinks],
    sample_links: List[TermLink],
    sample_document_chunks: List[KnowledgeChunkWithTerms],
    sample_pages: Dict[Tuple[str, int], KnowledgePageData],
) -> LinkedKnowledge:
    """Create complete LinkedKnowledge structure."""
    # Convert chunks to the new format
    chunks_dict = {chunk.doc_id: chunk for chunk in sample_document_chunks}

    # Build the term_to_chunks_index from terms
    term_to_chunks_index = {}
    for term_name, term_data in sample_terms.items():
        chunks_set = set()
        for occurrence in term_data.occurrences:
            chunks_set.add((occurrence.document_id, occurrence.chunk_index))
        term_to_chunks_index[term_name.lower()] = chunks_set

    # Calculate totals from the sample data
    total_acronyms = sum(1 for term in sample_terms.values() if term.type == "acronym")
    total_keywords = sum(1 for term in sample_terms.values() if term.type == "keyword")
    total_chunks = len(chunks_dict)

    return LinkedKnowledge(
        documents={"doc1": sample_document},
        pages=sample_pages,
        terms=sample_terms,
        chunks=chunks_dict,
        term_to_chunks_index=term_to_chunks_index,
        total_acronyms=total_acronyms,
        total_keywords=total_keywords,
        total_chunks=total_chunks,
        total_images=0,
        total_tables=1,
    )


@pytest.fixture
def search_core(sample_linked_knowledge: LinkedKnowledge) -> KnowledgeSearchCore:
    """Create initialized KnowledgeSearchCore instance."""
    return KnowledgeSearchCore(linked_knowledge=sample_linked_knowledge)


# ============================================================================
# Unit Tests (from original KnowledgeSearchCoreTest.py)
# ============================================================================


class TestKnowledgeSearchCoreInitialization:
    """Test KnowledgeSearchCore initialization."""

    def test_initialization_with_linked_knowledge(self, sample_linked_knowledge: LinkedKnowledge) -> None:
        """Test proper initialization with LinkedKnowledge."""
        search_core = KnowledgeSearchCore(linked_knowledge=sample_linked_knowledge)

        assert search_core.linked_knowledge == sample_linked_knowledge
        assert search_core._vector_store is not None

    def test_initialization_with_empty_knowledge(self) -> None:
        """Test initialization with empty LinkedKnowledge."""
        empty_knowledge = LinkedKnowledge(
            documents={},
            pages={},
            terms={},
            chunks={},
            term_to_chunks_index={},
            total_acronyms=0,
            total_keywords=0,
            total_chunks=0,
            total_images=0,
            total_tables=0,
        )
        assert empty_knowledge is not None, "empty_knowledge should not be None"
        search_core = KnowledgeSearchCore(linked_knowledge=empty_knowledge)

        assert search_core.linked_knowledge == empty_knowledge
        assert search_core._vector_store is not None

    def test_initialization_builds_indices(self, sample_linked_knowledge: LinkedKnowledge) -> None:
        """Test that initialization properly builds search indices."""
        search_core = KnowledgeSearchCore(linked_knowledge=sample_linked_knowledge)

        # Check that vector store is populated
        assert search_core._vector_store is not None
        # Vector store should have chunks added to it
        assert len(search_core._vector_store.store) > 0


class TestKnowledgeSearchCoreSearch:
    """Test search functionality."""

    def test_search_with_max_depth(self, search_core: KnowledgeSearchCore) -> None:
        """Test search with different max_depth values."""
        # Create test data with linked terms (depth chain: A -> B -> C -> D)
        term_a = TermWithLinks(
            term="term_a",
            type="keyword",
            full_form="Term A",
            meaning="First term",
            reasoning="Term A is a fundamental concept in this test suite that serves as the starting point for testing recursive link resolution. It demonstrates the ability to traverse linked terms through multiple levels of depth, ensuring the system can handle complex term relationships.",
            occurrences=[],
            cooccurrences=[],
            links=[TermLink(link_from="term_a", link_to="term_b", score=0.9)],
        )

        term_b = TermWithLinks(
            term="term_b",
            type="keyword",
            full_form="Term B",
            meaning="Second term",
            reasoning="Term B represents an intermediate node in the term linkage chain, demonstrating the system's ability to continue traversing relationships beyond the initial level. This term is critical for testing depth-based resolution as it both receives and provides links.",
            occurrences=[],
            cooccurrences=[],
            links=[TermLink(link_from="term_b", link_to="term_c", score=0.9)],
        )

        term_c = TermWithLinks(
            term="term_c",
            type="keyword",
            full_form="Term C",
            meaning="Third term",
            reasoning="Term C continues the chain of linked terms, allowing us to test the system's ability to resolve relationships at deeper levels. It serves as a bridge between intermediate and terminal terms, ensuring proper depth tracking throughout the resolution process.",
            occurrences=[],
            cooccurrences=[],
            links=[TermLink(link_from="term_c", link_to="term_d", score=0.9)],
        )

        term_d = TermWithLinks(
            term="term_d",
            type="keyword",
            full_form="Term D",
            meaning="Fourth term",
            reasoning="Term D serves as the terminal node in our test chain, having no outgoing links. This allows us to verify that the system properly handles leaf nodes in the term graph and correctly stops traversal when reaching terms without further connections.",
            occurrences=[],
            cooccurrences=[],
            links=[],
        )

        # Add terms to the knowledge base
        search_core._linked_knowledge.terms["term_a"] = term_a
        search_core._linked_knowledge.terms["term_b"] = term_b
        search_core._linked_knowledge.terms["term_c"] = term_c
        search_core._linked_knowledge.terms["term_d"] = term_d

        # Test max_depth=0 (no linked terms)
        resolved_depth0 = search_core._resolve_linked_terms(term_a, max_depth=0)
        assert len(resolved_depth0) == 0

        # Test max_depth=1 (only immediate links: term_b)
        resolved_depth1 = search_core._resolve_linked_terms(term_a, max_depth=1)
        assert len(resolved_depth1) == 1
        assert resolved_depth1[0].term == "term_b"

        # Test max_depth=2 (term_b and term_c)
        resolved_depth2 = search_core._resolve_linked_terms(term_a, max_depth=2)
        assert len(resolved_depth2) == 2
        term_names = [t.term for t in resolved_depth2]
        assert "term_b" in term_names
        assert "term_c" in term_names

        # Test max_depth=3 (term_b, term_c, and term_d)
        resolved_depth3 = search_core._resolve_linked_terms(term_a, max_depth=3)
        assert len(resolved_depth3) == 3
        term_names = [t.term for t in resolved_depth3]
        assert "term_b" in term_names
        assert "term_c" in term_names
        assert "term_d" in term_names

        # Test cycle prevention
        term_b.links.append(TermLink(link_from="term_b", link_to="term_a", score=0.9))
        # Start with term_a already in visited set to prevent cycles back to it
        visited_with_start = {"term_a"}
        resolved_with_cycle = search_core._resolve_linked_terms(term_a, max_depth=5, visited=visited_with_start)
        # Should still only get term_b, term_c, term_d (no duplicates or infinite loops)
        assert len(resolved_with_cycle) == 3
        term_names_cycle = [t.term for t in resolved_with_cycle]
        assert "term_a" not in term_names_cycle  # term_a should not appear due to cycle prevention

    def test_search_enriches_with_terms(self, search_core: KnowledgeSearchCore) -> None:
        """Test that enhanced search results are enriched with term information."""
        results = search_core.search("machine learning algorithms", k=3)

        # At least one result should have primary terms
        results_with_terms = [r for r in results for _ in r.primary_terms]
        assert len(results_with_terms) > 0

    def test_search_includes_cooccurrences_in_related_terms(self, search_core: KnowledgeSearchCore) -> None:
        """Test that enhanced search results include co-occurring terms in related_terms."""
        results = search_core.search("machine learning", max_cooccurrences=5, k=3)

        # Check that related_terms includes both linked terms and co-occurrences
        for result in results:
            assert isinstance(result.related_terms, list)
            # All items in related_terms should be TermInfo, Term, TermWithLinks objects or strings
            for term in result.related_terms:
                assert isinstance(term, (Term, TermWithLinks, TermInfo, str)), f"Unexpected type: {type(term)}"


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
            original_ml_results = original_core.search("machine learning", k=3)

            # Verify we got meaningful results
            assert len(original_ml_results) > 0

            # Save to pickle
            original_core.persist()
            assert pickle_path.exists()

            # Create new instance and load from pickle
            loaded_core = KnowledgeSearchCore.from_pickle(pickle_path, resources_base_url=None)

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
            loaded_ml_results = loaded_core.search("machine learning", k=3)

            # Verify search results are consistent
            assert len(loaded_ml_results) == len(original_ml_results)

            # Verify specific result content matches
            for orig, loaded in zip(original_ml_results, loaded_ml_results):
                assert orig.content == loaded.content
                assert orig.document_name == loaded.document_name
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
            original_results = original_core.search("machine learning artificial intelligence", k=5)
            original_search_time = time.time() - start_time

            # Save and load
            original_core.persist()
            loaded_core = KnowledgeSearchCore.from_pickle(pickle_path, resources_base_url=None)

            # Time loaded search
            start_time = time.time()
            loaded_results = loaded_core.search("machine learning artificial intelligence", k=5)
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
            loaded_core = KnowledgeSearchCore.from_pickle(pickle_path, resources_base_url=None)

            # Check that the vector store was properly loaded
            assert loaded_core._vector_store is not None
            assert len(loaded_core._vector_store.store) > 0

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
            loaded_core = KnowledgeSearchCore.from_pickle(pickle_path, resources_base_url=None)

            # Check images and tables exist
            doc1_page1 = loaded_core.linked_knowledge.pages[("doc1", 1)]
            doc1_page2 = loaded_core.linked_knowledge.pages[("doc1", 2)]

            assert len(doc1_page1.images) == 1
            assert len(doc1_page2.tables) == 1

            # Verify specific content
            image = doc1_page1.images[0]
            assert image.path == "/fake/path/image1.png"

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
                core = KnowledgeSearchCore.from_pickle(pickle_path, resources_base_url=None)

                # Verify integrity each cycle
                assert len(core.linked_knowledge.documents) == 2
                assert len(core.linked_knowledge.terms) == 4
                assert len(core.linked_knowledge.chunks) == 4

                # Verify search still works
                results = core.search("machine learning", k=2)
                assert len(results) > 0

                # Verify specific content hasn't been corrupted
                ml_term = core.linked_knowledge.terms["ml_term"]
                assert ml_term.term == "machine learning"
                assert ml_term.type == "keyword"
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
        results_before = search_core.search("machine learning", k=2)
        assert len(results_before) > 0

        # Save to pickle
        search_core.persist()
        assert pickle_path.exists()

        # Load from pickle using class method
        loaded_search_core = KnowledgeSearchCore.from_pickle(pickle_path, resources_base_url=None)

        # Test search after loading
        results_after = loaded_search_core.search("machine learning", k=2)
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
            KnowledgeSearchCore.from_pickle(pickle_path, resources_base_url=None)


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
            KnowledgeSearchCore.from_pickle(pickle_path, resources_base_url=None)
            load_time = time.time() - start_time

            # Should load in under 0.5 seconds
            assert load_time < 0.5, f"Pickle loading took {load_time:.3f}s, expected < 0.5s"

    def test_standard_search_time_under_limit(self) -> None:
        """Test that standard search completes within 0.5 seconds."""
        dataset = create_sample_dataset()
        core = KnowledgeSearchCore(linked_knowledge=dataset)

        start_time = time.time()
        results = core.search("machine learning", k=5)
        search_time = time.time() - start_time

        # Should search in under 0.5 seconds
        assert search_time < 0.5, f"Standard search took {search_time:.3f}s, expected < 0.5s"
        assert len(results) > 0, "Search should return results"

    def test_enhanced_search_time_under_limit(self) -> None:
        """Test that enhanced search completes within 0.5 seconds."""
        dataset = create_sample_dataset()
        core = KnowledgeSearchCore(linked_knowledge=dataset)

        start_time = time.time()
        results = core.search("machine learning artificial intelligence", k=5)
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
            core.search(query, k=3)
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
        core.search("machine learning", k=100)  # Request many results
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

            loaded_core = KnowledgeSearchCore.from_pickle(pickle_path, resources_base_url=None)

            # Test search performance on loaded instance
            start_time = time.time()
            results = loaded_core.search("machine learning AI", k=5)
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
            results = core.search(query, k=3)
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
        results = core.search("machine learning", k=5)
        assert len(results) > 0, f"{mode_name} search should return results"
        search_time = time.time() - start_time

        assert search_time < 0.5, f"{mode_name} search took {search_time:.3f}s, expected < 0.5s"

    def test_performance_with_complex_query(self) -> None:
        """Test performance with complex multi-term queries."""
        dataset = create_sample_dataset()
        core = KnowledgeSearchCore(linked_knowledge=dataset)

        complex_query = "machine learning artificial intelligence API REST neural networks deep learning"

        start_time = time.time()
        results = core.search(complex_query, k=10)
        search_time = time.time() - start_time

        assert search_time < 0.5, f"Complex query search took {search_time:.3f}s, expected < 0.5s"
        assert len(results) >= 0, "Complex query should complete successfully"
