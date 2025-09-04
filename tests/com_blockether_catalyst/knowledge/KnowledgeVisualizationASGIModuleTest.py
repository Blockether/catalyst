"""Tests for KnowledgeVisualizationASGIModule."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from com_blockether_catalyst.knowledge.internal.KnowledgeExtractionTypes import (
    KnowledgeChunkWithTerms,
    LinkedKnowledge,
    Term,
)
from com_blockether_catalyst.knowledge.KnowledgeVisualizationASGIModule import (
    KnowledgeVisualizationASGIModule,
)


class TestKnowledgeVisualizationASGIModule:
    """Test suite for KnowledgeVisualizationASGIModule."""

    def test_initialization(self) -> None:
        """Test KnowledgeVisualizationASGIModule initialization."""
        viz = KnowledgeVisualizationASGIModule()
        assert viz.linked_knowledge is None
        assert viz.output_dir.exists()

    def test_load_from_pickle(self, tmp_path: Path) -> None:
        """Test loading LinkedKnowledge from pickle file."""
        # Create a real LinkedKnowledge instance instead of mock for pickling
        knowledge = LinkedKnowledge(
            documents={},
            terms={},
            links=[],
            chunks={},
            total_acronyms=0,
            total_keywords=0,
            total_chunks=0,
        )

        # Save to pickle
        import pickle

        pickle_path = tmp_path / "test.pkl"
        with open(pickle_path, "wb") as f:
            pickle.dump(knowledge, f)

        # Load and verify
        viz = KnowledgeVisualizationASGIModule()
        viz.load_from_pickle(pickle_path)
        assert viz.linked_knowledge is not None
        assert isinstance(viz.linked_knowledge, LinkedKnowledge)

    def test_api_stats_no_data(self) -> None:
        """Test stats API endpoint with no data loaded."""
        viz = KnowledgeVisualizationASGIModule()
        client = TestClient(viz.app)

        response = client.get("/knowledge/api/stats")
        assert response.status_code == 200
        assert response.json() == {"error": "No data loaded"}

    def test_api_terms_search_functionality(self) -> None:
        """Test the search functionality in the terms API endpoint."""
        viz = KnowledgeVisualizationASGIModule()

        # Create mock LinkedKnowledge with test data
        mock_knowledge = MagicMock(spec=LinkedKnowledge)

        # Create test terms
        test_acronym = Term(
            term="API",
            term_type="acronym",
            full_form="Application Programming Interface",
            meaning="A set of protocols for building software",
            total_count=10,
            reasoning="Test acronym",
        )

        test_keyword = Term(
            term="database",
            term_type="keyword",
            full_form="database",
            meaning="A structured collection of data",
            total_count=5,
            mean_score=0.85,
            reasoning="Test keyword",
        )

        mock_knowledge.terms = {
            "term1": test_acronym,
            "term2": test_keyword,
        }

        viz.linked_knowledge = mock_knowledge
        client = TestClient(viz.app)

        # Test 1: Search for "API" should find the acronym
        response = client.get("/knowledge/api/terms?search=API")
        assert response.status_code == 200
        assert "API" in response.text
        assert "Application Programming Interface" in response.text
        assert "database" not in response.text

        # Test 2: Search for "interface" should find the acronym (via full form)
        response = client.get("/knowledge/api/terms?search=interface")
        assert response.status_code == 200
        assert "API" in response.text
        assert "Application Programming Interface" in response.text
        assert "database" not in response.text

        # Test 3: Search for "data" should find the keyword
        response = client.get("/knowledge/api/terms?search=data")
        assert response.status_code == 200
        assert "database" in response.text
        assert "API" not in response.text

        # Test 4: Search with filters - only acronyms
        response = client.get("/knowledge/api/terms?search=&show_acronyms=true&show_keywords=false")
        assert response.status_code == 200
        assert "API" in response.text
        assert "database" not in response.text

        # Test 5: Search with filters - only keywords
        response = client.get("/knowledge/api/terms?search=&show_acronyms=false&show_keywords=true")
        assert response.status_code == 200
        assert "database" in response.text
        assert "API" not in response.text

        # Test 6: Case-insensitive search
        response = client.get("/knowledge/api/terms?search=api")
        assert response.status_code == 200
        assert "API" in response.text

        # Test 7: Search in meaning field
        response = client.get("/knowledge/api/terms?search=protocols")
        assert response.status_code == 200
        assert "API" in response.text
        assert "database" not in response.text

    def test_api_documents_no_data(self) -> None:
        """Test documents API endpoint with no data loaded."""
        viz = KnowledgeVisualizationASGIModule()
        client = TestClient(viz.app)

        response = client.get("/knowledge/api/documents")
        assert response.status_code == 200
        assert "No data loaded" in response.text

    def test_api_document_details(self) -> None:
        """Test document details view with chunks."""
        viz = KnowledgeVisualizationASGIModule()

        # Create mock LinkedKnowledge with document and chunks

        mock_knowledge = MagicMock(spec=LinkedKnowledge)

        # Create test document
        test_doc = MagicMock()
        test_doc.filename = "test.pdf"
        test_doc.total_pages = 10
        test_doc.total_chunks = 3
        test_doc.total_tables = 1
        test_doc.total_acronyms = 5
        test_doc.total_keywords = 8

        # Create test chunks
        test_chunks = {
            "doc1_p1_c0": KnowledgeChunkWithTerms(
                document_id="doc1",
                document_name="test.pdf",
                doc_id="doc1_p1_c0",
                index=0,
                text="This is the first chunk of text from the document.",
                page=1,
                terms=[],
            ),
            "doc1_p2_c1": KnowledgeChunkWithTerms(
                document_id="doc1",
                document_name="test.pdf",
                doc_id="doc1_p2_c1",
                index=1,
                text="This is the second chunk with more content that goes on for a while. " * 20,
                page=2,
                terms=[],
            ),
            "doc1_p3_c2": KnowledgeChunkWithTerms(
                document_id="doc1",
                document_name="test.pdf",
                doc_id="doc1_p3_c2",
                index=2,
                text="Final chunk from page 3.",
                page=3,
                terms=[],
            ),
        }

        mock_knowledge.documents = {"doc1": test_doc}
        mock_knowledge.chunks = test_chunks
        mock_knowledge.terms = {}  # Add empty terms dict to mock
        mock_knowledge.links = []  # Add empty links list to mock
        mock_knowledge.document_to_chunk_ids_index = {
            "doc1": {"doc1_p1_c0", "doc1_p2_c1", "doc1_p3_c2"}
        }  # Add chunk index

        viz.linked_knowledge = mock_knowledge
        client = TestClient(viz.app)

        # Test viewing document details
        response = client.get("/knowledge/api/documents/doc1")
        assert response.status_code == 200
        assert "test.pdf" in response.text
        assert "Chunk 1" in response.text
        assert "Chunk 2" in response.text
        assert "Chunk 3" in response.text
        assert "This is the first chunk" in response.text
        assert "Page 1" in response.text
        assert "← Back" in response.text

        # Test non-existent document
        response = client.get("/knowledge/api/documents/nonexistent")
        assert response.status_code == 200
        assert "Document not found" in response.text

    def test_api_links_no_data(self) -> None:
        """Test links API endpoint with no data loaded."""
        viz = KnowledgeVisualizationASGIModule()
        client = TestClient(viz.app)

        response = client.get("/knowledge/api/links")
        assert response.status_code == 200
        assert "No data loaded" in response.text

    def test_main_page_renders(self) -> None:
        """Test that the main page renders correctly."""
        viz = KnowledgeVisualizationASGIModule()
        client = TestClient(viz.app)

        response = client.get("/knowledge/")
        assert response.status_code == 200
        assert "Knowledge Visualization" in response.text
        assert "search-terms" in response.text
        assert "hx-get" in response.text
