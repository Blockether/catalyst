"""
Tests for KnowledgeExtractionCore, specifically testing edge cases
for term extraction to ensure IndexError fix works correctly.
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import anyio
import pytest

from blockether_catalyst.knowledge.extraction.ExtractionCore import (
    KnowledgeExtractionCore,
)
from blockether_catalyst.knowledge.extraction.internal.KnowledgeExtractionCallBase import (
    BaseChunkContentClassificationCall,
    BaseDocumentChunkingCall,
    BaseTermExtractionCall,
    ExtractionCallsSettings,
)
from blockether_catalyst.knowledge.KnowledgeTypes import (
    KnowledgeExtractionResultWithChunks,
    KnowledgeProcessorSettings,
    RawKnowledgeChunk,
)


class TestKnowledgeExtractionCore:
    """Test suite for KnowledgeExtractionCore focusing on edge cases."""

    @pytest.fixture
    def mock_calls_settings(self):
        """Create mock extraction calls settings."""
        mock_term_extraction = MagicMock(spec=BaseTermExtractionCall)
        mock_document_chunking = MagicMock(spec=BaseDocumentChunkingCall)
        mock_chunk_classification = MagicMock(spec=BaseChunkContentClassificationCall)

        return ExtractionCallsSettings(
            term_extraction_call=mock_term_extraction,
            document_chunking_call=mock_document_chunking,
            chunk_content_classification_call=mock_chunk_classification,
        )

    @pytest.fixture
    def processor_settings(self, tmp_path):
        """Create processor settings with temporary output directory."""
        return KnowledgeProcessorSettings(
            extraction_output_dir=tmp_path / "test_output",
            linking_threshold=0.7,
            encoding_model="cl100k_base",
        )

    @pytest.fixture
    def extractor(self, mock_calls_settings, processor_settings):
        """Create KnowledgeExtractionCore instance for testing."""
        return KnowledgeExtractionCore(calls=mock_calls_settings, settings=processor_settings)

    @pytest.mark.anyio
    async def test_extract_terms_from_document_with_only_acronyms(self, extractor):
        """
        Test extraction with documents containing only acronyms.
        """
        from blockether_catalyst.knowledge.KnowledgeTypes import DocumentMetadata

        document = KnowledgeExtractionResultWithChunks(
            id="test-acronyms",
            document_filename="test_acronyms.pdf",
            document_metadata=DocumentMetadata(document_path="test_acronyms.pdf"),
            source_type="pdf",
            chunks=[
                RawKnowledgeChunk(
                    document_id="test-acronyms",
                    document_name="test_acronyms.pdf",
                    doc_id="test-acronyms-1-0",
                    index=0,
                    page=1,
                    text="NASA FBI CIA NATO UNICEF",  # Only acronyms
                )
            ],
        )

        terms = await extractor._extract_terms_from_document(document)

        assert isinstance(terms, list)
        # Should extract the acronyms
        acronyms = [t for t in terms if t.type == "acronym"]
        assert len(acronyms) > 0

        # Check that some expected acronyms were found
        acronym_texts = [t.term.upper() for t in acronyms]
        assert any(acronym in acronym_texts for acronym in ["NASA", "FBI", "CIA", "NATO"])

    @pytest.mark.anyio
    async def test_extract_terms_from_empty_document(self, extractor):
        """
        Test extraction with empty document (no chunks).
        """
        from blockether_catalyst.knowledge.KnowledgeTypes import DocumentMetadata

        document = KnowledgeExtractionResultWithChunks(
            id="test-empty",
            document_filename="test_empty.pdf",
            document_metadata=DocumentMetadata(document_path="test_empty.pdf"),
            source_type="pdf",
            chunks=[],  # Empty chunks
        )

        terms = await extractor._extract_terms_from_document(document)

        assert isinstance(terms, list)
        assert len(terms) == 0

    @pytest.mark.anyio
    async def test_extract_terms_from_document_with_empty_text(self, extractor):
        """
        Test extraction with chunks containing empty text.
        """
        from blockether_catalyst.knowledge.KnowledgeTypes import DocumentMetadata

        document = KnowledgeExtractionResultWithChunks(
            id="test-empty-text",
            document_filename="test_empty_text.pdf",
            document_metadata=DocumentMetadata(document_path="test_empty_text.pdf"),
            source_type="pdf",
            chunks=[
                RawKnowledgeChunk(
                    document_id="test-empty-text",
                    document_name="test_empty_text.pdf",
                    doc_id="test-empty-text-1-0",
                    index=0,
                    page=1,
                    text="",  # Empty text
                ),
                RawKnowledgeChunk(
                    document_id="test-empty-text",
                    document_name="test_empty_text.pdf",
                    doc_id="test-empty-text-1-1",
                    index=1,
                    page=1,
                    text="   ",  # Only whitespace
                ),
            ],
        )

        terms = await extractor._extract_terms_from_document(document)

        assert isinstance(terms, list)
        assert len(terms) == 0

    @pytest.mark.anyio
    async def test_extract_terms_from_document_with_mixed_content(self, extractor):
        """
        Test extraction with documents containing both keywords and acronyms.
        """
        from blockether_catalyst.knowledge.KnowledgeTypes import DocumentMetadata

        document = KnowledgeExtractionResultWithChunks(
            id="test-mixed",
            document_filename="test_mixed.pdf",
            document_metadata=DocumentMetadata(document_path="test_mixed.pdf"),
            source_type="pdf",
            chunks=[
                RawKnowledgeChunk(
                    document_id="test-mixed",
                    document_name="test_mixed.pdf",
                    doc_id="test-mixed-1-0",
                    index=0,
                    page=1,
                    text="NASA developed machine learning algorithms for analyzing data from ISS experiments",
                )
            ],
        )

        terms = await extractor._extract_terms_from_document(document)

        assert isinstance(terms, list)
        assert len(terms) > 0

        # Should have both types
        acronyms = [t for t in terms if t.type == "acronym"]
        keywords = [t for t in terms if t.type == "keyword"]

        assert len(acronyms) > 0
        assert len(keywords) > 0

        # Check specific terms
        acronym_texts = [t.term.upper() for t in acronyms]
        assert "NASA" in acronym_texts or "ISS" in acronym_texts

    @pytest.mark.anyio
    async def test_extract_terms_handles_special_characters(self, extractor):
        """
        Test that special characters in acronyms are handled correctly.
        """
        from blockether_catalyst.knowledge.KnowledgeTypes import DocumentMetadata

        document = KnowledgeExtractionResultWithChunks(
            id="test-special",
            document_filename="test_special.pdf",
            document_metadata=DocumentMetadata(document_path="test_special.pdf"),
            source_type="pdf",
            chunks=[
                RawKnowledgeChunk(
                    document_id="test-special",
                    document_name="test_special.pdf",
                    doc_id="test-special-1-0",
                    index=0,
                    page=1,
                    text="API-KEY AWS_SECRET CI/CD ML-OPS",  # Acronyms with special chars
                )
            ],
        )

        terms = await extractor._extract_terms_from_document(document)
        assert isinstance(terms, list)
        # Should handle acronyms with hyphens and underscores
        acronyms = [t for t in terms if t.type == "acronym"]
        assert len(acronyms) > 0

    @pytest.mark.anyio
    async def test_extract_terms_with_large_document(self, extractor):
        """
        Test extraction with a larger document to ensure performance.
        """
        # Create a document with multiple chunks
        chunks = []
        for i in range(10):
            chunks.append(
                RawKnowledgeChunk(
                    document_id="test-large",
                    document_name="test_large.pdf",
                    doc_id=f"test-large-{i + 1}-{i}",
                    index=i,
                    page=i + 1,
                    text=f"Page {i + 1}: Machine learning models like GPT and BERT use transformer architectures. "
                    f"NASA and ESA collaborate on space missions. "
                    f"API endpoints handle HTTP requests efficiently.",
                )
            )

        from blockether_catalyst.knowledge.KnowledgeTypes import DocumentMetadata

        document = KnowledgeExtractionResultWithChunks(
            id="test-large",
            document_filename="test_large.pdf",
            document_metadata=DocumentMetadata(document_path="test_large.pdf"),
            source_type="pdf",
            chunks=chunks,
        )

        terms = await extractor._extract_terms_from_document(document)

        assert isinstance(terms, list)
        assert len(terms) > 0

        # Should extract both keywords and acronyms from all chunks
        acronyms = [t for t in terms if t.type == "acronym"]
        keywords = [t for t in terms if t.type == "keyword"]

        assert len(acronyms) > 0
        assert len(keywords) > 0

        # Terms should come from multiple pages
        pages = set(t.page for t in terms)
        assert len(pages) > 1
