"""
Comprehensive tests for document chunking functionality.
Tests the chunking logic, boundary detection, and chunk creation.
"""

from typing import List

import pytest

from blockether_catalyst.consensus.ConsensusTypes import ConsensusResult
from blockether_catalyst.knowledge.KnowledgeExtractionCallBase import (
    BaseChunkContentClassificationCall,
    BaseDocumentChunkingCall,
    BaseTermExtractionCall,
    ExtractionCallsSettings,
)
from blockether_catalyst.knowledge.KnowledgeExtractionCore import (
    KnowledgeExtractionCore,
)
from blockether_catalyst.knowledge.KnowledgeTypes import (
    ChunkingDecisionResponse,
    ChunkOutput,
    DocumentMetadata,
    KnowledgeChunk,
    KnowledgePageDataWithRawText,
    KnowledgeProcessorSettings,
)


class TestChunkingConstants:
    """Constants for chunking tests."""

    DEFAULT_MAX_CHUNK_LENGTH = 500
    MIN_CHUNK_LENGTH = 50
    OVERLAP_LENGTH = 20

    # Test document samples
    SHORT_TEXT = "This is a short text. It should fit in a single chunk."

    PARAGRAPH_TEXT = """Introduction: This is the introduction section.

    Methods: Here we describe the methodology used in the study.
    
    Results: The results show significant improvements.
    
    Conclusion: In conclusion, the approach is effective."""

    LONG_TEXT_SENTENCES = 50
    SENTENCE_TEMPLATE = "Sentence number {} contains important information"

    MIXED_CONTENT_TEXT = "Here is some text. [TABLE: data table]. More text here. [FIGURE 1: diagram]. Final text."

    SPECIAL_CHARS_TEXT = """• Bullet point one
    • Bullet point two
    
    Code example:
    ```python
    def hello():
        return "world"
    ```
    
    Mathematical formula: E = mc²"""


class RealDocumentChunkingCall(BaseDocumentChunkingCall):
    """Real implementation of document chunking for testing."""

    MAX_CHUNK_LENGTH = TestChunkingConstants.DEFAULT_MAX_CHUNK_LENGTH
    MIN_CHUNK_LENGTH = TestChunkingConstants.MIN_CHUNK_LENGTH

    def __init__(self) -> None:
        """Initialize with a mock consensus for testing."""
        # Create a simple mock consensus that's not used in our deterministic tests
        from unittest.mock import MagicMock

        mock_consensus = MagicMock()
        super().__init__(consensus=mock_consensus)

    async def execute(
        self,
        page: KnowledgePageDataWithRawText,
        document_name: str,
        metadata: DocumentMetadata,
    ) -> ConsensusResult:
        """Execute real chunking logic."""
        chunks = self._create_chunks(page.text)

        response = ChunkingDecisionResponse(
            reasoning=(
                f"Analyzed page {page.page} and created {len(chunks)} chunks based on semantic boundaries. "
                "The chunking process used paragraph breaks as primary boundaries and ensured each chunk "
                "stays within the maximum length limit while preserving semantic meaning."
            ),
            chunks=chunks,
        )

        return ConsensusResult(
            reasoning=f"Test chunking completed for {document_name} page {page.page}. " * 5,  # Ensure 150+ chars
            consensus_achieved=True,
            final_response=response,
            rounds=[],
            total_rounds=1,
            convergence_score=1.0,
            participating_models=["test-model"],
        )

    def _create_chunks(self, text: str) -> List[ChunkOutput]:
        """Create chunks from text based on semantic boundaries."""
        chunks = []

        # Handle empty text
        if not text or not text.strip():
            return chunks

        # Split by paragraphs first
        paragraphs = text.split("\n\n")
        current_chunk_text = ""

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            # If paragraph is too long by itself, split it by sentences
            if len(paragraph) > self.MAX_CHUNK_LENGTH:
                # Split overly long paragraph by sentences
                sentences = paragraph.split(". ")
                for i, sentence in enumerate(sentences):
                    if i < len(sentences) - 1:  # Add period back except for last sentence
                        sentence += "."

                    # Check if adding this sentence exceeds max length
                    combined_length = len(current_chunk_text) + len(sentence)
                    if current_chunk_text and combined_length > self.MAX_CHUNK_LENGTH:
                        # Save current chunk
                        chunks.append(ChunkOutput(text=current_chunk_text.strip()))
                        current_chunk_text = sentence
                    else:
                        # Add to current chunk
                        if current_chunk_text:
                            current_chunk_text += " " + sentence
                        else:
                            current_chunk_text = sentence
            else:
                # Check if adding this paragraph exceeds max length
                combined_length = len(current_chunk_text) + len(paragraph)
                if current_chunk_text and combined_length > self.MAX_CHUNK_LENGTH:
                    # Save current chunk
                    chunks.append(ChunkOutput(text=current_chunk_text.strip()))
                    current_chunk_text = paragraph
                else:
                    # Add to current chunk
                    if current_chunk_text:
                        current_chunk_text += "\n\n" + paragraph
                    else:
                        current_chunk_text = paragraph

        # Add remaining text as final chunk
        if current_chunk_text.strip():
            chunks.append(ChunkOutput(text=current_chunk_text.strip()))

        return chunks

    def fill_template(
        self,
        page: KnowledgePageDataWithRawText,
        document_name: str,
        metadata: DocumentMetadata,
    ) -> str:
        """Simple template filling for testing."""
        return "Test chunking template"


class TestChunking:
    """Test suite for document chunking functionality."""

    @pytest.fixture
    def real_chunking_call(self):
        """Create a real chunking call."""
        return RealDocumentChunkingCall()

    @pytest.fixture
    def mock_calls_settings(self, real_chunking_call):
        """Create extraction calls settings with real chunking."""
        from unittest.mock import MagicMock

        # These remain as mocks since we're only testing chunking
        mock_term_extraction = MagicMock(spec=BaseTermExtractionCall)
        mock_chunk_classification = MagicMock(spec=BaseChunkContentClassificationCall)

        return ExtractionCallsSettings(
            term_extraction_call=mock_term_extraction,
            document_chunking_call=real_chunking_call,
            chunk_content_classification_call=mock_chunk_classification,
        )

    @pytest.fixture
    def processor_settings(self, tmp_path):
        """Create processor settings."""
        return KnowledgeProcessorSettings(
            extraction_output_dir=tmp_path / "test_output",
            linking_threshold=0.7,
            encoding_model="cl100k_base",
        )

    @pytest.mark.anyio
    async def test_chunking_with_short_text(self, real_chunking_call) -> None:
        """Test chunking with short text that fits in one chunk."""
        page = KnowledgePageDataWithRawText(
            page=1,
            text=TestChunkingConstants.SHORT_TEXT,
            images=[],
            tables=[],
        )

        result = await real_chunking_call.execute(page, "test.pdf", DocumentMetadata(document_path="test.pdf"))

        assert result.final_response is not None
        assert len(result.final_response.chunks) == 1
        assert result.final_response.chunks[0].text == TestChunkingConstants.SHORT_TEXT
        assert result.consensus_achieved is True
        assert result.convergence_score == 1.0

    @pytest.mark.anyio
    async def test_chunking_with_long_text(self, real_chunking_call) -> None:
        """Test chunking with long text requiring multiple chunks."""
        # Create exactly 50 sentences
        sentences = [
            TestChunkingConstants.SENTENCE_TEMPLATE.format(i) for i in range(TestChunkingConstants.LONG_TEXT_SENTENCES)
        ]
        long_text = ". ".join(sentences) + "."

        page = KnowledgePageDataWithRawText(page=1, text=long_text, images=[], tables=[])

        result = await real_chunking_call.execute(page, "test.pdf", DocumentMetadata(document_path="test.pdf"))

        assert result.final_response is not None
        # With 50 sentences of ~40 chars each = 2000 chars, expect 6 chunks with 500 char limit
        assert len(result.final_response.chunks) == 6

        # Verify all text is preserved
        all_chunk_text = " ".join(chunk.text for chunk in result.final_response.chunks)
        assert "Sentence number 0" in all_chunk_text
        assert "Sentence number 49" in all_chunk_text

    @pytest.mark.anyio
    async def test_chunking_preserves_semantic_boundaries(self, real_chunking_call) -> None:
        """Test that chunking respects semantic boundaries."""
        page = KnowledgePageDataWithRawText(
            page=1,
            text=TestChunkingConstants.PARAGRAPH_TEXT,
            images=[],
            tables=[],
        )

        result = await real_chunking_call.execute(page, "test.pdf", DocumentMetadata(document_path="test.pdf"))

        assert result.final_response is not None
        chunks = result.final_response.chunks

        # Should create 1 chunk since total length is under 500 chars
        assert len(chunks) == 1

        # Verify all sections are preserved
        chunk_text = chunks[0].text
        assert "Introduction:" in chunk_text
        assert "Methods:" in chunk_text
        assert "Results:" in chunk_text
        assert "Conclusion:" in chunk_text

    @pytest.mark.anyio
    async def test_chunking_with_tables_and_images(self, real_chunking_call) -> None:
        """Test chunking with mixed content (text, tables, images)."""
        page = KnowledgePageDataWithRawText(
            page=1,
            text=TestChunkingConstants.MIXED_CONTENT_TEXT,
            images=[],
            tables=[],
        )

        result = await real_chunking_call.execute(page, "test.pdf", DocumentMetadata(document_path="test.pdf"))

        assert result.final_response is not None
        assert len(result.final_response.chunks) == 1

        chunk_text = result.final_response.chunks[0].text
        assert "[TABLE: data table]" in chunk_text
        assert "[FIGURE 1: diagram]" in chunk_text
        assert chunk_text == TestChunkingConstants.MIXED_CONTENT_TEXT

    @pytest.mark.anyio
    async def test_chunking_empty_page(self, real_chunking_call) -> None:
        """Test chunking with empty page."""
        page = KnowledgePageDataWithRawText(
            page=1,
            text="",
            images=[],
            tables=[],
        )

        result = await real_chunking_call.execute(page, "test.pdf", DocumentMetadata(document_path="test.pdf"))

        assert result.final_response is not None
        assert len(result.final_response.chunks) == 0
        assert result.consensus_achieved is True

    @pytest.mark.anyio
    async def test_chunking_special_characters(self, real_chunking_call) -> None:
        """Test chunking with special characters and formatting."""
        page = KnowledgePageDataWithRawText(
            page=1,
            text=TestChunkingConstants.SPECIAL_CHARS_TEXT,
            images=[],
            tables=[],
        )

        result = await real_chunking_call.execute(page, "test.pdf", DocumentMetadata(document_path="test.pdf"))

        assert result.final_response is not None
        chunks = result.final_response.chunks

        assert len(chunks) == 1
        chunk_text = chunks[0].text

        # Verify special characters are preserved exactly
        assert "•" in chunk_text
        assert "²" in chunk_text
        assert "```python" in chunk_text
        assert "def hello():" in chunk_text

    @pytest.mark.anyio
    async def test_chunk_creation_from_decisions(self, real_chunking_call) -> None:
        """Test creation of KnowledgeChunk objects from ChunkOutputs."""
        document_id = "test-doc-123"
        document_name = "test.pdf"

        # Create specific test chunks
        decisions = [
            ChunkOutput(text="First chunk text"),
            ChunkOutput(text="Second chunk text"),
        ]

        # Create KnowledgeChunk objects
        chunks = []
        for idx, decision in enumerate(decisions):
            chunk = KnowledgeChunk(
                document_id=document_id,
                document_name=document_name,
                doc_id=f"{document_id}-1-{idx}",
                index=idx,
                page=1,
                text=decision.text,
                content_types=["text"],
                semantic_types=["general"],
            )
            chunks.append(chunk)

        # Exact assertions
        assert len(chunks) == 2
        assert chunks[0].text == "First chunk text"
        assert chunks[1].text == "Second chunk text"
        assert chunks[0].doc_id == "test-doc-123-1-0"
        assert chunks[1].doc_id == "test-doc-123-1-1"
        assert chunks[0].index == 0
        assert chunks[1].index == 1

    @pytest.mark.anyio
    async def test_chunking_whitespace_handling(self, real_chunking_call) -> None:
        """Test chunking handles various whitespace correctly."""
        text_with_whitespace = "  First paragraph.  \n\n  Second paragraph.  \n\n\n  Third paragraph.  "

        page = KnowledgePageDataWithRawText(
            page=1,
            text=text_with_whitespace,
            images=[],
            tables=[],
        )

        result = await real_chunking_call.execute(page, "test.pdf", DocumentMetadata(document_path="test.pdf"))

        assert result.final_response is not None
        chunks = result.final_response.chunks

        assert len(chunks) == 1
        chunk_text = chunks[0].text

        # Verify whitespace is normalized
        assert "First paragraph." in chunk_text
        assert "Second paragraph." in chunk_text
        assert "Third paragraph." in chunk_text
        # Should not have excessive whitespace
        assert "  \n\n\n  " not in chunk_text
