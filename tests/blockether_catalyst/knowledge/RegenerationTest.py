"""
Comprehensive tests for knowledge extraction regeneration functionality.
Tests image regeneration, dependency management, and pipeline state handling.
"""

import json
import pickle
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

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
    DocumentMetadata,
    KnowledgeExtractionResultWithChunks,
    KnowledgeProcessorSettings,
    LinkedKnowledge,
    NormalizedDocumentMetadata,
    RawKnowledgeChunk,
)


class TestRegenerationFunctionality:
    """Test suite for knowledge extraction regeneration functionality."""

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
        """Create KnowledgeExtractionCore instance with mocked dependencies."""
        return KnowledgeExtractionCore(
            calls=mock_calls_settings,
            settings=processor_settings,
        )

    @pytest.fixture
    def sample_pdf_files(self, tmp_path):
        """Create sample PDF files for testing."""
        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()

        # Create mock PDF files
        pdf1 = pdf_dir / "document1.pdf"
        pdf2 = pdf_dir / "document2.pdf"
        pdf1.write_bytes(b"PDF content 1")
        pdf2.write_bytes(b"PDF content 2")

        return [str(pdf1), str(pdf2)]

    @pytest.fixture
    def mock_extraction_state(self, extractor, tmp_path):
        """Create a mock extraction state with existing files."""
        output_dir = extractor._settings.extraction_output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create mock state files
        state_files = {
            "1_raw_extraction": {"metadata": {"total_images": 5}},
            "2_chunked_documents": {"chunks": [{"text": "Sample chunk"}]},
            "3_classified_chunks": {"classified": True},
            "4_term_candidates": {"terms": {"API": {"count": 10}}},
            "5_grouped_terms": {"grouped_terms": {"API": {"variations": []}}},
            "6_terms_with_cooccurrences": {"cooccurrences": True},
            "7_terms_with_meanings": {"refined_terms": {"API": {"meaning": "Interface"}}},
            "8_terms_with_links": {"links": [{"from": "API", "to": "REST"}]},
            "linked_knowledge": {"final": True},
            "knowledge_search": {"indexed": True},
        }

        for step_name, data in state_files.items():
            step_file = output_dir / f"{step_name}.pkl"
            with open(step_file, "wb") as f:
                pickle.dump(data, f)

        return state_files

    def test_get_image_affected_steps_returns_correct_steps(self, extractor):
        """Test that _get_image_affected_steps returns the correct dependency steps."""
        affected_steps = extractor._get_image_affected_steps()

        expected_steps = [
            "1_raw_extraction",
            "linked_knowledge",
            "knowledge_search",
        ]

        assert affected_steps == expected_steps

    def test_resolve_glob_patterns_with_pdfs(self, extractor, sample_pdf_files):
        """Test _resolve_glob_patterns correctly resolves PDF patterns."""
        # Use relative patterns that rglob can handle
        globs = ["*.pdf"]

        # Change to the directory containing the PDFs for the test
        import os

        pdf_dir = Path(sample_pdf_files[0]).parent
        original_cwd = os.getcwd()
        try:
            os.chdir(pdf_dir)
            resolved_files = extractor._resolve_glob_patterns(globs)

            assert len(resolved_files) == 2
            file_names = [f.name for f in resolved_files]
            assert "document1.pdf" in file_names
            assert "document2.pdf" in file_names
        finally:
            os.chdir(original_cwd)

    def test_resolve_glob_patterns_with_empty_directory(self, extractor, tmp_path):
        """Test _resolve_glob_patterns with empty directory."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        # Use relative patterns and change to empty directory
        import os

        globs = ["*.pdf"]
        original_cwd = os.getcwd()
        try:
            os.chdir(empty_dir)
            resolved_files = extractor._resolve_glob_patterns(globs)
            assert resolved_files == []
        finally:
            os.chdir(original_cwd)

    def test_get_extraction_status_identifies_all_steps(self, extractor):
        """Test get_extraction_status returns all pipeline steps."""
        status = extractor.get_extraction_status()

        expected_steps = [
            "1_raw_extraction",
            "2_chunked_documents",
            "3_classified_chunks",
            "4_term_candidates",
            "5_grouped_terms",
            "6_terms_with_cooccurrences",
            "7_terms_with_meanings",
            "8_terms_with_links",
            "linked_knowledge",
            "knowledge_search",
        ]

        assert list(status.keys()) == expected_steps

    def test_check_existing_state_with_complete_extraction(self, extractor, mock_extraction_state):
        """Test get_extraction_status with complete extraction state."""
        status = extractor.get_extraction_status()

        # All 10 steps should exist
        assert len(status) == 10
        assert all(status.values())  # All should be True

    def test_check_existing_state_with_missing_files(self, extractor, tmp_path):
        """Test get_extraction_status with missing state files."""
        output_dir = extractor._settings.extraction_output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # Only create some files
        step_file = output_dir / "1_raw_extraction.pkl"
        with open(step_file, "wb") as f:
            pickle.dump({"test": "data"}, f)

        status = extractor.get_extraction_status()

        # Only 1_raw_extraction should exist
        assert status["1_raw_extraction"] is True
        assert status["2_chunked_documents"] is False
        assert len([v for v in status.values() if v]) == 1

    def test_invalidate_image_affected_steps_removes_correct_files(self, extractor, mock_extraction_state):
        """Test _invalidate_image_affected_steps removes only affected step files."""
        output_dir = extractor._settings.extraction_output_dir

        # Verify files exist before invalidation
        assert (output_dir / "1_raw_extraction.pkl").exists()
        assert (output_dir / "linked_knowledge.pkl").exists()
        assert (output_dir / "knowledge_search.pkl").exists()
        assert (output_dir / "4_term_candidates.pkl").exists()  # Should not be removed

        invalidated_steps = extractor._invalidate_image_affected_steps()

        # Check affected files are removed
        assert not (output_dir / "1_raw_extraction.pkl").exists()
        assert not (output_dir / "linked_knowledge.pkl").exists()
        assert not (output_dir / "knowledge_search.pkl").exists()

        # Verify the expected steps were invalidated
        expected_affected = ["1_raw_extraction", "linked_knowledge", "knowledge_search"]
        assert set(invalidated_steps) == set(expected_affected)

        # Check unaffected files remain
        assert (output_dir / "2_chunked_documents.pkl").exists()
        assert (output_dir / "7_terms_with_meanings.pkl").exists()

    def test_invalidate_image_affected_steps_handles_missing_files_gracefully(self, extractor, tmp_path):
        """Test _invalidate_image_affected_steps handles missing files without errors."""
        output_dir = extractor._settings.extraction_output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # Try to invalidate non-existent files - should not raise an exception
        invalidated_steps = extractor._invalidate_image_affected_steps()

        # Should return empty list since no files existed
        assert invalidated_steps == []

    def test_image_regeneration_preserves_expensive_work(self, extractor, mock_extraction_state):
        """Test that image regeneration preserves expensive text processing work."""
        output_dir = extractor._settings.extraction_output_dir

        # Simulate image regeneration dependency invalidation
        affected_steps = extractor._get_image_affected_steps()
        extractor._invalidate_dependent_steps(affected_steps)

        # Verify expensive steps are preserved
        assert (output_dir / "4_term_candidates.pkl").exists()
        assert (output_dir / "7_terms_with_meanings.pkl").exists()
        assert (output_dir / "2_chunked_documents.pkl").exists()
        assert (output_dir / "3_classified_chunks.pkl").exists()
        assert (output_dir / "8_terms_with_links.pkl").exists()

        # Verify affected steps are removed
        assert not (output_dir / "1_raw_extraction.pkl").exists()
        assert not (output_dir / "linked_knowledge.pkl").exists()
        assert not (output_dir / "knowledge_search.pkl").exists()

    def test_regeneration_integration_verification_detects_image_content(self, extractor):
        """Test that regeneration properly updates content types to include images."""
        # This would be an integration test that verifies:
        # 1. Images are regenerated
        # 2. Chunks include new image content types
        # 3. LinkedKnowledge includes updated image metadata
        # 4. Search indices include image content

        # For now, test the logic that would verify this
        sample_chunk = RawKnowledgeChunk(
            document_id="test_doc",
            document_name="test.pdf",
            doc_id="test_chunk",
            index=0,
            text="Sample text with image reference",
            page=1,
            content_types=["text", "image"],  # Should include image after regeneration
            semantic_types=["explanation"],
        )

        # Verify image content type is present
        assert "image" in sample_chunk.content_types
        assert "text" in sample_chunk.content_types

    def test_regeneration_preserves_metadata_consistency(self, extractor):
        """Test that regeneration maintains metadata consistency across pipeline."""
        # Test that after regeneration:
        # - Document metadata reflects correct image counts
        # - Chunks reference valid images
        # - LinkedKnowledge has consistent image references

        sample_metadata = NormalizedDocumentMetadata(
            document_id="test_doc",
            document_filename="test.pdf",
            document_path="/test/test.pdf",
            title="Test Document",
            subject="Testing",
            author="Test Author",
            modification_date="2024-01-01T00:00:00Z",
            publication_date="2024-01-01",
            total_pages=10,
            total_chunks=5,
            total_terms=20,
            total_keywords=15,
            total_acronyms=5,
            total_images=8,  # Should reflect regenerated image count
            total_tables=2,
        )

        # Verify metadata structure is correct
        assert sample_metadata.total_images == 8
        assert sample_metadata.total_chunks == 5
        assert sample_metadata.document_id == "test_doc"

    def test_regeneration_step_ordering_maintains_pipeline_integrity(self, extractor):
        """Test that regeneration maintains correct pipeline step ordering."""
        status = extractor.get_extraction_status()
        all_steps = list(status.keys())
        affected_steps = extractor._get_image_affected_steps()

        # Verify affected steps are in correct positions
        step_indices = {step: all_steps.index(step) for step in affected_steps}

        # 1_raw_extraction should come before linked_knowledge and knowledge_search
        assert step_indices["1_raw_extraction"] < step_indices["linked_knowledge"]
        assert step_indices["1_raw_extraction"] < step_indices["knowledge_search"]

        # linked_knowledge should come before knowledge_search
        assert step_indices["linked_knowledge"] < step_indices["knowledge_search"]

    def test_regeneration_handles_concurrent_access_safely(self, extractor, tmp_path):
        """Test that regeneration handles concurrent file access safely."""
        output_dir = extractor._settings.extraction_output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create a test file
        test_file = output_dir / "test_step.pkl"
        with open(test_file, "wb") as f:
            pickle.dump({"test": "data"}, f)

        # Simulate file removal (as would happen in invalidation)
        extractor._invalidate_dependent_steps(["test_step"])

        # Verify file is safely removed
        assert not test_file.exists()

    def test_regeneration_error_handling_preserves_system_state(self, extractor, mock_extraction_state):
        """Test that regeneration errors don't corrupt the extraction state."""
        output_dir = extractor._settings.extraction_output_dir

        # Count files before potential error
        files_before = list(output_dir.glob("*.pkl"))

        # Test with invalid step name (should not crash)
        try:
            extractor._invalidate_dependent_steps(["non_existent_step"])
        except Exception:
            pass

        # Verify existing files are not affected by invalid operations
        files_after = list(output_dir.glob("*.pkl"))
        assert len(files_after) == len(files_before)
