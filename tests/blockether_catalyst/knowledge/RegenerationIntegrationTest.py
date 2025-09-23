"""
Integration tests for knowledge extraction regeneration functionality.
Tests the complete regeneration workflow with real pipeline interactions.
"""

import json
import pickle
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import anyio
import pytest

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
    DocumentMetadata,
    ImageMetadata,
    KnowledgeChunk,
    KnowledgeChunkWithTerms,
    KnowledgeExtractionResultWithChunks,
    KnowledgeProcessorSettings,
    LinkedKnowledge,
    NormalizedDocumentMetadata,
    TermOccurrence,
    TermWithLinks,
)


class TestRegenerationIntegration:
    """Integration test suite for complete regeneration workflows."""

    EXPECTED_STEP_COUNT = 10
    EXPECTED_IMAGE_AFFECTED_STEPS = 3
    EXPECTED_PRESERVED_STEPS = 7

    @pytest.fixture
    def mock_calls_settings(self):
        """Create comprehensive mock extraction calls settings."""
        mock_term_extraction = MagicMock(spec=BaseTermExtractionCall)
        mock_document_chunking = MagicMock(spec=BaseDocumentChunkingCall)
        mock_chunk_classification = MagicMock(spec=BaseChunkContentClassificationCall)

        # Configure mock responses
        mock_term_extraction.execute = AsyncMock(
            return_value={"terms": [{"term": "API", "type": "acronym", "count": 5}]}
        )
        mock_document_chunking.execute = AsyncMock(return_value={"chunks": [{"text": "Sample chunk", "page": 1}]})
        mock_chunk_classification.execute = AsyncMock(
            return_value={"classification": {"semantic_types": ["explanation"]}}
        )

        return ExtractionCallsSettings(
            term_extraction_call=mock_term_extraction,
            document_chunking_call=mock_document_chunking,
            chunk_content_classification_call=mock_chunk_classification,
        )

    @pytest.fixture
    def processor_settings(self, tmp_path):
        """Create processor settings with temporary directories."""
        return KnowledgeProcessorSettings(
            extraction_output_dir=tmp_path / "extraction_output",
            linking_threshold=0.7,
            encoding_model="cl100k_base",
        )

    @pytest.fixture
    def extractor(self, mock_calls_settings, processor_settings):
        """Create KnowledgeExtractionCore instance for integration testing."""
        return KnowledgeExtractionCore(
            calls=mock_calls_settings,
            settings=processor_settings,
        )

    @pytest.fixture
    def complete_extraction_state(self, extractor):
        """Create a complete extraction state with realistic data."""
        output_dir = extractor._settings.extraction_output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create realistic extraction state
        extraction_data = self._create_realistic_extraction_data()

        for step_name, data in extraction_data.items():
            step_file = output_dir / f"{step_name}.pkl"
            with open(step_file, "wb") as f:
                pickle.dump(data, f)

        return extraction_data

    def _create_realistic_extraction_data(self) -> Dict[str, Any]:
        """Create realistic extraction data for testing."""
        return {
            "1_raw_extraction": {
                "documents": {
                    "doc1_hash": NormalizedDocumentMetadata(
                        document_id="doc1_hash",
                        document_filename="technical_guide.pdf",
                        document_path="/docs/technical_guide.pdf",
                        title="Technical Guide",
                        subject="Documentation",
                        author="Engineering Team",
                        modification_date="2024-01-15T10:00:00Z",
                        publication_date="2024-01-15",
                        total_pages=25,
                        total_chunks=15,
                        total_terms=85,
                        total_keywords=65,
                        total_acronyms=20,
                        total_images=8,
                        total_tables=3,
                    )
                },
                "images": {
                    "doc1_hash": [
                        ImageMetadata(
                            document_name="document1.pdf",
                            page=5,
                            path="/images/diagram_001.png",
                            caption="System architecture diagram",
                        ),
                        ImageMetadata(
                            document_name="document1.pdf",
                            page=12,
                            path="/images/flowchart_001.png",
                            caption="Process flowchart",
                        ),
                    ]
                },
            },
            "2_chunked_documents": {
                "chunks": {
                    "doc1_hash": [
                        KnowledgeChunk(
                            document_id="doc1_hash",
                            document_name="technical_guide.pdf",
                            doc_id="doc1_hash_p5_c3",
                            index=3,
                            text="The API provides secure authentication mechanisms for user verification.",
                            page=5,
                            content_types=["text", "image"],
                            semantic_types=["explanation"],
                        )
                    ]
                }
            },
            "3_classified_chunks": {
                "classified_chunks": {
                    "doc1_hash": [
                        KnowledgeChunkWithTerms(
                            document_id="doc1_hash",
                            document_name="technical_guide.pdf",
                            doc_id="doc1_hash_p5_c3",
                            index=3,
                            text="The API provides secure authentication mechanisms for user verification.",
                            page=5,
                            content_types=["text", "image"],
                            semantic_types=["explanation"],
                            terms={"API": 1, "authentication": 1},
                        )
                    ]
                }
            },
            "4_term_candidates": {
                "terms": {
                    "API": {
                        "term": "API",
                        "type": "acronym",
                        "full_form": "Application Programming Interface",
                        "total_occurrences": 15,
                        "documents": ["doc1_hash"],
                    },
                    "authentication": {
                        "term": "authentication",
                        "type": "keyword",
                        "total_occurrences": 8,
                        "documents": ["doc1_hash"],
                    },
                }
            },
            "5_grouped_terms": {"grouped_terms": {"API": {"variations": []}}},
            "6_terms_with_cooccurrences": {"cooccurrences": True},
            "7_terms_with_meanings": {
                "refined_terms": {
                    "API": TermWithLinks(
                        term="API",
                        type="acronym",
                        full_form="Application Programming Interface",
                        meaning="A set of protocols and tools for building software applications",
                        total=15,
                        reasoning="API is identified as an acronym standing for Application Programming Interface. It appears frequently in technical documentation and refers to protocols and tools used for building software applications. This term is fundamental to software development and integration discussions.",
                        occurrences=[
                            TermOccurrence(
                                document_id="doc1_hash",
                                document_name="technical_guide.pdf",
                                page=5,
                                chunk_index=3,
                                total=3,
                            )
                        ],
                        cooccurrences=[],
                        links=[],
                    )
                }
            },
            "8_terms_with_links": {
                "linked_terms": {
                    "API": TermWithLinks(
                        term="API",
                        type="acronym",
                        full_form="Application Programming Interface",
                        meaning="A set of protocols and tools for building software applications",
                        total=15,
                        reasoning="API is identified as an acronym standing for Application Programming Interface. It appears frequently in technical documentation and refers to protocols and tools used for building software applications. This term is fundamental to software development and integration discussions.",
                        occurrences=[
                            TermOccurrence(
                                document_id="doc1_hash",
                                document_name="technical_guide.pdf",
                                page=5,
                                chunk_index=3,
                                total=3,
                            )
                        ],
                        cooccurrences=[],
                        links=[],
                    )
                }
            },
            "linked_knowledge": {
                "final_result": LinkedKnowledge(
                    documents={
                        "doc1_hash": NormalizedDocumentMetadata(
                            document_id="doc1_hash",
                            document_filename="technical_guide.pdf",
                            document_path="/docs/technical_guide.pdf",
                            title="Technical Guide",
                            subject="Documentation",
                            author="Engineering Team",
                            modification_date="2024-01-15T10:00:00Z",
                            publication_date="2024-01-15",
                            total_pages=25,
                            total_chunks=15,
                            total_terms=85,
                            total_keywords=65,
                            total_acronyms=20,
                            total_images=8,
                            total_tables=3,
                        )
                    },
                    terms={
                        "API": TermWithLinks(
                            term="API",
                            type="acronym",
                            full_form="Application Programming Interface",
                            meaning="A set of protocols and tools for building software applications",
                            total=15,
                            reasoning="API is identified as an acronym standing for Application Programming Interface. It appears frequently in technical documentation and refers to protocols and tools used for building software applications. This term is fundamental to software development and integration discussions.",
                            occurrences=[],
                            cooccurrences=[],
                            links=[],
                        )
                    },
                    chunks={
                        "doc1_hash_p5_c3": KnowledgeChunkWithTerms(
                            document_id="doc1_hash",
                            document_name="technical_guide.pdf",
                            doc_id="doc1_hash_p5_c3",
                            index=3,
                            text="The API provides secure authentication mechanisms.",
                            page=5,
                            content_types=["text", "image"],
                            semantic_types=["explanation"],
                            terms={"API": 1},
                        )
                    },
                    total_chunks=15,
                    total_keywords=65,
                    total_acronyms=20,
                    total_images=8,
                    total_tables=3,
                    extraction_timestamp=datetime.now().isoformat(),
                    processing_duration="45s",
                )
            },
            "knowledge_search": {
                "search_indices": {
                    "term_index": {"API": ["doc1_hash_p5_c3"]},
                    "content_index": {"authentication": ["doc1_hash_p5_c3"]},
                    "image_index": {"diagram": ["img_001", "img_002"]},
                }
            },
        }

    def test_complete_regeneration_workflow(self, extractor, complete_extraction_state, tmp_path):
        """Test the complete image regeneration workflow from start to finish."""
        output_dir = extractor._settings.extraction_output_dir

        # Verify initial state
        initial_preserved, initial_total = extractor._check_existing_state()
        assert initial_preserved == self.EXPECTED_STEP_COUNT
        assert initial_total == self.EXPECTED_STEP_COUNT

        # Create mock PDF files
        pdf_files = self._create_mock_pdf_files(tmp_path)

        # Perform image regeneration
        with patch(
            "blockether_catalyst.knowledge.PDFKnowledgeExtractor.PDFKnowledgeExtractor.regenerate_all_images"
        ) as mock_regen:
            mock_regen.return_value = None

            extractor._regenerate_images_with_dependencies(pdf_files)

        # Verify affected steps were invalidated
        affected_steps = extractor._get_image_affected_steps()
        for step in affected_steps:
            step_file = output_dir / f"{step}.pkl"
            assert not step_file.exists(), f"Step {step} should have been invalidated"

        # Verify preserved steps remain
        preserved_steps = [
            "2_chunked_documents",
            "3_classified_chunks",
            "4_term_candidates",
            "5_grouped_terms",
            "6_terms_with_cooccurrences",
            "7_terms_with_meanings",
            "8_terms_with_links",
        ]
        for step in preserved_steps:
            step_file = output_dir / f"{step}.pkl"
            assert step_file.exists(), f"Step {step} should have been preserved"

        # Verify final state
        final_preserved, final_total = extractor._check_existing_state()
        assert final_preserved == self.EXPECTED_PRESERVED_STEPS
        assert final_total == self.EXPECTED_STEP_COUNT

    def test_regeneration_maintains_pipeline_integrity(self, extractor, complete_extraction_state):
        """Test that regeneration maintains overall pipeline integrity."""
        # Get initial step ordering
        all_steps = extractor._get_extraction_steps()
        affected_steps = extractor._get_image_affected_steps()
        preserved_steps = [step for step in all_steps if step not in affected_steps]

        # Perform regeneration
        extractor._invalidate_dependent_steps(affected_steps)

        # Verify preserved steps maintain logical order
        for i, step in enumerate(preserved_steps[:-1]):
            current_index = all_steps.index(step)
            next_step = preserved_steps[i + 1]
            next_index = all_steps.index(next_step)

            # Verify no affected steps between preserved steps would break dependencies
            intermediate_steps = all_steps[current_index + 1 : next_index]
            conflicting_steps = [s for s in intermediate_steps if s not in affected_steps]
            assert len(conflicting_steps) == 0, f"Found conflicting preserved steps: {conflicting_steps}"

    def test_regeneration_with_missing_images_directory(self, extractor, complete_extraction_state, tmp_path):
        """Test regeneration behavior when images directory is missing."""
        pdf_files = self._create_mock_pdf_files(tmp_path)

        with patch(
            "blockether_catalyst.knowledge.PDFKnowledgeExtractor.PDFKnowledgeExtractor.regenerate_all_images"
        ) as mock_regen:
            # Simulate missing images directory
            mock_regen.side_effect = FileNotFoundError("Images directory not found")

            with pytest.raises(FileNotFoundError):
                extractor._regenerate_images_with_dependencies(pdf_files)

        # Verify that even with failure, dependencies were still invalidated
        output_dir = extractor._settings.extraction_output_dir
        affected_steps = extractor._get_image_affected_steps()

        for step in affected_steps:
            step_file = output_dir / f"{step}.pkl"
            assert not step_file.exists(), f"Step {step} should have been invalidated even on failure"

    def _create_mock_pdf_files(self, tmp_path) -> list[str]:
        """Create mock PDF files for testing."""
        pdf_dir = tmp_path / "test_pdfs"
        pdf_dir.mkdir()

        pdf_files = []
        for i in range(2):
            pdf_file = pdf_dir / f"document_{i}.pdf"
            pdf_file.write_bytes(b"mock PDF content")
            pdf_files.append(str(pdf_file))

        return pdf_files
