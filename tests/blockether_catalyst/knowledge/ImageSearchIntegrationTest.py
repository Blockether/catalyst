from pathlib import Path
from typing import Dict, List, Tuple
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from blockether_catalyst.knowledge.extraction.ExtractionCore import (
    KnowledgeExtractionCore,
)
from blockether_catalyst.knowledge.extraction.internal.KnowledgeExtractionCallBase import (
    ExtractionCallsSettings,
)
from blockether_catalyst.knowledge.KnowledgeProcessingUtils import (
    KnowledgeProcessingUtils,
)
from blockether_catalyst.knowledge.KnowledgeTypes import (
    DocumentMetadata,
    ImageMetadata,
    KnowledgePageData,
    KnowledgeProcessorSettings,
    RawKnowledgeChunk,
)


class TestImageSearchIntegration:
    """Test that image captions are included in searchable chunks."""

    def _create_test_extraction_core(self) -> KnowledgeExtractionCore:
        """Create a properly configured KnowledgeExtractionCore for testing."""
        # Create mock calls settings
        mock_calls = Mock(spec=ExtractionCallsSettings)
        mock_calls.term_extraction_call = Mock()
        mock_calls.document_chunking_call = Mock()
        mock_calls.chunk_content_classification_call = Mock()

        # Create mock processor settings
        mock_settings = Mock(spec=KnowledgeProcessorSettings)
        mock_settings.extraction_output_dir = Path("/tmp/test_output")
        mock_settings.model_dump.return_value = {}
        mock_settings.image_optimization_level = 3
        mock_pdf_settings = Mock()
        mock_pdf_settings.extraction_mode = "standard"
        mock_settings.pdf_settings = mock_pdf_settings

        with (
            patch("pathlib.Path.mkdir"),
            patch("pathlib.Path.exists", return_value=True),
            patch.object(Path, "__new__", side_effect=lambda cls, *args: object.__new__(cls)),
        ):
            with patch(
                "blockether_catalyst.knowledge.optimization.ImageOptimizer.ImageOptimizer.__init__",
                return_value=None,
            ):
                with patch(
                    "blockether_catalyst.knowledge.extraction.internal.ImageExtractor.ImageRecognition.__init__",
                    return_value=None,
                ):
                    return KnowledgeExtractionCore(calls=mock_calls, settings=mock_settings)

    @pytest.mark.anyio
    async def test_image_captions_included_in_chunks(self) -> None:
        """Verify that image captions are included in page text for chunking."""
        extraction_core = self._create_test_extraction_core()

        # Create a page with text and images
        test_page = KnowledgePageData(
            page=1,
            text="This is the main text content of the page.",
            images=[
                ImageMetadata(
                    document_name="test.pdf",
                    page=1,
                    path="images/img1.png",
                    caption="A diagram showing the system architecture with multiple components",
                ),
                ImageMetadata(
                    document_name="test.pdf",
                    page=1,
                    path="images/img2.png",
                    caption="Performance graph displaying throughput over time",
                ),
            ],
            tables=[],
        )

        # Test the enrichment method
        enriched_text = extraction_core._enrich_page_text_with_images(test_page)

        # Verify that the original text is preserved
        assert "This is the main text content of the page." in enriched_text

        # Verify that image captions are included with markers
        assert (
            "<<<IMAGE_CAPTION_START>>>A diagram showing the system architecture with multiple components<<<IMAGE_CAPTION_END>>>"
            in enriched_text
        )
        assert (
            "<<<IMAGE_CAPTION_START>>>Performance graph displaying throughput over time<<<IMAGE_CAPTION_END>>>"
            in enriched_text
        )

        # Verify the structure is correct
        expected_text = (
            "This is the main text content of the page.\n\n"
            "<<<IMAGE_CAPTION_START>>>A diagram showing the system architecture with multiple components<<<IMAGE_CAPTION_END>>>\n"
            "<<<IMAGE_CAPTION_START>>>Performance graph displaying throughput over time<<<IMAGE_CAPTION_END>>>"
        )
        assert enriched_text == expected_text

        # Test that markers can be cleaned for display
        cleaned = KnowledgeProcessingUtils.clean_image_markers_from_text(enriched_text)
        assert cleaned == "This is the main text content of the page."
        assert "<<<IMAGE_CAPTION_START>>>" not in cleaned
        assert "<<<IMAGE_CAPTION_END>>>" not in cleaned

    @pytest.mark.anyio
    async def test_page_without_images(self) -> None:
        """Verify that pages without images work correctly."""
        extraction_core = self._create_test_extraction_core()

        # Create a page without images
        test_page = KnowledgePageData(page=1, text="This is a page with no images.", images=[], tables=[])

        # Test the enrichment method
        enriched_text = extraction_core._enrich_page_text_with_images(test_page)

        # Verify that the text is unchanged when there are no images
        assert enriched_text == "This is a page with no images."

    @pytest.mark.anyio
    async def test_images_without_captions(self) -> None:
        """Verify that images without captions are handled correctly."""
        extraction_core = self._create_test_extraction_core()

        # Create a page with images that have no captions
        test_page = KnowledgePageData(
            page=1,
            text="Main page content.",
            images=[
                ImageMetadata(
                    document_name="test.pdf",
                    page=1,
                    path="images/img1.png",
                    caption="",  # Empty caption
                ),
                ImageMetadata(
                    document_name="test.pdf",
                    page=1,
                    path="images/img2.png",
                    caption="",  # Empty caption
                ),
            ],
            tables=[],
        )

        # Test the enrichment method
        enriched_text = extraction_core._enrich_page_text_with_images(test_page)

        # Verify that the text is unchanged when images have no captions
        assert enriched_text == "Main page content."

    @pytest.mark.anyio
    async def test_mixed_images_with_and_without_captions(self) -> None:
        """Verify that mixed images (some with captions, some without) are handled correctly."""
        extraction_core = self._create_test_extraction_core()

        # Create a page with mixed images
        test_page = KnowledgePageData(
            page=1,
            text="Content with mixed images.",
            images=[
                ImageMetadata(
                    document_name="test.pdf",
                    page=1,
                    path="images/img1.png",
                    caption="",  # Empty caption
                ),
                ImageMetadata(
                    document_name="test.pdf",
                    page=1,
                    path="images/img2.png",
                    caption="This image has a caption",
                ),
                ImageMetadata(
                    document_name="test.pdf",
                    page=1,
                    path="images/img3.png",
                    caption="",  # Empty caption
                ),
                ImageMetadata(
                    document_name="test.pdf",
                    page=1,
                    path="images/img4.png",
                    caption="Another caption for searching",
                ),
            ],
            tables=[],
        )

        # Test the enrichment method
        enriched_text = extraction_core._enrich_page_text_with_images(test_page)

        # Verify original text is preserved
        assert "Content with mixed images." in enriched_text

        # Verify only images with captions are included
        assert "<<<IMAGE_CAPTION_START>>>This image has a caption<<<IMAGE_CAPTION_END>>>" in enriched_text
        assert "<<<IMAGE_CAPTION_START>>>Another caption for searching<<<IMAGE_CAPTION_END>>>" in enriched_text

        # Verify images without captions are not included
        assert "img1.png" not in enriched_text
        assert "img3.png" not in enriched_text

        # Verify the exact structure
        expected_text = (
            "Content with mixed images.\n\n"
            "<<<IMAGE_CAPTION_START>>>This image has a caption<<<IMAGE_CAPTION_END>>>\n"
            "<<<IMAGE_CAPTION_START>>>Another caption for searching<<<IMAGE_CAPTION_END>>>"
        )
        assert enriched_text == expected_text

        # Verify cleaning works
        cleaned = KnowledgeProcessingUtils.clean_image_markers_from_text(enriched_text)
        assert cleaned == "Content with mixed images."
        assert "<<<IMAGE_CAPTION_START>>>" not in cleaned
