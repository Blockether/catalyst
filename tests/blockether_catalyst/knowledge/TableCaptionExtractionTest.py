"""
Tests for table caption extraction functionality.
"""

import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from blockether_catalyst.consensus.ConsensusTypes import ConsensusResult
from blockether_catalyst.knowledge.extraction.ConcreteExtractionCalls import (
    ConcreteTableCaptionExtractionCall,
)
from blockether_catalyst.knowledge.extraction.internal.KnowledgeExtractionCallBase import (
    BaseTableCaptionExtractionCall,
)
from blockether_catalyst.knowledge.extraction.ModelSettings import (
    ConsensusSettings,
    ModelSettings,
)
from blockether_catalyst.knowledge.extraction.internal.PDFExtractor import (
    PDFKnowledgeExtractor,
    PDFTableData,
)
from blockether_catalyst.knowledge.KnowledgeTypes import (
    KnowledgeProcessorSettings,
    TableCaptionExtractionResponse,
)
from blockether_catalyst.knowledge.extraction.internal.ExtractionTypes import (
    PDFKnowledgeProcessorSettings,
    PDFProcessorTableExtractionSettings,
    PDFProcessorTextExtractionSettings,
)


class MockTableCaptionCall(BaseTableCaptionExtractionCall):
    """Mock table caption extraction call for testing."""

    def __init__(self, mock_response: TableCaptionExtractionResponse):
        """Initialize with a mock response."""
        self._mock_response = mock_response
        # Create a mock consensus that returns our response
        mock_consensus = MagicMock()
        mock_consensus.call = AsyncMock(
            return_value=ConsensusResult(
                final_response=mock_response,
                rounds=[],
                reasoning="Mock consensus reasoning for testing purposes with detailed explanation of the consensus process and validation steps taken during table caption extraction",
                consensus_achieved=True,
                total_rounds=1,
                convergence_score=1.0,
                participating_models=["mock-model"]
            )
        )
        super().__init__(consensus=mock_consensus)

    def fill_template(
        self,
        table_content: str,
        document_name: str,
        page_number: int,
    ) -> str:
        """Mock template filling."""
        return f"Table: {table_content[:50]}... in {document_name} page {page_number}"


class TestTableCaptionExtraction:
    """Test suite for table caption extraction."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        # Create settings
        self.pdf_settings = PDFKnowledgeProcessorSettings(
            pdf_table_extraction=PDFProcessorTableExtractionSettings(
                text_tolerance=3,
                text_x_tolerance=3,
                text_y_tolerance=3,
                edge_min_length=10,
                min_words_vertical=3,
                min_words_horizontal=1,
                keep_blank_chars=False,
                snap_tolerance=3.0,
                snap_x_tolerance=None,
                snap_y_tolerance=None,
                join_tolerance=3.0,
                join_x_tolerance=None,
                join_y_tolerance=None,
            ),
            pdf_text_extraction=PDFProcessorTextExtractionSettings(
                use_text_flow=True,
                paragraph_threshold=0.05,
                word_margin=0.2,
                char_margin=2.0,
                line_margin=0.5,
                boxes_flow=0.5,
            ),
        )

        self.knowledge_settings = KnowledgeProcessorSettings(
            pdf_settings=self.pdf_settings,
        )

        # Create image output directory
        self.image_output_dir = Path("/tmp/test_images")

    @pytest.mark.anyio
    async def test_generate_table_caption_with_consensus_call(self) -> None:
        """Test generating table caption with consensus call."""
        # Create mock caption response
        mock_response = TableCaptionExtractionResponse(
            reasoning="This table contains financial data with quarterly revenue metrics for 2024 and includes growth percentages showing business performance trends",
            caption="Quarterly Revenue Report 2024",
        )

        # Create mock table caption call
        mock_caption_call = MockTableCaptionCall(mock_response)

        # Create PDF extractor with mock call
        extractor = PDFKnowledgeExtractor(
            image_output_dir=self.image_output_dir,
            knowledge_settings=self.knowledge_settings,
            table_caption_call=mock_caption_call,
        )

        # Test table with content
        table_data = PDFTableData(
            page=5,
            data=[
                ["Q1", "Q2", "Q3", "Q4"],
                ["$1.2M", "$1.5M", "$1.8M", "$2.1M"],
                ["Revenue", "Growth", "15%", "20%"],
            ],
            rows=3,
            columns=4,
            caption="",
            bbox=(0, 0, 100, 100),
        )

        caption = await extractor._generate_table_caption(
            table_data=table_data,
            document_name="financial_report.pdf",
            page_number=5,
        )

        # Verify the caption contains the expected result
        assert "Quarterly Revenue Report 2024" in caption
        assert "financial_report.pdf" in caption
        assert "Page 5" in caption

    @pytest.mark.anyio
    async def test_generate_table_caption_no_call_provided(self) -> None:
        """Test generating table caption when no consensus call is provided."""
        # Create PDF extractor without caption call
        extractor = PDFKnowledgeExtractor(
            image_output_dir=self.image_output_dir,
            knowledge_settings=self.knowledge_settings,
            table_caption_call=None,  # No call provided
        )

        # Test table
        table_data = PDFTableData(
            page=1,
            data=[
                ["Header1", "Header2"],
                ["Value1", "Value2"],
            ],
            rows=2,
            columns=2,
            caption="",
            bbox=(0, 0, 50, 50),
        )

        caption = await extractor._generate_table_caption(
            table_data=table_data,
            document_name="test_doc.pdf",
            page_number=1,
        )

        # Should return basic caption
        assert caption == "Table - test_doc.pdf (Page 1)"

    @pytest.mark.anyio
    async def test_generate_table_caption_empty_table(self) -> None:
        """Test generating caption for empty table."""
        # Create mock caption response
        mock_response = TableCaptionExtractionResponse(
            reasoning="Empty table structure detected with no visible content or data rows present in document section requiring analysis",
            caption="Empty Data Table Structure",
        )

        # Create mock table caption call
        mock_caption_call = MockTableCaptionCall(mock_response)

        # Create PDF extractor with mock call
        extractor = PDFKnowledgeExtractor(
            image_output_dir=self.image_output_dir,
            knowledge_settings=self.knowledge_settings,
            table_caption_call=mock_caption_call,
        )

        # Test empty table
        table_data = PDFTableData(
            page=1,
            data=[],
            rows=0,
            columns=0,
            caption="",
            bbox=(0, 0, 0, 0),
        )

        caption = await extractor._generate_table_caption(
            table_data=table_data,
            document_name="empty_doc.pdf",
            page_number=1,
        )

        # Should handle empty table gracefully
        assert "Empty Data Table Structure" in caption
        assert "empty_doc.pdf" in caption

    @pytest.mark.anyio
    async def test_generate_table_caption_with_exception(self) -> None:
        """Test handling exceptions during caption generation."""
        # Create a mock call that raises an exception
        mock_caption_call = MagicMock()
        mock_caption_call.execute = AsyncMock(side_effect=Exception("API Error"))

        # Create PDF extractor with faulty call
        extractor = PDFKnowledgeExtractor(
            image_output_dir=self.image_output_dir,
            knowledge_settings=self.knowledge_settings,
            table_caption_call=mock_caption_call,
        )

        # Test table
        table_data = PDFTableData(
            page=3,
            data=[
                ["Test", "Data"],
                ["Value", "123"],
            ],
            rows=2,
            columns=2,
            caption="",
            bbox=(0, 0, 75, 75),
        )

        # Should handle exception and return basic caption
        with patch.object(extractor._logger, "warning") as mock_logger:
            caption = await extractor._generate_table_caption(
                table_data=table_data,
                document_name="error_doc.pdf",
                page_number=3,
            )

            # Check that warning was logged
            mock_logger.assert_called_once()
            assert "Failed to generate table caption" in str(mock_logger.call_args)

            # Should return fallback caption
            assert caption == "Table - error_doc.pdf (Page 3)"

    @pytest.mark.anyio
    async def test_table_caption_integration_with_failed_consensus(self) -> None:
        """Test handling when consensus fails to achieve agreement."""
        # Create a mock call that returns a failed consensus result
        mock_caption_call = MagicMock()
        mock_caption_call.execute = AsyncMock(
            return_value=ConsensusResult(
                final_response=TableCaptionExtractionResponse(
                    reasoning="Failed to generate caption due to processing error or insufficient table content data available for analysis",
                    caption="Unable to Generate Caption"
                ),
                rounds=[],
                reasoning="Mock consensus failure case for testing error handling scenarios with comprehensive validation of edge cases and exception management",
                consensus_achieved=False,
                total_rounds=1,
                convergence_score=0.0,
                participating_models=["mock-model"]
            )
        )

        # Create PDF extractor with mock call
        extractor = PDFKnowledgeExtractor(
            image_output_dir=self.image_output_dir,
            knowledge_settings=self.knowledge_settings,
            table_caption_call=mock_caption_call,
        )

        # Test table
        table_data = PDFTableData(
            page=1,
            data=[["Data"]],
            rows=1,
            columns=1,
            caption="",
            bbox=(0, 0, 25, 25),
        )

        caption = await extractor._generate_table_caption(
            table_data=table_data,
            document_name="none_result.pdf",
            page_number=1,
        )

        # Should handle failed consensus and return the failed response
        assert "Unable to Generate Caption" in caption
        assert "none_result.pdf" in caption

    @pytest.mark.anyio
    async def test_table_caption_markdown_conversion(self) -> None:
        """Test that table data is properly converted to markdown format."""
        called_with_content = None

        class CaptureContentCall(BaseTableCaptionExtractionCall):
            """Capture the table content passed to fill_template."""

            def __init__(self) -> None:
                mock_consensus = MagicMock()
                mock_consensus.call = AsyncMock(
                    return_value=ConsensusResult(
                        final_response=TableCaptionExtractionResponse(
                            reasoning="Test reasoning for table caption generation with detailed analysis of structured content and format",
                            caption="Comprehensive Test Caption"
                        ),
                        rounds=[],
                        reasoning="Mock consensus reasoning for testing content capture functionality with detailed validation of markdown conversion and content processing workflows",
                        consensus_achieved=True,
                        total_rounds=1,
                        convergence_score=1.0,
                        participating_models=["mock-model"]
                    )
                )
                super().__init__(consensus=mock_consensus)

            def fill_template(self, table_content: str, document_name: str, page_number: int) -> str:
                nonlocal called_with_content
                called_with_content = table_content
                return "test prompt"

        # Create PDF extractor with capture call
        capture_call = CaptureContentCall()
        extractor = PDFKnowledgeExtractor(
            image_output_dir=self.image_output_dir,
            knowledge_settings=self.knowledge_settings,
            table_caption_call=capture_call,
        )

        # Test table with various content
        table_data = PDFTableData(
            page=1,
            data=[
                ["Column A", "Column B", "Column C"],
                ["Value 1", "Value 2", "Value 3"],
                ["Long Value Here", "123.45", "Short"],
            ],
            rows=3,
            columns=3,
            caption="",
            bbox=(0, 0, 200, 100),
        )

        await extractor._generate_table_caption(
            table_data=table_data,
            document_name="test.pdf",
            page_number=1,
        )

        # Verify markdown format was created
        assert called_with_content is not None
        assert "Column A" in called_with_content
        assert "Column B" in called_with_content
        assert "|" in called_with_content  # Markdown table separator
        assert "-" in called_with_content  # Markdown header separator

    def test_concrete_table_caption_call_with_real_models(self) -> None:
        """Test ConcreteTableCaptionExtractionCall with real models."""
        # Create consensus settings with real models as requested
        settings = ConsensusSettings(
            models=[
                ModelSettings(
                    model="gpt-4o",
                    api_url="http://localhost:3005/v1",
                    api_key="test-key",
                    temperature=0.7,
                    weight=1.0,
                    perspective="default",
                ),
                ModelSettings(
                    model="gpt-4o",
                    api_url="http://localhost:3005/v1",
                    api_key="test-key",
                    temperature=0.5,
                    weight=1.0,
                    perspective="conservative",
                ),
                ModelSettings(
                    model="gpt-4o",
                    api_url="http://localhost:3005/v1",
                    api_key="test-key",
                    temperature=0.6,
                    weight=1.0,
                    perspective="balanced",
                )
            ],
            consensus_threshold=0.8,
            max_rounds=3,
        )

        # Create concrete call - this should initialize without errors
        call = ConcreteTableCaptionExtractionCall(settings=settings)

        # Verify initialization
        assert call.settings == settings
        assert call.consensus is not None

        # Test fill_template method
        prompt = call.fill_template(
            table_content="| Header 1 | Header 2 |\n|----------|----------|\n| Value 1  | Value 2  |",
            document_name="test_document.pdf",
            page_number=5,
        )

        # Verify prompt contains expected information
        assert "Header 1" in prompt
        assert "Header 2" in prompt
        assert "test_document.pdf" in prompt
        assert "5" in prompt
        assert "caption" in prompt.lower()