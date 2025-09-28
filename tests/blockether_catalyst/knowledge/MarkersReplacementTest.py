"""
Tests for image and table marker replacement in search results.

This module tests the functionality of replacing special markers
(IMAGE_CAPTION_START/END and TABLE_START/END) with proper markdown
for display in search results.
"""

from typing import List

import pytest

from blockether_catalyst.knowledge.KnowledgeProcessingUtils import (
    KnowledgeProcessingUtils,
)
from blockether_catalyst.knowledge.KnowledgeTypes import ImageInfo, TableInfo


class TestImageMarkerReplacement:
    """Test cases for image marker replacement functionality."""

    def test_image_markers_constants_defined(self):
        """Test that IMAGE_CAPTION_START and IMAGE_CAPTION_END constants are correctly defined."""
        assert KnowledgeProcessingUtils.IMAGE_CAPTION_START == "<<<IMAGE_CAPTION_START>>>"
        assert KnowledgeProcessingUtils.IMAGE_CAPTION_END == "<<<IMAGE_CAPTION_END>>>"

    def test_replace_single_image_marker(self):
        """Test replacing a single image marker with markdown."""
        text = """
# Report

Here is some content.

<<<IMAGE_CAPTION_START>>>Financial chart showing growth<<<IMAGE_CAPTION_END>>>

More content here.
"""
        images = [
            ImageInfo(
                caption="Financial chart showing growth",
                href="/resources/chart.png",
                page=1,
                document_name="test_document.pdf",
            )
        ]

        result = KnowledgeProcessingUtils.replace_image_markers_with_markdown(text, images)

        assert "![Financial chart showing growth](/resources/chart.png)" in result
        assert "*Financial chart showing growth*" in result
        assert "<<<IMAGE_CAPTION_START>>>" not in result
        assert "<<<IMAGE_CAPTION_END>>>" not in result

    def test_replace_multiple_image_markers(self):
        """Test replacing multiple image markers with markdown."""
        text = """
# Annual Report

<<<IMAGE_CAPTION_START>>>CEO presenting results<<<IMAGE_CAPTION_END>>>

Some text here.

<<<IMAGE_CAPTION_START>>>Revenue growth chart<<<IMAGE_CAPTION_END>>>

Final text.
"""
        images = [
            ImageInfo(
                caption="CEO presenting results",
                href="/img/ceo.jpg",
                page=1,
                document_name="test.pdf",
            ),
            ImageInfo(
                caption="Revenue growth chart",
                href="/img/revenue.png",
                page=2,
                document_name="test.pdf",
            ),
        ]

        result = KnowledgeProcessingUtils.replace_image_markers_with_markdown(text, images)

        assert "![CEO presenting results](/img/ceo.jpg)" in result
        assert "![Revenue growth chart](/img/revenue.png)" in result
        assert "<<<IMAGE_CAPTION_START>>>" not in result
        assert "<<<IMAGE_CAPTION_END>>>" not in result

    def test_image_markers_without_matching_images(self):
        """Test handling when no images are provided for markers."""
        text = """<<<IMAGE_CAPTION_START>>>Missing image<<<IMAGE_CAPTION_END>>>"""

        # No images provided
        result = KnowledgeProcessingUtils.replace_image_markers_with_markdown(text, None)

        # Markers should be removed
        assert "<<<IMAGE_CAPTION_START>>>" not in result
        assert "<<<IMAGE_CAPTION_END>>>" not in result
        assert "Missing image" not in result

    def test_image_caption_without_url(self):
        """Test handling image with caption but no URL."""
        text = """<<<IMAGE_CAPTION_START>>>Image without URL<<<IMAGE_CAPTION_END>>>"""

        images = [
            ImageInfo(
                caption="Different caption",
                href="/img/other.jpg",
                page=1,
                document_name="test.pdf",
            )
        ]

        result = KnowledgeProcessingUtils.replace_image_markers_with_markdown(text, images)

        # Should show as italic text when no matching URL found
        assert "*Image without URL*" in result
        assert "![Image without URL]" not in result


class TestTableMarkerReplacement:
    """Test cases for table marker replacement functionality."""

    def test_table_markers_constants_defined(self):
        """Test that TABLE_START and TABLE_END constants are correctly defined."""
        assert KnowledgeProcessingUtils.TABLE_START == "<<<TABLE_START>>>"
        assert KnowledgeProcessingUtils.TABLE_END == "<<<TABLE_END>>>"

    def test_replace_single_table_marker(self):
        """Test replacing a single table marker with markdown table."""
        text = """
# Sales Report

Monthly data:

<<<TABLE_START>>>Month     Units    Revenue

January   150      $15,000
February  200      $20,000<<<TABLE_END>>>

Summary follows.
"""
        markdown_table = """| Month    | Units | Revenue |
|----------|-------|---------|
| January  | 150   | $15,000 |
| February | 200   | $20,000 |"""

        tables = [TableInfo(content=markdown_table)]

        result = KnowledgeProcessingUtils.replace_table_markers_with_markdown(text, tables)

        assert "| Month    | Units | Revenue |" in result
        assert "|----------|-------|---------|" in result
        assert "| January  | 150   | $15,000 |" in result
        assert "<<<TABLE_START>>>" not in result
        assert "<<<TABLE_END>>>" not in result

    def test_replace_multiple_table_markers(self):
        """Test replacing multiple table markers with markdown tables."""
        text = """
# Report

## Revenue
<<<TABLE_START>>>Q1 $1.2M
Q2 $1.5M<<<TABLE_END>>>

## Expenses
<<<TABLE_START>>>Salaries $800K
Marketing $200K<<<TABLE_END>>>

Done.
"""
        revenue_table = """| Quarter | Revenue |
|---------|---------|
| Q1      | $1.2M   |
| Q2      | $1.5M   |"""

        expenses_table = """| Category  | Amount |
|-----------|--------|
| Salaries  | $800K  |
| Marketing | $200K  |"""

        tables = [TableInfo(content=revenue_table), TableInfo(content=expenses_table)]

        result = KnowledgeProcessingUtils.replace_table_markers_with_markdown(text, tables)

        assert "| Quarter | Revenue |" in result
        assert "| Category  | Amount |" in result
        assert "<<<TABLE_START>>>" not in result
        assert "<<<TABLE_END>>>" not in result

    def test_table_markers_without_tables(self):
        """Test handling when no tables are provided for markers."""
        text = """<<<TABLE_START>>>Some table data<<<TABLE_END>>>"""

        result = KnowledgeProcessingUtils.replace_table_markers_with_markdown(text, None)

        # Markers and content should be removed
        assert "<<<TABLE_START>>>" not in result
        assert "<<<TABLE_END>>>" not in result
        assert "Some table data" not in result

    def test_more_markers_than_tables(self):
        """Test handling when there are more markers than tables provided."""
        text = """
<<<TABLE_START>>>Table 1<<<TABLE_END>>>
<<<TABLE_START>>>Table 2<<<TABLE_END>>>
<<<TABLE_START>>>Table 3<<<TABLE_END>>>
"""
        tables = [
            TableInfo(content="| Col1 | Col2 |\n|------|------|"),
            TableInfo(content="| A | B |\n|---|---|"),
        ]

        result = KnowledgeProcessingUtils.replace_table_markers_with_markdown(text, tables)

        # First two should be replaced
        assert "| Col1 | Col2 |" in result
        assert "| A | B |" in result

        # Third marker should be removed
        assert "Table 3" not in result
        assert "<<<TABLE_START>>>" not in result


class TestCombinedMarkersReplacement:
    """Test cases for combined image and table marker replacement."""

    def test_replace_both_image_and_table_markers(self):
        """Test replacing both image and table markers in the same text."""
        text = """
# Annual Report 2024

<<<IMAGE_CAPTION_START>>>CEO presenting results<<<IMAGE_CAPTION_END>>>

## Revenue Analysis

<<<TABLE_START>>>Quarter  Revenue    YoY Growth

Q1       $12.5M     18%
Q2       $14.2M     22%<<<TABLE_END>>>

<<<IMAGE_CAPTION_START>>>Revenue growth chart<<<IMAGE_CAPTION_END>>>

## Summary

Strong performance overall.
"""
        images = [
            ImageInfo(
                caption="CEO presenting results",
                href="/img/ceo.jpg",
                page=1,
                document_name="report.pdf",
            ),
            ImageInfo(
                caption="Revenue growth chart",
                href="/img/chart.png",
                page=2,
                document_name="report.pdf",
            ),
        ]

        revenue_table = """| Quarter | Revenue | YoY Growth |
|---------|---------|------------|
| Q1      | $12.5M  | 18%        |
| Q2      | $14.2M  | 22%        |"""

        tables = [TableInfo(content=revenue_table)]

        # First replace images
        content_with_images = KnowledgeProcessingUtils.replace_image_markers_with_markdown(text, images)
        # Then replace tables
        final_result = KnowledgeProcessingUtils.replace_table_markers_with_markdown(content_with_images, tables)

        # Check images are replaced
        assert "![CEO presenting results](/img/ceo.jpg)" in final_result
        assert "![Revenue growth chart](/img/chart.png)" in final_result

        # Check table is replaced
        assert "| Quarter | Revenue | YoY Growth |" in final_result

        # Check no markers remain
        assert "<<<IMAGE_CAPTION_START>>>" not in final_result
        assert "<<<TABLE_START>>>" not in final_result

    def test_clean_all_markers(self):
        """Test cleaning all markers from text."""
        text = """
Content here.

<<<IMAGE_CAPTION_START>>>An image<<<IMAGE_CAPTION_END>>>

<<<TABLE_START>>>Some table data<<<TABLE_END>>>

More content.
"""
        cleaned = KnowledgeProcessingUtils.clean_all_markers_from_text(text)

        assert "<<<IMAGE_CAPTION_START>>>" not in cleaned
        assert "<<<TABLE_START>>>" not in cleaned
        assert "An image" not in cleaned
        assert "Some table data" not in cleaned
        assert "Content here." in cleaned
        assert "More content." in cleaned


class TestEnrichPageText:
    """Test cases for enriching page text with image captions."""

    def test_enrich_page_with_image_captions(self):
        """Test enriching page text with searchable image captions."""
        page_text = "This is the main page content."
        captions = ["Chart showing revenue growth", "Diagram of system architecture"]

        enriched = KnowledgeProcessingUtils.enrich_page_text_with_images(page_text, captions)

        assert "This is the main page content." in enriched
        assert "<<<IMAGE_CAPTION_START>>>Chart showing revenue growth<<<IMAGE_CAPTION_END>>>" in enriched
        assert "<<<IMAGE_CAPTION_START>>>Diagram of system architecture<<<IMAGE_CAPTION_END>>>" in enriched

    def test_enrich_with_empty_captions(self):
        """Test enriching when some captions are empty."""
        page_text = "Main content"
        captions = ["Valid caption", "", "Another caption", None]

        enriched = KnowledgeProcessingUtils.enrich_page_text_with_images(page_text, captions)

        assert "<<<IMAGE_CAPTION_START>>>Valid caption<<<IMAGE_CAPTION_END>>>" in enriched
        assert "<<<IMAGE_CAPTION_START>>>Another caption<<<IMAGE_CAPTION_END>>>" in enriched
        # Empty captions should not be included
        assert enriched.count("<<<IMAGE_CAPTION_START>>>") == 2
