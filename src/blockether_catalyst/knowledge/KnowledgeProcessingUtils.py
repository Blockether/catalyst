"""
Centralized utilities for knowledge processing to avoid code duplication and circular imports.

This module provides common text processing functions used across the knowledge extraction
and search components.
"""

import re
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from blockether_catalyst.knowledge.KnowledgeTypes import ImageInfo, TableInfo


class KnowledgeProcessingUtils:
    """Utility functions for knowledge processing and text normalization."""

    # Emoji pattern for removing emojis from text
    EMOJI_PATTERN = re.compile(
        "["
        "\U0001f600-\U0001f64f"  # emoticons
        "\U0001f300-\U0001f5ff"  # symbols & pictographs
        "\U0001f680-\U0001f6ff"  # transport & map symbols
        "\U0001f1e0-\U0001f1ff"  # flags (iOS)
        "\U0001f900-\U0001f9ff"  # supplemental symbols and pictographs
        "\U00002702-\U000027b0"  # dingbats
        "\U0000fe0f"  # variation selector-16 (emoji presentation)
        "\U0000fe0e"  # variation selector-15 (text presentation)
        "\U0000200d"  # zero-width joiner
        "\U000024c2-\U0001f251"  # enclosed characters
        "]+",
        flags=re.UNICODE,
    )

    # Image caption markers for searchable but hideable content
    IMAGE_CAPTION_START = "<<<IMAGE_CAPTION_START>>>"
    IMAGE_CAPTION_END = "<<<IMAGE_CAPTION_END>>>"
    IMAGE_CAPTION_PATTERN = re.compile(rf"{re.escape(IMAGE_CAPTION_START)}.*?{re.escape(IMAGE_CAPTION_END)}", re.DOTALL)

    # Table markers for searchable but replaceable content
    TABLE_START = "<<<TABLE_START>>>"
    TABLE_END = "<<<TABLE_END>>>"
    TABLE_PATTERN = re.compile(rf"{re.escape(TABLE_START)}.*?{re.escape(TABLE_END)}", re.DOTALL)

    @staticmethod
    def normalize_term(term: str) -> str:
        """
        Normalize a term by lowercasing and removing unwanted characters.

        This function:
        - Converts to lowercase
        - Removes emojis
        - Removes parenthetical content
        - Removes list markers
        - Normalizes whitespace
        - Strips special characters

        Args:
            term: The term to normalize

        Returns:
            Normalized term string
        """
        # Convert to lowercase
        normalized = term.lower()

        # Remove emojis using pattern
        normalized = KnowledgeProcessingUtils.EMOJI_PATTERN.sub("", normalized)

        # Remove parenthetical content (e.g., "ROI (Return on Investment)" -> "ROI")
        normalized = re.sub(r"\s*\([^)]*\)", "", normalized)

        # Remove bullets and list markers - keep applying until no more markers found
        previous = ""
        while previous != normalized:
            previous = normalized
            normalized = re.sub(r"^[\s]*[-•·*▪▸◦‣⁃]\s*", "", normalized)  # Unordered list markers
            normalized = re.sub(r"^[\s]*\d+[.)]\s*", "", normalized)  # Ordered list markers (1. or 1))
            normalized = re.sub(r"^[\s]*[a-z][.)]\s*", "", normalized)  # Lettered lists (a. or a))
            normalized = re.sub(r"^[\s]*[ivxlcdm]+[.)]\s*", "", normalized)  # Roman numerals

        # Remove multiple spaces and newlines
        normalized = re.sub(r"\s+", " ", normalized)
        normalized = normalized.rstrip("/")

        # Remove trailing and leading hyphens
        normalized = normalized.strip("-")

        # Strip leading/trailing whitespace
        normalized = normalized.strip()

        return normalized

    @staticmethod
    def clean_image_markers_from_text(text: str) -> str:
        """
        Remove image caption markers from text for display purposes.

        Args:
            text: Text potentially containing image caption markers

        Returns:
            Clean text without image caption markers
        """
        if KnowledgeProcessingUtils.IMAGE_CAPTION_START not in text:
            return text

        # Remove all image caption markers and their content
        cleaned = KnowledgeProcessingUtils.IMAGE_CAPTION_PATTERN.sub("", text)

        # Clean up any extra newlines that might be left
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

        return cleaned.strip()

    @staticmethod
    def clean_all_markers_from_text(text: str) -> str:
        """
        Remove both image and table markers from text for display purposes.

        Args:
            text: Text potentially containing markers

        Returns:
            Clean text without any markers
        """
        # Remove image caption markers
        if KnowledgeProcessingUtils.IMAGE_CAPTION_START in text:
            text = KnowledgeProcessingUtils.IMAGE_CAPTION_PATTERN.sub("", text)

        # Remove table markers
        if KnowledgeProcessingUtils.TABLE_START in text:
            text = KnowledgeProcessingUtils.TABLE_PATTERN.sub("", text)

        # Clean up any extra newlines that might be left
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    @staticmethod
    def enrich_page_text_with_images(page_text: str, image_captions: list[str]) -> str:
        """
        Enrich page text with image captions to make them searchable.

        Uses special marker format that can be easily filtered from display
        while keeping content searchable.

        Args:
            page_text: Original page text
            image_captions: List of image captions

        Returns:
            Enriched text including image captions with special markers
        """
        enriched_text = page_text

        # Filter out empty captions (but they should always be provided)
        valid_captions = [caption for caption in image_captions if caption]

        if valid_captions:
            # Add image captions with markers
            image_descriptions = [
                f"{KnowledgeProcessingUtils.IMAGE_CAPTION_START}{caption}{KnowledgeProcessingUtils.IMAGE_CAPTION_END}"
                for caption in valid_captions
            ]

            # Append all image descriptions at the end of the page text
            enriched_text = f"{page_text}\n\n" + "\n".join(image_descriptions)

        return enriched_text

    @staticmethod
    def remove_emojis(text: str) -> str:
        """
        Remove emojis from text.

        Args:
            text: Text potentially containing emojis

        Returns:
            Text with emojis removed
        """
        return KnowledgeProcessingUtils.EMOJI_PATTERN.sub("", text)

    @staticmethod
    def replace_image_markers_with_markdown(text: str, images: Optional[List["ImageInfo"]] = None) -> str:
        """
        Replace image caption markers with markdown image syntax.

        This method converts the special image caption markers to proper markdown
        images with links, making them displayable in the UI.

        Args:
            text: Text potentially containing image caption markers
            images: List of ImageInfo objects with caption and href attributes

        Returns:
            Text with image markers replaced by markdown images
        """
        if KnowledgeProcessingUtils.IMAGE_CAPTION_START not in text:
            return text

        if not images:
            # If no images provided, just remove the markers
            return KnowledgeProcessingUtils.clean_image_markers_from_text(text)

        # Create a mapping from captions to image URLs
        # Images are ImageInfo objects with caption and href as required fields
        caption_to_url = {img.caption: img.href for img in images}

        # Find all image caption markers and replace them with markdown
        def replace_marker(match: re.Match[str]) -> str:
            full_match = match.group(0)
            # Extract caption between the markers
            caption = full_match.replace(KnowledgeProcessingUtils.IMAGE_CAPTION_START, "")
            caption = caption.replace(KnowledgeProcessingUtils.IMAGE_CAPTION_END, "")

            # Find the corresponding URL
            if caption in caption_to_url and caption_to_url[caption]:
                # Return markdown image syntax
                return f"\n\n![{caption}]({caption_to_url[caption]})\n*{caption}*\n"
            else:
                # If no URL found, just return the caption as italic text
                return f"\n\n*{caption}*\n"

        # Replace all image markers with markdown
        result = KnowledgeProcessingUtils.IMAGE_CAPTION_PATTERN.sub(replace_marker, text)

        # Clean up any extra newlines
        result = re.sub(r"\n{3,}", "\n\n", result)

        return result.strip()

    @staticmethod
    def replace_table_markers_with_markdown(text: str, tables: Optional[List["TableInfo"]] = None) -> str:
        """
        Replace table markers with markdown table syntax.

        This method converts the special table markers to proper markdown
        tables, making them displayable in the UI.

        Args:
            text: Text potentially containing table markers
            tables: List of TableInfo objects with markdown content

        Returns:
            Text with table markers replaced by markdown tables
        """
        if KnowledgeProcessingUtils.TABLE_START not in text:
            return text

        if not tables:
            # If no tables provided, just remove the markers and content
            return KnowledgeProcessingUtils.TABLE_PATTERN.sub("", text).strip()

        # Tables are TableInfo objects with content field containing markdown
        markdown_tables = [table.content for table in tables if table.content]

        if not markdown_tables:
            # No valid markdown tables, just remove markers
            return KnowledgeProcessingUtils.TABLE_PATTERN.sub("", text).strip()

        # Replace each table marker with corresponding markdown table
        table_index = 0

        def replace_table_marker(match: re.Match[str]) -> str:
            nonlocal table_index
            if table_index < len(markdown_tables):
                markdown = markdown_tables[table_index]
                table_index += 1
                # Return markdown table with some spacing
                return f"\n\n{markdown}\n\n"
            else:
                # No more tables available, remove the marker
                return ""

        # Replace all table markers with markdown
        result = KnowledgeProcessingUtils.TABLE_PATTERN.sub(replace_table_marker, text)

        # Clean up any extra newlines
        result = re.sub(r"\n{3,}", "\n\n", result)

        return result.strip()
