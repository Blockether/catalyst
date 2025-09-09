"""
PDF Processing Algorithm using pdfplumber for sophisticated table extraction.
"""

import base64
import hashlib
import io
import logging
import re
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pdfplumber
import torch
from pdfplumber import page
from pdfplumber.display import PageImage
from PIL import Image

from .ImageRecognition import ImageRecognition

from .KnowledgeExtractionTypes import (
    ImageMetadata,
    KnowledgeExtractionResult,
    KnowledgeMetadata,
    KnowledgePageData,
    KnowledgePageDataWithRawText,
    KnowledgeProcessorSettings,
    KnowledgeTableData,
)
from .PDKnowledgeExtractorTypes import (
    PDFImageProcessingSettings,
    PDFKnowledgeProcessorSettings,
    PDFPageCropOffset,
    PDFProcessorTableExtractionSettings,
    PDFProcessorTextExtractionSettings,
)


class PDFTableData(KnowledgeTableData):
    """Extended table data with bounding box for PDF processing."""

    bbox: Tuple[float, float, float, float]


class PDFKnowledgeExtractor:
    """Advanced PDF processor using pdfplumber for sophisticated extraction."""

    def __init__(
        self,
        knowledge_settings: KnowledgeProcessorSettings
    ) -> None:
        """
        Initialize PDF processor with optional configuration.

        Args:
            settings: Unified PDF processor settings
        """
        self._image_recognition = ImageRecognition()
        self._logger = logging.getLogger(__name__)
        self._knowledge_settings = knowledge_settings
        self._settings = knowledge_settings.pdf_settings
        self._table_settings = self._settings.pdf_table_extraction or PDFProcessorTableExtractionSettings()
        self._text_extraction_settings = self._settings.pdf_text_extraction or PDFProcessorTextExtractionSettings()
        self._pdf_image_processing_settings = self._settings.pdf_image_processing or PDFImageProcessingSettings()
        self._current_document: Optional[str] = None
        self._current_document_path: Optional[Path] = None

    @property
    def settings(self) -> PDFKnowledgeProcessorSettings:
        """Get the PDF processor settings."""
        return self._settings

    @property
    def knowledge_settings(self) -> KnowledgeProcessorSettings:
        """Get the knowledge processor settings."""
        return self._knowledge_settings

    def extract(self, source: Path) -> KnowledgeExtractionResult:
        """Synchronous PDF extraction with pdfplumber."""

        if not source.exists():
            raise FileNotFoundError(f"PDF file not found: {source}")

        id = self._calculate_id(source)
        result = KnowledgeExtractionResult(filename=source.name, id=id, source_type="pdf")

        # Store document name and path for logging and image naming
        self._current_document = source.name
        self._current_document_path = source

        with pdfplumber.open(source) as pdf:
            # Extract metadata
            if pdf.metadata:
                result.metadata = KnowledgeMetadata(
                    title=pdf.metadata.get("Title", None),
                    author=pdf.metadata.get("Author", None),
                    subject=pdf.metadata.get("Subject", None),
                    creation_date=str(pdf.metadata.get("CreationDate", "")),
                    modification_date=str(pdf.metadata.get("ModDate", "")),
                )

            result.total_pages = len(pdf.pages)

            # Process each page (pdfplumber pages are 1-indexed)
            all_raw_text = []
            for page in pdf.pages:  # page.page_number is 1-indexed
                page_image = None

                page = self._crop_page(page, page_image)
                page_data = self._process_page(page)

                # Add processed page to results
                result.pages.append(page_data)
                result.total_tables += len(page_data.tables)
                result.total_images += len(page_data.images)

                # Collect raw text for combined result
                # Use raw_text if available, otherwise fall back to text field
                page_text = page_data.raw_text
                if page_text:
                    all_raw_text.append(page_text)

            # Combine all raw text
            result.raw = "\n".join(all_raw_text)

        return result

    def _process_page(
        self,
        page: page.Page,
    ) -> KnowledgePageDataWithRawText:
        """Process a single PDF page."""
        # Extract tables
        tables = self._extract_tables_from_page(page)

        # Get table bounding boxes
        table_bboxes = [table.bbox for table in tables]
        bbox_not_within_bboxes = partial(self._not_within_bboxes, bboxes=table_bboxes)
        page_without_tables = page.filter(bbox_not_within_bboxes)

        # Extract base text (without tables)
        base_text = page_without_tables.extract_text(**self._text_extraction_settings.model_dump()) or ""

        # Extract image
        images = self._extract_images_from_page(page, base_text)

        # Fix hyphenated line breaks immediately after extraction
        base_text = self._fix_hyphenated_line_breaks(base_text)

        # Build raw text
        raw_text = self._build_raw_text(base_text, tables)

        # Calculate text statistics
        lines = base_text.split("\n")

        # Convert PDFTableData to KnowledgeTableData for the result
        knowledge_tables = [
            KnowledgeTableData(
                page=t.page,
                data=t.data,
                rows=t.rows,
                columns=t.columns,
            )
            for t in tables
        ]

        return KnowledgePageDataWithRawText(
            page=page.page_number,
            text=base_text,
            raw_text=raw_text,
            tables=knowledge_tables,
            images=images,
            lines=len(lines),
        )

    def _crop_page(self, page: page.Page, page_image: PageImage | None = None) -> page.Page:
        """
        Crop the page based on the provided offsets.

        Args:
            page: The pdfplumber page object to crop.

        Returns:
            Cropped page object.
        """
        x0, y0, x1, y1 = page.bbox  # (left, top, right, bottom)
        crop_offset = self._settings.pdf_page_crop_offset

        if not crop_offset:
            return page

        # Only crop if offsets are non-zero and would not create invalid bounding box
        if crop_offset.top > 0 and y0 + crop_offset.top < y1:
            header_crop = (x0, y0, x1, y0 + crop_offset.top)
            if page_image:
                page_image.draw_rects([header_crop], stroke="red")
            page = page.outside_bbox(header_crop)

        if crop_offset.bottom > 0 and y1 - crop_offset.bottom > y0:
            footer_crop = (x0, y1 - crop_offset.bottom, x1, y1)
            if page_image:
                page_image.draw_rects([footer_crop], stroke="red")
            page = page.outside_bbox(footer_crop)

        return page

    def _filter_invisible_lines(self, obj: Dict[str, Any]) -> bool:
        """
        If the object is a ``rect`` type, keep it only if the lines are visible.

        A visible line is the one having ``non_stroking_color`` as 0.
        """
        if obj["object_type"] == "rect":
            return bool(obj["non_stroking_color"] == 0)
        return True

    def _not_within_bboxes(self, obj: Dict[str, Any], bboxes: List[Tuple[float, float, float, float]]) -> bool:
        """Check if the object is in any of the table's bbox."""

        def obj_in_bbox(_bbox: Tuple[float, float, float, float]) -> bool:
            """Define objects in box.

            See https://github.com/jsvine/pdfplumber/blob/stable/pdfplumber/table.py#L404
            """
            v_mid = (obj["top"] + obj["bottom"]) / 2
            h_mid = (obj["x0"] + obj["x1"]) / 2
            x0, top, x1, bottom = _bbox
            return bool((h_mid >= x0) and (h_mid < x1) and (v_mid >= top) and (v_mid < bottom))

        return bool(not any(obj_in_bbox(__bbox) for __bbox in bboxes))

    def _extract_tables_from_page(
        self,
        page: page.Page,
    ) -> List[PDFTableData]:
        """Extract all tables from a page using pdfplumber."""
        tables = []
        page = page.filter(self._filter_invisible_lines)

        # Use pdfplumber's table finder
        found_tables = page.find_tables(table_settings=self._table_settings.model_dump())

        for table in found_tables:
            try:
                # Extract table data
                table_data = table.extract()

                if table_data and len(table_data) > 1:
                    # Use PDFTableData which extends KnowledgeTableData with bbox
                    pdf_table = PDFTableData(
                        page=page.page_number,
                        data=table_data,
                        rows=len(table_data),
                        columns=len(table_data[0]) if table_data else 0,
                        bbox=table.bbox,
                    )

                    tables.append(pdf_table)

            except Exception as e:
                self._logger.warning(f"[{self._current_document}] Error extracting table on page {page.page_number}: {e}")

        return tables

    def _extract_images_from_page(
        self,
        page: page.Page,
        base_text: str,
    ) -> List[ImageMetadata]:
        """Extract non-decorative images from page and save to files

        Returns:
            List of image metadata
        """
        images: List[ImageMetadata] = []

        # Extract images using pdfplumber's image extraction

        for idx, img_obj in enumerate(page.images):
            try:
                # Skip images smaller than 64px in height (likely decorative elements)
                img_width = img_obj.get("width", 0)
                img_height = img_obj.get("height", 0)
                if img_height < 64:
                    self._logger.info(
                        f"[{self._current_document}] Skipped small image on page {page.page_number}: "
                        f"{img_width}x{img_height} pixels (height < 64px)"
                    )
                    continue

                # Use pdfplumber's within_bbox to crop and extract the image area
                img_bbox = (
                    img_obj["x0"],
                    img_obj["top"],
                    img_obj["x1"],
                    img_obj["bottom"],
                )
                cropped = page.crop(img_bbox)

                # Convert to image
                if cropped:
                    page_image = cropped.to_image(resolution=300, antialias=True)

                    # Create filename based on PDF name and page number
                    pdf_stem = self._current_document_path.stem if self._current_document_path else "document"
                    image_filename = f"{pdf_stem}_page_{page.page_number}_img_{idx + 1}.png"
                    output_dir = Path(self._knowledge_settings.extraction_output_dir) / "images"
                    output_dir.mkdir(parents=True, exist_ok=True)
                    image_path = output_dir / image_filename
                    image = page_image.original

                    metadata = ImageMetadata(
                        document_name=self._current_document or "",
                        page=page.page_number,
                        href=str(image_path),
                        caption=self._image_recognition.caption_for_image(image, context=base_text)
                    )

                    # Save the image
                    page_image.save(str(image_path), format="PNG")

                    # Store the filename (not full path) in the results
                    images.append(metadata)

                    self._logger.info(
                        f"[{self._current_document}] Saved image from page {page.page_number}: {image_filename}"
                    )

            except Exception as e:
                self._logger.warning(
                    f"[{self._current_document}] Error processing image {idx} on page {page.page_number}: {e}"
                )

        return images

    def _calculate_id(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def _fix_hyphenated_line_breaks(self, text: str) -> str:
        """Fix words that are hyphenated at line breaks.

        Args:
            text: Text that may contain hyphenated line breaks

        Returns:
            Text with hyphenated line breaks fixed
        """
        # Pattern to match word characters followed by hyphen and newline, then more word characters
        # This handles cases like "credit policy devia-\ntion" -> "credit policy deviation"
        fixed_text = re.sub(r"([a-zA-Z]+)-\n([a-zA-Z]+)", r"\1\2", text)
        return fixed_text

    def _build_raw_text(self, base_text: str, tables: List[PDFTableData]) -> str:
        """Build raw text with tables inline

        Args:
            page: PDF page
            tables: Extracted tables
        Returns:
            Raw text with everything inline
        """
        # Text already has hyphenated line breaks fixed
        raw_text = base_text

        # Append tables as HTML for full nesting support
        for table in tables:
            raw_text += "\n" + table.to_ascii_table() + "\n"

        return raw_text
