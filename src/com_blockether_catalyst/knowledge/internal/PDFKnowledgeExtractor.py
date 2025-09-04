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

import easyocr
import numpy as np
import pdfplumber
import torch
from pdfplumber import page
from pdfplumber.display import PageImage
from PIL import Image

from .KnowledgeExtractionBaseTypes import KnowledgeTableData
from .KnowledgeExtractionTypes import (
    KnowledgeExtractionResult,
    KnowledgeMetadata,
    KnowledgePageData,
    KnowledgePageDataWithRawText,
    KnowledgeProcessorSettings,
)
from .PDKnowledgeExtractorTypes import (
    PDFImageProcessingSettings,
    PDFPageCropOffset,
    PDFProcessorTableExtractionSettings,
    PDFProcessorTextExtractionSettings,
)


class PDFTableData(KnowledgeTableData):
    """Extended table data with bounding box for PDF processing."""

    bbox: Tuple[float, float, float, float]


class PDFKnowledgeExtractor:
    """Advanced PDF processor using pdfplumber for sophisticated extraction."""

    def __init__(self, settings: KnowledgeProcessorSettings):
        """
        Initialize PDF processor with optional configuration.

        Args:
            settings: Unified PDF processor settings
        """
        self.logger = logging.getLogger(__name__)
        self.settings = settings
        self.table_settings = settings.pdf_table_extraction or PDFProcessorTableExtractionSettings()
        self.text_extraction_settings = settings.pdf_text_extraction or PDFProcessorTextExtractionSettings()
        self.pdf_image_processing = settings.pdf_image_processing or PDFImageProcessingSettings()
        self.ocr_reader: Optional[Any] = None

        # Initialize OCR if needed
        self._initialize_ocr()

    def extract(self, source: Path) -> KnowledgeExtractionResult:
        """Synchronous PDF extraction with pdfplumber."""

        if not source.exists():
            raise FileNotFoundError(f"PDF file not found: {source}")

        id = self._calculate_id(source)
        result = KnowledgeExtractionResult(filename=source.name, id=id, source_type="pdf")

        # Store document name for logging
        self.current_document = source.name

        last_toc_page_idx: Optional[int] = None
        toc_page_last_y0: Optional[float] = None

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
                page, last_toc_page_idx, toc_page_last_y0 = self.filter_out_toc(
                    page,
                    page.page_number - 1,
                    page_image,
                    last_toc_page_idx,
                    toc_page_last_y0,
                )

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

        # Extract image
        images = self._extract_images_from_page(page)

        # Get table bounding boxes
        table_bboxes = [table.bbox for table in tables]
        bbox_not_within_bboxes = partial(self._not_within_bboxes, bboxes=table_bboxes)
        page_without_tables = page.filter(bbox_not_within_bboxes)

        # Extract base text (without tables)
        base_text = page_without_tables.extract_text(**self.text_extraction_settings.model_dump()) or ""

        # Fix hyphenated line breaks immediately after extraction
        base_text = self._fix_hyphenated_line_breaks(base_text)

        # Clean TOC artifacts from the text
        base_text = self._clean_toc_artifacts(base_text)

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
        crop_offset = self.settings.pdf_page_crop_offset

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

    def filter_out_toc(
        self,
        page: page.Page,
        page_idx: int,
        page_image: PageImage | None = None,
        last_toc_page_idx: Optional[int] = None,
        toc_page_last_y0: Optional[float] = None,
        toc_page_break_pages_gap_threshold: int = 40,
    ) -> Tuple[page.Page, Optional[int], Optional[float]]:
        toc_header_regex = re.compile(r"table\s+of\s+contents?", re.IGNORECASE)
        crop_offset = self.settings.pdf_page_crop_offset or PDFPageCropOffset(top=0, bottom=0)

        # Patterns for TOC entries:
        # r'([\d.]+)\s+(.*?)\s*[\s.]{2,}\s*(\d+)' => "1.1 Introduction ........ 6" or "1.1 Introduction      6"
        # r'^\s*[o▪•]\s*.*' => "• Intro; o Table of Content"
        toc_page_regex = re.compile(r"([\d.]+)\s+(.*?)\s*[\s.]{2,}\s*(\d+)|^\s*[o▪•]\s*.*", re.MULTILINE)

        toc_header = page.search(toc_header_regex)

        x0, y0, x1, y1 = page.bbox  # (left, top, right, bottom)

        if toc_header:
            last_toc_page_idx = page_idx

            toc_header_top = toc_header[0]["top"]

            toc_page = page.search(toc_page_regex)

            if toc_page:
                toc_page_last = toc_page[-1]
                toc_page_last_bottom = toc_page_last["bottom"]
                toc_page_last_y0 = toc_page_last["chars"][0]["y0"] - crop_offset.bottom

                if page_image:
                    page_image.draw_rect(
                        (
                            page.bbox[0],
                            toc_header_top,
                            page.bbox[2],
                            toc_page_last_bottom,
                        ),
                        stroke="blue",
                    )

                return (
                    page.outside_bbox((x0, toc_header_top, x1, toc_page_last_bottom)),
                    last_toc_page_idx,
                    toc_page_last_y0,
                )

        if last_toc_page_idx and page_idx == last_toc_page_idx + 1:
            toc_page = page.search(toc_page_regex)

            if toc_page:
                last_toc_page_idx = page_idx

                toc_page_first = toc_page[0]
                toc_page_last = toc_page[-1]

                toc_page_first_element_top = toc_page_first["top"] - crop_offset.top

                if (
                    toc_page_last_y0 is not None
                    and toc_page_first_element_top + toc_page_last_y0 < toc_page_break_pages_gap_threshold
                ):
                    last_bottom = toc_page_last["bottom"]
                    toc_page_last_y0 = toc_page_last["chars"][0]["y0"] - crop_offset.bottom

                    if page_image:
                        page_image.draw_rect(
                            (
                                page.bbox[0],
                                toc_page_first["top"],
                                page.bbox[2],
                                last_bottom,
                            ),
                            stroke="blue",
                        )

                    return (
                        page.outside_bbox((x0, toc_page_first["top"], x1, last_bottom)),
                        last_toc_page_idx,
                        toc_page_last_y0,
                    )

        return page, last_toc_page_idx, toc_page_last_y0

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
        found_tables = page.find_tables(table_settings=self.table_settings.model_dump())

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
                self.logger.warning(f"[{self.current_document}] Error extracting table on page {page.page_number}: {e}")

        return tables

    def _extract_images_from_page(
        self,
        page: page.Page,
    ) -> List[str]:
        """Extract non-decorative images from page with base64 encoding

        Returns:
            List of base64 encoded images
        """
        images = []

        # Extract images using pdfplumber's image extraction

        for idx, img_obj in enumerate(page.images):
            try:
                # Skip images smaller than 64px in height (likely decorative elements)
                img_width = img_obj.get("width", 0)
                img_height = img_obj.get("height", 0)
                if img_height < 64:
                    self.logger.info(
                        f"[{self.current_document}] Skipped small image on page {page.page_number}: "
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
                cropped = page.within_bbox(img_bbox)

                # Convert to image
                if cropped:
                    page_image = cropped.to_image(resolution=150)

                    # Convert to base64 for non-decorative images
                    img_buffer = io.BytesIO()
                    page_image.save(img_buffer, format="PNG")
                    img_buffer.seek(0)
                    base64_data = base64.b64encode(img_buffer.getvalue()).decode("utf-8")
                    images.append(base64_data)

            except Exception as e:
                self.logger.warning(
                    f"[{self.current_document}] Error processing image {idx} on page {page.page_number}: {e}"
                )

        return images

    def _calculate_id(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def _initialize_ocr(self) -> None:
        """Initialize OCR reader."""

        try:
            # Initialize EasyOCR with English language
            self.ocr_reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available(), verbose=False)
            self.logger.info(f"OCR reader initialized (GPU: {torch.cuda.is_available()})")
        except Exception as e:
            self.logger.error(f"Failed to initialize OCR: {e}")
            self.ocr_reader = None

    def _extract_text_with_ocr(self, pil_image: Image.Image) -> str:
        """Extract text from image using OCR.

        Args:
            pil_image: PIL Image to extract text from

        Returns:
            Extracted text or empty string if OCR fails
        """
        if self.ocr_reader is None:
            return ""

        try:
            # Convert PIL image to numpy array
            img_array = np.array(pil_image)

            # Perform OCR
            results = self.ocr_reader.readtext(img_array)

            # Combine all text results
            if results:
                texts = [result[1] for result in results]
                return " ".join(texts)

            return ""

        except Exception as e:
            self.logger.info(f"[{getattr(self, 'current_document', 'Unknown')}] OCR extraction failed: {e}")
            return ""

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

    def _clean_toc_artifacts(self, text: str) -> str:
        """Remove table of contents artifacts and page numbering from text.

        Removes:
        - Lines with excessive dots (.........) used as leaders in TOCs
        - Page number references at the end of TOC entries
        - Broken TOC entries that span multiple lines
        - Page numbering patterns like "Page N of X"

        Args:
            text: Text to clean

        Returns:
            Cleaned text without TOC artifacts and page numbers
        """
        lines = text.split("\n")
        cleaned_lines = []

        for line in lines:
            # Skip lines that are likely TOC entries:
            # 1. Lines with 5+ consecutive dots (TOC leaders)
            if re.search(r"\.{5,}", line):
                continue

            # 2. Lines that look like TOC entries with section numbers and page numbers
            # Pattern: "6.4.2 Some Title .... 17" or "6.5 Title    17"
            if re.match(r"^\s*\d+(\.\d+)*\s+.*?\s+\d+\s*$", line):
                # Check if it has typical TOC patterns
                if re.search(r"[\s.]{3,}\d+\s*$", line):  # Ends with spaces/dots and page number
                    continue

            # 3. Lines that are just dots and page numbers (broken TOC entries)
            # Pattern: ".......................... 17"
            if re.match(r"^\s*\.+\s*\d+\s*$", line):
                continue

            # 4. Lines with section numbers followed by dots
            # Pattern: "6.4.2 Top-Down Limit setting ...."
            if re.match(r"^\s*\d+(\.\d+)*\s+.*?\.{3,}", line):
                continue

            # 5. Skip page numbering patterns (case insensitive)
            # Patterns: "Page N of X", "Page N/X", "N of X", "Page N", "- N -", etc.
            if re.match(r"^\s*page\s+\d+\s*(of|/)\s*\d+\s*$", line, re.IGNORECASE):
                continue
            if re.match(r"^\s*\d+\s*(of|/)\s*\d+\s*$", line):
                continue
            if re.match(r"^\s*page\s+\d+\s*$", line, re.IGNORECASE):
                continue
            if re.match(r"^\s*-\s*\d+\s*-\s*$", line):  # Pattern: "- 17 -"
                continue
            if re.match(r"^\s*\[\s*\d+\s*\]\s*$", line):  # Pattern: "[17]"
                continue

            # Keep the line if it doesn't match TOC or page number patterns
            cleaned_lines.append(line)

        return "\n".join(cleaned_lines)

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
