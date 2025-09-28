"""
PDF Processing Algorithm using pdfplumber for sophisticated table extraction.
"""

import hashlib
import logging
import re
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from .KnowledgeExtractionCallBase import BaseTableCaptionExtractionCall

import pdfplumber
from pdfplumber import page
from pdfplumber.display import PageImage
import time

from blockether_catalyst.knowledge.KnowledgeTypes import (
    DocumentMetadata,
    ImageMetadata,
    KnowledgeExtractionResult,
    KnowledgePageData,
    KnowledgeProcessorSettings,
    KnowledgeTableData,
)

from .ExtractionTypes import (
    PDFKnowledgeProcessorSettings,
)
from .ImageExtractor import ImageRecognition, ImageRecognitionSettings


class PDFTableData(KnowledgeTableData):
    """Extended table data with bounding box for PDF processing."""

    bbox: Tuple[float, float, float, float]


class PDFKnowledgeExtractor:
    """Advanced PDF processor using pdfplumber for sophisticated extraction."""

    def __init__(
        self,
        image_output_dir: Path,
        knowledge_settings: KnowledgeProcessorSettings,
        table_caption_call: Optional["BaseTableCaptionExtractionCall"] = None,
    ) -> None:
        """
        Initialize PDF processor with optional configuration.

        Args:
            image_output_dir: Directory to save extracted images
            knowledge_settings: Unified PDF processor settings
            table_caption_call: Optional extraction call for generating table captions
        """
        self._image_recognition = ImageRecognition(ImageRecognitionSettings(max_tokens=256))
        self._logger = logging.getLogger(__name__)
        self._knowledge_settings = knowledge_settings
        self._settings = knowledge_settings.pdf_settings
        self._table_settings = self._settings.pdf_table_extraction
        self._text_extraction_settings = self._settings.pdf_text_extraction
        self._table_caption_call = table_caption_call

        self._image_output_dir = image_output_dir

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
        import os

        self._image_output_dir.mkdir(parents=True, exist_ok=True)

        if not source.exists():
            raise FileNotFoundError(f"PDF file not found: {source}")

        id = self._calculate_id(source)
        result = KnowledgeExtractionResult(
            document_filename=source.name,
            document_metadata=DocumentMetadata(),
            id=id,
            source_type="pdf",
        )

        # Pass document info as parameters for thread-safety
        current_document = source.name
        current_document_path = source

        with pdfplumber.open(source) as pdf:
            # Extract metadata
            if pdf.metadata:
                result.document_metadata = DocumentMetadata(
                    title=pdf.metadata.get("Title", None),
                    author=pdf.metadata.get("Author", None),
                    subject=pdf.metadata.get("Subject", None),
                    creation_date=str(pdf.metadata.get("CreationDate", "")),
                    modification_date=str(pdf.metadata.get("ModDate", "")),
                    document_path=f"{self._knowledge_settings.extraction_output_dir}/source_documents/{source.name}",
                )

            result.total_pages = len(pdf.pages)

            # Log extraction start with page count
            self._logger.info(f"[{current_document}] Starting extraction of {result.total_pages} pages")

            # Process each page (pdfplumber pages are 1-indexed)
            all_raw_text = []
            for page_idx, page in enumerate(pdf.pages):  # page.page_number is 1-indexed
                page_image = None

                # Log progress every 10 pages or for single-digit page counts
                if result.total_pages < 10 or (page_idx + 1) % 10 == 0 or page_idx == 0:
                    self._logger.info(f"[{current_document}] Processing page {page_idx + 1}/{result.total_pages}")

                page = self._crop_page(page, page_image)
                page_data = self._process_page(page, current_document, current_document_path)

                # Add processed page to results
                result.pages.append(page_data)
                result.total_tables += len(page_data.tables)
                result.total_images += len(page_data.images)

                # Collect text for combined result
                page_text = page_data.text
                if page_text:
                    all_raw_text.append(page_text)

            # Combine all raw text
            result.raw = "\n".join(all_raw_text)

            # Log extraction completion with summary
            self._logger.info(
                f"[{current_document}] Extraction completed - "
                f"Pages: {result.total_pages}, Tables: {result.total_tables}, Images: {result.total_images}"
            )

        return result

    def _process_page(
        self,
        page: page.Page,
        current_document: str,
        current_document_path: Path,
    ) -> KnowledgePageData:
        """Process a single PDF page."""
        # Extract tables
        tables = self._extract_tables_from_page(page, current_document)

        # Get table bounding boxes
        table_bboxes = [table.bbox for table in tables]
        bbox_not_within_bboxes = partial(self._not_within_bboxes, bboxes=table_bboxes)
        page_without_tables = page.filter(bbox_not_within_bboxes)

        # Extract base text (without tables)
        base_text = page_without_tables.extract_text(**self._text_extraction_settings.model_dump()) or ""

        # Extract image
        images = self._extract_images_from_page(
            page,
            context=base_text,
            current_document=current_document,
            current_document_path=current_document_path,
        )

        # Fix hyphenated line breaks immediately after extraction
        base_text = self._fix_hyphenated_line_breaks(base_text)

        # Build raw text
        raw_text = self._build_raw_text(base_text, tables)

        # Convert PDFTableData to KnowledgeTableData with intelligent captions
        knowledge_tables = []
        for t in tables:
            # Generate intelligent caption for the table
            # Use sync fallback if no async call is configured
            if self._table_caption_call:
                # For now, use a simple sync caption since PDFExtractor is not async
                # This will use a basic caption for now - proper async support would require
                # making the entire extraction pipeline async
                caption = f"Table - {current_document} (Page {page.page_number})"
            else:
                caption = f"Table - {current_document} (Page {page.page_number})"

            knowledge_tables.append(
                KnowledgeTableData(
                    page=t.page,
                    data=t.data,
                    rows=t.rows,
                    columns=t.columns,
                    caption=caption,
                )
            )

        # Combine base_text and raw_text into single text field for KnowledgePageData
        # The text field now contains the complete content with tables included
        return KnowledgePageData(
            page=page.page_number,
            text=raw_text,  # Use raw_text which includes tables
            tables=knowledge_tables,
            images=images,
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
        current_document: str,
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
                        caption="",  # Will be generated later
                        bbox=table.bbox,
                    )

                    tables.append(pdf_table)

            except Exception as e:
                self._logger.warning(f"[{current_document}] Error extracting table on page {page.page_number}: {e}")

        return tables

    def _extract_images_from_page(
        self,
        page: page.Page,
        context: str,
        current_document: str,
        current_document_path: Path,
    ) -> List[ImageMetadata]:
        """Extract non-decorative images from page and save to files

        Returns:
            List of image metadata
        """
        images: List[ImageMetadata] = []

        # Extract images using pdfplumber's image extraction
        # Use a separate counter for successfully processed images to ensure sequential numbering
        successful_image_count = 0

        for idx, img_obj in enumerate(page.images):
            try:
                # Skip images smaller than 64px in height (likely decorative elements)
                img_width = img_obj.get("width", 0)
                img_height = img_obj.get("height", 0)
                if img_height < 64:
                    self._logger.info(
                        f"[{current_document}] Skipped small image {idx + 1} on page {page.page_number}: {img_width}x{img_height} pixels (height < 64px)"
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
                    # Increment counter only for successfully processed images
                    successful_image_count += 1

                    self._logger.debug(
                        f"[{current_document}] Successfully cropped image {idx + 1} (output #{successful_image_count}) on page {page.page_number}: {img_width}x{img_height}px"
                    )
                    page_image = cropped.to_image(resolution=300, antialias=True)

                    # Create filename based on PDF name and page number using successful image count
                    pdf_stem = current_document_path.stem if current_document_path else "document"
                    image_filename = f"{pdf_stem}_page_{page.page_number}_img_{successful_image_count}.png"
                    image_path = self._image_output_dir / image_filename
                    image = page_image.original

                    if not current_document:
                        raise ValueError("Current document name is not set for image naming.")

                    # Try to generate caption with retries
                    caption = "Image without caption"
                    max_retries = 3
                    retry_delay = 2  # seconds

                    for attempt in range(max_retries):
                        try:
                            self._logger.info(
                                f"[{current_document}] Generating caption for image {successful_image_count} on page {page.page_number} (attempt {attempt + 1}/{max_retries})..."
                            )
                            caption = self._image_recognition.caption_for_image(image, context=context)
                            self._logger.info(
                                f"[{current_document}] Caption generated successfully for image {successful_image_count}"
                            )
                            break  # Success, exit retry loop
                        except Exception as caption_error:
                            if attempt < max_retries - 1:
                                self._logger.warning(
                                    f"[{current_document}] Caption generation attempt {attempt + 1} failed for image {successful_image_count} on page {page.page_number}: {caption_error}"
                                )
                                self._logger.info(f"[{current_document}] Retrying in {retry_delay} seconds...")

                                time.sleep(retry_delay)
                                retry_delay *= 2  # Exponential backoff
                            else:
                                self._logger.error(
                                    f"[{current_document}] All caption generation attempts failed for image {successful_image_count} on page {page.page_number}: {caption_error}"
                                )
                                self._logger.warning(f"[{current_document}] Using default caption for image {successful_image_count}")

                    metadata = ImageMetadata(
                        document_name=current_document,
                        page=page.page_number,
                        path=str(image_path),
                        caption=caption,
                    )

                    # Save the image
                    page_image.save(str(image_path), format="PNG")

                    # Store the filename (not full path) in the results
                    images.append(metadata)

                    self._logger.info(
                        f"[{current_document}] Saved image from page {page.page_number}: {image_filename}"
                    )
                else:
                    self._logger.warning(
                        f"[{current_document}] Failed to crop image {idx + 1} on page {page.page_number}: bbox={img_bbox}, size={img_width}x{img_height}px - cropped result is None/empty"
                    )

            except Exception as e:
                self._logger.warning(
                    f"[{current_document}] Error processing image {idx + 1} on page {page.page_number}: {e}"
                )

        return images

    def _calculate_id(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    async def _generate_table_caption(self, table_data: "PDFTableData", document_name: str, page_number: int) -> str:
        """
        Generate an intelligent caption for a table using the injected LLM call.

        Args:
            table_data: The PDFTableData object containing the table
            document_name: Name of the document
            page_number: Page number where table appears

        Returns:
            Generated caption describing the table content
        """
        # If no table caption call is configured, return basic caption
        if not self._table_caption_call:
            return f"Table - {document_name} (Page {page_number})"

        try:
            # Get first 500 characters of the table markdown for better context
            if table_data.data and len(table_data.data) > 0:
                # Convert to markdown format for better readability
                markdown_lines = []

                # Add headers
                if table_data.data:
                    headers = [str(cell) if cell else "" for cell in table_data.data[0]]
                    markdown_lines.append(" | ".join(headers))
                    markdown_lines.append(" | ".join(["-" * min(len(h), 3) for h in headers]))

                    # Add data rows
                    for row in table_data.data[1:]:
                        cells = [str(cell)[:50] if cell else "" for cell in row]
                        markdown_lines.append(" | ".join(cells))

                # Join and take first 500 characters
                full_table = "\n".join(markdown_lines)
                context = full_table[:500]

                # Add ellipsis if truncated
                if len(full_table) > 500:
                    context += "..."
            else:
                context = "Table with no readable content"

            # Use the injected table caption extraction call
            result = await self._table_caption_call.execute(
                table_content=context,
                document_name=document_name,
                page_number=page_number,
            )

            # Extract caption from consensus result and combine with document info
            caption = result.final_response.caption if result.final_response else "Table"
            return f"{caption} - {document_name} (Page {page_number})"

        except Exception as e:
            self._logger.warning(f"Failed to generate table caption with consensus call: {e}")
            # Return a basic caption as last resort
            return f"Table - {document_name} (Page {page_number})"

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
        from blockether_catalyst.knowledge.KnowledgeProcessingUtils import (
            KnowledgeProcessingUtils,
        )

        # Text already has hyphenated line breaks fixed
        raw_text = base_text

        # Append tables with markers for later replacement with markdown
        for table in tables:
            # Store ASCII version within markers for searchability
            raw_text += f"\n{KnowledgeProcessingUtils.TABLE_START}{table.to_ascii_table()}{KnowledgeProcessingUtils.TABLE_END}\n"

        return raw_text

    def regenerate_all_images(self, pdf_files: List[str]) -> None:
        """Regenerate all images from the given PDF files.

        Args:
            pdf_files: List of PDF file paths to process

        Raises:
            FileNotFoundError: If images directory or PDF files not found
        """
        # Check if image output directory exists
        if not self._image_output_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self._image_output_dir}")

        # Process each PDF file
        for pdf_file in pdf_files:
            pdf_path = Path(pdf_file)
            if not pdf_path.exists():
                raise FileNotFoundError(f"PDF file not found: {pdf_file}")

            # This would need to be implemented with proper PDF processing
            # For now, just validate the files exist
            pass
