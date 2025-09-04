"""
Pydantic models for PDF-specific processing settings.
"""

from typing import Any, Dict, List, Optional, Tuple, TypedDict

from pydantic import BaseModel, Field


class PDFProcessorTableExtractionSettings(BaseModel):
    """Settings for table extraction passed to pdfplumber."""

    vertical_strategy: str = Field(default="lines", description="Strategy for detecting vertical lines")
    horizontal_strategy: str = Field(default="lines", description="Strategy for detecting horizontal lines")
    snap_tolerance: float = Field(default=3, description="Tolerance for snapping lines together")
    edge_min_length: float = Field(default=3, description="Minimum length for table edges")
    min_words_vertical: int = Field(default=3, description="Minimum words for vertical alignment")
    min_words_horizontal: int = Field(default=1, description="Minimum words for horizontal alignment")


class PDFProcessorTextExtractionSettings(BaseModel):
    """Settings for text extraction passed to pdfplumber."""

    layout: bool = Field(default=False, description="Maintain layout in extracted text")
    x_tolerance: float = Field(default=3, description="Horizontal tolerance for grouping text")
    y_tolerance: float = Field(default=3, description="Vertical tolerance for grouping text")
    use_text_flow: bool = Field(default=True, description="Use natural reading order")


class PDFImageProcessingSettings(BaseModel):
    """Settings for image processing with CLIP and OCR."""

    device: Optional[str] = Field(default=None, description="Device to use (cuda, mps, cpu). Auto-detect if None")
    torch_dtype: str = Field(default="auto", description="Torch dtype (float16, float32, auto)")
    confidence_threshold: float = Field(default=0.7, description="Confidence threshold for decorative image detection")


class PDFPageCropOffset(BaseModel):
    """Page cropping offsets in pixels."""

    top: int
    bottom: int
