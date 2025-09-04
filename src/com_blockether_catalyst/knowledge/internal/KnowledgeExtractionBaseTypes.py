"""
Base types for knowledge extraction that are shared between other modules.

This intermediate file breaks circular dependencies between KnowledgeExtractionTypes
and KnowledgeExtractionCallBase by providing common types both modules need.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

from pydantic import BaseModel, Field, computed_field

from com_blockether_catalyst.consensus import TypedCallBaseForConsensus
from com_blockether_catalyst.consensus.internal.VotingComparison import (
    ComparisonStrategy,
    VotingField,
)


class ExtractedKeyword(BaseModel):
    """A keyword extracted from text."""

    term: str = VotingField(comparison=ComparisonStrategy.EXACT, description="The extracted keyword")


class ExtractedAcronym(BaseModel):
    """An acronym extracted from text with its full form."""

    term: str = VotingField(comparison=ComparisonStrategy.EXACT, description="The acronym (e.g., 'API')")
    full_form: str = VotingField(
        comparison=ComparisonStrategy.SEMANTIC,
        threshold=0.85,
        description="The full form (e.g., 'Application Programming Interface')",
    )


class ChunkKeywordExtractionResponse(TypedCallBaseForConsensus):
    """Response from chunk-level keyword extraction."""

    keywords: List[ExtractedKeyword] = VotingField(
        default_factory=list,
        comparison=ComparisonStrategy.SEQUENCE_UNORDERED_ALIKE,
        threshold=0.6,
        description="List of keywords found in the chunk",
    )


class ChunkAcronymExtractionResponse(TypedCallBaseForConsensus):
    """Response from chunk-level acronym extraction."""

    acronyms: List[ExtractedAcronym] = VotingField(
        default_factory=list,
        comparison=ComparisonStrategy.SEQUENCE_UNORDERED_ALIKE,
        threshold=0.6,
        description="List of acronyms found in the chunk",
    )


class AcronymMeaningExtractionResponse(TypedCallBaseForConsensus):
    """Response model for LLM acronym meaning extraction."""

    is_valid: bool = VotingField(
        comparison=ComparisonStrategy.EXACT,
        description="Whether this is a valid acronym",
    )
    meaning: Optional[str] = VotingField(
        default=None,
        comparison=ComparisonStrategy.SEMANTIC,
        threshold=0.8,
        description="The extracted meaning of the acronym if found",
    )
    full_form: Optional[str] = VotingField(
        default=None,
        comparison=ComparisonStrategy.SEMANTIC,
        threshold=0.85,
        description="The full form of the acronym if found",
    )


class KeywordMeaningExtractionResponse(TypedCallBaseForConsensus):
    """Response model for LLM keyword meaning extraction."""

    is_valid: bool = VotingField(
        comparison=ComparisonStrategy.EXACT,
        description="Whether this is a valid keyword worth keeping",
    )
    meaning: Optional[str] = VotingField(
        default=None,
        comparison=ComparisonStrategy.SEMANTIC,
        threshold=0.8,
        description="The extracted meaning/definition of the keyword if found",
    )
    full_form: Optional[str] = VotingField(
        default=None,
        comparison=ComparisonStrategy.EXACT,
        description="For keywords, always equals to the keyword term (set during post-processing)",
    )


class ChunkOutput(BaseModel):
    """A single chunk with metadata."""

    text: str = VotingField(
        comparison=ComparisonStrategy.EXACT,
        description="The actual chunk text (must be valid markdown)",
    )
    start_position: int = VotingField(
        comparison=ComparisonStrategy.EXACT,
        description="Starting character position in original text",
    )
    end_position: int = VotingField(
        comparison=ComparisonStrategy.EXACT,
        description="Ending character position in original text",
    )
    preserves_structure: bool = VotingField(
        default=True,
        comparison=ComparisonStrategy.EXACT,
        description="Whether this chunk preserves tables, lists, and other structures",
    )


class ChunkingDecision(TypedCallBaseForConsensus):
    """Response model for intelligent chunking of document pages."""

    chunks: List[ChunkOutput] = VotingField(
        comparison=ComparisonStrategy.EXACT,
        description="Sequence of chunks to create from the provided text, each with proper boundaries",
    )

    @property
    @computed_field
    def total_chunks(self) -> int:
        """Get the total number of chunks created."""
        return len(self.chunks)


class TermCooccurrence(BaseModel):
    """Represents co-occurrence of terms."""

    term: str = Field(description="The co-occurring term")
    frequency: int = Field(default=1, description="Number of times terms co-occur")
    confidence: float = Field(default=0.0, description="Confidence score of co-occurrence (0-1)")


class KnowledgeTableData(BaseModel):
    """Represents table data with metadata."""

    page: int = Field(description="Page number where table is located")
    data: List[List[Optional[str]]] = Field(description="Table data as 2D list")
    rows: int = Field(description="Number of rows in table")
    columns: int = Field(description="Number of columns in table")

    def to_html_table(self) -> str:
        """Convert table data to HTML format with full nesting support.

        Returns:
            HTML formatted table string that supports nested tables
        """
        if not self.data:
            return ""

        html = ['<table border="1" cellpadding="4" cellspacing="0">']

        # Add thead with first row as header
        if self.data:
            html.append("  <thead>")
            html.append("    <tr>")
            for cell in self.data[0]:
                cell_value = "" if cell is None else str(cell)
                # Escape HTML entities except if it looks like a nested table
                if not cell_value.startswith("<table"):
                    cell_value = cell_value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                html.append(f"      <th>{cell_value}</th>")
            html.append("    </tr>")
            html.append("  </thead>")

        # Add tbody with remaining rows
        if len(self.data) > 1:
            html.append("  <tbody>")
            for row in self.data[1:]:
                html.append("    <tr>")
                # Ensure row has same number of columns as header
                cells = list(row) if row else []
                while len(cells) < len(self.data[0]):
                    cells.append(None)
                cells = cells[: len(self.data[0])]

                for cell in cells:
                    cell_value = "" if cell is None else str(cell)
                    # Check if cell contains a nested table
                    if not cell_value.startswith("<table"):
                        cell_value = cell_value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                    html.append(f"      <td>{cell_value}</td>")
                html.append("    </tr>")
            html.append("  </tbody>")

        html.append("</table>")
        return "\n".join(html)

    def to_ascii_table(self) -> str:
        """Convert table data to clean whitespace-separated format.

        Returns:
            Clean ASCII table with aligned columns using spaces only
        """
        if not self.data:
            return ""

        # Calculate column widths
        col_widths = []
        for col_idx in range(self.columns):
            max_width = 0
            for row in self.data:
                if col_idx < len(row):
                    cell_value = "" if row[col_idx] is None else str(row[col_idx])
                    max_width = max(max_width, len(cell_value))
            col_widths.append(max_width)

        lines = []

        # Process all rows
        for row_idx, row in enumerate(self.data):
            cells = []
            for col_idx, width in enumerate(col_widths):
                cell_value = ""
                if col_idx < len(row):
                    cell_value = "" if row[col_idx] is None else str(row[col_idx])

                # Left-align and pad with spaces
                cells.append(cell_value.ljust(width))

            # Join with double spaces for better readability
            lines.append("  ".join(cells))

            # Add blank line after header for clarity
            if row_idx == 0 and len(self.data) > 1:
                lines.append("")

        return "\n".join(lines)

    def to_markdown(self) -> str:
        """Convert table data to Markdown format.

        Returns:
            Markdown formatted table string
        """
        if not self.data or not self.data[0]:
            return ""

        lines = []

        # Create header row
        header = "| " + " | ".join(str(cell) if cell else "" for cell in self.data[0]) + " |"
        lines.append(header)

        # Create separator row
        separator = "| " + " | ".join("---" for _ in self.data[0]) + " |"
        lines.append(separator)

        # Add data rows
        for row in self.data[1:]:
            row_cells = []
            for i in range(len(self.data[0])):
                cell = str(row[i]) if i < len(row) and row[i] else ""
                # Escape pipe characters in cell content
                cell = cell.replace("|", "\\|")
                row_cells.append(cell)
            lines.append("| " + " | ".join(row_cells) + " |")

        return "\n".join(lines)


class KnowledgePageData(BaseModel):
    """Represents a processed page."""

    page: int = Field(description="Page number (1-indexed)")
    text: str = Field(description="Extracted text content without tables or images")
    tables: List[KnowledgeTableData] = Field(default_factory=list, description="Tables found on page")
    images: List[str] = Field(default_factory=list, description="Images found on page")
    lines: int = Field(default=0, description="Line count (calculated from text)")


class KnowledgeMetadata(BaseModel):
    """metadata information."""

    title: Optional[str] = Field(default=None, description="title")
    author: Optional[str] = Field(default=None, description="author")
    subject: Optional[str] = Field(default=None, description="subject")
    creation_date: Optional[str] = Field(default=None, description="creation date")
    modification_date: Optional[str] = Field(default=None, description="modification date")
