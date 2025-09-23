"""
Unified type definitions for the knowledge extraction and search system.

This module consolidates all type definitions from extraction, search, and processing
to reduce duplication and maintain a single source of truth for all types.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, computed_field

from blockether_catalyst.consensus.VotingComparison import (
    BaseModelWithReasoning,
    ComparisonStrategy,
    VotingField,
)

from .PDKnowledgeExtractorTypes import PDFKnowledgeProcessorSettings

# ============================================================================
# Core Document Types
# ============================================================================

DocumentSourceType = Literal["pdf", "docx", "txt"]


class DocumentMetadata(BaseModel):
    """Core document metadata used during extraction."""

    title: Optional[str] = VotingField(default=None, description="Document title", comparison=ComparisonStrategy.EXACT)
    author: Optional[str] = VotingField(
        default=None, description="Document author", comparison=ComparisonStrategy.EXACT
    )
    subject: Optional[str] = VotingField(
        default=None,
        description="Document subject",
        comparison=ComparisonStrategy.EXACT,
    )
    document_path: str = Field(description="Path to the document file")
    creation_date: Optional[str] = Field(default=None, description="Document creation date")
    modification_date: Optional[str] = Field(default=None, description="Document last modification date")


class NormalizedDocumentMetadata(BaseModel):
    """Extended document metadata with statistics for LinkedKnowledge."""

    # Core identification
    document_id: str = Field(description="SHA-256 hash identifier of the document")
    document_filename: str = Field(description="Original filename of the document")
    document_path: str = Field(description="Path to the document file")

    # Metadata fields
    title: str = Field(description="Document title")
    subject: Optional[str] = Field(description="Document subject")
    author: Optional[str] = Field(description="Document author")
    modification_date: Optional[str] = Field(description="Document last modification date")
    publication_date: Optional[str] = Field(description="Document creation/publication date")

    # Statistics
    total_pages: int = Field(description="Total number of pages in document")
    total_chunks: int = Field(description="Total number of chunks created from document")
    total_terms: int = Field(description="Total number of unique terms found in document")
    total_tables: int = Field(description="Total number of tables found in document")
    total_images: int = Field(default=0, description="Total number of images found in document")
    total_acronyms: int = Field(description="Total number of unique acronyms found in document")
    total_keywords: int = Field(description="Total number of unique keywords found in document")


# ============================================================================
# Image and Table Types
# ============================================================================


class ImageMetadata(BaseModel):
    """Metadata for an extracted image from a document."""

    document_name: str = Field(description="Name of the source document")
    page: int = Field(description="Page number where the image was found")
    path: str = Field(description="Path or reference to the saved image file")
    caption: Optional[str] = VotingField(
        default=None,
        description="AI-generated caption describing the image",
        comparison=ComparisonStrategy.SEMANTIC,
    )


class KnowledgeTableData(BaseModel):
    """Represents table data with metadata and rendering methods."""

    page: int = Field(description="Page number where table is located")
    data: List[List[Optional[str]]] = Field(description="Table data as 2D list")
    rows: int = Field(description="Number of rows in table")
    columns: int = Field(description="Number of columns in table")

    def to_html_table(self) -> str:
        """Convert table data to HTML format with full nesting support."""
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
        """Convert table data to clean whitespace-separated format."""
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
        """Convert table data to markdown format using MarkdownGenerator.

        Returns:
            Markdown formatted table string
        """
        from .MarkdownGenerator import MarkdownGenerator

        if not self.data or len(self.data) < 2:
            return ""

        # Extract headers and rows
        headers = [str(cell) if cell else "" for cell in self.data[0]]
        rows = []
        for row in self.data[1:]:
            row_cells = []
            for i in range(len(self.data[0])):
                cell = str(row[i]) if i < len(row) and row[i] else ""
                # Escape pipe characters in cell content
                cell = cell.replace("|", "\\|")
                row_cells.append(cell)
            rows.append(row_cells)

        return MarkdownGenerator.create_table(headers, rows)


# ============================================================================
# Page Data Types
# ============================================================================


class KnowledgePageData(BaseModel):
    """Data for a single page in a document."""

    page: int = Field(description="1-indexed page number")
    text: str = Field(default="", description="Extracted text content of the page")
    images: List[ImageMetadata] = Field(
        default_factory=list,
        description="List of images extracted from the page",
    )
    tables: List[KnowledgeTableData] = Field(
        default_factory=list,
        description="List of tables extracted from the page",
    )


class KnowledgePageDataWithRawText(KnowledgePageData):
    """Page data with additional raw text information."""

    raw_text: str = Field(default="", description="Raw text including tables and OCRed image text inline")
    lines: int = Field(default=0, description="Line count (calculated from raw_text)")


# ============================================================================
# Chunk Types
# ============================================================================


class KnowledgeChunk(BaseModel):
    """Represents a text chunk with keyword indexing."""

    document_id: str = Field(description="SHA-256 hash identifier of the document")
    document_name: str = Field(description="Name of the source document")
    doc_id: str = Field(description="Unique chunk identifier combining document_id, page, and chunk_index")
    index: int = Field(description="Index of the chunk")
    text: str = VotingField(
        description="Text content of the chunk",
        comparison=ComparisonStrategy.SEMANTIC,
    )
    page: int = Field(description="Page number (1-indexed) - can be used to retrieve images/tables from page data")
    content_types: List[str] = VotingField(
        default_factory=list,
        description="Types of content in this chunk: 'text', 'image', 'table'",
        comparison=ComparisonStrategy.EXACT,
    )
    semantic_types: List[str] = VotingField(
        default_factory=list,
        description="Semantic classifications: 'table_of_contents', 'summary', 'rule', 'explanation', 'example', 'general'",
        comparison=ComparisonStrategy.EXACT,
    )


class KnowledgeChunkWithTerms(KnowledgeChunk):
    """Knowledge chunk with terms for search and indexing."""

    terms: Dict[str, int] = Field(
        default_factory=dict,
        description="Terms (keywords/acronyms) found in this chunk",
    )


class ChunkOutput(BaseModel):
    """Output model for a single chunk created from a page."""

    text: str = VotingField(
        description="Text content of the chunk",
        comparison=ComparisonStrategy.SEMANTIC,
    )


class ChunkingDecisionResponse(BaseModelWithReasoning):
    """Response model for intelligent chunking of document pages."""

    chunks: List[ChunkOutput] = VotingField(
        description="Sequence of chunks to create from the provided text, each with proper boundaries",
        comparison=ComparisonStrategy.EXACT,
    )

    @property
    @computed_field
    def total_chunks(self) -> int:
        """Get the total number of chunks created."""
        return len(self.chunks)


class ChunkContentClassification(BaseModelWithReasoning):
    """Classification of chunk content based on semantic analysis."""

    semantic_types: List[
        Literal[
            "table_of_contents",
            "summary",
            "rule",
            "explanation",
            "example",
            "reference",
            "general",
        ]
    ] = VotingField(
        min_length=1,
        comparison=ComparisonStrategy.EXACT,
        description="The semantic classifications of the chunk content (can be multiple, min_length=1)",
    )
    confidence_scores: Dict[str, float] = VotingField(
        default_factory=dict,
        comparison=ComparisonStrategy.IGNORE,
        description="Confidence score for each classification (0-1)",
    )
    key_indicators: Dict[str, List[str]] = VotingField(
        default_factory=dict,
        comparison=ComparisonStrategy.IGNORE,
        description="Key phrases or patterns that led to each classification",
    )


# ============================================================================
# Term Types
# ============================================================================


class TermOccurrence(BaseModel):
    """Represents a term occurrence within a document chunk."""

    document_id: str = Field(description="SHA-256 hash identifier of the document")
    document_name: str = Field(description="Name of the source document")
    page: int = Field(description="Page number (1-indexed) where the term was found")
    chunk_index: int = Field(description="Index of the chunk where the term was found")
    total: int = Field(description="Total occurrences of the term in this specific chunk")


class TermCooccurrence(BaseModel):
    """Represents co-occurrence of terms."""

    term: str = Field(description="The co-occurring term")
    score: float = Field(default=0.0, description="Confidence score of co-occurrence (0-1)")


class TermLink(BaseModel):
    """Represents a link between an acronym and its corresponding keyword/term."""

    link_from: str = Field(description="Term text from where the link originates")
    link_to: str = Field(description="Term text to which term is linked")
    score: float = Field(description="Matching score between acronym full form and keyword (0-1)")


class TermCandidate(BaseModel):
    """Candidate term during extraction phase (before consolidation)."""

    document_filename: str = Field(description="Name of the document")
    document_id: str = Field(description="SHA-256 identifier of the document")
    term: str = Field(description="The term text")
    page: int = Field(description="Page number where this term appears")
    chunk: int = Field(description="Chunk index where this term appears")
    total: int = Field(description="Total occurrences of this term in the specific chunk")
    type: Literal["acronym", "keyword", "unknown"] = Field(description="Type of term")


class TermCandidateGrouped(BaseModel):
    """Grouped term candidates across documents."""

    type: Literal["acronym", "keyword", "unknown"] = Field(description="Type of term")
    term: str = Field(description="The term text")
    occurrences: List[TermOccurrence] = Field(
        default_factory=list,
        description="List of occurrences of this term across documents",
    )
    cooccurrences: List[TermCooccurrence] = Field(
        default_factory=list,
        description="Terms that frequently co-occur with this term",
    )
    total: int = Field(default=0, description="Total occurrences across all documents")


class Term(BaseModelWithReasoning):
    """Represents a consolidated term (acronym or keyword) with validation and meaning."""

    term: str = Field(description="The term text (acronym or keyword)")
    type: str = Field(description="Type of term: 'acronym' or 'keyword'")
    occurrences: list[TermOccurrence] = Field(
        default_factory=list,
        description="All occurrences of this term across documents",
    )
    cooccurrences: list[TermCooccurrence] = Field(
        default_factory=list,
        description="Terms that frequently co-occur with this term",
    )
    total: int = Field(default=0, description="Total number of times this term appears")
    full_form: str = Field(description="The expanded full form (same as term for keywords)")
    meaning: str = Field(description="The extracted meaning of the term")


class TermWithLinks(Term):
    """Term with links to related terms."""

    links: List[TermLink] = Field(
        default_factory=list,
        description="Links from this term to related terms",
    )


class TermMeaningExtractionResponse(BaseModelWithReasoning):
    """Response model for term meaning extraction."""

    term: str = VotingField(
        description="The term text (acronym or keyword)",
        comparison=ComparisonStrategy.EXACT,
    )

    full_form: Optional[str] = VotingField(
        description="The expanded full form (same as term for keywords)",
        comparison=ComparisonStrategy.EXACT,
    )

    meaning: str = VotingField(
        description="The extracted meaning of the term. Maximum 1200 characters.",
        comparison=ComparisonStrategy.SEMANTIC,
        min_length=150,
        max_length=1200,
        threshold=0.65,
    )

    type: Literal["acronym", "keyword"] = VotingField(
        description="Type of term: 'acronym' or 'keyword'. ACRONYM if all uppercase, or mixed case with periods; typically expanding to a longer phrase. KEYWORD otherwise.",
        comparison=ComparisonStrategy.EXACT,
    )

    meaning_status: Literal["meaningful", "generic", "unknown"] = VotingField(
        description="Status: 'meaningful' if a definition was extracted either from direct mention or context, 'generic' if no distinct definition, 'unknown' if undetermined",
        comparison=ComparisonStrategy.EXACT,
    )


# ============================================================================
# Extraction Result Types
# ============================================================================


class KnowledgeExtractionResultBase(BaseModel):
    """Base class for extraction results."""

    id: str = Field(description="SHA256 hash of the file")
    document_filename: str = Field(description="filename")
    document_metadata: DocumentMetadata = Field(description="Document metadata extracted from PDF")
    total_pages: int = Field(default=0, description="Total number of pages")
    total_images: int = Field(default=0, description="Total number of images found")
    total_acronyms: int = Field(default=0, description="Total number of acronyms found")
    total_keywords: int = Field(default=0, description="Total number of keywords found")
    total_tables: int = Field(default=0, description="Total number of tables found")
    source_type: DocumentSourceType = Field(description="Source type of the document")


class KnowledgeExtractionResult(KnowledgeExtractionResultBase):
    """Complete extraction result."""

    pages: list[KnowledgePageDataWithRawText] = Field(default_factory=list, description="Processed pages")
    raw: str = Field(
        default="",
        description="Full text with OCR text replacing images and tables inline",
    )


class KnowledgeExtractionResultWithChunks(KnowledgeExtractionResult):
    """Extraction result with chunks for indexing."""

    chunks: list[KnowledgeChunk] = Field(
        default_factory=list,
        description="List of text chunks with keyword indexing",
    )
    total_chunks: int = Field(default=0, description="Total number of chunks created")


class KnowledgeExtractionItem(BaseModel):
    """Represents a single PDF extraction result or error."""

    result: Optional[KnowledgeExtractionResult] = Field(default=None, description="Successful extraction result")
    error: Optional[str] = Field(default=None, description="Error message if extraction failed")


class KnowledgeExtractionOutput(BaseModel):
    """Complete output from knowledge extraction process."""

    pdf: Optional[list[KnowledgeExtractionItem]] = Field(default=None)
    # Future: Add other file types as needed
    # docx: Optional[Sequence[KnowledgeExtractionItem]] = Field(default=None)
    # txt: Optional[Sequence[KnowledgeExtractionItem]] = Field(default=None)
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="ISO format timestamp of when extraction was performed",
    )


class KnowledgeExtractionOutputWithChunks(KnowledgeExtractionItem):
    """Extraction output with chunking information."""

    chunk_size: int = Field(default=0, description="Size of each chunk in characters")
    chunk_overlap: int = Field(default=0, description="Not used for agentic chunking (always 0)")


# ============================================================================
# Search Types
# ============================================================================


class TypedPydanticModel(BaseModel):
    """Base model with markdown rendering capability."""

    def markdown(self) -> str:
        """
        Generate a markdown representation of the model.

        Returns:
            Markdown formatted string of the model data
        """
        lines = []
        model_name = self.__class__.__name__
        lines.append(f"## {model_name}")

        # Iterate through fields and format them
        for field_name, field_value in self.model_dump(exclude_none=True).items():
            if field_value is None or (isinstance(field_value, (list, dict, set)) and not field_value):
                continue

            # Format field name
            formatted_name = field_name.replace("_", " ").title()

            if isinstance(field_value, list):
                lines.append(f"\n### {formatted_name}")
                for item in field_value:
                    if hasattr(item, "markdown"):
                        lines.append(item.markdown())
                    else:
                        lines.append(f"- {item}")
            elif isinstance(field_value, dict):
                lines.append(f"\n### {formatted_name}")
                for key, value in field_value.items():
                    lines.append(f"- **{key}**: {value}")
            else:
                lines.append(f"**{formatted_name}**: {field_value}")

        return "\n".join(lines)


class TermInfo(TypedPydanticModel):
    """Information about a term."""

    term: str = Field(description="The term text")
    meaning: Optional[str] = Field(default=None, description="Meaning of the term")
    term_type: Optional[str] = Field(default=None, description="Type of term (e.g., acronym, keyword)")
    total_times_occurred_in_knowledgebase: Optional[int] = Field(default=None, description="Total occurrences")
    linked_terms: List["LinkedTermInfo"] = Field(default_factory=list, description="Terms linked from this term")

    def markdown(self) -> str:
        """Generate markdown for a term."""
        lines = []

        # Main term with type
        if self.term_type:
            lines.append(f"- **{self.term}** ({self.term_type})")
        else:
            lines.append(f"- **{self.term}**")

        # Add meaning if present
        if self.meaning:
            meaning_preview = self.meaning[:150] + "..." if len(self.meaning) > 150 else self.meaning
            lines.append(f"  - Meaning: {meaning_preview}")

        # Add occurrences if present
        if self.total_times_occurred_in_knowledgebase:
            lines.append(f"  - Occurrences: {self.total_times_occurred_in_knowledgebase}")

        # Add linked terms if present
        if self.linked_terms:
            lines.append("  - Related:")
            for linked_term in self.linked_terms[:3]:  # Limit to 3
                lines.append(f"    - {linked_term.term}")

        return "\n".join(lines)


class LinkedTermInfo(TermInfo):
    """Information about a term that is linked from another term (extends TermInfo with link_score)."""

    link_score: float = Field(description="Score of the link to this term (strength of relationship, 0.0-1.0)")
    linked_terms: List["LinkedTermInfo"] = Field(
        default_factory=list, description="Nested linked terms (usually empty)"
    )


class ImageInfo(TypedPydanticModel):
    """Information about an image."""

    caption: Optional[str] = Field(default=None, description="Image caption")
    href: str = Field(description="URL to the image")
    page: Optional[int] = Field(default=None, description="Page number")
    document_name: Optional[str] = Field(default=None, description="Document containing the image")

    def markdown(self) -> str:
        """Generate markdown for an image."""
        caption_text = self.caption or "Image"
        if self.document_name and self.page is not None:
            return f"\n![{caption_text} - {self.document_name} - (Page {self.page})]({self.href})\n<center>{caption_text} - {self.document_name} - (Page {self.page})</center>\n"
        elif self.page is not None:
            return f"\n![{caption_text} - (Page {self.page})]({self.href})\n<center>{caption_text} - (Page {self.page})</center>\n"
        else:
            return f"\n![{caption_text}]({self.href})\n<center>{caption_text}</center>\n"


class TableInfo(TypedPydanticModel):
    """Information about a table."""

    markdown_content: str = Field(description="Table content in markdown format")
    page: Optional[int] = Field(default=None, description="Page number")

    def markdown(self) -> str:
        """Generate markdown for a table."""
        lines = []
        if self.page is not None:
            lines.append(f"**Table (Page {self.page})**")
        lines.append(self.markdown_content)
        return "\n".join(lines)


class NormalizedSearchResult(TypedPydanticModel):
    """Normalized search result ready for consumption."""

    # Core fields
    score: float = Field(description="Relevance score")
    content: str = Field(description="The main text content")
    document_name: str = Field(description="Name of the document")
    page: Optional[int] = Field(default=None, description="Page number")

    # Metadata
    author: Optional[str] = Field(default=None, description="Document author")
    publication_date: Optional[str] = Field(default=None, description="Publication date")
    modified_date: Optional[str] = Field(default=None, description="Modification date")
    href: Optional[str] = Field(default=None, description="Link to document")

    # Extracted content
    images: List[ImageInfo] = Field(default_factory=list, description="Images in the result")
    tables: List[TableInfo] = Field(default_factory=list, description="Tables in the result")
    primary_terms: List[TermInfo] = Field(default_factory=list, description="Primary terms")
    related_terms: List[TermInfo] = Field(default_factory=list, description="Related terms")

    def markdown(self) -> str:
        """Generate markdown representation of the search result."""
        lines = []

        # Document header with metadata
        lines.append(
            f"## THIS IS PART OF {self.document_name} on page {self.page}, RELEVANCE of this excerpt: {self.score:.2%}"
        )

        # Metadata section
        metadata_parts = []
        if self.author:
            metadata_parts.append(f"Author: {self.author}")
        if self.publication_date:
            metadata_parts.append(f"Published: {self.publication_date}")

        if metadata_parts:
            lines.append("")
            lines.append("### Metadata:")
            lines.append(f"*{' | '.join(metadata_parts)}*")
            lines.append("")

        # Document link if available
        if self.href:
            lines.append(f"[View Document]({self.href})")
            lines.append("")

        # Main content
        lines.append("### Content")
        lines.append(self.content)
        lines.append("")

        # Primary terms section
        if self.primary_terms:
            lines.append("### 🔑 Key Terms")
            for term in self.primary_terms[:4]:
                lines.append(term.markdown())
            lines.append("")

        # Related terms section
        if self.related_terms:
            lines.append("### 🔗 Related Terms")
            for term in self.related_terms[:5]:
                term_line = f"`- {term.term} (type: {term.term_type or 'unknown'})`"
                if term.meaning:
                    term_line += f", meaning: {term.meaning}"
                lines.append(term_line)
            lines.append("")

        # Images section
        if self.images:
            lines.append("### 🖼️ Images")
            for img in self.images[:3]:  # Limit to 3 images
                lines.append(img.markdown())
            lines.append("")

        # Tables section
        if self.tables:
            lines.append("### 📊 Tables")
            for idx, table in enumerate(self.tables[:2], 1):  # Limit to 2 tables
                lines.append(f"**Table {idx}**")
                lines.append(table.markdown())
                lines.append("")

        lines.append("---")
        return "\n".join(lines)


class CompactSearchResult(TypedPydanticModel):
    """Search result without duplicated term data."""

    # Core fields
    score: float = Field(description="Relevance score")
    content: str = Field(description="The main text content")
    document_name: str = Field(description="Name of the document")
    page: Optional[int] = Field(default=None, description="Page number")

    # Metadata
    author: Optional[str] = Field(default=None, description="Document author")
    publication_date: Optional[str] = Field(default=None, description="Publication date")
    modified_date: Optional[str] = Field(default=None, description="Modification date")
    href: Optional[str] = Field(default=None, description="Link to document")

    # References to terms by their keys
    primary_term_keys: List[str] = Field(default_factory=list, description="Keys of primary terms")
    related_term_keys: List[str] = Field(default_factory=list, description="Keys of related terms")

    # Content that is unique per result
    images: List[ImageInfo] = Field(default_factory=list, description="Images in the result")
    tables: List[TableInfo] = Field(default_factory=list, description="Tables in the result")


class OptimizedSearchResponse(TypedPydanticModel):
    """Optimized search response with deduplicated terms."""

    # Search results without duplicated term data
    results: List[CompactSearchResult] = Field(description="Search results")

    # Deduplicated terms dictionary
    terms: Dict[str, TermInfo] = Field(default_factory=dict, description="All unique terms referenced in results")

    # Search metadata
    total_results: int = Field(description="Total number of results found")
    query: str = Field(description="The original search query")
    search_type: str = Field(default="hybrid", description="Type of search performed")

    def to_standard_format(self) -> List[NormalizedSearchResult]:
        """Convert back to standard format if needed for compatibility."""
        standard_results = []
        for result in self.results:
            # Reconstruct the full result with term data
            primary_terms = [self.terms[key] for key in result.primary_term_keys if key in self.terms]
            related_terms = [self.terms[key] for key in result.related_term_keys if key in self.terms]

            standard_result = NormalizedSearchResult(
                score=result.score,
                content=result.content,
                document_name=result.document_name,
                page=result.page,
                author=result.author,
                publication_date=result.publication_date,
                modified_date=result.modified_date,
                href=result.href,
                images=result.images,
                tables=result.tables,
                primary_terms=primary_terms,
                related_terms=related_terms,
            )
            standard_results.append(standard_result)
        return standard_results


# Update imports for forward references
TermInfo.model_rebuild()


class SearchResultMetadata(BaseModel):
    """Metadata for a search result."""

    chunk_id: str
    document_id: str
    document_name: str
    document_path: str
    document_title: str
    page: Optional[int] = None
    chunk_index: Optional[int] = None
    author: Optional[str] = None
    publication_date: Optional[str] = None
    modified_date: Optional[str] = None
    document_subject: Optional[str] = None
    terms: Dict[str, int] = Field(
        default_factory=dict,
        description="Terms (keywords/acronyms) found in this chunk",
    )


class SearchResult(BaseModel):
    """Simple search result matching the expected interface."""

    text: str
    score: float
    doc_id: str
    metadata: SearchResultMetadata


@dataclass
class TopTermsResult:
    """Result containing top keywords and acronyms from TF-IDF analysis."""

    keywords: List[Tuple[str, float]]  # List of (keyword, score) tuples ordered by n-gram size then score
    acronyms: List[Tuple[str, float]]  # List of (acronym, score) tuples ordered by score


class KnowledgeSearchResult(BaseModel):
    """Enhanced search result with term analysis and statistics."""

    # Core fields
    text: str = Field(description="Matching text content")
    score: float = Field(description="Relevance score")
    document_id: str = Field(description="Document identifier")
    document_name: str = Field(description="Document filename")
    document_path: str = Field(description="Path to the document in the extraction output directory")
    page: int = Field(description="Page number where content appears")
    chunk_index: int = Field(description="Chunk index where content appears")
    metadata: SearchResultMetadata = Field(description="Additional metadata")

    # Term analysis
    primary_terms: List[TermWithLinks] = Field(default_factory=list, description="Primary terms with links")
    related_terms: List[TermWithLinks] = Field(default_factory=list, description="Related terms with links")
    all_terms: Set[str] = Field(default_factory=set, description="All unique terms")

    # Term frequency statistics
    term_frequencies: Dict[str, int] = Field(default_factory=dict, description="Term -> frequency in query")
    term_relevance_score: float = Field(default=0.0, description="Combined relevance based on term frequencies")
    final_score: float = Field(default=0.0, description="Composite score including all boosts and weights")

    # Page content
    images: List[ImageMetadata] = Field(default_factory=list, description="Images in the result")
    tables: List[KnowledgeTableData] = Field(default_factory=list, description="Tables in the result")


# ============================================================================
# Aggregation Types
# ============================================================================


class RawExtractionData(BaseModel):
    """Raw extraction data used to build LinkedKnowledge objects."""

    results_with_chunks: list[KnowledgeExtractionResultWithChunks] = Field(
        description="List of document results with their chunks"
    )
    terms: Dict[str, TermWithLinks] = Field(description="Dictionary of all validated terms with links")
    document_to_chunks_index: Dict[str, list[KnowledgeChunk]] = Field(
        description="Index mapping document IDs to their chunks for efficient lookup"
    )


class LinkedKnowledge(BaseModel):
    """Comprehensive container for all extracted knowledge and relationships."""

    # Core extracted content - simplified to just metadata
    documents: Dict[str, NormalizedDocumentMetadata] = Field(
        default_factory=dict,
        description="Document metadata indexed by document_id",
    )

    # Pages index - key is (document_id, page_number) tuple
    pages: Dict[Tuple[str, int], KnowledgePageData] = Field(
        default_factory=dict,
        description="All pages indexed by (document_id, page_number) tuple for fast lookup",
    )

    terms: Dict[str, TermWithLinks] = Field(
        default_factory=dict,
        description="All validated terms (keywords and acronyms) with their meanings",
    )

    # Flattened chunks structure - direct access by doc_id
    chunks: Dict[str, KnowledgeChunkWithTerms] = Field(
        default_factory=dict,
        description="All chunks indexed by their unique doc_id (document_id_p\\{page\\}_c\\{chunk_index\\})",
    )

    # Keep only the term-to-chunks index for O(1) term lookups
    term_to_chunks_index: Dict[str, Set[Tuple[str, int]]] = Field(
        default_factory=dict,
        description="Index mapping terms to (document_id, chunk_index) tuples for fast lookup",
    )

    # Summary statistics - mandatory fields
    total_acronyms: int = Field(description="Total count of acronyms across all documents")
    total_keywords: int = Field(description="Total count of keywords across all documents")
    total_chunks: int = Field(description="Total count of chunks across all documents")
    total_images: Optional[int] = Field(default=0, description="Total count of images across all documents")
    total_tables: Optional[int] = Field(default=0, description="Total count of tables across all documents")

    # Processing metadata
    extraction_timestamp: Optional[str] = Field(default=None, description="Timestamp when extraction was completed")
    processing_duration: Optional[str] = Field(default=None, description="Total processing duration for the extraction")

    def get_extraction_details(self, base_url: str = "") -> Dict[str, Any]:
        """Get comprehensive extraction details in a structured format.

        Args:
            base_url: Optional base URL for document hrefs

        Returns:
            Dictionary containing all extraction details
        """
        from urllib.parse import quote

        return {
            "terms_count": len(self.terms),
            "documents_count": len(self.documents),
            "all_keywords_count": self.total_keywords,
            "all_acronyms_count": self.total_acronyms,
            "all_images_count": self.total_images,
            "all_tables_count": self.total_tables,
            "all_chunks_count": self.total_chunks,
            "documents": [
                {
                    "document_filename": doc.document_filename,
                    "document_author": doc.author,
                    "document_title": doc.title,
                    "document_pages": doc.total_pages,
                    "document_chunks": doc.total_chunks,
                    "document_images": doc.total_images,
                    "document_tables": doc.total_tables,
                    "document_terms": doc.total_terms,
                    "document_publication_date": doc.publication_date,
                    "document_href": (
                        f"{base_url}/{quote(doc.document_path, safe='/')}" if base_url else doc.document_path
                    ),
                }
                for doc in self.documents.values()
                if doc
            ],
        }

    def get_extraction_summary(self, detailed: bool = False, base_url: str = "") -> str:
        """Get a formatted markdown summary of the extraction using MarkdownGenerator.

        Args:
            detailed: If True, include document-level details and top terms
            base_url: Base URL for document links (optional)

        Returns:
            Formatted markdown summary
        """
        from .MarkdownGenerator import MarkdownGenerator

        # Use the comprehensive report generator for detailed view
        if detailed:
            return MarkdownGenerator.create_extraction_report(linked_knowledge=self, include_all_sections=True)

        # For non-detailed, create a simpler summary using MarkdownGenerator
        from urllib.parse import quote

        lines = []
        lines.append("# 📚 Knowledge Base Summary")
        lines.append("")

        # Overall statistics section using MarkdownGenerator
        stats = {
            "Total Documents": len(self.documents),
            "Total Chunks": self.total_chunks,
            "Total Terms": len(self.terms),
            "Keywords": self.total_keywords,
            "Acronyms": self.total_acronyms,
            "Total Images": self.total_images,
            "Total Tables": self.total_tables,
        }

        lines.append(MarkdownGenerator.create_statistics_card("📊 Overall Statistics", stats))
        lines.append("")

        # Content density metrics
        if self.documents:
            avg_chunks_per_doc = self.total_chunks / len(self.documents)
            avg_terms_per_doc = len(self.terms) / len(self.documents)
            total_pages = sum(doc.total_pages for doc in self.documents.values())
            avg_pages_per_doc = total_pages / len(self.documents) if self.documents else 0

            content_stats = {
                "Average chunks per document": f"{avg_chunks_per_doc:.1f}",
                "Average terms per document": f"{avg_terms_per_doc:.1f}",
                "Average pages per document": f"{avg_pages_per_doc:.1f}",
                "Total pages across all documents": f"{total_pages:,}",
            }

            lines.append(MarkdownGenerator.create_statistics_card("📈 Content Metrics", content_stats))
            lines.append("")

        # Documents table using MarkdownGenerator
        if self.documents:
            lines.append("## 📄 Document Details")
            lines.append("")

            headers = [
                "Document",
                "Title",
                "Author",
                "Published",
                "Pages",
                "Chunks",
                "Keywords",
                "Acronyms",
                "Images",
                "Tables",
            ]
            rows = []

            for doc in sorted(self.documents.values(), key=lambda d: d.document_filename):
                # Create link if base_url provided
                if base_url:
                    doc_link = f"[{doc.document_filename}]({base_url}/{quote(doc.document_path, safe='/')})"
                else:
                    doc_link = doc.document_filename

                # Handle N/A values
                title = doc.title if doc.title and doc.title != "N/A" else "-"
                author = doc.author if doc.author and doc.author != "N/A" else "-"
                pub_date = doc.publication_date if doc.publication_date and doc.publication_date != "N/A" else "-"

                rows.append(
                    [
                        doc_link,
                        title,
                        author,
                        pub_date,
                        str(doc.total_pages),
                        str(doc.total_chunks),
                        str(doc.total_keywords),
                        str(doc.total_acronyms),
                        str(doc.total_images),
                        str(doc.total_tables),
                    ]
                )

            lines.append(MarkdownGenerator.create_table(headers, rows))
            lines.append("")

        # Processing metadata if available
        if self.extraction_timestamp or self.processing_duration:
            lines.append("## ⏱️ Processing Information")
            lines.append("")
            processing_info = []
            if self.extraction_timestamp:
                processing_info.append(f"**Extraction completed:** {self.extraction_timestamp}")
            if self.processing_duration:
                processing_info.append(f"**Processing duration:** {self.processing_duration}")
            lines.extend([f"- {info}" for info in processing_info])
            lines.append("")

        lines.append("---")
        lines.append("")
        lines.append("*Generated by Catalyst Knowledge Extraction System*")

        return "\n".join(lines)

    @staticmethod
    def _normalize_term(term: str) -> str:
        """Normalize a term by lowercasing and removing unwanted characters."""
        # Convert to lowercase
        normalized = term.lower()

        # Remove emojis using pattern
        emoji_pattern = re.compile(
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
            "]+",
            flags=re.UNICODE,
        )
        normalized = emoji_pattern.sub("", normalized)

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

    @classmethod
    def _build_term_to_chunks_index(cls, terms: Dict[str, TermWithLinks]) -> Dict[str, Set[Tuple[str, int]]]:
        """Build inverted index for O(1) term-to-chunk lookups."""
        term_to_chunks: Dict[str, Set[Tuple[str, int]]] = defaultdict(set)

        # Build inverted index from term occurrences
        for term_name, term_data in terms.items():
            # Normalize the term name for consistent lookups
            normalized_term = cls._normalize_term(term_name)

            # Add all occurrences for this term
            for occurrence in term_data.occurrences:
                # Add to chunks index
                term_to_chunks[normalized_term].add((occurrence.document_id, occurrence.chunk_index))

        # Convert defaultdict to regular dict for serialization
        return dict(term_to_chunks)

    @classmethod
    def _build_chunks_with_terms(
        cls,
        terms: Dict[str, TermWithLinks],
        document_chunks: Dict[str, list[KnowledgeChunk]],
    ) -> Dict[str, KnowledgeChunkWithTerms]:
        """Build flattened chunks structure with term metadata."""
        chunks_dict: Dict[str, KnowledgeChunkWithTerms] = {}

        for doc_id, chunk_list in document_chunks.items():
            for chunk in chunk_list:
                # Find terms that appear in this chunk
                chunk_terms: Dict[str, int] = {}
                for term_name, term_data in terms.items():
                    # Check if this term appears in this specific chunk
                    for occurrence in term_data.occurrences:
                        if (
                            occurrence.document_id == doc_id
                            and occurrence.page == chunk.page
                            and occurrence.chunk_index == chunk.index
                        ):
                            maybe_count = chunk_terms.get(term_name)
                            current_count = maybe_count if maybe_count else 0
                            chunk_terms[term_name] = current_count + 1

                # Create KnowledgeChunkWithTerms with all fields from base chunk
                chunk_with_meta = KnowledgeChunkWithTerms(
                    document_id=chunk.document_id,
                    document_name=chunk.document_name,
                    doc_id=chunk.doc_id,
                    index=chunk.index,
                    text=chunk.text,
                    page=chunk.page,
                    terms=chunk_terms,
                )

                # Add to flattened chunks dict
                chunks_dict[chunk.doc_id] = chunk_with_meta

        return chunks_dict

    @classmethod
    def from_extraction_data(cls, data: RawExtractionData) -> "LinkedKnowledge":
        """Build LinkedKnowledge from raw extraction data."""
        documents: Dict[str, NormalizedDocumentMetadata] = {}
        pages_index: Dict[Tuple[str, int], KnowledgePageData] = {}

        # Build chunks with their terms
        chunks = cls._build_chunks_with_terms(data.terms, data.document_to_chunks_index)

        # Build the term-to-chunks index
        term_to_chunks_index = cls._build_term_to_chunks_index(data.terms)

        # Process results to build documents and pages
        for result in data.results_with_chunks:
            # Build pages index
            for page in result.pages:
                # Create page key: (document_id, page_number) tuple
                page_key = (result.id, page.page)

                # Store page without raw_text in the pages index
                page_data = KnowledgePageData(
                    page=page.page,
                    text=page.text,
                    tables=page.tables,
                    images=page.images,
                )
                pages_index[page_key] = page_data

            # Calculate total tables and images from pages
            total_tables = sum(len(page.tables) for page in result.pages)
            total_images = sum(len(page.images) for page in result.pages)

            # Calculate term counts for this document
            doc_terms = set()
            acronyms_count = 0
            keywords_count = 0

            for term_key, term in data.terms.items():
                # Check if this term appears in this document
                if any(occ.document_id == result.id for occ in term.occurrences):
                    doc_terms.add(term_key)
                    if term.type == "acronym":
                        acronyms_count += 1
                    elif term.type == "keyword":
                        keywords_count += 1

            documents[result.id] = NormalizedDocumentMetadata(
                document_id=result.id,
                document_filename=result.document_filename,
                document_path=result.document_metadata.document_path,
                total_pages=len(result.pages),
                total_chunks=result.total_chunks,
                total_terms=len(doc_terms),
                title=(
                    result.document_metadata.title
                    if result.document_metadata and result.document_metadata.title
                    else "N/A"
                ),
                subject=(
                    result.document_metadata.subject
                    if result.document_metadata and result.document_metadata.subject
                    else "N/A"
                ),
                author=(
                    result.document_metadata.author
                    if result.document_metadata and result.document_metadata.author
                    else "N/A"
                ),
                modification_date=result.document_metadata.modification_date if result.document_metadata else "N/A",
                publication_date=result.document_metadata.creation_date if result.document_metadata else "N/A",
                total_acronyms=acronyms_count,
                total_keywords=keywords_count,
                total_tables=total_tables,
                total_images=total_images,
            )

        # Calculate total statistics across all documents
        total_acronyms_count = sum(1 for term in data.terms.values() if term.type == "acronym")
        total_keywords_count = sum(1 for term in data.terms.values() if term.type == "keyword")
        total_chunks_count = len(chunks)
        total_images_count = sum(doc.total_images for doc in documents.values())
        total_tables_count = sum(doc.total_tables for doc in documents.values())

        # Create and return LinkedKnowledge object with only necessary indices
        return cls(
            documents=documents,
            pages=pages_index,
            terms=data.terms,
            chunks=chunks,
            term_to_chunks_index=term_to_chunks_index,
            total_acronyms=total_acronyms_count,
            total_keywords=total_keywords_count,
            total_chunks=total_chunks_count,
            total_images=total_images_count,
            total_tables=total_tables_count,
        )


# ============================================================================
# Settings Types
# ============================================================================


class KnowledgeProcessorSettings(BaseModel):
    """Unified settings for knowledge processing (PDF, DOCX, TXT, etc.)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    pdf_settings: PDFKnowledgeProcessorSettings = Field(
        default_factory=lambda: PDFKnowledgeProcessorSettings(),
        description="Settings specific to PDF processing",
    )

    min_term_score: float = Field(
        default=0.0,
        description="Minimum term score for keyword extraction",
    )

    encoding_model: str = Field(
        default="o200k_base",
        description="Encoding model to use for text splitting and tokenization",
    )

    extraction_output_dir: Path = Field(
        default=Path("public/knowledge_extraction"),
        description="Directory to store extraction output",
    )

    max_display_occurrences: int = Field(
        default=15,
        description="Maximum occurrences to display per term",
        ge=1,
    )

    max_display_cooccurrences: int = Field(
        default=5,
        description="Maximum cooccurrences to display per term",
        ge=1,
    )

    image_optimization_level: int = Field(
        default=6,
        description="PNG optimization level (0-6) for extracted images",
        ge=0,
        le=6,
    )

    keywords_min_df: int = Field(
        default=5,
        description="Minimum document frequency for keywords to be considered",
        ge=4,
    )

    acronyms_min_df: int = Field(
        default=1,
        description="Minimum document frequency for acronyms to be considered",
        ge=1,
    )

    linking_threshold: float = Field(
        default=0.70,
        description="Minimum similarity score to link acronyms with keywords (0.0-1.0)",
        ge=0.0,
        le=1.0,
    )
