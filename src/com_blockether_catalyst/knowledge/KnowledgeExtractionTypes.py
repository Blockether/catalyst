from __future__ import annotations

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

from pydantic import BaseModel, Field, RootModel, computed_field

from com_blockether_catalyst.consensus.ConsensusTypes import TypedCallBaseForConsensus
from com_blockether_catalyst.consensus.VotingComparison import BaseModelWithReasoning, ComparisonStrategy, VotingField


from .PDKnowledgeExtractorTypes import PDFKnowledgeProcessorSettings


class TermCooccurrence(BaseModel):
    """Represents co-occurrence of terms."""

    term: str = Field(description="The co-occurring term")
    score: float = Field(default=0.0, description="Confidence score of co-occurrence (0-1)")


class TermMeaningExtractionResponse(TypedCallBaseForConsensus):
    """Response model for term meaning extraction."""

    term: str = VotingField(
        comparison=ComparisonStrategy.EXACT,
        description="The term text (acronym or keyword)",
        threshold=1.0
    )

    full_form: Optional[str] = VotingField(
        comparison=ComparisonStrategy.EXACT,
        description="The expanded full form (same as term for keywords)",
        threshold=1.0
    )

    meaning: Optional[str] = VotingField(
        comparison=ComparisonStrategy.SEMANTIC,
        description="The extracted meaning of the term",
        threshold=0.6
    )

    type: Literal["acronym", "keyword", "unknown"] = VotingField(
        comparison=ComparisonStrategy.EXACT,
        description="Type of term: 'acronym' or 'keyword'",
        threshold=1.0
    )


class ChunkOutput(RootModel):
    """Output model for a single chunk created from a page."""
    root: str = VotingField(
        comparison=ComparisonStrategy.SEMANTIC,
        threshold=0.75,
        description="Text content of the chunk")


class ChunkingDecision(TypedCallBaseForConsensus):
    """Response model for intelligent chunking of document pages."""

    chunks: List[ChunkOutput] = VotingField(
        comparison=ComparisonStrategy.DERIVED,
        description="Sequence of chunks to create from the provided text, each with proper boundaries",
        threshold=0.7
    )

    @property
    @computed_field
    def total_chunks(self) -> int:
        """Get the total number of chunks created."""
        return len(self.chunks)


class KnowledgeMetadata(BaseModel):
    """Document metadata extracted from PDF."""

    title: Optional[str] = Field(default=None,
                                 description="Document title")
    author: Optional[str] = Field(default=None,
                                  description="Document author")
    subject: Optional[str] = Field(default=None,
                                   description="Document subject")
    creation_date: str = Field(default="",
                               description="Document creation date")
    modification_date: str = Field(default="",
                                   description="Document last modification date")


class KnowledgePageData(BaseModel):
    images: List[ImageMetadata] = Field(
        default_factory=list,
        description="List of images extracted from the page",
    )
    tables: List[KnowledgeTableData] = Field(
        default_factory=list,
        description="List of tables extracted from the page",
    )
    page: int = Field(description="1-indexed page number")
    text: str = Field(default="", description="Extracted text content of the page")


class KnowledgePageDataWithRawText(KnowledgePageData):
    raw_text: str = Field(default="", description="Raw text including tables and OCRed image text inline")
    lines: int = Field(default=0, description="Line count (calculated from raw_text)")


DocumentSourceType = Literal["pdf", "docx", "txt"]


class KnowledgeExtractionResultBase(BaseModel):
    id: str = Field(description="SHA256 hash of the file")
    filename: str = Field(description="filename")
    metadata: KnowledgeMetadata = Field(default_factory=KnowledgeMetadata, description="metadata")
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


class KnowledgeExtractionItem(BaseModel):
    """Represents a single PDF extraction result or error."""

    result: Optional[KnowledgeExtractionResult] = Field(default=None, description="Successful extraction result")
    error: Optional[str] = Field(default=None, description="Error message if extraction failed")


class KnowledgeExtractionOutput(BaseModel):
    """Complete output from knowledge extraction process."""

    pdf: Optional[Sequence[KnowledgeExtractionItem]] = Field(default=None)
    # Future: Add other file types as needed
    # docx: Optional[Sequence[KnowledgeExtractionItem]] = field(default=None)
    # txt: Optional[Sequence[KnowledgeExtractionItem]] = field(default=None)
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="ISO format timestamp of when extraction was performed",
    )


class KnowledgeChunk(BaseModel):
    """Represents a text chunk with keyword indexing."""

    document_id: str = Field(description="SHA-256 hash identifier of the document")
    document_name: str = Field(description="Name of the source document")
    doc_id: str = Field(description="Unique chunk identifier combining document_id, page, and chunk_index")
    index: int = Field(description="Index of the chunk")
    text: str = Field(description="Text content of the chunk")
    page: int = Field(description="Page number (1-indexed) - can be used to retrieve images/tables from page data")


class ImageMetadata(BaseModel):
    """Metadata for an extracted image from a document."""

    document_name: str = Field(description="Name of the source document")
    page: int = Field(description="Page number where the image was found")
    href: str = Field(description="Path or reference to the saved image file")
    caption: Optional[str] = Field(default=None, description="AI-generated caption describing the image")


class KnowledgeExtractionResultWithChunks(KnowledgeExtractionResult):
    """
    Represents a chunked extraction result with keyword indexing.

    Inherits from KnowledgeExtractionResult and adds chunking information.
    Each chunk contains indexed keywords, acronyms, and key terms.

    Attributes:
        chunks: Sequence of text chunks with keyword indexing
        chunk_size: Size of each chunk in characters
        chunk_overlap: Not used for agentic chunking (always 0)
    """

    chunks: Sequence[KnowledgeChunk] = Field(
        default_factory=list,
        description="Sequence of text chunks with keyword indexing",
    )
    total_chunks: int = Field(default=0, description="Total number of chunks created")


class TermOccurrence(BaseModel):
    """Represents a term occurrence within a document chunk."""

    document_id: str = Field(description="SHA-256 hash identifier of the document")
    document_name: str = Field(description="Name of the source document")
    page: int = Field(description="Page number (1-indexed) where the term was found")
    chunk_index: int = Field(description="Index of the chunk where the term was found")
    total: int = Field(description="Total occurrences of the term in the document")


class KnowledgeExtractionOutputWithChunks(KnowledgeExtractionItem):
    """
    Represents the output of the knowledge extraction process with chunking and keyword indexing.

    Inherits from KnowledgeExtractionItem and adds chunking information.
    Final stage where documents are chunked with keywords indexed in each chunk.

    Attributes:
        chunk_size: Size of each chunk in characters
        chunk_overlap: Not used for agentic chunking (always 0)
    """

    chunk_size: int = Field(default=0, description="Size of each chunk in characters")
    chunk_overlap: int = Field(default=0, description="Not used for agentic chunking (always 0)")


class KnowledgeProcessorSettings(BaseModel):
    """Unified settings for knowledge processing (PDF, DOCX, TXT, etc.)."""

    model_config = {"arbitrary_types_allowed": True}

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
        default=0.65,
        description="Minimum similarity score to link acronyms with keywords (0.0-1.0)",
        ge=0.0,
        le=1.0,
    )


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


class TermCandidate(BaseModel):
    """Candidate term during extraction phase (before consolidation)."""

    document_name: str = Field(description="Name of the document")
    document_id: str = Field(description="SHA-256 identifier of the document")
    term: str = Field(description="The term text")
    page: int = Field(description="Page number where this term appears")
    chunk: int = Field(description="Chunk index where this term appears")
    total: int = Field(description="Total occurrences of this term in the document")
    type: Literal["acronym", "keyword", "unknown"] = Field(description="Type of term")


class TermCandidateGrouped(BaseModel):
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
    occurrences: Sequence[TermOccurrence] = Field(
        default_factory=list,
        description="All occurrences of this term across documents",
    )
    cooccurrences: Sequence[TermCooccurrence] = Field(
        default_factory=list,
        description="Terms that frequently co-occur with this term",
    )
    total: int = Field(default=0, description="Total number of times this term appears")
    full_form: str = Field(description="The expanded full form (same as term for keywords)")
    meaning: Optional[str] = Field(default=None, description="The extracted meaning of the term")


class TermLink(BaseModel):
    """Represents a link between an acronym and its corresponding keyword/term."""

    link_from: str = Field(description="Term text from where the link originates")
    link_to: str = Field(description="Term text to which term is linked")
    score: float = Field(description="Matching score between acronym full form and keyword (0-1)")


class TermWithLinks(Term):
    links: List[TermLink] = Field(
        default_factory=list,
        description="Links from this acronym to its corresponding keywords")


class KnowledgeChunkWithTerms(KnowledgeChunk):
    """Knowledge chunk with terms for search and indexing."""

    terms: Sequence[str] = Field(
        default_factory=list,
        description="Terms (keywords/acronyms) found in this chunk",
    )


class DocumentMetadata(BaseModel):
    """Simplified document metadata for normalized graph structure."""

    document_id: str = Field(description="SHA-256 hash identifier of the document")
    filename: str = Field(description="Original filename of the document")
    total_pages: int = Field(description="Total number of pages in document")
    total_chunks: int = Field(description="Total number of chunks created from document")
    total_terms: int = Field(description="Total number of unique terms found in document")
    total_tables: int = Field(description="Total number of tables found in document")
    total_acronyms: int = Field(description="Total number of unique acronyms found in document")
    total_keywords: int = Field(description="Total number of unique keywords found in document")


class LinkedKnowledge(BaseModel):
    """Comprehensive container for all extracted knowledge and relationships."""

    # Core extracted content - simplified to just metadata
    documents: Dict[str, DocumentMetadata] = Field(
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

    # Chunk lookup indices
    document_to_chunk_ids_index: Dict[str, Set[str]] = Field(
        default_factory=dict,
        description="Index mapping document_id to set of chunk doc_ids for that document",
    )

    document_page_to_chunks_index: Dict[Tuple[str, int], Set[str]] = Field(
        default_factory=dict,
        description="Index mapping (document_id, page) to set of chunk doc_ids on that page",
    )

    # Inverted indices for O(1) term-to-chunk lookups
    term_to_chunks_index: Dict[str, Set[Tuple[str, int]]] = Field(
        default_factory=dict,
        description="Index mapping terms to (document_id, chunk_index) tuples for fast lookup",
    )

    term_to_document_with_page_index: Dict[str, Set[Tuple[str, int]]] = Field(
        default_factory=dict,
        description="Index mapping terms to (document_id, page) tuples for fast page-level lookup",
    )

    term_to_documents_index: Dict[str, Set[str]] = Field(
        default_factory=dict,
        description="Index mapping terms to document IDs for fast lookup",
    )

    document_to_terms_index: Dict[str, Set[str]] = Field(
        default_factory=dict,
        description="Index mapping document IDs to terms for fast lookup",
    )

    # Summary statistics - mandatory fields
    total_acronyms: int = Field(description="Total count of acronyms across all documents")
    total_keywords: int = Field(description="Total count of keywords across all documents")
    total_chunks: int = Field(description="Total count of chunks across all documents")
