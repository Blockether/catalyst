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

from pydantic import BaseModel, Field, computed_field

from com_blockether_catalyst.consensus import TypedCallBaseForConsensus

# Import base types to avoid duplication and circular dependencies
from .KnowledgeExtractionBaseTypes import (
    KnowledgeMetadata,
    KnowledgePageData,
    TermCooccurrence,
)

# Import base call classes directly (no circular dependency now)
from .KnowledgeExtractionCallBase import (
    BaseAcronymExtractionCall,
    BaseChunkAcronymExtractionCall,
    BaseChunkingCall,
    BaseChunkKeywordExtractionCall,
    BaseKeywordExtractionCall,
)
from .PDKnowledgeExtractorTypes import (
    PDFImageProcessingSettings,
    PDFPageCropOffset,
    PDFProcessorTableExtractionSettings,
    PDFProcessorTextExtractionSettings,
)


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


class AgenticChunkingRequest(BaseModel):
    """Request model for agentic chunking of full pages."""

    pages_text: str = Field(description="Full text of 1-2 pages to be chunked")
    page_numbers: Sequence[int] = Field(description="Page numbers being chunked (e.g., [1, 2] or [3])")
    target_chunk_size: int = Field(description="Target size for each chunk in characters")
    max_chunk_size: int = Field(description="Maximum allowed chunk size")
    document_context: str = Field(description="Metadata and context about the document")
    preserve_structures: bool = Field(
        default=True,
        description="Whether to preserve tables, lists, and other structures intact",
    )


class KnowledgeChunk(BaseModel):
    """Represents a text chunk with keyword indexing."""

    document_id: str = Field(description="SHA-256 hash identifier of the document")
    document_name: str = Field(description="Name of the source document")
    doc_id: str = Field(description="Unique chunk identifier combining document_id, page, and chunk_index")
    index: int = Field(description="Index of the chunk")
    text: str = Field(description="Text content of the chunk")
    page: int = Field(description="Page number (1-indexed) - can be used to retrieve images/tables from page data")


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

    # PDF-specific settings with proper typing
    pdf_table_extraction: Optional[PDFProcessorTableExtractionSettings] = Field(
        default_factory=lambda: None,
        description="PDF table extraction settings",
    )
    pdf_text_extraction: Optional[PDFProcessorTextExtractionSettings] = Field(
        default_factory=lambda: None,
        description="PDF text extraction settings",
    )
    pdf_image_processing: Optional[PDFImageProcessingSettings] = Field(
        default_factory=lambda: None,
        description="PDF image processing settings",
    )
    pdf_page_crop_offset: Optional[PDFPageCropOffset] = Field(
        default_factory=lambda: None,
        description="PDF page crop offsets in pixels for headers/footers",
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

    acronym_extraction_call: BaseAcronymExtractionCall = Field(
        description="User-implemented call for acronym validation/meaning extraction",
    )

    keyword_extraction_call: BaseKeywordExtractionCall = Field(
        description="User-implemented call for keyword validation/meaning extraction",
    )

    chunking_call: BaseChunkingCall = Field(
        description="User-implemented call for document chunking",
    )

    chunk_acronym_extraction_call: BaseChunkAcronymExtractionCall = Field(
        description="User-implemented call for initial acronym discovery in chunks (MANDATORY)",
    )

    chunk_keyword_extraction_call: BaseChunkKeywordExtractionCall = Field(
        description="User-implemented call for initial keyword discovery in chunks (MANDATORY)",
    )

    # LLM prompt limits
    max_llm_tokens: int = Field(
        default=100000,
        description="Maximum tokens for LLM prompts (default: 100k for GPT-4)",
        ge=1000,
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

    linking_threshold: float = Field(
        default=0.65,
        description="Minimum similarity score to link acronyms with keywords (0.0-1.0)",
        ge=0.0,
        le=1.0,
    )


class AcronymCandidate(BaseModel):
    document_id: str = Field(description="SHA-256 identifier of the document")
    document_name: str = Field(description="Name of the document")
    term: str = Field(description="The term text (acronym candidate)")
    chunk: int = Field(description="Chunk number where the term is found")
    page: int = Field(description="Page number where the term is found")


class TermOccurrence(BaseModel):
    """Represents a single occurrence of a term (keyword or acronym) in a document."""

    document_id: str = Field(description="SHA-256 identifier of the document")
    document_name: str = Field(description="Name of the document")
    chunk_index: int = Field(description="Index of the chunk where term appears")
    page: int = Field(description="Page number where term appears")


class KeywordCandidate(BaseModel):
    """Candidate term during extraction phase (before consolidation)."""

    document_name: str = Field(description="Name of the document")
    document_id: str = Field(description="SHA-256 identifier of the document")
    term: str = Field(description="The term text")
    score: float = Field(description="Relevance score of the term")
    page: int = Field(description="Page number where this term appears")
    chunk: int = Field(description="Chunk index where this term appears")


class GroupedAcronym(BaseModel):
    term: str = Field(description="The term text (acronym candidate)")
    occurrences: List[TermOccurrence] = Field(
        default_factory=list,
        description="All occurrences of this term across documents",
    )
    cooccurrences: List[TermCooccurrence] = Field(
        default_factory=list,
        description="Terms that frequently co-occur with this term",
    )
    total_count: int = Field(default=0, description="Total number of times this term appears")
    mean_score: float = Field(default=0.0, description="Mean relevance score across all occurrences")


class Term(BaseModel):
    """Represents a consolidated term (acronym or keyword) with validation and meaning."""

    term: str = Field(description="The term text (acronym or keyword)")
    term_type: str = Field(description="Type of term: 'acronym' or 'keyword'")
    occurrences: Sequence[TermOccurrence] = Field(
        default_factory=list,
        description="All occurrences of this term across documents",
    )
    cooccurrences: Sequence[TermCooccurrence] = Field(
        default_factory=list,
        description="Terms that frequently co-occur with this term",
    )
    total_count: int = Field(default=0, description="Total number of times this term appears")
    mean_score: float = Field(default=0.0, description="Mean relevance score (for keywords)")
    reasoning: str = Field(default="", description="LLM reasoning for the term meaning extraction")
    full_form: str = Field(description="The expanded full form (same as term for keywords)")
    meaning: Optional[str] = Field(default=None, description="The extracted meaning of the term")


class GroupedKeyword(BaseModel):
    """Represents grouped term candidates with the same term text."""

    term: str = Field(description="The term text")
    occurrences: List[TermOccurrence] = Field(
        default_factory=list,
        description="All occurrences of this term across documents",
    )
    cooccurrences: List[TermCooccurrence] = Field(
        default_factory=list,
        description="Terms that frequently co-occur with this term",
    )
    total_count: int = Field(default=0, description="Total number of times this term appears")
    mean_score: float = Field(default=0.0, description="Mean relevance score across all occurrences")


class ExtractedAcronym(BaseModel):
    """Single extracted acronym from chunk."""

    term: str = Field(description="The acronym itself (e.g., 'API', 'HTTP')")
    full_form: str = Field(description="The full form of the acronym (e.g., 'Application Programming Interface')")
    rationale: str = Field(description="Explanation of why this is considered an acronym")


class ExtractedKeyword(BaseModel):
    """Single extracted keyword from chunk."""

    term: str = Field(description="The keyword/term itself")
    rationale: str = Field(description="Explanation of why this is considered an important keyword")


class ChunkAcronymExtractionResponse(TypedCallBaseForConsensus):
    """Response for extracting acronyms from a chunk of text."""

    acronyms: List[ExtractedAcronym] = Field(
        default_factory=list,
        description="List of acronyms found in the chunk with their full forms",
    )


class TermLink(BaseModel):
    """Represents a link between an acronym and its corresponding keyword/term."""

    acronym: str = Field(description="Acronym text of the linked acronym")
    keyword: str = Field(description="Key of the linked keyword/term")
    match_score: float = Field(description="Matching score between acronym full form and keyword (0-1)")


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

    # All terms stored as Union type - can be either Keyword or Acronym
    terms: Dict[str, Term] = Field(
        default_factory=dict,
        description="All validated terms (keywords and acronyms) with their meanings",
    )

    # Relationships between terms
    links: Sequence[TermLink] = Field(
        default_factory=list,
        description="Links between acronyms and their corresponding keywords",
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

    def model_dump(self, **kwargs: Any) -> Dict[str, Any]:
        """Custom serialization to handle tuple keys for JSON compatibility."""
        data = super().model_dump(**kwargs)

        # Convert tuple keys in pages to string keys
        if "pages" in data:
            pages_with_string_keys = {}
            for key, value in data["pages"].items():
                # Convert tuple key to string format
                string_key = f"{key[0]}_{key[1]}" if isinstance(key, tuple) else str(key)
                pages_with_string_keys[string_key] = value
            data["pages"] = pages_with_string_keys

        # Convert tuple keys in document_page_to_chunks_index
        if "document_page_to_chunks_index" in data:
            converted = {}
            for key, value in data["document_page_to_chunks_index"].items():
                string_key = f"{key[0]}_{key[1]}" if isinstance(key, tuple) else str(key)
                # Convert set to list for JSON serialization
                converted[string_key] = list(value) if isinstance(value, set) else value
            data["document_page_to_chunks_index"] = converted

        # Convert sets with tuples in term_to_chunks_index
        if "term_to_chunks_index" in data:
            converted = {}
            for term, chunks_set in data["term_to_chunks_index"].items():
                # Convert set of tuples to list of strings
                converted[term] = [f"{doc}_{idx}" for doc, idx in chunks_set] if chunks_set else []
            data["term_to_chunks_index"] = converted

        # Convert sets with tuples in term_to_document_with_page_index
        if "term_to_document_with_page_index" in data:
            converted = {}
            for term, docs_set in data["term_to_document_with_page_index"].items():
                # Convert set of tuples to list of strings
                converted[term] = [f"{doc}_{page}" for doc, page in docs_set] if docs_set else []
            data["term_to_document_with_page_index"] = converted

        # Convert sets to lists for other indices
        for field in [
            "document_to_chunk_ids_index",
            "term_to_documents_index",
            "document_to_terms_index",
        ]:
            if field in data:
                for key, value in data[field].items():
                    if isinstance(value, set):
                        data[field][key] = list(value)

        return data
