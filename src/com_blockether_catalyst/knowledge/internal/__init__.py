# Import from base types
from .KnowledgeExtractionBaseTypes import (
    AcronymMeaningExtractionResponse,
    ChunkAcronymExtractionResponse,
    ChunkingDecision,
    ChunkKeywordExtractionResponse,
    ChunkOutput,
    ExtractedAcronym,
    ExtractedKeyword,
    KeywordMeaningExtractionResponse,
    KnowledgeMetadata,
    KnowledgePageData,
    KnowledgeTableData,
    TermCooccurrence,
)
from .KnowledgeExtractionCallBase import (
    BaseAcronymExtractionCall,
    BaseChunkAcronymExtractionCall,
    BaseChunkingCall,
    BaseChunkKeywordExtractionCall,
    BaseKeywordExtractionCall,
)

# Import from main types
from .KnowledgeExtractionTypes import (
    AgenticChunkingRequest,
    KnowledgeChunk,
    KnowledgeExtractionItem,
    KnowledgeExtractionOutput,
    KnowledgeExtractionResult,
    KnowledgeExtractionResultWithChunks,
    KnowledgePageDataWithRawText,
    KnowledgeProcessorSettings,
    TermOccurrence,
)

# Conditional imports for PDF extraction (requires easyocr, pdf2image, etc.)
try:
    from .PDFKnowledgeExtractor import PDFKnowledgeExtractor
    from .PDKnowledgeExtractorTypes import (
        PDFImageProcessingSettings,
        PDFPageCropOffset,
        PDFProcessorTableExtractionSettings,
        PDFProcessorTextExtractionSettings,
    )

    _PDF_AVAILABLE = True
except ImportError:
    # PDF extraction dependencies not available
    PDFKnowledgeExtractor = None  # type: ignore
    PDFImageProcessingSettings = None  # type: ignore
    PDFPageCropOffset = None  # type: ignore
    PDFProcessorTableExtractionSettings = None  # type: ignore
    PDFProcessorTextExtractionSettings = None  # type: ignore
    _PDF_AVAILABLE = False

# Core exports (always available)
__all__ = [
    "KnowledgeMetadata",
    "KnowledgeTableData",
    "KnowledgePageData",
    "KnowledgeExtractionResult",
    "KnowledgeExtractionItem",
    "KnowledgeExtractionOutput",
    "KnowledgeChunk",
    "KnowledgeExtractionResultWithChunks",
    "KnowledgePageDataWithRawText",
    "KnowledgeProcessorSettings",
    "TermCooccurrence",
    "TermOccurrence",
    "KeywordMeaningExtractionResponse",
    "AcronymMeaningExtractionResponse",
    "AgenticChunkingRequest",
    "ChunkingDecision",
    "ChunkOutput",
    "BaseAcronymExtractionCall",
    "BaseChunkAcronymExtractionCall",
    "BaseChunkKeywordExtractionCall",
    "BaseKeywordExtractionCall",
    "BaseChunkingCall",
    "ChunkAcronymExtractionResponse",
    "ChunkKeywordExtractionResponse",
    "ExtractedAcronym",
    "ExtractedKeyword",
]

# Add PDF-related exports if available
if _PDF_AVAILABLE:
    __all__.extend(
        [
            "PDFKnowledgeExtractor",
            "PDFImageProcessingSettings",
            "PDFPageCropOffset",
            "PDFProcessorTableExtractionSettings",
            "PDFProcessorTextExtractionSettings",
        ]
    )
