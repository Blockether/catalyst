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
    ExtractionCallsSettings,
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
    LinkedKnowledge,
    Term,
    TermLink,
    TermOccurrence,
)

# Conditional imports for PDF extraction (requires easyocr, pdf2image, etc.)
try:
    from .PDFKnowledgeExtractor import PDFKnowledgeExtractor
    from .PDKnowledgeExtractorTypes import (
        PDFImageProcessingSettings,
        PDFKnowledgeProcessorSettings,
        PDFPageCropOffset,
        PDFProcessorTableExtractionSettings,
        PDFProcessorTextExtractionSettings,
    )

    PDF_AVAILABLE = True
except ImportError:
    # PDF extraction dependencies not available
    PDFKnowledgeExtractor = None  # type: ignore
    PDFImageProcessingSettings = None  # type: ignore
    PDFPageCropOffset = None  # type: ignore
    PDFProcessorTableExtractionSettings = None  # type: ignore
    PDFProcessorTextExtractionSettings = None  # type: ignore
    PDF_AVAILABLE = False

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
    "LinkedKnowledge",
    "Term",
    "TermLink",
    "BaseAcronymExtractionCall",
    "BaseChunkAcronymExtractionCall",
    "BaseChunkKeywordExtractionCall",
    "BaseKeywordExtractionCall",
    "BaseChunkingCall",
    "ChunkAcronymExtractionResponse",
    "ChunkKeywordExtractionResponse",
    "ExtractedAcronym",
    "ExtractedKeyword",
    "PDF_AVAILABLE",
    "ExtractionCallsSettings",
]

# Add PDF-related exports if available
if PDF_AVAILABLE:
    __all__.extend(
        [
            "PDFKnowledgeExtractor",
            "PDFImageProcessingSettings",
            "PDFPageCropOffset",
            "PDFProcessorTableExtractionSettings",
            "PDFProcessorTextExtractionSettings",
            "PDFKnowledgeProcessorSettings",
        ]
    )
