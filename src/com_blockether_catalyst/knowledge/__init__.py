"""
Knowledge extraction and management module for Catalyst.

This module provides functionality for extracting, processing, and managing
knowledge from various document sources including PDFs, DOCX, and text files.

Core functionality (always available):
- KnowledgeSearchCore: Search and retrieval from extracted knowledge
- LinkedKnowledge: Knowledge graph data structure
- Types: Various data types for knowledge representation

Optional functionality (requires extraction dependencies):
- KnowledgeExtractionCore: Extract knowledge from documents

To install extraction dependencies:

    # If installed from PyPI:
    pip install com_blockether_catalyst[extraction]

    # If installed from GitHub:
    pip install "com_blockether_catalyst[extraction] @ git+https://github.com/blockether/com_blockether_catalyst.git"

    # If using uv from local/cloned repo:
    uv add --optional extraction
"""

# Core types that don't require extraction dependencies
from .internal import (  # Base extraction calls; Response types
    PDF_AVAILABLE,
    AcronymMeaningExtractionResponse,
    BaseAcronymExtractionCall,
    BaseChunkAcronymExtractionCall,
    BaseChunkingCall,
    BaseChunkKeywordExtractionCall,
    BaseKeywordExtractionCall,
    ChunkAcronymExtractionResponse,
    ChunkingDecision,
    ChunkKeywordExtractionResponse,
    ChunkOutput,
    ExtractionCallsSettings,
    KeywordMeaningExtractionResponse,
    KnowledgeExtractionResult,
    KnowledgeExtractionResultWithChunks,
    KnowledgeMetadata,
    KnowledgePageData,
    KnowledgeProcessorSettings,
    LinkedKnowledge,
    PDFKnowledgeProcessorSettings,
    Term,
    TermCooccurrence,
    TermLink,
    TermOccurrence,
)
from .KnowledgeSearchCore import KnowledgeSearchCore

if PDF_AVAILABLE:
    from .internal import PDFPageCropOffset
    from .KnowledgeExtractionCore import KnowledgeExtractionCore


__all__ = [
    # Core search and data structures
    "KnowledgeSearchCore",
    "LinkedKnowledge",
    "Term",
    "TermLink",
    "TermOccurrence",
    "TermCooccurrence",
    "KnowledgeMetadata",
    "KnowledgePageData",
    # Settings
    "ExtractionCallsSettings",
    "KnowledgeProcessorSettings",
    # Base extraction calls
    "BaseAcronymExtractionCall",
    "BaseKeywordExtractionCall",
    "BaseChunkingCall",
    "BaseChunkAcronymExtractionCall",
    "BaseChunkKeywordExtractionCall",
    # Response types
    "AcronymMeaningExtractionResponse",
    "ChunkAcronymExtractionResponse",
    "ChunkKeywordExtractionResponse",
    "ChunkOutput",
    "KeywordMeaningExtractionResponse",
    "ChunkingDecision",
]

if PDF_AVAILABLE:
    __all__.extend(
        [
            "KnowledgeExtractionCore",
            "KnowledgeExtractionResult",
            "KnowledgeExtractionResultWithChunks",
            "PDFKnowledgeProcessorSettings",
            "PDFPageCropOffset",
        ]
    )
