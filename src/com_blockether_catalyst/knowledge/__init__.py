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
from .internal.KnowledgeExtractionTypes import (
    KnowledgeExtractionResult,
    KnowledgeExtractionResultWithChunks,
    KnowledgeMetadata,
    LinkedKnowledge,
    Term,
    TermCooccurrence,
    TermLink,
    TermOccurrence,
)
from .KnowledgeSearchCore import KnowledgeSearchCore

RAPID_FUZZ_AVAILABLE = False

try:
    import rapidfuzz  # type: ignore

    RAPID_FUZZ_AVAILABLE = True

    from .internal.KnowledgeExtractionTypes import (
        KnowledgeExtractionResult,
        KnowledgeExtractionResultWithChunks,
    )
    from .KnowledgeExtractionCore import KnowledgeExtractionCore
except ImportError:
    RAPID_FUZZ_AVAILABLE = False

__all__ = [
    "KnowledgeSearchCore",
    "LinkedKnowledge",
    "Term",
    "TermLink",
    "TermOccurrence",
    "TermCooccurrence",
    "KnowledgeMetadata",
]

if RAPID_FUZZ_AVAILABLE:
    __all__.extend(
        [
            "KnowledgeExtractionCore",
            "KnowledgeExtractionResult",
            "KnowledgeExtractionResultWithChunks",
        ]
    )
