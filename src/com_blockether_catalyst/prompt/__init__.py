"""
Prompt alignment and optimization module.

This module provides functionality for automatically aligning and optimizing
prompts based on feedback, using TypedCalls for structured interactions.
"""

from .PromptAlignmentCore import (
    AlignmentResult,
    PromptAlignmentCore,
    PromptConfiguration,
)

__all__ = [
    "PromptAlignmentCore",
    "PromptConfiguration",
    "AlignmentResult",
]
