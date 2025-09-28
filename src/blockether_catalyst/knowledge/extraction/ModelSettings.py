"""
Model settings for knowledge extraction calls.

This module defines the configuration for LLM calls in the extraction pipeline.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class ModelSettings(BaseModel):
    """Settings for a single model in extraction calls."""

    model: str = Field(description="Model identifier (e.g., 'gpt-4o')")
    api_url: Optional[str] = Field(default=None, description="Optional API URL override")
    api_key: Optional[str] = Field(default=None, description="Optional API key override")
    temperature: float = Field(default=0.0, description="Temperature for sampling")
    weight: float = Field(default=1.0, description="Weight multiplier for consensus")
    perspective: str = Field(description="Model perspective for prompts")


class ConsensusSettings(BaseModel):
    """Settings for consensus-based extraction calls."""

    models: List[ModelSettings] = Field(description="List of models to use for consensus")
    consensus_threshold: float = Field(default=0.8, description="Threshold for consensus agreement")
    max_rounds: int = Field(default=3, description="Maximum consensus rounds")


class ExtractionModelSettings(BaseModel):
    """Complete model settings for all extraction calls."""

    term_extraction: ConsensusSettings = Field(description="Settings for term extraction calls")
    document_chunking: ConsensusSettings = Field(description="Settings for document chunking calls")
    chunk_classification: ConsensusSettings = Field(description="Settings for chunk content classification calls")
    table_caption: ConsensusSettings = Field(description="MANDATORY settings for table caption extraction")

    @classmethod
    def default_settings(cls) -> "ExtractionModelSettings":
        """Create default extraction settings with 3 models for consensus."""
        # Create 3 models with different perspectives for meaningful consensus
        model1 = ModelSettings(
            model="gpt-4o",
            api_url="http://localhost:3005/v1",
            temperature=0.0,
            weight=1.0,
            perspective="Focus on extracting concrete facts and precise definitions. Be extremely succinct - use minimal words while preserving accuracy. Prioritize technical precision.",
        )

        model2 = ModelSettings(
            model="gpt-4o",
            api_url="http://localhost:3005/v1",
            temperature=0.1,
            weight=1.0,
            perspective="Extract factual information with emphasis on contextual relevance. Keep responses brief and concrete. Avoid generalizations - focus on specific, verifiable details.",
        )

        model3 = ModelSettings(
            model="gpt-4o",
            api_url="http://localhost:3005/v1",
            temperature=0.0,
            weight=1.0,
            perspective="Identify key facts and structural relationships. Provide concise, objective summaries. Every word must add value - eliminate redundancy and filler content.",
        )

        three_model_consensus = ConsensusSettings(
            models=[model1, model2, model3],
            consensus_threshold=0.65,  # 2 out of 3 models must agree
            max_rounds=3,
        )

        return cls(
            term_extraction=three_model_consensus,
            document_chunking=three_model_consensus,
            chunk_classification=three_model_consensus,
            table_caption=three_model_consensus,  # MANDATORY table caption extraction
        )
