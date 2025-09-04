"""
Type definitions for prompt alignment system.

This module defines all the structured types used in the prompt alignment
process, following the patterns established in the codebase.
"""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, RootModel

from ...consensus.internal.ConsensusTypes import TypedCallBaseForConsensus
from ...consensus.internal.VotingComparison import ComparisonStrategy, VotingField


class AlignmentPrinciple(BaseModel):
    """A principle extracted from feedback for alignment."""

    principle: str = Field(description="The principle to apply for alignment")
    importance: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Importance weight of this principle",
    )


class SemanticStringList(RootModel[List[str]]):
    """A list of strings that should be compared semantically."""

    root: List[str] = VotingField(
        comparison=ComparisonStrategy.SEMANTIC,
        default_factory=list,
        threshold=0.8,  # Use semantic comparison for individual strings
    )


class EvaluationResult(TypedCallBaseForConsensus):
    """Result of evaluating a prompt against target behavior."""

    alignment_score: float = VotingField(
        comparison=ComparisonStrategy.EXACT,
        ge=0.0,
        le=1.0,
        description="How well the prompt aligns with target behavior (0-1)",
    )
    feedback: str = VotingField(
        comparison=ComparisonStrategy.SEMANTIC,
        min_length=10,
        description="Detailed feedback on the prompt's alignment",
    )
    strengths: SemanticStringList = VotingField(
        comparison=ComparisonStrategy.DERIVED,
        default_factory=SemanticStringList,
        description="Identified strengths of the prompt",
    )
    weaknesses: SemanticStringList = VotingField(
        comparison=ComparisonStrategy.DERIVED,
        default_factory=SemanticStringList,
        description="Identified weaknesses of the prompt",
    )
    suggested_improvements: SemanticStringList = VotingField(
        comparison=ComparisonStrategy.DERIVED,
        default_factory=SemanticStringList,
        description="Specific improvements suggested",
    )
    reasoning: str = VotingField(
        comparison=ComparisonStrategy.IGNORE,
        min_length=50,
        description="Detailed reasoning behind the evaluation",
    )


class AlignmentPrincipleList(RootModel[List[AlignmentPrinciple]]):
    """A list of alignment principles with DERIVED comparison."""

    root: List[AlignmentPrinciple] = VotingField(
        comparison=ComparisonStrategy.SEQUENCE_UNORDERED_DERIVED,
        default_factory=list,
    )


class AlignmentFeedback(TypedCallBaseForConsensus):
    """Feedback from alignment model for prompt improvement."""

    overall_assessment: str = VotingField(
        comparison=ComparisonStrategy.SEMANTIC,
        min_length=20,
        description="Overall assessment of the prompt",
    )
    specific_issues: SemanticStringList = VotingField(
        comparison=ComparisonStrategy.DERIVED,
        default_factory=SemanticStringList,
        description="Specific issues identified in the prompt",
    )
    improvement_suggestions: SemanticStringList = VotingField(
        comparison=ComparisonStrategy.DERIVED,
        default_factory=SemanticStringList,
        description="Concrete suggestions for improvement",
    )
    principles_to_apply: AlignmentPrincipleList = VotingField(
        comparison=ComparisonStrategy.DERIVED,
        default_factory=AlignmentPrincipleList,
        description="Principles that should be applied",
    )
    revised_prompt_suggestion: Optional[str] = VotingField(
        comparison=ComparisonStrategy.SEMANTIC,
        default=None,
        description="Direct suggestion for revised prompt",
    )
    confidence_score: float = VotingField(
        comparison=ComparisonStrategy.EXACT,
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Confidence in the feedback provided",
    )
    reasoning: str = VotingField(
        comparison=ComparisonStrategy.IGNORE,
        min_length=50,
        description="Detailed reasoning behind the feedback",
    )


class PromptEvolution(BaseModel):
    """Record of how a prompt evolved during alignment."""

    iteration: int = Field(
        ge=0,
        description="Iteration number in the alignment process",
    )
    prompt: str = Field(
        min_length=1,
        description="The prompt at this iteration",
    )
    score: float = Field(
        ge=0.0,
        le=1.0,
        description="Alignment score at this iteration",
    )
    feedback: str = Field(
        description="Feedback received at this iteration",
    )
    improvements_made: List[str] = Field(
        default_factory=list,
        description="Improvements made in this iteration",
    )
    principles_applied: List[str] = Field(
        default_factory=list,
        description="Principles applied in this iteration",
    )


class AlignmentMetrics(BaseModel):
    """Metrics for the alignment process."""

    total_iterations: int = Field(
        ge=0,
        description="Total number of iterations performed",
    )
    average_improvement: float = Field(
        description="Average score improvement per iteration",
    )
    final_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Final alignment score achieved",
    )
    convergence_rate: float = Field(
        description="Rate of convergence to target score",
    )
    stability_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Stability of improvements across iterations",
    )


class PromptTemplate(BaseModel):
    """Template for structured prompts."""

    template: str = Field(
        description="The prompt template with placeholders",
    )
    variables: List[str] = Field(
        default_factory=list,
        description="List of variables in the template",
    )
    constraints: List[str] = Field(
        default_factory=list,
        description="Constraints that should be maintained",
    )
    context_preservation_zones: List[str] = Field(
        default_factory=list,
        description="Parts of the prompt that should be preserved",
    )
