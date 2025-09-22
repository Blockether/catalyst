"""
Types for voting-based consensus mechanism.

This module provides types and configuration for the majority voting
consensus system with optional judge-based tie-breaking.
"""

import threading
from collections import deque
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Deque, Dict, Generic, List, Optional, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from ..utils.TypedCalls import ArityOneTypedCall
from .VotingComparison import (
    BaseModelWithReasoning,
)

T = TypeVar("T", bound=BaseModelWithReasoning)


class ModelConfiguration(BaseModel, Generic[T]):
    """Configuration for a model in the consensus system."""

    id: str = Field(description="Unique identifier for the model")
    executor: ArityOneTypedCall[str, T] = Field(description="The typed call implementation for this model")
    perspective: str = Field(
        description="The perspective or role the model should take (e.g., 'As a mathematician', 'From a security perspective')",
    )
    weight_multiplier: float = Field(
        default=1.0,
        ge=0.0,
        description="Weight multiplier for this model's vote in consensus",
    )
    model_config = ConfigDict(arbitrary_types_allowed=True)


class ConsensusQualityMetrics(BaseModel):
    """Metrics for consensus quality assessment using voting."""

    convergence_speed: float = Field(description="How quickly consensus was reached (rounds/max_rounds)")
    agreement_strength: float = Field(
        ge=0.0,
        le=1.0,
        description="Weighted score combining voting proportion, field consistency, and vote stability (0-1)",
    )
    field_consistency: float = Field(
        ge=0.0,
        le=1.0,
        default=0.0,
        description="Proportion of fields that reached consensus (0-1)",
    )
    vote_stability: float = Field(
        ge=0.0,
        le=1.0,
        default=1.0,
        description="How stable votes were across rounds (0-1, higher is more stable)",
    )
    vote_distribution: Dict[str, int] = Field(
        default_factory=dict,
        description="Distribution of votes in final round",
    )
    outlier_models: List[str] = Field(
        default_factory=list,
        description="Models that voted differently from the majority",
    )
    judge_invoked: bool = Field(
        default=False,
        description="Whether judge was used for tie-breaking",
    )

    @property
    def overall_quality_score(self) -> float:
        """Calculate overall consensus quality score."""
        return self.agreement_strength


class FieldChangeValue(BaseModel):
    """Represents a change in a field value.

    Note: Uses Any type because field values can be of any type from the model.
    This is intentional as we're tracking changes in arbitrary model fields.
    """

    old_value: Any = Field(description="Old field value")  # type: ignore[type-arg]
    new_value: Any = Field(description="New field value")  # type: ignore[type-arg]


class ResponseEvolution(BaseModel, Generic[T]):
    """Tracks how a response evolved between rounds."""

    id: str = Field(description="Model whose response evolved")
    round_from: int = Field(description="Starting round")
    round_to: int = Field(description="Ending round")
    field_changes: Dict[str, FieldChangeValue] = Field(
        default_factory=dict,
        description="Map of field names to change values",
    )
    vote_changed: bool = Field(default=False, description="Whether the model changed its vote")
    reasoning_evolution: str = Field(default="", description="How the reasoning changed")
    influenced_by: List[str] = Field(default_factory=list, description="Model IDs that influenced this evolution")


class DisagreementAnalysis(BaseModel):
    """Analysis of disagreements between model responses."""

    disagreement_fields: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Fields with disagreements mapped to string representations of different values",
    )
    consensus_fields: List[str] = Field(default_factory=list, description="Fields where all models agree")
    ignored_fields: List[str] = Field(
        default_factory=list,
        description="Fields that are ignored in consensus (IGNORE strategy)",
    )


class GossipHistory(BaseModel):
    """Tracks the gossip history and information flow."""

    round_number: int = Field(description="The round number")
    refined_from_peers: bool = Field(default=False, description="Whether this round involved peer refinement")
    peer_models_seen: List[str] = Field(default_factory=list, description="Peer models whose responses were seen")


class ResponseMetadata(BaseModel):
    """Metadata for model responses."""

    initial_response: bool = Field(default=False, description="Whether this is an initial response")
    refined: bool = Field(default=False, description="Whether this response was refined from peers")
    round: Optional[int] = Field(default=None, description="Round number for refinement")
    judge_decision: bool = Field(default=False, description="Whether this is a judge decision")
    resolved_tie: bool = Field(default=False, description="Whether this resolved a tie")


class ModelResponse(BaseModel, Generic[T]):
    """Represents a response from a reasoning model with gossip metadata."""

    id: str = Field(description="Unique identifier for the model")
    round_number: int = Field(description="The consensus round number")
    content: T = Field(description="The model's structured response")
    metadata: ResponseMetadata = Field(
        default_factory=ResponseMetadata,
        description="Additional metadata about the response",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the response was generated",
    )
    gossip_history: List[GossipHistory] = Field(default_factory=list, description="History of information flow")


class ConsensusRound(BaseModel, Generic[T]):
    """Represents a single round in the consensus process."""

    round_number: int = Field(description="The round number (0-indexed)")
    responses: List[ModelResponse[T]] = Field(default_factory=list, description="All model responses for this round")
    consensus_achieved: bool = Field(default=False, description="Whether consensus was reached this round")
    consensus_response: Optional[T] = Field(default=None, description="The consensus response if achieved")
    information_flow: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Tracks which models influenced each other",
    )
    response_evolutions: List[ResponseEvolution[T]] = Field(
        default_factory=list, description="How responses evolved in this round"
    )
    disagreement_analysis: Optional[DisagreementAnalysis] = Field(
        default=None, description="Analysis of disagreements in this round"
    )
    vote_groups: Optional[Dict[str, List[ModelResponse[T]]]] = Field(
        default=None,
        description="Computed vote groups for this round (group_key -> responses)",
    )


class ModelMetrics(BaseModel):
    """Metrics for model behavior and contribution to consensus."""

    id: str = Field(description="Model identifier")
    total_rounds: int = Field(default=0, description="Total rounds participated")
    consistency_score: float = Field(default=1.0, description="Consistency across rounds")
    contribution_score: float = Field(default=1.0, description="Overall contribution to consensus quality")


class ConsensusMetrics(BaseModel):
    """Metrics for consensus results."""

    duration_ms: float = Field(description="Duration of consensus process in milliseconds")
    rounds_to_convergence: int = Field(description="Number of rounds to reach convergence")
    total_model_calls: int = Field(description="Total number of model calls made")
    convergence_achieved: bool = Field(description="Whether convergence was achieved")
    dissent_rate: float = Field(ge=0.0, le=1.0, description="Rate of dissent among models")
    model_contributions: Dict[str, float] = Field(
        default_factory=dict, description="Individual model contribution scores"
    )
    consensus_confidence: float = Field(ge=0.0, le=1.0, description="Confidence in consensus result")
    convergence_indicator: float = Field(ge=0.0, le=1.0, description="Convergence indicator score")
    total_refinements: int = Field(default=0, description="Total number of refinements across all rounds")
    avg_refinements_per_round: float = Field(default=0.0, description="Average refinements per round")
    information_flows: List[Dict[str, List[str]]] = Field(
        default_factory=list, description="Information flow between models per round"
    )
    fallback_method: Optional[str] = Field(
        default=None,
        description="Fallback method used if consensus not achieved (judge/majority/none)",
    )


class ConsensusResult(BaseModelWithReasoning, Generic[T]):
    """Result of the consensus process."""

    consensus_achieved: bool = Field(description="Whether consensus was reached")
    final_response: T = Field(description="The final consensus response")
    rounds: List[ConsensusRound[T]] = Field(description="All rounds in the consensus process")
    total_rounds: int = Field(description="Total number of rounds executed")
    convergence_score: float = Field(description="Final convergence score (0-1)")
    participating_models: List[str] = Field(description="List of models that participated")
    dissenting_models: List[str] = Field(default_factory=list, description="Models that didn't converge")
    model_contributions: Dict[str, float] = Field(
        default_factory=dict, description="Contribution scores for each model"
    )
    metrics: ConsensusMetrics = Field(
        default_factory=lambda: ConsensusMetrics(
            duration_ms=0.0,
            rounds_to_convergence=0,
            total_model_calls=0,
            convergence_achieved=False,
            dissent_rate=0.0,
            model_contributions={},
            consensus_confidence=0.0,
            convergence_indicator=0.0,
            total_refinements=0,
            avg_refinements_per_round=0.0,
            information_flows=[],
        ),
        description="Consensus metrics",
    )


class VerbosityLevel(Enum):
    """Logging verbosity levels for consensus operations."""

    SILENT = 0  # No logging at all
    COMPACT = 1  # Single line summary with key metrics
    VERBOSE = 2  # Full comprehensive output at the end  # Full comprehensive output at the end


class ConsensusSettings(BaseModel):
    """Configuration settings for consensus using majority voting."""

    threshold: float = Field(
        default=0.5,
        description="Threshold for consensus acceptance (e.g., 0.5 = simple majority, 0.7 = 70% of votes needed)",
    )
    first_round_threshold: float = Field(
        default=0.75,
        description="Threshold for first round consensus (higher bar for initial agreement, e.g., 0.75 = 75% super-majority)",
    )
    max_rounds: int = Field(default=5, ge=1, le=20, description="Maximum number of consensus rounds")
    max_concurrent_calls: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of concurrent model calls for performance control",
    )
    verbosity: VerbosityLevel = Field(
        default=VerbosityLevel.VERBOSE,
        description="Logging verbosity level for consensus operations",
    )


class RoundMetric(BaseModel):
    """Metrics for a single consensus round."""

    round_number: int = Field(description="The round number (0-indexed)")
    convergence_score: float = Field(ge=0.0, le=1.0, description="Convergence score for this round")
    consensus_achieved: bool = Field(description="Whether consensus was achieved in this round")
    num_responses: int = Field(ge=0, description="Number of model responses in this round")
    unique_votes: int = Field(ge=0, description="Number of unique voting patterns in this round")
