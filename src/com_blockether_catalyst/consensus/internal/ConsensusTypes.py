"""
Types for voting-based consensus mechanism.

This module provides types and configuration for the majority voting
consensus system with optional judge-based tie-breaking.
"""

import hashlib
import json
import math
import threading
from abc import ABC
from collections import deque
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Deque, Dict, Generic, List, Optional, TypeVar, Union

from pydantic import BaseModel, ConfigDict, Field, SkipValidation, field_validator

from ...utils.TypedCalls import ArityOneTypedCall
from .VotingComparison import ComparisonStrategy, VotingField, VotingMetadata


class TypedCallBaseForConsensus(BaseModel):
    """Base class for consensus-compatible typed calls.

    Subclasses can define voting comparison strategies for their fields
    by using the VotingField from VotingComparison module.
    """

    reasoning: str = VotingField(
        min_length=50,
        comparison=ComparisonStrategy.IGNORE,
        description=(
            "Detailed reasoning that explains the thought process and rationale "
            "behind ALL values provided in other fields. This should include: "
            "1) Key observations that influenced the analysis, "
            "2) Justification for scores or ratings given, "
            "3) Explanation of why specific issues or points were identified, "
            "4) Rationale for any recommendations or conclusions. "
            "This field is MANDATORY to enable meaningful consensus building."
        ),
    )

    def get_voting_key(self) -> str:
        """Generate a voting key for this response based on field comparison strategies.

        This method can be overridden by subclasses for custom voting logic.
        """
        voting_data: Dict[str, Union[str, int, float, List, bool]] = {}
        response_dict = self.model_dump()

        # Check each field for voting comparison metadata
        for field_name, field_value in response_dict.items():
            # Use actual field value from model for derived strategies
            actual_field_value = getattr(self, field_name, field_value)
            # Get field info from the model class
            field_info = self.__class__.model_fields.get(field_name)

            # Parse voting metadata from field info
            voting_meta = VotingMetadata()  # Default values
            if field_info and field_info.json_schema_extra:
                extra = field_info.json_schema_extra
                if isinstance(extra, dict) and "voting_comparison" in extra:
                    voting_comparison = extra["voting_comparison"]
                    if isinstance(voting_comparison, dict):
                        # Parse the dict into VotingMetadata - safely extract fields
                        voting_meta_kwargs: Dict[str, Union[ComparisonStrategy, float, int]] = {}
                        if "strategy" in voting_comparison and isinstance(voting_comparison["strategy"], (str, int)):
                            voting_meta_kwargs["strategy"] = ComparisonStrategy(voting_comparison["strategy"])
                        if "tolerance" in voting_comparison and isinstance(
                            voting_comparison["tolerance"], (int, float)
                        ):
                            voting_meta_kwargs["tolerance"] = float(voting_comparison["tolerance"])
                        if "decimal_places" in voting_comparison and isinstance(
                            voting_comparison["decimal_places"], (int, float)
                        ):
                            voting_meta_kwargs["decimal_places"] = int(voting_comparison["decimal_places"])
                        if "threshold" in voting_comparison and isinstance(
                            voting_comparison["threshold"], (int, float)
                        ):
                            voting_meta_kwargs["threshold"] = float(voting_comparison["threshold"])
                        voting_meta = VotingMetadata(**voting_meta_kwargs)  # type: ignore[arg-type]

            strategy = voting_meta.strategy

            if strategy == ComparisonStrategy.IGNORE:
                continue  # Skip this field for voting
            elif strategy == ComparisonStrategy.SEMANTIC:
                # Use semantic hashing for text similarity
                if isinstance(field_value, str):
                    # Import here to avoid circular dependency
                    from ..internal.Consensus import Consensus

                    voting_data[field_name] = Consensus._semantic_hash(field_value, threshold=voting_meta.threshold)
                else:
                    voting_data[field_name] = field_value
            elif strategy == ComparisonStrategy.RANGE:
                # For range, we use logarithmic binning for relative tolerance
                if isinstance(field_value, (int, float)):
                    # The tolerance parameter directly controls bin width.
                    # For example, tolerance=0.2 creates bins that are 20% wide,
                    # which allows values within ~10% of each other to fall in the same bin.
                    # Users can adjust tolerance based on their needs:
                    # - Smaller tolerance (0.1) = tighter matching
                    # - Larger tolerance (0.3) = more forgiving matching

                    if field_value != 0:
                        # Use logarithmic binning for better relative tolerance
                        # Take log to convert multiplicative tolerance to additive
                        log_value = math.log(abs(field_value))
                        # Use tolerance directly as the bin width multiplier
                        log_bin_width = math.log(1 + voting_meta.tolerance)
                        # Calculate bin number
                        bin_number = int(log_value / log_bin_width)
                        # Include sign
                        if field_value < 0:
                            bin_number = -bin_number
                        voting_data[field_name] = f"range_log_bin_{bin_number}"
                    else:
                        voting_data[field_name] = "range_zero"
                else:
                    voting_data[field_name] = field_value
            elif strategy in (
                ComparisonStrategy.SEQUENCE_ORDERED_DERIVED,
                ComparisonStrategy.SEQUENCE_UNORDERED_DERIVED,
                ComparisonStrategy.DERIVED,
                ComparisonStrategy.SEQUENCE_ORDERED_ALIKE,
                ComparisonStrategy.SEQUENCE_UNORDERED_ALIKE,
            ):
                # For derived and alike strategies, create a structural hash using actual model objects
                voting_data[field_name] = self._create_derived_hash(actual_field_value, strategy, voting_meta)
            else:
                # EXACT or CUSTOM - use exact value
                voting_data[field_name] = field_value

        # Create deterministic hash from voting data
        voting_json = json.dumps(voting_data, sort_keys=True, default=str)
        hash_result = hashlib.sha256(voting_json.encode()).hexdigest()[:16]
        return hash_result

    def _create_derived_hash(
        self,
        field_value: Any,
        strategy: "ComparisonStrategy",
        voting_meta: "VotingMetadata",
    ) -> str:
        """Create a structural hash for derived comparison strategies."""
        if strategy == ComparisonStrategy.SEQUENCE_ORDERED_DERIVED:
            return self._hash_sequence_ordered(field_value, voting_meta.threshold)
        elif strategy == ComparisonStrategy.SEQUENCE_UNORDERED_DERIVED:
            return self._hash_sequence_unordered(field_value, voting_meta.threshold)
        elif strategy == ComparisonStrategy.DERIVED:
            return self._hash_model_structure(field_value, voting_meta.threshold)
        elif strategy == ComparisonStrategy.SEQUENCE_ORDERED_ALIKE:
            return self._hash_sequence_ordered_alike(field_value, voting_meta.threshold)
        elif strategy == ComparisonStrategy.SEQUENCE_UNORDERED_ALIKE:
            return self._hash_sequence_unordered_alike(field_value, voting_meta.threshold)
        else:
            return str(field_value)

    def _hash_sequence_ordered(self, seq: Any, threshold: float) -> str:
        """Hash a sequence preserving order for comparison."""
        if not isinstance(seq, (list, tuple)):
            return str(seq)

        # Create hash based on structural elements that matter for comparison
        elements = []
        for item in seq:
            if hasattr(item, "model_fields"):  # Pydantic model
                elements.append(self._hash_model_structure(item, threshold))
            else:
                elements.append(str(item))

        return f"ordered:{len(seq)}:{':'.join(elements)}"

    def _hash_sequence_unordered(self, seq: Any, threshold: float) -> str:
        """Hash a sequence ignoring order for comparison."""
        if not isinstance(seq, (list, tuple)):
            return str(seq)

        # Create hash based on sorted structural elements
        elements = set()  # Use set to ensure uniqueness and order-independence
        for item in seq:
            if hasattr(item, "model_fields"):  # Pydantic model
                elements.add(self._hash_model_structure(item, threshold))
            else:
                elements.add(str(item))

        # Convert to sorted list for deterministic hashing
        sorted_elements = sorted(elements)
        return f"unordered:{len(seq)}:{':'.join(sorted_elements)}"

    def _hash_model_structure(self, obj: Any, threshold: float) -> str:
        """Hash a BaseModel based on its voting-relevant fields."""
        if not hasattr(obj, "model_fields"):
            return str(obj)

        # Extract only voting-relevant fields
        relevant_fields = {}
        for field_name, field_info in obj.__class__.model_fields.items():
            # Extract voting metadata
            if field_info and field_info.json_schema_extra:
                extra = field_info.json_schema_extra
                if isinstance(extra, dict) and "voting_comparison" in extra:
                    voting_comparison = extra["voting_comparison"]
                    if isinstance(voting_comparison, dict):
                        if "strategy" in voting_comparison:
                            strategy_val = voting_comparison["strategy"]
                            # Convert to ComparisonStrategy enum if it's a string
                            if isinstance(strategy_val, str):
                                strategy_val = ComparisonStrategy(strategy_val)
                            if strategy_val != ComparisonStrategy.IGNORE:
                                relevant_fields[field_name] = getattr(obj, field_name)

        # Create sorted hash of relevant fields
        field_items = sorted(relevant_fields.items())
        field_strs = [f"{k}:{v}" for k, v in field_items]
        return f"model:{obj.__class__.__name__}:{':'.join(field_strs)}"

    def _hash_sequence_ordered_alike(self, seq: Any, threshold: float) -> str:
        """Hash for sequences that can partially match with order preservation."""
        if not isinstance(seq, (list, tuple)):
            return str(seq)

        if not seq:
            return "ordered_alike:empty"

        # Extract ordered content fingerprint
        content_elements = []
        for item in seq[: int(len(seq) * max(threshold, 0.5))]:  # Use prefix based on threshold
            if hasattr(item, "__class__") and hasattr(item.__class__, "__name__"):
                if hasattr(item, "term"):
                    content_elements.append(f"term:{item.term}")
                elif hasattr(item, "text"):
                    text_preview = item.text[:20] if len(item.text) > 20 else item.text
                    content_elements.append(f"text:{text_preview}")
                elif hasattr(item, "__dict__"):
                    first_field = list(item.__dict__.items())[0] if item.__dict__ else None
                    if first_field:
                        content_elements.append(f"{first_field[0]}:{str(first_field[1])[:20]}")
                else:
                    content_elements.append(str(item)[:20])
            else:
                content_elements.append(str(item)[:20])

        # Keep order for ordered comparison
        content_sig = ":".join(content_elements)

        # Size bucket - same logic as unordered
        seq_len = len(seq)
        if threshold <= 0.5:
            if seq_len <= 2:
                size_bucket = "small"
            elif seq_len <= 5:
                size_bucket = "medium"
            else:
                size_bucket = "large"
        elif threshold <= 0.7:
            if seq_len <= 1:
                size_bucket = "tiny"
            elif seq_len <= 3:
                size_bucket = "small"
            elif seq_len <= 5:
                size_bucket = "medium"
            else:
                size_bucket = "large"
        else:
            size_bucket = f"exact_{seq_len}"

        return f"ordered_alike:size_{size_bucket}:content_{content_sig}"

    def _hash_sequence_unordered_alike(self, seq: Any, threshold: float) -> str:
        """Hash for sequences that can partially match ignoring order."""
        if not isinstance(seq, (list, tuple)):
            return str(seq)

        if not seq:
            return "unordered_alike:empty"

        # Extract content fingerprint from the sequence
        content_elements = []
        for item in seq:
            if hasattr(item, "__class__") and hasattr(item.__class__, "__name__"):
                # For Pydantic models, extract key identifying fields
                if hasattr(item, "term"):
                    # For ExtractedKeyword/ExtractedAcronym
                    content_elements.append(f"term:{item.term}")
                elif hasattr(item, "text"):
                    # For ChunkOutput - use first 20 chars as fingerprint
                    text_preview = item.text[:20] if len(item.text) > 20 else item.text
                    content_elements.append(f"text:{text_preview}")
                elif hasattr(item, "__dict__"):
                    # Generic Pydantic model - use first field value
                    first_field = list(item.__dict__.items())[0] if item.__dict__ else None
                    if first_field:
                        content_elements.append(f"{first_field[0]}:{str(first_field[1])[:20]}")
                else:
                    content_elements.append(str(item)[:20])
            else:
                content_elements.append(str(item)[:20])

        # Sort for consistent hashing regardless of order
        content_elements.sort()

        # Create content signature - represents what's in the list
        # Use only first few elements to allow partial matching
        max_elements = max(1, int(len(content_elements) * threshold))
        content_sig = ":".join(content_elements[:max_elements])

        # Size bucket for approximate size matching
        seq_len = len(seq)
        # For threshold=0.5: [1,2] should be same bucket since 1/2=0.5 >= threshold
        # Simplest approach: use coarse buckets that group similar sizes
        if threshold <= 0.5:
            # Very permissive - group [1-2], [3-4], [5-8], etc.
            if seq_len <= 2:
                size_bucket = "small"
            elif seq_len <= 5:
                size_bucket = "medium"
            else:
                size_bucket = "large"
        elif threshold <= 0.7:
            # Moderate - group [1], [2-3], [4-5], etc.
            if seq_len <= 1:
                size_bucket = "tiny"
            elif seq_len <= 3:
                size_bucket = "small"
            elif seq_len <= 5:
                size_bucket = "medium"
            else:
                size_bucket = "large"
        else:
            # Strict - smaller buckets
            size_bucket = f"exact_{seq_len}"

        hash_key = f"unordered_alike:size_{size_bucket}:content_{content_sig}"
        return hash_key


T = TypeVar("T", bound=TypedCallBaseForConsensus)


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
    agreement_strength: float = Field(ge=0.0, le=1.0, description="Proportion of models voting together (0-1)")
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
        # Simple quality based on agreement and speed
        return (self.agreement_strength + (1.0 - self.convergence_speed)) / 2.0


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


class ConsensusResult(BaseModel, Generic[T]):
    """Result of the consensus process."""

    reasoning: str = Field(description="Reasoning behind the consensus")
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

    SILENT = 0  # No logging except errors
    NORMAL = 1  # Key milestones only
    VERBOSE = 2  # Full metrics between rounds


class ConsensusSettings(BaseModel):
    """Configuration settings for consensus using majority voting."""

    threshold: float = Field(
        default=0.85,
        description="Threshold for plurality acceptance (e.g., 0.85 = 85% of votes needed for consensus)",
    )
    max_rounds: int = Field(default=5, ge=1, le=20, description="Maximum number of consensus rounds")
    max_concurrent_calls: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of concurrent model calls for performance control",
    )
    verbosity: VerbosityLevel = Field(
        default=VerbosityLevel.SILENT,
        description="Logging verbosity level for consensus operations",
    )
    state_tracking: bool = Field(default=False, description="Whether to track consensus call history in state")
    max_history_size: Optional[int] = Field(
        default=None,
        description="Maximum number of consensus calls to store in history (None = unlimited)",
    )


class ConsensusCallRecord(BaseModel, Generic[T]):
    """Record of a single consensus call for state tracking."""

    timestamp: datetime = Field(description="When this consensus call was made")
    input_prompt: str = Field(description="The input prompt for this consensus call")
    result: ConsensusResult[T] = Field(description="The complete consensus result")
    round_metrics: List[Dict[str, Any]] = Field(default_factory=list, description="Metrics for each round of consensus")
    convergence_path: List[float] = Field(default_factory=list, description="Convergence scores throughout the rounds")
    duration_ms: float = Field(description="Total duration of this consensus call")


class ConsensusState(BaseModel, Generic[T]):
    """Thread-safe state storage for consensus call history."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, max_history_size: Optional[int] = None, enabled: bool = False):
        """Initialize consensus state with optional history limit."""
        super().__init__()
        self._lock = threading.Lock()
        self._call_history: Deque[ConsensusCallRecord[T]] = (
            deque(maxlen=max_history_size) if max_history_size else deque()
        )
        self._max_history_size = max_history_size
        self._total_calls = 0
        self._total_duration_ms = 0.0
        self._enabled = enabled

    def add_call(self, record: ConsensusCallRecord[T]) -> None:
        """Thread-safe addition of a consensus call record."""
        if not self._enabled:
            return

        with self._lock:
            self._call_history.append(record)
            self._total_calls += 1
            self._total_duration_ms += record.duration_ms

    def get_history(self) -> List[ConsensusCallRecord[T]]:
        """Get a snapshot of call history (thread-safe copy)."""
        with self._lock:
            return list(self._call_history)

    def get_stats(self) -> Dict[str, Any]:
        """Get overall statistics about consensus calls."""
        with self._lock:
            return {
                "total_calls": self._total_calls,
                "total_duration_ms": self._total_duration_ms,
                "avg_duration_ms": self._total_duration_ms / self._total_calls if self._total_calls > 0 else 0,
                "history_size": len(self._call_history),
                "enabled": self._enabled,
            }

    def reset(self) -> None:
        """Reset all state data."""
        with self._lock:
            self._call_history.clear()
            self._total_calls = 0
            self._total_duration_ms = 0.0

    def enable(self) -> None:
        """Enable state tracking."""
        self._enabled = True

    def disable(self) -> None:
        """Disable state tracking."""
        self._enabled = False
