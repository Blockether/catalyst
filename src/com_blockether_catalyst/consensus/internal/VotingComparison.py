"""
Voting comparison strategies for consensus fields.

This module provides flexible field comparison options for voting-based consensus,
allowing fields to be compared in different ways (exact match, range, ignored, etc.)
"""

from enum import Enum
from typing import Any, Callable, Dict, Optional, TypeVar, Union, overload

from pydantic import BaseModel
from pydantic import Field as PydanticField
from pydantic.fields import FieldInfo


class ComparisonStrategy(str, Enum):
    """Strategies for comparing field values during voting."""

    # Equality check x1 = x2
    EXACT = "exact"  # Fields must match exactly (default)

    # Equality check omitted x1 <=> x2
    IGNORE = "ignore"

    # Numeric fields within a range are considered equal
    RANGE = "range"


    SEMANTIC = "semantic"  # Semantic similarity using embeddings (for text)
    CUSTOM = "custom"  # Use a custom comparison function

    # Recursive/derived comparison strategies
    SEQUENCE_ORDERED_DERIVED = (
        "sequence_ordered_derived"  # List[T], Tuple[T] - order matters, compare each T recursively
    )
    SEQUENCE_UNORDERED_DERIVED = "sequence_unordered_derived"  # List[T] as set - order ignored, find best matches
    DERIVED = "derived"  # BaseModel - recursive field comparison using each field's strategy

    # Partial/alike comparison strategies - support different sized collections
    SEQUENCE_ORDERED_ALIKE = "sequence_ordered_alike"  # List[T] - order matters, allows partial overlap
    SEQUENCE_UNORDERED_ALIKE = "sequence_unordered_alike"  # List[T] - order ignored, allows partial overlap


class VotingMetadata(BaseModel):
    """Metadata for field voting comparison."""

    strategy: ComparisonStrategy = PydanticField(
        default=ComparisonStrategy.EXACT,
        description="Comparison strategy to use for this field",
    )
    tolerance: float = PydanticField(
        default=0.2,
        description="For RANGE strategy, the bin width as a fraction (0.2 = 20% bins for ~10% matching)",
    )
    threshold: float = PydanticField(
        default=0.8,
        description="For SEMANTIC and derived strategies, similarity threshold",
    )


class FieldComparator:
    """Handles field comparison logic based on strategy."""

    @staticmethod
    def compare_fields(
        value1: Any,
        value2: Any,
        strategy: ComparisonStrategy = ComparisonStrategy.EXACT,
        tolerance: Optional[float] = None,
        decimal_places: Optional[int] = None,
        threshold: Optional[float] = None,
        custom_comparator: Optional[Callable[[Any, Any], bool]] = None,
    ) -> bool:
        """
        Compare two field values according to the specified strategy.

        Args:
            value1: First value to compare
            value2: Second value to compare
            strategy: Comparison strategy to use
            tolerance: For RANGE strategy, the acceptable difference
            threshold: For SEMANTIC and derived strategies, similarity threshold
            custom_comparator: For CUSTOM strategy, the comparison function

        Returns:
            True if values are considered equal according to strategy
        """
        if strategy == ComparisonStrategy.IGNORE:
            return True  # Always considered equal

        if strategy == ComparisonStrategy.EXACT:
            return bool(value1 == value2)

        if strategy == ComparisonStrategy.RANGE:
            if not isinstance(value1, (int, float)) or not isinstance(value2, (int, float)):
                return bool(value1 == value2)  # Fall back to exact for non-numeric
            tolerance = tolerance or 0.1  # Default 10% tolerance
            return abs(value1 - value2) <= tolerance * max(abs(value1), abs(value2), 1)

        if strategy == ComparisonStrategy.CUSTOM:
            if custom_comparator:
                return custom_comparator(value1, value2)
            return bool(value1 == value2)

        if strategy == ComparisonStrategy.SEQUENCE_ORDERED_DERIVED:
            return FieldComparator._compare_sequence_ordered_derived(value1, value2, threshold or 0.8)

        if strategy == ComparisonStrategy.SEQUENCE_UNORDERED_DERIVED:
            return FieldComparator._compare_sequence_unordered_derived(value1, value2, threshold or 0.8)

        if strategy == ComparisonStrategy.DERIVED:
            return FieldComparator._compare_model_derived(value1, value2, threshold or 0.8)

        if strategy == ComparisonStrategy.SEQUENCE_ORDERED_ALIKE:
            return FieldComparator._compare_sequence_ordered_alike(value1, value2, threshold or 0.8)

        if strategy == ComparisonStrategy.SEQUENCE_UNORDERED_ALIKE:
            return FieldComparator._compare_sequence_unordered_alike(value1, value2, threshold or 0.8)

        # Default to exact comparison
        return bool(value1 == value2)

    @staticmethod
    def _compare_sequence_ordered_derived(seq1: Any, seq2: Any, threshold: float) -> bool:
        """Compare two sequences where order matters, using recursive field comparison."""
        if not isinstance(seq1, (list, tuple)) or not isinstance(seq2, (list, tuple)):
            return bool(seq1 == seq2)

        if len(seq1) != len(seq2):
            return False

        if not seq1 and not seq2:
            return True

        matches = 0
        for item1, item2 in zip(seq1, seq2):
            if FieldComparator._compare_items_recursively(item1, item2, threshold):
                matches += 1

        return (matches / len(seq1)) >= threshold if len(seq1) > 0 else True

    @staticmethod
    def _compare_sequence_unordered_derived(seq1: Any, seq2: Any, threshold: float) -> bool:
        """Compare two sequences ignoring order, finding best matches using recursive comparison."""
        if not isinstance(seq1, (list, tuple)) or not isinstance(seq2, (list, tuple)):
            return bool(seq1 == seq2)

        if not seq1 and not seq2:
            return True
        if not seq1 or not seq2:
            return False

        # Find best matches using Hungarian-style algorithm
        matches = 0
        used_indices = set()

        for item1 in seq1:
            best_match_score = 0.0
            best_match_idx = -1

            for idx, item2 in enumerate(seq2):
                if idx in used_indices:
                    continue

                if FieldComparator._compare_items_recursively(item1, item2, threshold):
                    score = FieldComparator._get_similarity_score(item1, item2, threshold)
                    if score > best_match_score:
                        best_match_score = score
                        best_match_idx = idx

            if best_match_score >= threshold and best_match_idx >= 0:
                matches += 1
                used_indices.add(best_match_idx)

        # Return True if enough items matched
        total = max(len(seq1), len(seq2))
        return (matches / total) >= threshold if total > 0 else True

    @staticmethod
    def _compare_model_derived(obj1: Any, obj2: Any, threshold: float) -> bool:
        """Compare two BaseModel objects using recursive field comparison."""
        # Import here to avoid circular dependency
        from pydantic import BaseModel

        if not isinstance(obj1, BaseModel) or not isinstance(obj2, BaseModel):
            return bool(obj1 == obj2)

        if type(obj1) is not type(obj2):
            return False

        # Compare each field using its own voting strategy
        field_scores = []
        for field_name, field_info in obj1.__class__.model_fields.items():
            # Extract voting metadata for this field
            voting_meta = FieldComparator._extract_voting_metadata(field_info)

            if voting_meta.strategy == ComparisonStrategy.IGNORE:
                continue  # Skip ignored fields

            val1 = getattr(obj1, field_name)
            val2 = getattr(obj2, field_name)

            # Recursively compare based on field's strategy
            field_matches = FieldComparator.compare_fields(
                val1,
                val2,
                strategy=voting_meta.strategy,
                tolerance=voting_meta.tolerance,
                decimal_places=voting_meta.decimal_places,
                threshold=voting_meta.threshold,
            )
            field_scores.append(1.0 if field_matches else 0.0)

        if not field_scores:
            return True  # All fields ignored

        # Return True if average field match score meets threshold
        avg_score = sum(field_scores) / len(field_scores)
        return avg_score >= threshold

    @staticmethod
    def _compare_items_recursively(item1: Any, item2: Any, threshold: float) -> bool:
        """Compare two items recursively based on their type."""
        # Import here to avoid circular dependency
        from pydantic import BaseModel

        if isinstance(item1, BaseModel) and isinstance(item2, BaseModel):
            return FieldComparator._compare_model_derived(item1, item2, threshold)
        elif isinstance(item1, (list, tuple)) and isinstance(item2, (list, tuple)):
            return FieldComparator._compare_sequence_unordered_derived(item1, item2, threshold)
        else:
            return bool(item1 == item2)

    @staticmethod
    def _get_similarity_score(item1: Any, item2: Any, threshold: float) -> float:
        """Get similarity score between two items (simplified version)."""
        if FieldComparator._compare_items_recursively(item1, item2, threshold):
            return 1.0
        else:
            return 0.0

    @staticmethod
    def _compare_sequence_ordered_alike(seq1: Any, seq2: Any, threshold: float) -> bool:
        """Compare sequences allowing size differences, preserving order for matching elements."""
        if not isinstance(seq1, (list, tuple)) or not isinstance(seq2, (list, tuple)):
            return bool(seq1 == seq2)

        if not seq1 and not seq2:
            return True
        if not seq1 or not seq2:
            # Empty vs non-empty - check if threshold allows it
            return threshold <= 0.0

        # Use sliding window to find best alignment
        smaller = seq1 if len(seq1) <= len(seq2) else seq2
        larger = seq2 if len(seq1) <= len(seq2) else seq1

        best_match_ratio = 0.0

        # Try different starting positions in the larger sequence
        for start_idx in range(len(larger) - len(smaller) + 1):
            matches = 0
            for i, item in enumerate(smaller):
                if FieldComparator._compare_items_recursively(item, larger[start_idx + i], threshold):
                    matches += 1

            match_ratio = matches / len(smaller)
            best_match_ratio = max(best_match_ratio, match_ratio)

        # Also check partial matches at boundaries
        # Check prefix match
        prefix_matches = 0
        for i in range(min(len(seq1), len(seq2))):
            if FieldComparator._compare_items_recursively(seq1[i], seq2[i], threshold):
                prefix_matches += 1
            else:
                break

        prefix_ratio = prefix_matches / max(len(seq1), len(seq2))
        best_match_ratio = max(best_match_ratio, prefix_ratio)

        return best_match_ratio >= threshold

    @staticmethod
    def _compare_sequence_unordered_alike(seq1: Any, seq2: Any, threshold: float) -> bool:
        """Compare sequences allowing size differences, ignoring order."""
        if not isinstance(seq1, (list, tuple)) or not isinstance(seq2, (list, tuple)):
            return bool(seq1 == seq2)

        if not seq1 and not seq2:
            return True
        if not seq1 or not seq2:
            # Empty vs non-empty - check if threshold allows it
            return threshold <= 0.0

        # Sort both sequences to ensure consistent comparison
        # Create sortable representations of items
        def get_sort_key(item: Any) -> tuple[str, Any]:
            if hasattr(item, "term"):
                return ("term", item.term)
            elif hasattr(item, "text"):
                return ("text", item.text[:20] if len(item.text) > 20 else item.text)
            elif hasattr(item, "__dict__"):
                first_field = list(item.__dict__.items())[0] if item.__dict__ else ("", "")
                return first_field
            else:
                return ("str", str(item))

        # Sort both sequences by their content
        sorted_seq1 = sorted(seq1, key=get_sort_key)
        sorted_seq2 = sorted(seq2, key=get_sort_key)

        # Find best matches between sorted items
        smaller = sorted_seq1 if len(sorted_seq1) <= len(sorted_seq2) else sorted_seq2
        larger = sorted_seq2 if len(sorted_seq1) <= len(sorted_seq2) else sorted_seq1

        matches = 0
        used_indices = set()

        # Match each item in smaller sequence with best match in larger
        for item in smaller:
            for idx, other_item in enumerate(larger):
                if idx in used_indices:
                    continue

                if FieldComparator._compare_items_recursively(item, other_item, threshold):
                    matches += 1
                    used_indices.add(idx)
                    break

        # Calculate match ratio based on the larger sequence (more restrictive)
        # This ensures that [1,2] vs [1,2,3,4,5] gives lower score than [1,2] vs [1,2,3]
        match_ratio = matches / max(len(seq1), len(seq2))

        return match_ratio >= threshold

    @staticmethod
    def _extract_voting_metadata(field_info: Any) -> "VotingMetadata":
        """Extract voting metadata from field info."""
        voting_meta = VotingMetadata()  # Default values

        if field_info and field_info.json_schema_extra:
            extra = field_info.json_schema_extra
            if isinstance(extra, dict) and "voting_comparison" in extra:
                voting_comparison = extra["voting_comparison"]
                if isinstance(voting_comparison, dict):
                    # Parse the dict into VotingMetadata - safely extract fields
                    voting_meta_kwargs: dict[str, Any] = {}
                    if "strategy" in voting_comparison and isinstance(voting_comparison["strategy"], (str, int)):
                        voting_meta_kwargs["strategy"] = ComparisonStrategy(voting_comparison["strategy"])
                    if "tolerance" in voting_comparison and isinstance(voting_comparison["tolerance"], (int, float)):
                        voting_meta_kwargs["tolerance"] = voting_comparison["tolerance"]
                    if "decimal_places" in voting_comparison and isinstance(
                        voting_comparison["decimal_places"], (int, float)
                    ):
                        voting_meta_kwargs["decimal_places"] = voting_comparison["decimal_places"]
                    if "threshold" in voting_comparison and isinstance(voting_comparison["threshold"], (int, float)):
                        voting_meta_kwargs["threshold"] = voting_comparison["threshold"]

                    voting_meta = VotingMetadata(**voting_meta_kwargs)

        return voting_meta


_T = TypeVar("_T")


# Overloads to match Pydantic's Field behavior for type checking
@overload
def VotingField(
    default: _T,
    *,
    comparison: ComparisonStrategy = ...,
    tolerance: Optional[float] = ...,
    threshold: Optional[float] = ...,
    decimal_places: Optional[int] = ...,
    custom_comparator: Optional[Callable[[Any, Any], bool]] = ...,
    **kwargs: Any,
) -> _T: ...


@overload
def VotingField(
    *,
    default_factory: Callable[[], _T],
    comparison: ComparisonStrategy = ...,
    tolerance: Optional[float] = ...,
    threshold: Optional[float] = ...,
    decimal_places: Optional[int] = ...,
    custom_comparator: Optional[Callable[[Any, Any], bool]] = ...,
    **kwargs: Any,
) -> _T: ...


@overload
def VotingField(
    *,
    comparison: ComparisonStrategy = ...,
    tolerance: Optional[float] = ...,
    threshold: Optional[float] = ...,
    decimal_places: Optional[int] = ...,
    custom_comparator: Optional[Callable[[Any, Any], bool]] = ...,
    **kwargs: Any,
) -> Any: ...


def VotingField(
    *args: Any,
    comparison: ComparisonStrategy = ComparisonStrategy.EXACT,
    tolerance: Optional[float] = None,
    threshold: Optional[float] = None,
    decimal_places: Optional[int] = None,
    custom_comparator: Optional[Callable[[Any, Any], bool]] = None,
    **kwargs: Any,
) -> Any:
    """
    Enhanced Field function that includes voting comparison metadata.

    Works exactly like Pydantic's Field() but adds voting comparison strategy.
    All standard Field parameters are supported (default, description, alias, etc.)

    Args:
        *args: Positional arguments passed to Field
        comparison: How this field should be compared during voting (default: EXACT)
        tolerance: For RANGE strategy, the bin width (e.g., 0.2 for 20% bins, allows ~10% variance)
        threshold: For SEMANTIC strategy, minimum cosine similarity (e.g., 0.8 for 80%)
        custom_comparator: For CUSTOM strategy, a function(value1, value2) -> bool
        **kwargs: All standard Pydantic Field keyword arguments

    Usage:
        class MyResponse(TypedCallBaseForConsensus):
            # Exact match required for this field (default behavior)
            answer: int = VotingField(description="The answer")

            # This field is ignored during voting
            confidence: float = VotingField(
                default=0.0,
                comparison=ComparisonStrategy.IGNORE,
                description="Model confidence"
            )

            # Values within 10% are considered equal
            score: float = VotingField(
                comparison=ComparisonStrategy.RANGE,
                tolerance=0.1,
                ge=0,  # Standard Field validation still works
                le=100
            )

            # Semantic similarity with custom threshold
            description: str = VotingField(
                comparison=ComparisonStrategy.SEMANTIC,
                threshold=0.75,  # 75% cosine similarity required
                description="Description with semantic matching"
            )
    """
    # Store comparison metadata in the field's json_schema_extra
    json_schema_extra: Dict[str, Any] = kwargs.get("json_schema_extra", {})  # Field expects Any for json_schema_extra

    # Build metadata dict with only non-None values
    voting_comparison: Dict[str, Union[ComparisonStrategy, float, int, Optional[Callable]]] = {"strategy": comparison}

    voting_comparison["tolerance"] = tolerance
    voting_comparison["threshold"] = threshold
    voting_comparison["decimal_places"] = decimal_places
    voting_comparison["custom_comparator"] = custom_comparator

    json_schema_extra["voting_comparison"] = voting_comparison
    kwargs["json_schema_extra"] = json_schema_extra

    # Pass through to standard Field with all arguments
    return PydanticField(*args, **kwargs)
