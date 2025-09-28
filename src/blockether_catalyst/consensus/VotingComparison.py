"""
Voting comparison strategies for consensus fields.

This module provides flexible field comparison options for voting-based consensus,
allowing fields to be compared in different ways (exact match, range, ignored, etc.)
"""

from enum import Enum
from textwrap import dedent
from typing import Any, Callable, Dict, Optional, TypeVar, Union, overload

from pydantic import BaseModel
from pydantic import Field as PydanticField
from pydantic import RootModel
from pydantic.fields import FieldInfo

from ..encoder.PotionEightEncoder import PotionEightEncoder


class ComparisonStrategy(str, Enum):
    """Strategies for comparing field values during voting."""

    # Equality check x1 = x2
    EXACT = "exact"  # Fields must match exactly (default)

    # Equality check omitted x1 <=> x2
    IGNORE = "ignore"

    # Numeric fields within a range are considered equal
    RANGE = "range"

    # Semantic similarity using embeddings (for text)
    SEMANTIC = "semantic"

    # Use a custom comparison function
    CUSTOM = "custom"

    # BaseModel - recursive field comparison using each field's strategy
    DERIVED = "derived"

    # Unordered sequence comparison with semantic matching
    DERIVED_UNORDERED = "derived_unordered"


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
        default=0.7,
        description="For SEMANTIC and derived strategies, similarity threshold",
    )
    decimal_places: Optional[int] = PydanticField(
        default=None,
        description="For numeric comparisons, number of decimal places to consider",
    )
    custom_comparator_fn: Optional[Callable[[Any, Any], bool]] = PydanticField(
        default=None,
        description="For CUSTOM strategy, the comparison function",
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
        custom_comparator_fn: Optional[Callable[[Any, Any], bool]] = None,
    ) -> bool:
        """
        Compare two field values according to the specified strategy.

        Args:
            value1: First value to compare
            value2: Second value to compare
            strategy: Comparison strategy to use
            tolerance: For RANGE strategy, the acceptable difference
            threshold: For SEMANTIC and derived strategies, similarity threshold
            custom_comparator_fn: For CUSTOM strategy, the comparison function

        Returns:
            True if values are considered equal according to strategy
        """
        if strategy == ComparisonStrategy.IGNORE:
            return True

        if strategy == ComparisonStrategy.EXACT:
            # Python's == handles lists, dicts, and most types correctly
            return bool(value1 == value2)

        if strategy == ComparisonStrategy.RANGE:
            if not isinstance(value1, (int, float)) or not isinstance(value2, (int, float)):
                return bool(value1 == value2)
            tolerance = tolerance or 0.1
            return abs(value1 - value2) <= tolerance * max(abs(value1), abs(value2), 1)

        if strategy == ComparisonStrategy.CUSTOM:
            if custom_comparator_fn:
                return custom_comparator_fn(value1, value2)
            return bool(value1 == value2)

        if strategy == ComparisonStrategy.SEMANTIC:
            # For semantic comparison using embeddings
            if isinstance(value1, str) and isinstance(value2, str):
                emb1 = PotionEightEncoder.encode_single(value1)
                emb2 = PotionEightEncoder.encode_single(value2)
                similarity = PotionEightEncoder.cosine_similarity(emb1, emb2)

                return similarity >= (threshold or 0.7)
            return bool(value1 == value2)

        if strategy == ComparisonStrategy.DERIVED:
            return FieldComparator._compare_model_derived(value1, value2, threshold or 0.8)

        if strategy == ComparisonStrategy.DERIVED_UNORDERED:
            return FieldComparator._compare_sequence_unordered_derived(value1, value2, threshold or 0.8)

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
        """Compare two sequences ignoring order, finding best matches using semantic comparison.

        Uses a bipartite matching algorithm to find the best alignment between items
        in two sequences, handling different lengths gracefully.
        """
        if not isinstance(seq1, (list, tuple)) or not isinstance(seq2, (list, tuple)):
            return bool(seq1 == seq2)

        if not seq1 and not seq2:
            return True
        if not seq1 or not seq2:
            return False

        # Calculate similarity matrix
        similarity_matrix = []
        for item1 in seq1:
            row = []
            for item2 in seq2:
                # Use recursive comparison which will use SemanticString's comparison if applicable
                similarity = FieldComparator._calculate_item_similarity(item1, item2)
                row.append(similarity)
            similarity_matrix.append(row)

        # Find best matches using greedy approach (could be optimized with Hungarian algorithm)
        matched_indices_2 = set()
        total_score = 0
        matched_count = 0

        for i, row in enumerate(similarity_matrix):
            best_j = -1
            best_score = threshold  # Only consider matches above threshold

            for j, score in enumerate(row):
                if j not in matched_indices_2 and score > best_score:
                    best_j = j
                    best_score = score

            if best_j >= 0:
                matched_indices_2.add(best_j)
                total_score += best_score
                matched_count += 1

        # Calculate final score considering unmatched items
        max_possible_matches = max(len(seq1), len(seq2))
        if max_possible_matches == 0:
            return True

        # Penalize for unmatched items
        final_score = total_score / max_possible_matches
        return final_score >= threshold

    @staticmethod
    def _calculate_item_similarity(item1: Any, item2: Any) -> float:
        """Calculate similarity score between two items (0.0 to 1.0)."""
        # Handle SemanticString comparison
        if hasattr(item1, "__class__") and hasattr(item2, "__class__"):
            if item1.__class__.__name__ == "SemanticString" and item2.__class__.__name__ == "SemanticString":
                # Use semantic comparison for SemanticString
                if hasattr(item1, "root") and hasattr(item2, "root"):
                    # Encode strings first, then calculate similarity
                    emb1 = PotionEightEncoder.encode(str(item1))
                    emb2 = PotionEightEncoder.encode(str(item2))
                    similarity = PotionEightEncoder.cosine_similarity(emb1, emb2)
                    return similarity

        # Handle BaseModel comparison recursively
        if isinstance(item1, BaseModel) and isinstance(item2, BaseModel):
            if type(item1) is type(item2):
                # Compare all fields and average their similarity
                total_similarity = 0
                field_count = 0
                for field_name in item1.model_fields:
                    if hasattr(item2, field_name):
                        val1 = getattr(item1, field_name)
                        val2 = getattr(item2, field_name)
                        # Recursively compare field values
                        if FieldComparator._compare_items_recursively(val1, val2, 0.7):
                            total_similarity += 1.0
                        field_count += 1
                return total_similarity / field_count if field_count > 0 else 0.0
            return 0.0

        # Handle string comparison
        if isinstance(item1, str) and isinstance(item2, str):
            # Encode strings first, then calculate similarity
            emb1 = PotionEightEncoder.encode(item1)
            emb2 = PotionEightEncoder.encode(item2)
            return PotionEightEncoder.cosine_similarity(emb1, emb2)

        # Exact comparison for other types
        return 1.0 if item1 == item2 else 0.0

    @staticmethod
    def _compare_model_derived(obj1: Any, obj2: Any, threshold: float) -> bool:
        """Compare two BaseModel objects using recursive field comparison."""
        # Import here to avoid circular dependency
        from pydantic import BaseModel, RootModel

        # Handle RootModel special case
        if isinstance(obj1, RootModel) and isinstance(obj2, RootModel):
            if type(obj1) is not type(obj2):
                return False
            # For RootModel, compare the root values directly
            # Check if the RootModel has any special comparison metadata
            root_field = obj1.__class__.model_fields.get("root")
            if root_field:
                voting_meta = FieldComparator._extract_voting_metadata(root_field)
                return FieldComparator.compare_fields(
                    obj1.root,
                    obj2.root,
                    strategy=voting_meta.strategy,
                    tolerance=voting_meta.tolerance,
                    threshold=voting_meta.threshold,
                    decimal_places=voting_meta.decimal_places,
                    custom_comparator_fn=voting_meta.custom_comparator_fn,
                )
            # Fallback to direct comparison
            return FieldComparator._compare_items_recursively(obj1.root, obj2.root, threshold)

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

        if isinstance(item1, RootModel) and isinstance(item2, RootModel):
            return FieldComparator._compare_model_derived(item1, item2, threshold)
        elif isinstance(item1, BaseModel) and isinstance(item2, BaseModel):
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
    def _extract_voting_metadata(field_info: FieldInfo) -> "VotingMetadata":
        """Extract voting metadata from field info."""
        voting_meta = VotingMetadata()  # Default values

        # Extract from the metadata list where VotingField stores it
        if field_info and field_info.metadata:
            for metadata_item in field_info.metadata:
                # Check both direct dict and nested json_schema_extra
                if isinstance(metadata_item, dict):
                    # First check for voting_comparison directly
                    if "voting_comparison" in metadata_item:
                        voting_comparison = metadata_item["voting_comparison"]
                    # Also check in json_schema_extra (where VotingField stores it for non-model types)
                    elif "json_schema_extra" in metadata_item and isinstance(metadata_item["json_schema_extra"], dict):
                        json_extra = metadata_item["json_schema_extra"]
                        if "voting_comparison" in json_extra:
                            voting_comparison = json_extra["voting_comparison"]
                        else:
                            continue
                    else:
                        continue

                    if isinstance(voting_comparison, dict):
                        # Parse the dict into VotingMetadata - safely extract fields
                        voting_meta_kwargs: dict[str, Any] = {}
                        if "strategy" in voting_comparison and isinstance(
                            voting_comparison["strategy"], (str, ComparisonStrategy)
                        ):
                            voting_meta_kwargs["strategy"] = ComparisonStrategy(voting_comparison["strategy"])
                        if "tolerance" in voting_comparison and isinstance(
                            voting_comparison["tolerance"], (int, float)
                        ):
                            voting_meta_kwargs["tolerance"] = voting_comparison["tolerance"]
                        if "decimal_places" in voting_comparison and isinstance(
                            voting_comparison["decimal_places"], (int, float)
                        ):
                            voting_meta_kwargs["decimal_places"] = voting_comparison["decimal_places"]
                        if "threshold" in voting_comparison and isinstance(
                            voting_comparison["threshold"], (int, float)
                        ):
                            voting_meta_kwargs["threshold"] = voting_comparison["threshold"]
                        if "custom_comparator_fn" in voting_comparison and callable(
                            voting_comparison["custom_comparator_fn"]
                        ):
                            voting_meta_kwargs["custom_comparator_fn"] = voting_comparison["custom_comparator_fn"]
                        elif "custom_comparator" in voting_comparison and callable(
                            voting_comparison["custom_comparator"]
                        ):
                            voting_meta_kwargs["custom_comparator_fn"] = voting_comparison["custom_comparator"]

                        voting_meta = VotingMetadata(**voting_meta_kwargs)
                        break  # Found our metadata, stop looking

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
    custom_comparator_fn: Optional[Callable[[Any, Any], bool]] = ...,
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
    custom_comparator_fn: Optional[Callable[[Any, Any], bool]] = ...,
    **kwargs: Any,
) -> _T: ...


@overload
def VotingField(
    *,
    comparison: ComparisonStrategy = ...,
    tolerance: Optional[float] = ...,
    threshold: Optional[float] = ...,
    decimal_places: Optional[int] = ...,
    custom_comparator_fn: Optional[Callable[[Any, Any], bool]] = ...,
    **kwargs: Any,
) -> Any: ...


def VotingField(
    *args: Any,
    comparison: ComparisonStrategy = ComparisonStrategy.EXACT,
    tolerance: Optional[float] = None,
    threshold: Optional[float] = None,
    decimal_places: Optional[int] = None,
    custom_comparator_fn: Optional[Callable[[Any, Any], bool]] = None,
    **kwargs: Any,
) -> Any:
    """
    Enhanced Field function that includes voting comparison metadata.

    Works with both regular BaseModel fields AND RootModel root fields!
    All standard Field parameters are supported (default, description, alias, etc.)

    Args:
        *args: Positional arguments passed to Field
        comparison: How this field should be compared during voting (default: EXACT)
        tolerance: For RANGE strategy, the bin width (e.g., 0.2 for 20% bins, allows ~10% variance)
        threshold: For SEMANTIC strategy, minimum cosine similarity (e.g., 0.8 for 80%)
        custom_comparator_fn: For CUSTOM strategy, a function(value1, value2) -> bool
        **kwargs: All standard Pydantic Field keyword arguments

    Usage:
        # Regular BaseModel field
        class MyResponse(BaseModelWithReasoning):
            answer: int = VotingField(description="The answer")
            confidence: float = VotingField(
                default=0.0,
                comparison=ComparisonStrategy.IGNORE,
                description="Model confidence"
            )

        # RootModel - VotingField works here too!
        class SemanticString(RootModel[str]):
            root: str = VotingField(
                comparison=ComparisonStrategy.SEMANTIC,
                threshold=0.75,
                description="Semantic string"
            )
    """
    import inspect
    import logging
    from typing import get_args, get_origin

    # Check if description is provided for List[ComplexType] or model references and warn
    if "description" in kwargs:
        # Get the calling frame to identify which field we're currently processing
        frame = inspect.currentframe()
        if frame and frame.f_back:
            # We're in a class definition, try to get the field name from the calling context
            caller_frame = frame.f_back
            caller_locals = caller_frame.f_locals

            # Try to get the current field name by checking what's being assigned
            # This is a heuristic - look at the line being executed
            try:
                import linecache

                filename = caller_frame.f_code.co_filename
                lineno = caller_frame.f_lineno
                line = linecache.getline(filename, lineno).strip()

                # Extract field name from assignment line like "field_name: Type = VotingField(...)"
                if ":" in line and "=" in line:
                    field_part = line.split(":")[0].strip()
                    field_name = field_part.split()[-1] if " " in field_part else field_part

                    # Check the field's type annotation
                    annotations = caller_locals.get("__annotations__", {})
                    if field_name in annotations:
                        annotation = annotations[field_name]
                        origin = get_origin(annotation)

                        # Check if this field is a List[ComplexType]
                        is_list_of_models = False
                        if origin is list or (hasattr(origin, "__name__") and origin.__name__ == "list"):
                            type_args = get_args(annotation)
                            if type_args:
                                inner_type = type_args[0]
                                try:
                                    if issubclass(inner_type, (BaseModel, RootModel)):
                                        logger = logging.getLogger(__name__)
                                        logger.warning(
                                            f"VotingField with description used for List[{inner_type.__name__}] field '{field_name}'. "
                                            f"OpenAI JSON Schema does not allow descriptions on $ref fields. "
                                            f"Consider moving description to agent instructions for array length preferences. "
                                            f"Description: '{kwargs['description']}'"
                                        )
                                        is_list_of_models = True
                                except TypeError:
                                    # Not a class, skip
                                    pass

                        # Check if this field directly references a BaseModel/RootModel (not wrapped in List)
                        if not is_list_of_models and origin is None:  # No generic wrapper, direct type reference
                            try:
                                # Check if the annotation is a class that inherits from BaseModel or RootModel
                                if isinstance(annotation, type) and issubclass(annotation, (BaseModel, RootModel)):
                                    logger = logging.getLogger(__name__)
                                    logger.warning(
                                        f"VotingField with description used for {annotation.__name__} field '{field_name}'. "
                                        f"OpenAI JSON Schema does not allow descriptions on $ref fields. "
                                        f"Move the description to the {annotation.__name__} model's fields or docstring. "
                                        f"Description: '{kwargs['description']}'"
                                    )
                            except (TypeError, AttributeError):
                                # Not a BaseModel/RootModel, this is fine
                                pass
            except Exception:
                # If anything fails in the heuristic, skip the warning
                pass

    # Build metadata dict with only non-None values
    voting_comparison: Dict[str, Union[ComparisonStrategy, float, int, Optional[Callable]]] = {"strategy": comparison}

    voting_comparison["tolerance"] = tolerance
    voting_comparison["threshold"] = threshold
    voting_comparison["decimal_places"] = decimal_places
    voting_comparison["custom_comparator_fn"] = custom_comparator_fn

    # Create the field with standard Pydantic Field
    field = PydanticField(*args, **kwargs)

    # Add our voting metadata to the field's metadata list
    # This is the proper way to attach custom metadata to a FieldInfo object
    # The metadata list is designed for this purpose and won't affect JSON schema
    if not field.metadata:
        field.metadata = []
    field.metadata.append({"voting_comparison": voting_comparison})

    return field


class BaseModelWithReasoning(BaseModel):
    """
    Base model that includes a reasoning field for explaining thought process.

    This base class can be inherited by any Pydantic model that needs to
    include reasoning or explanation for its outputs. The reasoning field
    is designed to capture the thought process behind the values in other fields.

    Uses VotingField with IGNORE comparison strategy so reasoning doesn't affect consensus,
    and semantic similarity threshold of 0.80 for when comparing reasoning text.
    """

    reasoning: str = VotingField(
        # Enforce minimum length to ensure detailed reasoning while balancing thoroughness with clarity.
        # The 100-character requirement compensates for LLMs' character counting limitations without encouraging verbosity.
        min_length=30,
        comparison=ComparisonStrategy.IGNORE,
        description=dedent(
            """
            Before answering, work through this step-by-step:

            1. UNDERSTAND: What is the core question being asked or what is the task?
            2. ANALYZE: What are the key factors/components involved?
            3. REASON: What logical connections can I make?
            4. SYNTHESIZE: How do these elements combine?
            5. CONCLUDE: What is the most accurate/helpful response?

            IMPORTANT: This reasoning field must be at least 100 characters long.
            Provide detailed explanations and avoid abbreviated responses.
            """
        ),
    )


class SemanticString(RootModel[str]):
    root: str = VotingField(
        comparison=ComparisonStrategy.SEMANTIC,
        threshold=0.75,
        description="Semantic similarity for the string",
    )

    def __repr__(self) -> str:
        return self.root

    def __str__(self) -> str:
        """Convert SemanticString to regular string."""
        return self.root

    @property
    def value(self) -> str:
        """Backwards compatibility alias for the root string value."""
        return self.root
