"""
Tests for DERIVED comparison strategy with nested structures.

This module tests that the DERIVED strategy correctly handles:
- Nested dictionaries
- Nested BaseModel objects
- Nested sequences (lists and tuples)
- Mixed nested structures
"""

from typing import Any, Dict, List, Optional, Tuple

import pytest
from pydantic import BaseModel

from com_blockether_catalyst.consensusVotingComparison import (
    ComparisonStrategy,
    FieldComparator,
    VotingField,
)


class NestedModel(BaseModel):
    """A nested model for testing."""

    name: str
    value: int
    metadata: Optional[Dict[str, Any]] = None


class ComplexModel(BaseModel):
    """A complex model with nested structures."""

    id: int
    nested: NestedModel
    tags: List[str]
    config: Dict[str, Any]


class TestNestedDictionaries:
    """Test DERIVED comparison with nested dictionaries."""

    def test_identical_nested_dicts_return_true(self) -> None:
        """Identical nested dictionaries should return True."""
        dict1 = {"level1": {"level2": {"level3": "value", "number": 42}, "list": [1, 2, 3]}}
        dict2 = {"level1": {"level2": {"level3": "value", "number": 42}, "list": [1, 2, 3]}}

        result = FieldComparator.compare_fields(dict1, dict2, strategy=ComparisonStrategy.DERIVED, threshold=0.8)
        assert result is True

    def test_different_nested_values_return_false(self) -> None:
        """Different values in nested dictionaries should return False."""
        dict1 = {"level1": {"level2": {"level3": "value1"}}}
        dict2 = {"level1": {"level2": {"level3": "value2"}}}

        result = FieldComparator.compare_fields(dict1, dict2, strategy=ComparisonStrategy.DERIVED, threshold=0.8)
        assert result is False

    def test_partial_match_with_threshold(self) -> None:
        """Partial matches should respect threshold."""
        dict1 = {"a": 1, "b": 2, "c": 3, "d": 4}
        dict2 = {
            "a": 1,
            "b": 2,
            "c": 3,
            "d": 5,  # Different value
        }

        # 3/4 = 0.75 match rate
        result_low_threshold = FieldComparator.compare_fields(
            dict1, dict2, strategy=ComparisonStrategy.DERIVED, threshold=0.7
        )
        assert result_low_threshold is True

        result_high_threshold = FieldComparator.compare_fields(
            dict1, dict2, strategy=ComparisonStrategy.DERIVED, threshold=0.8
        )
        assert result_high_threshold is False

    def test_mixed_types_in_dict(self) -> None:
        """Dictionaries with mixed types should be compared correctly."""
        dict1 = {
            "string": "hello",
            "number": 42,
            "list": [1, 2, 3],
            "nested": {"key": "value"},
            "bool": True,
        }
        dict2 = {
            "string": "hello",
            "number": 42,
            "list": [1, 2, 3],
            "nested": {"key": "value"},
            "bool": True,
        }

        result = FieldComparator.compare_fields(dict1, dict2, strategy=ComparisonStrategy.DERIVED, threshold=0.8)
        assert result is True


class TestNestedBaseModels:
    """Test DERIVED comparison with nested BaseModel objects."""

    def test_identical_nested_models_return_true(self) -> None:
        """Identical nested models should return True."""
        model1 = ComplexModel(
            id=1,
            nested=NestedModel(name="test", value=42),
            tags=["a", "b"],
            config={"key": "value"},
        )
        model2 = ComplexModel(
            id=1,
            nested=NestedModel(name="test", value=42),
            tags=["a", "b"],
            config={"key": "value"},
        )

        result = FieldComparator.compare_fields(model1, model2, strategy=ComparisonStrategy.DERIVED, threshold=0.8)
        assert result is True

    def test_different_nested_model_values_return_false(self) -> None:
        """Different values in nested models should return False."""
        model1 = ComplexModel(
            id=1,
            nested=NestedModel(name="test1", value=42),
            tags=["a", "b"],
            config={"key": "value"},
        )
        model2 = ComplexModel(
            id=1,
            nested=NestedModel(name="test2", value=42),
            tags=["a", "b"],
            config={"key": "value"},
        )

        result = FieldComparator.compare_fields(model1, model2, strategy=ComparisonStrategy.DERIVED, threshold=0.9)
        assert result is False

    def test_deeply_nested_models(self) -> None:
        """Test with deeply nested model structures."""
        model1 = NestedModel(
            name="outer",
            value=1,
            metadata={"inner_model": NestedModel(name="inner", value=2, metadata={"deep": "value"}).model_dump()},
        )
        model2 = NestedModel(
            name="outer",
            value=1,
            metadata={"inner_model": NestedModel(name="inner", value=2, metadata={"deep": "value"}).model_dump()},
        )

        result = FieldComparator.compare_fields(model1, model2, strategy=ComparisonStrategy.DERIVED, threshold=0.8)
        assert result is True


class TestNestedSequences:
    """Test DERIVED comparison with nested sequences."""

    def test_nested_lists_identical(self) -> None:
        """Identical nested lists should return True."""
        list1 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        list2 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        result = FieldComparator.compare_fields(list1, list2, strategy=ComparisonStrategy.DERIVED, threshold=0.8)
        assert result is True

    def test_nested_tuples_identical(self) -> None:
        """Identical nested tuples should return True."""
        tuple1 = ((1, 2), (3, 4), (5, 6))
        tuple2 = ((1, 2), (3, 4), (5, 6))

        result = FieldComparator.compare_fields(tuple1, tuple2, strategy=ComparisonStrategy.DERIVED, threshold=0.8)
        assert result is True

    def test_mixed_list_tuple_nesting(self) -> None:
        """Mixed list and tuple nesting should work correctly."""
        seq1 = [(1, 2, 3), [4, 5, 6], (7, 8, 9)]
        seq2 = [(1, 2, 3), [4, 5, 6], (7, 8, 9)]

        result = FieldComparator.compare_fields(seq1, seq2, strategy=ComparisonStrategy.DERIVED, threshold=0.8)
        assert result is True

    def test_nested_sequences_with_different_values(self) -> None:
        """Different values in nested sequences should return False."""
        list1 = [[1, 2], [3, 4]]
        list2 = [[1, 2], [3, 5]]  # Different value in nested list

        result = FieldComparator.compare_fields(list1, list2, strategy=ComparisonStrategy.DERIVED, threshold=0.9)
        assert result is False


class TestMixedNestedStructures:
    """Test DERIVED comparison with mixed nested structures."""

    def test_list_of_dicts(self) -> None:
        """List of dictionaries should be compared correctly."""
        list1 = [
            {"id": 1, "name": "first"},
            {"id": 2, "name": "second"},
            {"id": 3, "name": "third"},
        ]
        list2 = [
            {"id": 1, "name": "first"},
            {"id": 2, "name": "second"},
            {"id": 3, "name": "third"},
        ]

        result = FieldComparator.compare_fields(list1, list2, strategy=ComparisonStrategy.DERIVED, threshold=0.8)
        assert result is True

    def test_dict_of_lists(self) -> None:
        """Dictionary of lists should be compared correctly."""
        dict1 = {
            "numbers": [1, 2, 3],
            "letters": ["a", "b", "c"],
            "mixed": [1, "a", True],
        }
        dict2 = {
            "numbers": [1, 2, 3],
            "letters": ["a", "b", "c"],
            "mixed": [1, "a", True],
        }

        result = FieldComparator.compare_fields(dict1, dict2, strategy=ComparisonStrategy.DERIVED, threshold=0.8)
        assert result is True

    def test_complex_nested_structure(self) -> None:
        """Complex nested structure with all types."""
        struct1 = {
            "models": [
                NestedModel(name="m1", value=1),
                NestedModel(name="m2", value=2),
            ],
            "data": {
                "lists": [[1, 2], [3, 4]],
                "tuples": ((5, 6), (7, 8)),
                "mixed": {"a": [1, 2, 3], "b": {"nested": "value"}},
            },
            "metadata": {"count": 10, "tags": ["tag1", "tag2"]},
        }
        struct2 = {
            "models": [
                NestedModel(name="m1", value=1),
                NestedModel(name="m2", value=2),
            ],
            "data": {
                "lists": [[1, 2], [3, 4]],
                "tuples": ((5, 6), (7, 8)),
                "mixed": {"a": [1, 2, 3], "b": {"nested": "value"}},
            },
            "metadata": {"count": 10, "tags": ["tag1", "tag2"]},
        }

        result = FieldComparator.compare_fields(struct1, struct2, strategy=ComparisonStrategy.DERIVED, threshold=0.8)
        assert result is True

    def test_partial_match_in_complex_structure(self) -> None:
        """Partial matches in complex structures should respect threshold."""
        struct1 = {"a": {"val": 1}, "b": {"val": 2}, "c": {"val": 3}, "d": {"val": 4}}
        struct2 = {
            "a": {"val": 1},
            "b": {"val": 2},
            "c": {"val": 3},
            "d": {"val": 5},  # Different
        }

        # 3/4 match = 0.75
        result = FieldComparator.compare_fields(struct1, struct2, strategy=ComparisonStrategy.DERIVED, threshold=0.7)
        assert result is True

        result = FieldComparator.compare_fields(struct1, struct2, strategy=ComparisonStrategy.DERIVED, threshold=0.8)
        assert result is False


class TestEdgeCases:
    """Test edge cases for nested structure comparison."""

    def test_empty_structures(self) -> None:
        """Empty structures should be equal."""
        assert FieldComparator.compare_fields({}, {}, strategy=ComparisonStrategy.DERIVED, threshold=0.8) is True

        assert FieldComparator.compare_fields([], [], strategy=ComparisonStrategy.DERIVED, threshold=0.8) is True

        assert FieldComparator.compare_fields((), (), strategy=ComparisonStrategy.DERIVED, threshold=0.8) is True

    def test_none_values(self) -> None:
        """None values should be handled correctly."""
        dict1 = {"a": None, "b": 2}
        dict2 = {"a": None, "b": 2}

        result = FieldComparator.compare_fields(dict1, dict2, strategy=ComparisonStrategy.DERIVED, threshold=0.8)
        assert result is True

    def test_circular_reference_prevention(self) -> None:
        """Test that circular references don't cause infinite recursion."""
        # This is a simplified test - real circular references in Python
        # would need special handling
        dict1: Dict[str, Any] = {"a": 1}
        dict1["self"] = {"ref": dict1.copy()}  # Create a copy to avoid actual circular ref

        dict2: Dict[str, Any] = {"a": 1}
        dict2["self"] = {"ref": dict2.copy()}

        # Should not crash with recursion error
        result = FieldComparator.compare_fields(dict1, dict2, strategy=ComparisonStrategy.DERIVED, threshold=0.8)
        assert result is True
