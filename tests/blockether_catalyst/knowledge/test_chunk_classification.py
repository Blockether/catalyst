#!/usr/bin/env python3
"""Quick verification that chunk content classification works correctly."""

import asyncio

from blockether_catalyst.consensus.VotingComparison import ComparisonStrategy
from blockether_catalyst.knowledge.KnowledgeTypes import ChunkContentClassification


def test_chunk_classification():
    """Test that ChunkContentClassification model works correctly."""

    # Create instance with multiple semantic types
    classification = ChunkContentClassification(
        reasoning="This chunk contains a numbered list of items which suggests a table of contents structure. The numbering and hierarchical organization clearly indicate it's meant to guide readers through document sections. Additionally, it provides concrete examples of each item to illustrate the concepts being discussed.",
        semantic_types=["table_of_contents", "example"],
        confidence_scores={"table_of_contents": 0.85, "example": 0.75},
        key_indicators={
            "table_of_contents": ["numbered list", "sections"],
            "example": ["demonstrates", "shows"],
        },
    )

    print("✓ Created ChunkContentClassification with multiple semantic types")
    print(f"  Semantic types: {classification.semantic_types}")
    print(f"  Confidence scores: {classification.confidence_scores}")

    # Verify VotingField comparison works
    classification2 = ChunkContentClassification(
        reasoning="Different reasoning but same classifications. The content shows clear organizational structure with numbered items forming a table of contents. Each entry also includes detailed examples to help readers understand the concepts better, making it both navigational and illustrative.",
        semantic_types=["table_of_contents", "example"],  # EXACT match required
        confidence_scores={"table_of_contents": 0.90, "example": 0.70},  # IGNORED
        key_indicators={
            "table_of_contents": ["different"],
            "example": ["other"],
        },  # IGNORED
    )

    # Test that semantic_types use EXACT comparison
    from blockether_catalyst.consensus.VotingComparison import FieldComparator

    # Should be True - exact match on semantic_types
    result = FieldComparator.compare_fields(
        classification.semantic_types,
        classification2.semantic_types,
        strategy=ComparisonStrategy.EXACT,
    )
    print(f"✓ EXACT comparison of same semantic_types: {result}")
    assert result

    # Different order should still match with EXACT (lists are ordered)
    classification3 = ChunkContentClassification(
        reasoning="Order matters in list comparison. When using EXACT comparison strategy, the order of elements in the list is significant. This means that ['example', 'table_of_contents'] is different from ['table_of_contents', 'example'] even though they contain the same elements.",
        semantic_types=["example", "table_of_contents"],  # Different order
        confidence_scores={},
        key_indicators={},
    )

    result = FieldComparator.compare_fields(
        classification.semantic_types,
        classification3.semantic_types,
        strategy=ComparisonStrategy.EXACT,
    )
    print(f"✓ EXACT comparison of different order: {result}")
    assert not result  # Should be False because order matters with EXACT

    # Test with different semantic types
    classification4 = ChunkContentClassification(
        reasoning="This chunk has completely different semantic types. It contains a summary section that provides an overview of the content, and also includes specific rules and regulations that must be followed. These are distinct from table_of_contents and example classifications.",
        semantic_types=["summary", "rule"],
        confidence_scores={"summary": 0.8, "rule": 0.9},
        key_indicators={},
    )

    result = FieldComparator.compare_fields(
        classification.semantic_types,
        classification4.semantic_types,
        strategy=ComparisonStrategy.EXACT,
    )
    print(f"✓ EXACT comparison of different semantic_types: {result}")
    assert not result

    print("\n✅ All chunk classification tests passed!")


if __name__ == "__main__":
    test_chunk_classification()
