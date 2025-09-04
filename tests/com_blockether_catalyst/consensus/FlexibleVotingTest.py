"""
Test cases for flexible field comparison in voting consensus.

This test suite validates that the consensus mechanism correctly:
1. Uses field-specific comparison strategies
2. Ignores fields marked as IGNORE
"""

from typing import Any, List

import pytest

from com_blockether_catalyst.consensus.ConsensusCore import ConsensusCore
from com_blockether_catalyst.consensus.internal.ConsensusTypes import (
    ConsensusSettings,
    TypedCallBaseForConsensus,
)
from com_blockether_catalyst.consensus.internal.VotingComparison import (
    ComparisonStrategy,
    VotingField,
)
from com_blockether_catalyst.utils.TypedCalls import ArityOneTypedCall


class FlexibleResponse(TypedCallBaseForConsensus):
    """Response model with flexible field comparison."""

    # Core answer - must match exactly
    answer: int = VotingField(description="The main answer", comparison=ComparisonStrategy.EXACT)

    # Confidence is ignored for voting (metadata field)
    confidence: float = VotingField(
        default=0.8,
        comparison=ComparisonStrategy.IGNORE,
        description="Model confidence - not used for voting",
    )

    # Score within range considered same vote
    # tolerance=0.2 means 20% bin width, allowing ~10% variance
    score: float = VotingField(
        comparison=ComparisonStrategy.RANGE,
        tolerance=0.2,  # 20% bin width for ~10% matching
        description="Score with tolerance",
    )

    # Category compared using semantic similarity
    category: str = VotingField(
        comparison=ComparisonStrategy.SEMANTIC,
        threshold=0.9,
        description="Category name with semantic matching",
    )

    # Tags compared as sets (order doesn't matter)
    tags: List[str] = VotingField(
        default_factory=list,
        comparison=ComparisonStrategy.SEQUENCE_UNORDERED_ALIKE,
        description="Tags in any order",
    )


class MockFlexibleCall(ArityOneTypedCall[str, FlexibleResponse]):
    """Mock typed call for testing flexible voting."""

    def __init__(self, response: FlexibleResponse):
        self._response = response

    async def call(self, x: str) -> FlexibleResponse:
        return self._response


class TestFlexibleVoting:
    """Test flexible field comparison in voting."""

    @pytest.fixture
    def mock_judge(self) -> Any:
        """Create mock judge typed call for tie-breaking."""
        return MockFlexibleCall(
            FlexibleResponse(
                answer=42,
                confidence=0.9,
                score=95.0,
                category="Judge",
                tags=["judge"],
                reasoning="Judge decision for tie-breaking based on comprehensive analysis.",
            )
        )

    @pytest.mark.anyio
    async def test_ignore_confidence_field(self, mock_judge: Any) -> None:
        """Test that confidence field is ignored in voting."""
        # Same answer but different confidence - should vote together
        response1 = FlexibleResponse(
            answer=42,
            confidence=0.9,  # High confidence
            score=95.0,
            category="Math",
            tags=["algebra", "basic"],
            reasoning="The answer is 42 based on mathematical calculation of the given equation.",
        )
        response2 = FlexibleResponse(
            answer=42,
            confidence=0.5,  # Low confidence - should be ignored
            score=95.0,
            category="Math",
            tags=["algebra", "basic"],
            reasoning="The answer is 42 based on mathematical calculation of the given equation.",
        )
        response3 = FlexibleResponse(
            answer=50,  # Different answer
            confidence=0.9,
            score=95.0,
            category="Math",
            tags=["algebra", "basic"],
            reasoning="I believe the answer is 50 based on my interpretation of the problem.",
        )

        models = [
            ConsensusCore.configuration(
                id="model1",
                executor=MockFlexibleCall(response1),
                perspective="High confidence perspective",
            ),
            ConsensusCore.configuration(
                id="model2",
                executor=MockFlexibleCall(response2),
                perspective="Low confidence perspective",
            ),
            ConsensusCore.configuration(
                id="model3",
                executor=MockFlexibleCall(response3),
                perspective="Alternative perspective",
            ),
        ]

        consensus = ConsensusCore.consensus(
            models=models,
            judge=mock_judge,
            settings=ConsensusSettings(max_rounds=1),
        )

        result = await consensus.call("What is 40 + 2?")

        # Models 1 and 2 should vote together despite different confidence
        assert result.consensus_achieved is True
        assert result.final_response.answer == 42

    @pytest.mark.anyio
    async def test_range_comparison(self, mock_judge: Any) -> None:
        """Test that values within range tolerance vote together."""
        # Scores within 10% should vote together
        response1 = FlexibleResponse(
            answer=100,
            score=90.0,
            category="Test",
            tags=["test"],
            reasoning="Calculated score of 90 based on performance metrics and analysis criteria.",
        )
        response2 = FlexibleResponse(
            answer=100,
            score=88.0,  # Within 10% of 90
            category="Test",
            tags=["test"],
            reasoning="Calculated score of 88 which is very close to the expected range.",
        )
        response3 = FlexibleResponse(
            answer=100,
            score=70.0,  # Outside 10% range
            category="Test",
            tags=["test"],
            reasoning="Calculated score of 70 which indicates a different performance level.",
        )

        models = [
            ConsensusCore.configuration(
                id="model1",
                executor=MockFlexibleCall(response1),
                perspective="First scorer",
            ),
            ConsensusCore.configuration(
                id="model2",
                executor=MockFlexibleCall(response2),
                perspective="Second scorer",
            ),
            ConsensusCore.configuration(
                id="model3",
                executor=MockFlexibleCall(response3),
                perspective="Third scorer",
            ),
        ]

        consensus = ConsensusCore.consensus(
            models=models,
            judge=mock_judge,
            settings=ConsensusSettings(max_rounds=1),
        )

        result = await consensus.call("Calculate the score")

        # With logarithmic binning for RANGE comparison, models 1 and 2 should vote together
        # Scores 90 and 88 are within 10% tolerance and get the same bin
        # Score 70 is outside the range and gets a different bin
        assert result.consensus_achieved is True
        assert result.final_response.answer == 100
        # Score should be from the majority group (either 90 or 88)
        assert result.final_response.score in [90.0, 88.0]

    @pytest.mark.anyio
    async def test_case_insensitive_comparison(self, mock_judge: Any) -> None:
        """Test that string fields can be compared case-insensitively."""
        response1 = FlexibleResponse(
            answer=100,
            score=95.0,
            category="Mathematics",  # Title case
            tags=["test"],
            reasoning="Category is Mathematics with proper capitalization for formal presentation.",
        )
        response2 = FlexibleResponse(
            answer=100,
            score=95.0,
            category="MATHEMATICS",  # Upper case - should match
            tags=["test"],
            reasoning="Category is MATHEMATICS in all caps for emphasis and clarity.",
        )
        response3 = FlexibleResponse(
            answer=100,
            score=95.0,
            category="Science",  # Different category
            tags=["test"],
            reasoning="Category is Science which is a different field of study altogether.",
        )

        models = [
            ConsensusCore.configuration(
                id="model1",
                executor=MockFlexibleCall(response1),
                perspective="Title case",
            ),
            ConsensusCore.configuration(
                id="model2",
                executor=MockFlexibleCall(response2),
                perspective="Upper case",
            ),
            ConsensusCore.configuration(
                id="model3",
                executor=MockFlexibleCall(response3),
                perspective="Different category",
            ),
        ]

        consensus = ConsensusCore.consensus(
            models=models,
            judge=mock_judge,
            settings=ConsensusSettings(max_rounds=1),
        )

        result = await consensus.call("Categorize this")

        # Models 1 and 2 should vote together (same category, different case)
        assert result.consensus_achieved is True
        assert result.final_response.category.lower() == "mathematics"

    @pytest.mark.anyio
    async def test_set_equality_comparison(self, mock_judge: Any) -> None:
        """Test that lists are compared as sets (order doesn't matter)."""
        response1 = FlexibleResponse(
            answer=42,
            score=95.0,
            category="Test",
            tags=["alpha", "beta", "gamma"],  # Order 1
            reasoning="Tags are alpha, beta, gamma in alphabetical order for consistency.",
        )
        response2 = FlexibleResponse(
            answer=42,
            score=95.0,
            category="Test",
            tags=["gamma", "alpha", "beta"],  # Different order - should match
            reasoning="Tags are gamma, alpha, beta in a different but equivalent order.",
        )
        response3 = FlexibleResponse(
            answer=42,
            score=95.0,
            category="Test",
            tags=["alpha", "delta"],  # Different tags
            reasoning="Tags are alpha and delta which represents a different classification.",
        )

        models = [
            ConsensusCore.configuration(
                id="model1",
                executor=MockFlexibleCall(response1),
                perspective="Ordered tags",
            ),
            ConsensusCore.configuration(
                id="model2",
                executor=MockFlexibleCall(response2),
                perspective="Reordered tags",
            ),
            ConsensusCore.configuration(
                id="model3",
                executor=MockFlexibleCall(response3),
                perspective="Different tags",
            ),
        ]

        consensus = ConsensusCore.consensus(
            models=models,
            judge=mock_judge,
            settings=ConsensusSettings(max_rounds=1),
        )

        result = await consensus.call("Tag this item")

        # Models 1 and 2 should vote together (same tags, different order)
        assert result.consensus_achieved is True
        assert set(result.final_response.tags) == {"alpha", "beta", "gamma"}

    @pytest.mark.anyio
    async def test_combined_strategies(self, mock_judge: Any) -> None:
        """Test multiple comparison strategies working together."""
        # These should all vote together despite differences
        response1 = FlexibleResponse(
            answer=42,
            confidence=0.9,  # Ignored
            score=100.0,
            category="Math",  # Case will be normalized
            tags=["a", "b"],
            reasoning="Answer 42 with score 100 in Math category with tags a and b ordered.",
        )
        response2 = FlexibleResponse(
            answer=42,
            confidence=0.3,  # Different but ignored
            score=98.0,  # Within 10% range
            category="MATH",  # Different case but matches
            tags=["b", "a"],  # Different order but same set
            reasoning="Answer 42 with score 98 in MATH category with tags b and a reordered.",
        )

        models = [
            ConsensusCore.configuration(
                id="model1",
                executor=MockFlexibleCall(response1),
                perspective="Model 1",
            ),
            ConsensusCore.configuration(
                id="model2",
                executor=MockFlexibleCall(response2),
                perspective="Model 2",
            ),
        ]

        consensus = ConsensusCore.consensus(
            models=models,
            judge=mock_judge,
            settings=ConsensusSettings(max_rounds=1),
        )

        result = await consensus.call("Complex query")

        # With proper RANGE comparison using logarithmic binning,
        # scores 100 and 98 (within 10%) should get the same voting key
        # Combined with other matching fields, consensus should be achieved
        assert result.consensus_achieved is True
        assert result.final_response.answer == 42
        assert result.total_rounds == 1
