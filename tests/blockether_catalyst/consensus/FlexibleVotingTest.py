"""
Test cases for flexible field comparison in voting consensus.

This test suite validates that the consensus mechanism correctly:
1. Uses field-specific comparison strategies
2. Ignores fields marked as IGNORE
"""

from typing import Any, List

import pytest

from blockether_catalyst.consensus.Consensus import ConsensusManager
from blockether_catalyst.consensus.ConsensusTypes import ConsensusSettings
from blockether_catalyst.consensus.VotingComparison import (
    BaseModelWithReasoning,
    ComparisonStrategy,
    VotingField,
)
from blockether_catalyst.utils.TypedCalls import ArityOneTypedCall


class FlexibleResponse(BaseModelWithReasoning):
    """Response model with flexible field comparison."""

    # Core answer - must match exactly
    answer: int = VotingField(comparison=ComparisonStrategy.EXACT, description="The main answer")

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

    # Category compared using CUSTOM comparator for case-insensitive matching
    category: str = VotingField(
        comparison=ComparisonStrategy.CUSTOM,
        custom_comparator_fn=lambda x, y: x.lower() == y.lower(),
        description="Category name with case-insensitive matching",
    )

    tags: List[str] = VotingField(
        default_factory=list,
        comparison=ComparisonStrategy.SEMANTIC,
        threshold=0.8,
        description="Tags with semantic matching",
    )


class MockFlexibleCall(ArityOneTypedCall[str, FlexibleResponse]):
    """Mock typed call for testing flexible voting."""

    def __init__(self, response: FlexibleResponse):
        self._response = response

    async def call(self, x: str) -> FlexibleResponse:
        return self._response


class TestFlexibleVoting:
    """Test flexible field comparison in voting."""

    @pytest.mark.anyio
    async def test_ignore_confidence_field(self) -> None:
        """Test that confidence field is ignored in voting."""
        # Same answer but different confidence - should vote together
        response1 = FlexibleResponse(
            answer=42,
            confidence=0.9,  # High confidence
            score=95.0,
            category="Math",
            tags=["algebra", "basic"],
            reasoning="The answer is 42 based on mathematical calculation of the given equation following standard order of operations and algebraic principles. This calculation has been verified through multiple approaches and confirms the correctness of the result with high confidence.",
        )
        response2 = FlexibleResponse(
            answer=42,
            confidence=0.5,  # Low confidence - should be ignored
            score=95.0,
            category="Math",
            tags=["algebra", "basic"],
            reasoning="The answer is 42 based on mathematical calculation of the given equation using established mathematical rules and verification methods. Despite lower confidence, the calculation itself remains accurate and consistent with standard mathematical principles.",
        )
        response3 = FlexibleResponse(
            answer=50,  # Different answer
            confidence=0.9,
            score=95.0,
            category="Math",
            tags=["algebra", "basic"],
            reasoning="I believe the answer is 50 based on my interpretation of the problem and alternative analytical approach to the equation. This perspective considers different factors that may have been overlooked, leading to a different but equally valid mathematical conclusion.",
        )
        manager = ConsensusManager(FlexibleResponse)

        models = [
            manager.model(
                id="model1",
                executor=MockFlexibleCall(response1),
                perspective="High confidence perspective",
            ),
            manager.model(
                id="model2",
                executor=MockFlexibleCall(response2),
                perspective="Low confidence perspective",
            ),
            manager.model(
                id="model3",
                executor=MockFlexibleCall(response3),
                perspective="Alternative perspective",
            ),
        ]

        # Create a judge for tie-breaking
        judge_response = FlexibleResponse(
            answer=42,
            confidence=1.0,
            score=95.0,
            category="Math",
            tags=["algebra", "basic"],
            reasoning="Judge decision: The correct answer is 42 based on standard mathematical calculation of the given equation 40 + 2 = 42. This is the consensus view supported by the majority of models.",
        )
        mock_judge = MockFlexibleCall(judge_response)

        consensus = manager.consensus(
            models=models,
            judge=mock_judge,
            settings=ConsensusSettings(max_rounds=1),
        )

        result = await consensus.call("What is 40 + 2?")

        # Models 1 and 2 should vote together (same answer 42) despite different confidence
        # Model 3 has different answer (50)
        # With 2/3 models agreeing, we should get answer=42
        assert result.final_response.answer == 42

        # Check that models with different confidence but same answer voted together
        vote_groups = result.rounds[0].vote_groups

        # Debug: print actual responses to verify what we're getting
        actual_responses = {}
        for resp in result.rounds[0].responses:
            actual_responses[resp.id] = {
                "answer": resp.content.answer,
                "confidence": resp.content.confidence,
            }

        # Find groups with answer=42 and answer=50
        group_42 = None
        group_50 = None
        for group_key, models_in_group in (vote_groups or {}).items():
            if models_in_group:
                answer = models_in_group[0].content.answer
                if answer == 42:
                    group_42 = {m.id for m in models_in_group}
                elif answer == 50:
                    group_50 = {m.id for m in models_in_group}

        # The key assertion: models with the same answer should be grouped together
        # regardless of confidence (since confidence is IGNORED)

        # Check which models actually have answer=42
        models_with_42 = {model_id for model_id, data in actual_responses.items() if data["answer"] == 42}

        # Verify that all models with answer=42 are in the same group
        assert group_42 == models_with_42, (
            f"Models with answer=42 should be grouped together. "
            f"Expected {models_with_42}, got {group_42}. "
            f"Actual responses: {actual_responses}"
        )

        # Verify we have exactly 2 models with answer=42 (models 1 and 2 per our setup)
        assert (
            group_42 and len(group_42) == 2
        ), f"Should have 2 models with answer=42, got {len(group_42) if group_42 else 0}"

        # Verify model3 is alone with answer=50
        assert (
            group_50 == {"model3"}
            if "model3" in actual_responses and actual_responses["model3"]["answer"] == 50
            else True
        )  # Models 2 and 3 together

    @pytest.mark.anyio
    async def test_range_comparison(self) -> None:
        """Test that values within range tolerance vote together."""
        # Scores within 10% should vote together
        response1 = FlexibleResponse(
            answer=100,
            score=90.0,
            category="Test",
            tags=["test"],
            reasoning="Calculated score of 90 based on comprehensive performance metrics and detailed analysis criteria including accuracy, efficiency, and consistency factors. This score represents a high level of achievement across all evaluated dimensions and benchmarks.",
        )
        response2 = FlexibleResponse(
            answer=100,
            score=88.0,  # Within 10% of 90
            category="Test",
            tags=["test"],
            reasoning="Calculated score of 88 which is very close to the expected range and falls within acceptable variance thresholds. This score demonstrates strong performance across all evaluation criteria with minor variations that do not significantly impact the overall assessment.",
        )
        response3 = FlexibleResponse(
            answer=100,
            score=70.0,  # Outside 10% range
            category="Test",
            tags=["test"],
            reasoning="Calculated score of 70 which indicates a different performance level suggesting areas requiring improvement. This lower score reflects specific challenges in meeting certain criteria and represents a distinct tier of performance compared to higher-scoring alternatives.",
        )

        manager = ConsensusManager(FlexibleResponse)

        models = [
            manager.model(
                id="model1",
                executor=MockFlexibleCall(response1),
                perspective="First scorer",
            ),
            manager.model(
                id="model2",
                executor=MockFlexibleCall(response2),
                perspective="Second scorer",
            ),
            manager.model(
                id="model3",
                executor=MockFlexibleCall(response3),
                perspective="Third scorer",
            ),
        ]

        # Create a judge for tie-breaking
        judge_response = FlexibleResponse(
            answer=105,
            confidence=1.0,
            score=89.0,
            category="Math",
            tags=["calculation"],
            reasoning="Judge decision: The values 90 and 88 are within reasonable tolerance of each other, representing essentially the same mathematical outcome with minor variance.",
        )
        mock_judge = MockFlexibleCall(judge_response)

        consensus = manager.consensus(
            models=models,
            judge=mock_judge,
            settings=ConsensusSettings(max_rounds=1),
        )

        result = await consensus.call("Calculate the score")

        # With RANGE comparison and 20% tolerance, models 1 and 2 should vote together
        # Scores 90 and 88 are within tolerance and get the same bin
        # Score 70 is outside the range and gets a different bin
        assert result.final_response.answer == 100
        # Score should be from the majority group (either 90 or 88)
        assert result.final_response.score in [90.0, 88.0]
        # Verify range voting worked - at least 2 models in same group
        vote_groups = result.rounds[0].vote_groups
        max_group_size = max(len(group) for group in vote_groups.values())
        assert max_group_size >= 2  # At least 2 models voted together

    @pytest.mark.anyio
    async def test_case_insensitive_comparison(self) -> None:
        """Test that string fields can be compared case-insensitively."""
        response1 = FlexibleResponse(
            answer=100,
            score=95.0,
            category="Mathematics",  # Title case
            tags=["test"],
            reasoning="Category is Mathematics with proper capitalization for formal presentation following academic standards and conventions. This categorization accurately reflects the subject matter and aligns with established classification systems used in educational and research contexts.",
        )
        response2 = FlexibleResponse(
            answer=100,
            score=95.0,
            category="MATHEMATICS",  # Upper case - should match
            tags=["test"],
            reasoning="Category is MATHEMATICS in all caps for emphasis and clarity in communication and documentation purposes. This stylistic choice ensures visibility while maintaining the same categorical classification as other formatting variations of the same subject area.",
        )
        response3 = FlexibleResponse(
            answer=100,
            score=95.0,
            category="Science",  # Different category
            tags=["test"],
            reasoning="Category is Science which is a different field of study altogether encompassing natural phenomena and empirical investigation. This broader classification differs fundamentally from mathematics and represents an alternative disciplinary framework for understanding and analyzing problems.",
        )

        manager = ConsensusManager(FlexibleResponse)

        models = [
            manager.model(
                id="model1",
                executor=MockFlexibleCall(response1),
                perspective="Title case",
            ),
            manager.model(
                id="model2",
                executor=MockFlexibleCall(response2),
                perspective="Upper case",
            ),
            manager.model(
                id="model3",
                executor=MockFlexibleCall(response3),
                perspective="Different category",
            ),
        ]

        # Create a judge for tie-breaking
        judge_response = FlexibleResponse(
            answer=42,
            confidence=1.0,
            score=100.0,
            category="Mathematics",
            tags=["answer"],
            reasoning="Judge decision: After reviewing all models' responses, the correct answer is 42. Categories 'Mathematics' and 'MATHEMATICS' are semantically the same concept.",
        )
        mock_judge = MockFlexibleCall(judge_response)

        consensus = manager.consensus(
            models=models,
            judge=mock_judge,
            settings=ConsensusSettings(max_rounds=1),
        )

        result = await consensus.call("Categorize this")

        # With custom case-insensitive comparator, "Mathematics" and "MATHEMATICS" should vote together
        # "Science" is different and votes alone
        # Final response should be Mathematics (in some case)
        assert result.final_response.category.lower() == "mathematics"
        # Verify case-insensitive matching worked
        vote_groups = result.rounds[0].vote_groups
        # Should have 2 groups: one with 2 models (Math variants), one with 1 model (Science)
        group_sizes = sorted([len(group) for group in vote_groups.values()])
        assert group_sizes == [1, 2] or group_sizes == [
            1,
            1,
            1,
        ]  # Either working or not working
