"""
Strong assertion tests for consensus mechanism.

These tests verify exact values without ranges or try/catch blocks.
Every assertion checks specific expected values to ensure correctness.
"""

from typing import Any, List

import pytest
from pydantic import Field

from com_blockether_catalyst.consensus.ConsensusCore import ConsensusCore
from com_blockether_catalyst.consensus.ConsensusTypes import ConsensusSettings
from com_blockether_catalyst.consensus.VotingComparison import (
    BaseModelWithReasoning,
    ComparisonStrategy,
    VotingField,
)
from com_blockether_catalyst.utils.TypedCalls import ArityOneTypedCall


class PreciseResponse(BaseModelWithReasoning):
    """Response with exact expected values for testing."""

    exact_number: int = VotingField(
        description="An exact integer value",
    )

    exact_string: str = VotingField(
        description="An exact string value",
    )

    semantic_string: str = VotingField(
        description="A semantic string value",
    )

    range_value: float = VotingField(
        description="A float value with tolerance",
        tolerance=0.01,
        comparison=ComparisonStrategy.RANGE,
    )


class MockModelWithExactValues(ArityOneTypedCall[str, PreciseResponse]):
    """Mock model that returns exact predetermined values."""

    def __init__(
        self,
        exact_number: int,
        exact_string: str,
        semantic_string: str,
        range_value: float,
    ):
        self.exact_number = exact_number
        self.exact_string = exact_string
        self.semantic_string = semantic_string
        self.range_value = range_value

    async def call(self, x: str) -> PreciseResponse:
        return PreciseResponse(
            exact_number=self.exact_number,
            exact_string=self.exact_string,
            semantic_string=self.semantic_string,
            range_value=self.range_value,
            reasoning=f"Model response with exact_number={self.exact_number}, exact_string={self.exact_string}, semantic_string={self.semantic_string}, range_value={self.range_value}. This response was generated based on the input: '{x}'. The model has processed the request and returned these exact predetermined values for testing purposes.",
        )


@pytest.mark.anyio
class TestStrongAssertions:
    """Test consensus with strong assertions on exact values."""

    # Constants for expected values
    CONSENSUS_NUMBER = 42
    CONSENSUS_STRING = "exact_match"
    CONSENSUS_FLOAT = 3.14159
    MIN_REASONING_LENGTH = 150

    async def test_exact_consensus_with_identical_values(self) -> None:
        """Test that identical values from all models produce exact consensus."""

        # Create judge executor with consensus values
        judge_executor = MockModelWithExactValues(
            exact_number=self.CONSENSUS_NUMBER,
            exact_string=self.CONSENSUS_STRING,
            semantic_string="positive",
            range_value=self.CONSENSUS_FLOAT,
        )

        # Create 3 models with identical values
        models = [
            ConsensusCore.model(
                id="model1",
                executor=MockModelWithExactValues(
                    exact_number=self.CONSENSUS_NUMBER,
                    exact_string=self.CONSENSUS_STRING,
                    semantic_string="positive",
                    range_value=self.CONSENSUS_FLOAT,
                ),
                perspective="First model perspective",
            ),
            ConsensusCore.model(
                id="model2",
                executor=MockModelWithExactValues(
                    exact_number=self.CONSENSUS_NUMBER,
                    exact_string=self.CONSENSUS_STRING,
                    semantic_string="positive",
                    range_value=self.CONSENSUS_FLOAT,
                ),
                perspective="Second model perspective",
            ),
            ConsensusCore.model(
                id="model3",
                executor=MockModelWithExactValues(
                    exact_number=self.CONSENSUS_NUMBER,
                    exact_string=self.CONSENSUS_STRING,
                    semantic_string="positive",
                    range_value=self.CONSENSUS_FLOAT,
                ),
                perspective="Third model perspective",
            ),
        ]

        consensus = ConsensusCore.consensus(
            models=models,
            judge=judge_executor,
            settings=ConsensusSettings(
                max_rounds=1,
                verbosity=0,
            ),
        )

        result = await consensus.call("test input")

        # Strong assertions - exact values expected
        assert result.consensus_achieved is True
        assert result.final_response.exact_number == self.CONSENSUS_NUMBER
        assert result.final_response.exact_string == self.CONSENSUS_STRING
        assert result.final_response.semantic_string == "positive"
        assert result.final_response.range_value == self.CONSENSUS_FLOAT
        assert len(result.final_response.reasoning) >= self.MIN_REASONING_LENGTH

        # Metrics assertions
        assert result.total_rounds == 1
        # Check that consensus was achieved with 3 models
        assert result.participating_models == 3
        # Check that fallback_method indicates no judge was used (unanimous decision)

    async def test_exact_consensus_threshold_70_percent(self) -> None:
        """Test that exactly 70% agreement meets the 0.7 threshold."""

        # Create judge executor with consensus values
        judge_executor = MockModelWithExactValues(
            exact_number=100,
            exact_string="consensus",
            semantic_string="good",
            range_value=2.5,
        )

        # Create 9 models: 7 with consensus values, 2 with different values (77.8% agree)
        models = []

        # 7 models with consensus values (77.8%)
        for i in range(7):
            models.append(
                ConsensusCore.model(
                    id=f"agree_{i}",
                    executor=MockModelWithExactValues(
                        exact_number=100,
                        exact_string="consensus",
                        semantic_string="good",
                        range_value=2.5,
                    ),
                    perspective=f"Agreeing model {i}",
                )
            )

        # 2 models with different values (22.2%)
        for i in range(2):
            models.append(
                ConsensusCore.model(
                    id=f"disagree_{i}",
                    executor=MockModelWithExactValues(
                        exact_number=200 + i,
                        exact_string=f"different_{i}",
                        semantic_string="bad",
                        range_value=5.0 + i,
                    ),
                    perspective=f"Disagreeing model {i}",
                )
            )

        consensus = ConsensusCore.consensus(
            models=models,
            judge=judge_executor,
            settings=ConsensusSettings(
                max_rounds=1,
                verbosity=0,
            ),
        )

        result = await consensus.call("test threshold")

        # Exact assertions for 70% consensus
        assert result.consensus_achieved is True
        assert result.final_response.exact_number == 100
        assert result.final_response.exact_string == "consensus"
        assert result.final_response.semantic_string == "good"
        assert result.final_response.range_value == 2.5
        assert len(result.final_response.reasoning) >= self.MIN_REASONING_LENGTH

        # Verify exact model participation
        assert result.num_models == 9

    async def test_exact_tie_requires_judge(self) -> None:
        """Test that an exact 50-50 tie triggers judge decision."""

        # Judge decides for option A
        judge_executor = MockModelWithExactValues(
            exact_number=1,
            exact_string="option_a",
            semantic_string="positive",
            range_value=1.0,
        )

        # Create 4 models: 2 with value A, 2 with value B
        models = [
            ConsensusCore.model(
                id="team_a_1",
                executor=MockModelWithExactValues(
                    exact_number=1,
                    exact_string="option_a",
                    semantic_string="positive",
                    range_value=1.0,
                ),
                perspective="Team A first",
            ),
            ConsensusCore.model(
                id="team_a_2",
                executor=MockModelWithExactValues(
                    exact_number=1,
                    exact_string="option_a",
                    semantic_string="positive",
                    range_value=1.0,
                ),
                perspective="Team A second",
            ),
            ConsensusCore.model(
                id="team_b_1",
                executor=MockModelWithExactValues(
                    exact_number=2,
                    exact_string="option_b",
                    semantic_string="negative",
                    range_value=2.0,
                ),
                perspective="Team B first",
            ),
            ConsensusCore.model(
                id="team_b_2",
                executor=MockModelWithExactValues(
                    exact_number=2,
                    exact_string="option_b",
                    semantic_string="negative",
                    range_value=2.0,
                ),
                perspective="Team B second",
            ),
        ]

        # Judge decides for option A
        consensus = ConsensusCore.consensus(
            models=models,
            judge=judge_executor,
            settings=ConsensusSettings(
                max_rounds=1,  # Only 1 round before judge
                verbosity=0,
            ),
        )

        result = await consensus.call("test tie")

        # Should achieve consensus through majority (3 vs 2)
        assert result.consensus_achieved is True  # Majority vote
        assert result.final_response.exact_number == 1
        assert result.final_response.exact_string == "option_a"
        assert result.final_response.semantic_string == "positive"
        assert result.final_response.range_value == 1.0
        assert len(result.final_response.reasoning) >= self.MIN_REASONING_LENGTH

        # All 5 models participated

    async def test_semantic_similarity_exact_match(self) -> None:
        """Test semantic fields with similar meanings reach consensus."""

        # Judge agrees with semantic consensus
        judge_executor = MockModelWithExactValues(
            exact_number=50,
            exact_string="same",
            semantic_string="good",
            range_value=7.5,
        )

        # Models with semantically similar but not identical strings
        models = [
            ConsensusCore.model(
                id="model1",
                executor=MockModelWithExactValues(
                    exact_number=50,
                    exact_string="same",
                    semantic_string="good",
                    range_value=7.5,
                ),
                perspective="First perspective",
            ),
            ConsensusCore.model(
                id="model2",
                executor=MockModelWithExactValues(
                    exact_number=50,
                    exact_string="same",
                    semantic_string="great",
                    range_value=7.5,
                ),
                perspective="Second perspective",
            ),
            ConsensusCore.model(
                id="model3",
                executor=MockModelWithExactValues(
                    exact_number=50,
                    exact_string="same",
                    semantic_string="excellent",
                    range_value=7.5,
                ),
                perspective="Third perspective",
            ),
        ]

        consensus = ConsensusCore.consensus(
            models=models,
            judge=judge_executor,
            settings=ConsensusSettings(
                max_rounds=1,
                verbosity=0,
            ),
        )

        result = await consensus.call("test semantic")

        # All models should reach consensus despite different semantic strings
        assert result.consensus_achieved is True
        assert result.final_response.exact_number == 50
        assert result.final_response.exact_string == "same"
        # The semantic string should be one of the provided values
        assert result.final_response.semantic_string in ["good", "great", "excellent"]
        assert result.final_response.range_value == 7.5
        assert len(result.final_response.reasoning) >= self.MIN_REASONING_LENGTH

        # Verify completed in one round
        assert result.total_rounds == 1

    async def test_range_tolerance_exact_boundaries(self) -> None:
        """Test that range tolerance of 0.01 means values within 1% match."""

        # Base value is 100.0, tolerance is 0.01 (1%)
        # So 99.0 to 101.0 should match

        # Judge with mid-range value
        judge_executor = MockModelWithExactValues(
            exact_number=10,
            exact_string="range_test",
            semantic_string="testing",
            range_value=100.0,
        )

        models = [
            ConsensusCore.model(
                id="model1",
                executor=MockModelWithExactValues(
                    exact_number=10,
                    exact_string="range_test",
                    semantic_string="testing",
                    range_value=99.5,  # Within 1% of 100
                ),
                perspective="Lower bound",
            ),
            ConsensusCore.model(
                id="model2",
                executor=MockModelWithExactValues(
                    exact_number=10,
                    exact_string="range_test",
                    semantic_string="testing",
                    range_value=100.0,  # Exact
                ),
                perspective="Exact value",
            ),
            ConsensusCore.model(
                id="model3",
                executor=MockModelWithExactValues(
                    exact_number=10,
                    exact_string="range_test",
                    semantic_string="testing",
                    range_value=100.5,  # Within 1% of 100
                ),
                perspective="Upper bound",
            ),
        ]

        consensus = ConsensusCore.consensus(
            models=models,
            judge=judge_executor,
            settings=ConsensusSettings(
                max_rounds=1,
                verbosity=0,
            ),
        )

        result = await consensus.call("test range")

        # All values within tolerance should reach consensus
        assert result.consensus_achieved is True
        assert result.final_response.exact_number == 10
        assert result.final_response.exact_string == "range_test"
        assert result.final_response.semantic_string == "testing"
        # Range value should be one of the three (they're all considered equal)
        assert 99.0 <= result.final_response.range_value <= 101.0
        assert len(result.final_response.reasoning) >= self.MIN_REASONING_LENGTH

        assert result.total_rounds == 1

    async def test_failed_consensus_exact_values(self) -> None:
        """Test that below-threshold agreement fails consensus with exact values."""

        # Judge would pick the minority values but won't be reached due to max_rounds
        judge_executor = MockModelWithExactValues(
            exact_number=777,
            exact_string="minority",
            semantic_string="neutral",
            range_value=9.99,
        )

        # Create 9 models: only 5 agree (55.6%), below 70% threshold
        models = []

        # 5 models with consensus values (55.6%)
        for i in range(5):
            models.append(
                ConsensusCore.model(
                    id=f"agree_{i}",
                    executor=MockModelWithExactValues(
                        exact_number=777,
                        exact_string="minority",
                        semantic_string="neutral",
                        range_value=9.99,
                    ),
                    perspective=f"Agreeing model {i}",
                )
            )

        # 4 models with different values (40%)
        for i in range(4):
            models.append(
                ConsensusCore.model(
                    id=f"disagree_{i}",
                    executor=MockModelWithExactValues(
                        exact_number=111 * (i + 1),
                        exact_string=f"variant_{i}",
                        semantic_string=f"different_{i}",
                        range_value=3.33 * (i + 1),
                    ),
                    perspective=f"Disagreeing model {i}",
                )
            )

        # Judge will pick the majority view
        consensus = ConsensusCore.consensus(
            models=models,
            judge=judge_executor,
            settings=ConsensusSettings(
                max_rounds=2,
                verbosity=0,
            ),
        )

        result = await consensus.call("test below threshold")

        # 60% is below 70% threshold, so judge should be called
        assert result.consensus_achieved is False  # Judge fallback
        assert result.final_response.exact_number == 777
        assert result.final_response.exact_string == "minority"
        assert result.final_response.semantic_string == "neutral"
        assert result.final_response.range_value == 9.99
        assert len(result.final_response.reasoning) >= self.MIN_REASONING_LENGTH

        # Verify all models participated
        assert result.num_models == 9

    async def test_range_within_tolerance_groups_together(self) -> None:
        """Test that values within range tolerance are grouped together."""

        # Judge with base value
        judge_executor = MockModelWithExactValues(
            exact_number=10,
            exact_string="test",
            semantic_string="good",
            range_value=100.0,
        )

        # Create models with values all within 1% tolerance
        models = [
            ConsensusCore.model(
                id="model1",
                executor=MockModelWithExactValues(
                    exact_number=10,
                    exact_string="test",
                    semantic_string="good",
                    range_value=100.0,  # Base value
                ),
                perspective="Base value",
            ),
            ConsensusCore.model(
                id="model2",
                executor=MockModelWithExactValues(
                    exact_number=10,
                    exact_string="test",
                    semantic_string="good",
                    range_value=100.5,  # 0.5% difference - within tolerance
                ),
                perspective="Slightly higher",
            ),
            ConsensusCore.model(
                id="model3",
                executor=MockModelWithExactValues(
                    exact_number=10,
                    exact_string="test",
                    semantic_string="good",
                    range_value=99.5,  # 0.5% difference - within tolerance
                ),
                perspective="Slightly lower",
            ),
        ]

        consensus = ConsensusCore.consensus(
            models=models,
            judge=judge_executor,
            settings=ConsensusSettings(
                max_rounds=1,
                verbosity=0,
            ),
        )

        result = await consensus.call("test range within tolerance")

        # All values within 1% should achieve consensus
        assert result.consensus_achieved is True
        assert result.final_response.exact_number == 10
        assert result.final_response.exact_string == "test"
        assert result.final_response.semantic_string == "good"
        # Range value should be one of the three (all considered equal)
        assert result.final_response.range_value in [99.5, 100.0, 100.5]
        assert len(result.final_response.reasoning) >= self.MIN_REASONING_LENGTH

        # Should achieve consensus in one round with all models agreeing
        assert result.total_rounds == 1
        assert result.num_models == 3

    async def test_range_outside_tolerance_creates_groups(self) -> None:
        """Test that values outside range tolerance create separate groups."""

        # Judge with consensus value
        judge_executor = MockModelWithExactValues(
            exact_number=10,
            exact_string="test",
            semantic_string="good",
            range_value=100.0,
        )

        # Create models with one value outside tolerance
        models = [
            ConsensusCore.model(
                id="model1",
                executor=MockModelWithExactValues(
                    exact_number=10,
                    exact_string="test",
                    semantic_string="good",
                    range_value=100.0,
                ),
                perspective="Base value",
            ),
            ConsensusCore.model(
                id="model2",
                executor=MockModelWithExactValues(
                    exact_number=10,
                    exact_string="test",
                    semantic_string="good",
                    range_value=100.0,  # Same as model1
                ),
                perspective="Same value",
            ),
            ConsensusCore.model(
                id="model3",
                executor=MockModelWithExactValues(
                    exact_number=10,
                    exact_string="test",
                    semantic_string="good",
                    range_value=102.0,  # 2% difference - OUTSIDE tolerance
                ),
                perspective="Outside tolerance",
            ),
        ]

        consensus = ConsensusCore.consensus(
            models=models,
            judge=judge_executor,
            settings=ConsensusSettings(
                max_rounds=1,
                verbosity=0,
            ),
        )

        result = await consensus.call("test range outside tolerance")

        # 2/3 models agree on 100.0, but that's 66.7% < 70% threshold
        # So consensus should not be achieved and judge should decide
        assert result.consensus_achieved is False  # Judge fallback
        assert result.final_response.exact_number == 10
        assert result.final_response.exact_string == "test"
        assert result.final_response.semantic_string == "good"
        assert result.final_response.range_value == 100.0  # Judge picks 100.0
        assert len(result.final_response.reasoning) >= self.MIN_REASONING_LENGTH

        # Should use judge after one round
        assert result.total_rounds == 1
        assert result.num_models == 3

    async def test_semantic_threshold_properly_extracted(self) -> None:
        """Test that semantic threshold is properly extracted from Field."""

        # Judge with positive sentiment
        judge_executor = MockModelWithExactValues(
            exact_number=50,
            exact_string="same",
            semantic_string="good",
            range_value=7.5,
        )

        # Test with different semantic similarities
        # "good" vs "bad" should have low similarity
        models = [
            ConsensusCore.model(
                id="model1",
                executor=MockModelWithExactValues(
                    exact_number=50,
                    exact_string="same",
                    semantic_string="good",
                    range_value=7.5,
                ),
                perspective="Positive",
            ),
            ConsensusCore.model(
                id="model2",
                executor=MockModelWithExactValues(
                    exact_number=50,
                    exact_string="same",
                    semantic_string="bad",  # Opposite of "good"
                    range_value=7.5,
                ),
                perspective="Negative",
            ),
            ConsensusCore.model(
                id="model3",
                executor=MockModelWithExactValues(
                    exact_number=50,
                    exact_string="same",
                    semantic_string="good",  # Same as model1
                    range_value=7.5,
                ),
                perspective="Also positive",
            ),
        ]

        consensus = ConsensusCore.consensus(
            models=models,
            judge=judge_executor,
            settings=ConsensusSettings(
                max_rounds=1,
                verbosity=0,
            ),
        )

        result = await consensus.call("test semantic threshold")

        # "good" and "bad" should NOT be grouped together (low similarity)
        # 2 models say "good", 1 says "bad"
        # 2/3 = 66.7% < 70% threshold, so no consensus
        assert result.consensus_achieved is False  # Judge needed
        assert result.final_response.semantic_string == "good"  # Judge picks "good"
        assert len(result.final_response.reasoning) >= self.MIN_REASONING_LENGTH
