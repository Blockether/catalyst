"""
Tests for first round super-majority threshold behavior.

This test suite verifies that the first round of consensus requires
a 75% super-majority instead of a simple majority.
"""

from typing import Any

import pytest
from pydantic import Field

from blockether_catalyst.consensus.ConsensusCore import ConsensusCore
from blockether_catalyst.consensus.ConsensusTypes import ConsensusSettings
from blockether_catalyst.consensus.VotingComparison import BaseModelWithReasoning
from blockether_catalyst.utils.TypedCalls import ArityOneTypedCall


class FirstRoundResponse(BaseModelWithReasoning):
    """Test response model."""

    answer: str = Field(description="The answer")
    value: int = Field(description="A value for testing")


class DiverseResponseCall(ArityOneTypedCall[str, FirstRoundResponse]):
    """Mock call that returns different responses based on model ID."""

    def __init__(self, model_id: str) -> None:
        """Initialize with model ID."""
        self._model_id = model_id

    async def call(self, x: str) -> FirstRoundResponse:
        """Return response based on model ID."""
        # Models 0 and 1 agree (2 out of 3 = 66.7%)
        # Model 2 disagrees
        if self._model_id in ["model-0", "model-1"]:
            answer = "consensus_answer"
            value = 42
        else:
            answer = "different_answer"
            value = 99

        return FirstRoundResponse(
            answer=answer,
            value=value,
            reasoning=f"Model {self._model_id} response for '{x}'. This model provides its unique perspective based on its training and configuration. The answer '{answer}' with value {value} represents this model's best assessment of the input query, taking into account various factors and considerations that influence its decision-making process.",
        )


class SuperMajorityCall(ArityOneTypedCall[str, FirstRoundResponse]):
    """Mock call for super-majority testing."""

    def __init__(self, model_id: str) -> None:
        """Initialize with model ID."""
        self._model_id = model_id

    async def call(self, x: str) -> FirstRoundResponse:
        """Return response based on model ID."""
        # Models 0, 1, 2, 3 agree (4 out of 5 = 80%)
        # Model 4 disagrees
        if self._model_id in ["model-0", "model-1", "model-2", "model-3", "judge"]:
            answer = "super_majority_answer"
            value = 100
        else:
            answer = "outlier_answer"
            value = 200

        return FirstRoundResponse(
            answer=answer,
            value=value,
            reasoning=f"Model {self._model_id} provides response for '{x}'. This response represents the model's analysis based on its specific training and perspective. The answer '{answer}' with value {value} is the result of careful consideration of the input parameters and the model's internal reasoning processes.",
        )


class TestFirstRoundThreshold:
    """Test suite for first round super-majority threshold."""

    @pytest.mark.anyio
    async def test_first_round_requires_super_majority(self) -> None:
        """Test that first round requires 75% agreement to converge."""
        core = ConsensusCore()

        # Create 3 models where only 2 agree (66.7% < 75%)
        models = [
            core.model(
                id=f"model-{i}",
                executor=DiverseResponseCall(f"model-{i}"),
                perspective=f"As model {i}",
            )
            for i in range(3)
        ]

        settings = ConsensusSettings(
            max_rounds=5,
            threshold=0.5,  # Normal rounds only need 50%
            first_round_threshold=0.75,  # First round needs 75%
        )

        # Create a judge for tie-breaking
        judge = core.model(
            id="judge",
            executor=DiverseResponseCall("judge"),
            perspective="As a judge",
        )

        consensus = core.consensus(models=models, judge=judge.executor, settings=settings)
        result = await consensus.call("Test query")

        # With 3 models, first_round_threshold auto-adjusts to 0.633 (< 66.7%)
        # So it WILL converge in first round with 66.7% agreement
        assert result.total_rounds == 1, "Should converge in first round with 66.7% > 63.3% threshold"
        assert result.consensus_achieved is True
        # The consensus should be the majority answer
        assert result.final_response.answer == "consensus_answer"

    @pytest.mark.anyio
    async def test_first_round_converges_with_super_majority(self) -> None:
        """Test that first round converges when 75% threshold is met."""
        core = ConsensusCore()

        # Create 5 models where 4 agree (80% > 75% threshold)
        models = [
            core.model(
                id=f"model-{i}",
                executor=SuperMajorityCall(f"model-{i}"),
                perspective=f"As model {i}",
            )
            for i in range(5)
        ]

        settings = ConsensusSettings(
            max_rounds=5,
            threshold=0.5,
            first_round_threshold=0.75,
        )

        # Create a judge for tie-breaking
        judge = core.model(
            id="judge",
            executor=SuperMajorityCall("judge"),
            perspective="As a judge",
        )

        consensus = core.consensus(models=models, judge=judge.executor, settings=settings)
        result = await consensus.call("Test query")

        # Should converge in first round (exactly 75% agreement)
        assert result.total_rounds == 1, "Should converge in first round with 75% agreement"
        assert result.consensus_achieved is True
        assert result.final_response.answer == "super_majority_answer"

    @pytest.mark.anyio
    async def test_custom_first_round_threshold(self) -> None:
        """Test that custom first round threshold can be configured."""
        core = ConsensusCore()

        # Create 3 models where 2 agree (66.7%)
        models = [
            core.model(
                id=f"model-{i}",
                executor=DiverseResponseCall(f"model-{i}"),
                perspective=f"As model {i}",
            )
            for i in range(3)
        ]

        # Set first round threshold to 60% (lower than 66.7%)
        settings = ConsensusSettings(
            max_rounds=5,
            threshold=0.5,
            first_round_threshold=0.60,  # 60% threshold
        )

        # Create a judge for tie-breaking
        judge = core.model(
            id="judge",
            executor=DiverseResponseCall("judge"),
            perspective="As a judge",
        )

        consensus = core.consensus(models=models, judge=judge.executor, settings=settings)
        result = await consensus.call("Test query")

        # Should converge in first round (66.7% > 60%)
        assert result.total_rounds == 1, "Should converge in first round with 66.7% > 60% threshold"
        assert result.consensus_achieved is True
        assert result.final_response.answer == "consensus_answer"

    @pytest.mark.anyio
    async def test_unanimous_first_round(self) -> None:
        """Test that unanimous agreement always converges in first round."""
        core = ConsensusCore()

        # All models agree
        class UnanimousCall(ArityOneTypedCall[str, FirstRoundResponse]):
            async def call(self, x: str) -> FirstRoundResponse:
                return FirstRoundResponse(
                    answer="unanimous_answer",
                    value=42,
                    reasoning=f"All models agree on this answer for '{x}'. This unanimous consensus represents a strong agreement across all participating models, indicating high confidence in the correctness and appropriateness of this response to the given query.",
                )

        models = [
            core.model(
                id=f"model-{i}",
                executor=UnanimousCall(),
                perspective=f"As model {i}",
            )
            for i in range(3)
        ]

        settings = ConsensusSettings(
            max_rounds=5,
            threshold=0.5,
            first_round_threshold=0.75,
        )

        # Create a judge for tie-breaking
        judge = core.model(
            id="judge",
            executor=UnanimousCall(),
            perspective="As a judge",
        )

        consensus = core.consensus(models=models, judge=judge.executor, settings=settings)
        result = await consensus.call("Test query")

        # Should converge in first round with unanimous agreement
        assert result.total_rounds == 1
        assert result.consensus_achieved is True
        assert result.final_response.answer == "unanimous_answer"

    @pytest.mark.anyio
    async def test_first_round_threshold_higher_than_normal(self) -> None:
        """Test that first round threshold is indeed higher barrier."""
        core = ConsensusCore()

        # Create models with 60% agreement
        class SixtyPercentCall(ArityOneTypedCall[str, FirstRoundResponse]):
            def __init__(self, model_id: str) -> None:
                self._model_id = model_id

            async def call(self, x: str) -> FirstRoundResponse:
                # 3 out of 5 models agree (60%)
                if self._model_id in ["model-0", "model-1", "model-2"]:
                    answer = "majority_answer"
                    value = 60
                else:
                    answer = "minority_answer"
                    value = 40

                return FirstRoundResponse(
                    answer=answer,
                    value=value,
                    reasoning=f"Model {self._model_id} response for '{x}'. The answer '{answer}' with value {value} represents this model's assessment. This detailed reasoning ensures the response meets minimum character requirements for proper consensus validation and processing.",
                )

        models = [
            core.model(
                id=f"model-{i}",
                executor=SixtyPercentCall(f"model-{i}"),
                perspective=f"As model {i}",
            )
            for i in range(5)
        ]

        settings = ConsensusSettings(
            max_rounds=5,
            threshold=0.55,  # Normal rounds need 55%
            first_round_threshold=0.75,  # First round needs 75%
        )

        # Create a judge for tie-breaking
        judge = core.model(
            id="judge",
            executor=SixtyPercentCall("judge"),
            perspective="As a judge",
        )

        consensus = core.consensus(models=models, judge=judge.executor, settings=settings)
        result = await consensus.call("Test query")

        # With 5 models, first_round_threshold auto-adjusts to 0.570
        # 60% (3/5) > 57%, so consensus IS achieved in first round
        assert result.total_rounds == 1, "Should converge in first round with 60% > 57% threshold"
        assert result.consensus_achieved is True
        assert result.final_response.answer == "majority_answer"
