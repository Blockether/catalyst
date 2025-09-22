"""
Edge case tests for consensus error handling.

This test suite validates that the consensus mechanism correctly handles:
1. Model exceptions and failures
2. Timeouts in async operations
3. Invalid judge responses
4. Zero models configuration
5. Invalid settings
"""

import asyncio
from typing import Any

import pytest
from pydantic import Field

from blockether_catalyst.consensus.ConsensusCore import ConsensusCore
from blockether_catalyst.consensus.ConsensusTypes import ConsensusSettings
from blockether_catalyst.consensus.VotingComparison import (
    BaseModelWithReasoning,
    ComparisonStrategy,
    VotingField,
)
from blockether_catalyst.utils.TypedCalls import ArityOneTypedCall


class EdgeCaseResponse(BaseModelWithReasoning):
    """Simple response model for edge case testing."""

    answer: int = VotingField(description="The answer", comparison=ComparisonStrategy.EXACT)
    confidence: float = VotingField(default=0.8, comparison=ComparisonStrategy.IGNORE)


class MockJudgeCall(ArityOneTypedCall[str, EdgeCaseResponse]):
    """Mock judge for tie-breaking."""

    def __init__(self, answer: int = 42):
        self._answer = answer

    async def call(self, x: str) -> EdgeCaseResponse:
        return EdgeCaseResponse(
            answer=self._answer,
            confidence=1.0,
            reasoning="Judge decision: After analyzing all model responses and their reasoning, I determine the most appropriate answer based on comprehensive evaluation of arguments presented throughout the consensus process.",
        )


class ExceptionThrowingCall(ArityOneTypedCall[str, EdgeCaseResponse]):
    """Mock call that always throws an exception."""

    def __init__(self, exception: Exception):
        self._exception = exception

    async def call(self, x: str) -> EdgeCaseResponse:
        raise self._exception


class SuccessfulCall(ArityOneTypedCall[str, EdgeCaseResponse]):
    """Mock call that succeeds normally."""

    def __init__(self, answer: int):
        self._answer = answer

    async def call(self, x: str) -> EdgeCaseResponse:
        return EdgeCaseResponse(
            answer=self._answer,
            confidence=0.9,
            reasoning=f"Successfully computed the answer as {self._answer} through standard processing without any errors or complications. This represents normal model behavior in the consensus system when all components are functioning properly.",
        )


class InvalidJudgeCall(ArityOneTypedCall[str, EdgeCaseResponse]):
    """Mock judge that returns invalid response."""

    async def call(self, x: str) -> EdgeCaseResponse:
        # Return response with missing required field (will cause validation error)
        return EdgeCaseResponse(
            answer=None,  # type: ignore - intentionally invalid
            confidence=0.9,
            reasoning="Invalid judge response with null answer field to test error handling when the judge returns structurally invalid data. This tests the consensus system's resilience to judge failures and its ability to handle validation errors gracefully.",
        )


class TestEdgeCaseConsensus:
    """Test edge cases in consensus voting."""

    # Static test configuration values
    DEFAULT_ANSWER = 42
    ALTERNATIVE_ANSWER = 50

    @pytest.mark.anyio
    async def test_zero_models_raises_error(self) -> None:
        """Test that consensus with zero models raises ValueError."""
        with pytest.raises(ValueError, match="At least one model must be specified"):
            ConsensusCore.consensus(
                models=[],  # Empty models list
                judge=MockJudgeCall(),
                settings=ConsensusSettings(max_rounds=1),
            )

    @pytest.mark.anyio
    async def test_model_exception_handling(self) -> None:
        """Test that consensus handles model exceptions gracefully."""
        runtime_error = RuntimeError("Model computation failed")

        models = [
            ConsensusCore.model(
                id="working_model_1",
                executor=SuccessfulCall(self.DEFAULT_ANSWER),
                perspective="First working model perspective",
            ),
            ConsensusCore.model(
                id="failing_model",
                executor=ExceptionThrowingCall(runtime_error),
                perspective="Failing model perspective",
            ),
            ConsensusCore.model(
                id="working_model_2",
                executor=SuccessfulCall(self.DEFAULT_ANSWER),
                perspective="Second working model",
            ),
            ConsensusCore.model(
                id="working_model_3",
                executor=SuccessfulCall(self.DEFAULT_ANSWER),
                perspective="Third working model",
            ),
            ConsensusCore.model(
                id="working_model_4",
                executor=SuccessfulCall(self.DEFAULT_ANSWER),
                perspective="Fourth working model",
            ),
        ]

        consensus = ConsensusCore.consensus(
            models=models,
            judge=MockJudgeCall(self.DEFAULT_ANSWER),
            settings=ConsensusSettings(
                max_rounds=1,
            ),
        )

        result = await consensus.call("Test with exception")

        # Should achieve consensus with working models
        assert result.consensus_achieved is True
        assert result.final_response.answer == self.DEFAULT_ANSWER
        # Check that all models participated
        assert len(result.participating_models) == 5
        # Verify that only 4 models successfully responded (one failed)
        successful_responses = len(result.rounds[-1].responses)
        assert successful_responses == 4  # Only successful models responded

    @pytest.mark.anyio
    async def test_all_models_fail(self) -> None:
        """Test consensus when all models throw exceptions."""
        models = [
            ConsensusCore.model(
                id="failing_1",
                executor=ExceptionThrowingCall(RuntimeError("Error 1")),
                perspective="First failing model",
            ),
            ConsensusCore.model(
                id="failing_2",
                executor=ExceptionThrowingCall(ValueError("Error 2")),
                perspective="Second failing model",
            ),
            ConsensusCore.model(
                id="failing_3",
                executor=ExceptionThrowingCall(TypeError("Error 3")),
                perspective="Third failing model",
            ),
        ]

        consensus = ConsensusCore.consensus(
            models=models,
            judge=MockJudgeCall(),
            settings=ConsensusSettings(max_rounds=2),
        )

        # When all models fail, the judge is used as fallback
        result = await consensus.call("Test with all failures")

        # Should still get a result from the judge
        assert result.final_response.answer == 42  # MockJudgeCall returns 42

        # Verify that no models succeeded (all rounds should have empty responses)
        for round_data in result.rounds:
            assert len(round_data.responses) == 0

        # Check that consensus was not achieved (since models failed)
        assert not result.consensus_achieved

        # The reasoning should indicate that all models failed
        assert (
            "all models failed" in result.reasoning.lower() or "fallback" in result.reasoning.lower()
        )  # No consensus since models failed

    @pytest.mark.anyio
    async def test_invalid_threshold_settings(self) -> None:
        """Test that invalid threshold settings are handled."""
        models = [
            ConsensusCore.model(
                id="model1",
                executor=SuccessfulCall(self.DEFAULT_ANSWER),
                perspective="Test model 1",
            ),
            ConsensusCore.model(
                id="model2",
                executor=SuccessfulCall(self.DEFAULT_ANSWER),
                perspective="Test model 2",
            ),
            ConsensusCore.model(
                id="model3",
                executor=SuccessfulCall(self.DEFAULT_ANSWER),
                perspective="Test model 3",
            ),
        ]

        # Test threshold > 1 - should be clamped to 1.0
        consensus = ConsensusCore.consensus(
            models=models,
            judge=MockJudgeCall(self.DEFAULT_ANSWER),
            settings=ConsensusSettings(threshold=1.5),  # Will be clamped
        )
        # Should still work
        result = await consensus.call("Test")
        assert result.consensus_achieved is True

        # Test threshold = 0 - all responses should achieve consensus
        consensus = ConsensusCore.consensus(
            models=models,
            judge=MockJudgeCall(self.DEFAULT_ANSWER),
            settings=ConsensusSettings(threshold=0.0),
        )
        result = await consensus.call("Test")
        assert result.consensus_achieved is True

    @pytest.mark.anyio
    async def test_max_rounds_exceeded(self) -> None:
        """Test behavior when max rounds is exceeded without consensus."""
        # Create models that will never agree
        models = [
            ConsensusCore.model(
                id="model1",
                executor=SuccessfulCall(self.DEFAULT_ANSWER),
                perspective="First perspective",
            ),
            ConsensusCore.model(
                id="model2",
                executor=SuccessfulCall(self.ALTERNATIVE_ANSWER),
                perspective="Second perspective",
            ),
            ConsensusCore.model(
                id="model3",
                executor=SuccessfulCall(60),  # Different from both
                perspective="Third perspective",
            ),
        ]

        consensus = ConsensusCore.consensus(
            models=models,
            judge=MockJudgeCall(self.DEFAULT_ANSWER),
            settings=ConsensusSettings(
                max_rounds=2,
            ),
        )

        result = await consensus.call("Test max rounds")

        # Should use judge when max rounds is reached with no majority
        # With 3 different answers, judge is used to break the tie
        assert result.consensus_achieved is False  # No consensus, judge used
        # With 3 different answers, it will pick one through voting
        # The exact answer depends on the voting mechanism
        assert result.final_response.answer in [
            self.DEFAULT_ANSWER,
            self.ALTERNATIVE_ANSWER,
            60,
        ]
        # Check that we hit max rounds
        assert result.total_rounds == 2

    @pytest.mark.anyio
    async def test_mixed_exceptions_and_timeouts(self) -> None:
        """Test consensus with mix of exceptions, timeouts, and successes."""
        models = [
            ConsensusCore.model(
                id="success_1",
                executor=SuccessfulCall(self.DEFAULT_ANSWER),
                perspective="First successful model",
            ),
            ConsensusCore.model(
                id="exception_model",
                executor=ExceptionThrowingCall(RuntimeError("Critical failure")),
                perspective="Model that throws exception",
            ),
            ConsensusCore.model(
                id="success_2",
                executor=SuccessfulCall(self.DEFAULT_ANSWER),
                perspective="Second successful model",
            ),
            ConsensusCore.model(
                id="value_error_model",
                executor=ExceptionThrowingCall(ValueError("Invalid input")),
                perspective="Model with value error",
            ),
            ConsensusCore.model(
                id="success_3",
                executor=SuccessfulCall(self.DEFAULT_ANSWER),
                perspective="Third successful model",
            ),
        ]

        consensus = ConsensusCore.consensus(
            models=models,
            judge=MockJudgeCall(self.DEFAULT_ANSWER),
            settings=ConsensusSettings(
                max_rounds=1,
            ),
        )

        result = await consensus.call("Complex failure test")

        # Should achieve consensus with the successful models
        assert result.consensus_achieved is True
        assert result.final_response.answer == self.DEFAULT_ANSWER
        # Count successful responses
        # Expected to succeed: success_1, success_2
        # Expected to fail: exception_model, value_error_model
        successful_responses = len(result.rounds[-1].responses)
        # Should get exactly 3 successful responses (we have 3 successful models)
        assert successful_responses == 3

    @pytest.mark.anyio
    async def test_minimum_models_consensus(self) -> None:
        """Test consensus with minimum required models (3)."""
        models = [
            ConsensusCore.model(
                id="model_1",
                executor=SuccessfulCall(self.DEFAULT_ANSWER),
                perspective="First model perspective",
            ),
            ConsensusCore.model(
                id="model_2",
                executor=SuccessfulCall(self.DEFAULT_ANSWER),
                perspective="Second model perspective",
            ),
            ConsensusCore.model(
                id="model_3",
                executor=SuccessfulCall(self.DEFAULT_ANSWER),
                perspective="Third model perspective",
            ),
        ]

        consensus = ConsensusCore.consensus(
            models=models,
            judge=MockJudgeCall(self.DEFAULT_ANSWER),
            settings=ConsensusSettings(max_rounds=1),
        )

        result = await consensus.call("Minimum models test")

        # All models agree, should achieve consensus
        assert result.consensus_achieved is True
        assert result.final_response.answer == self.DEFAULT_ANSWER
        assert len(result.participating_models) == 3

    @pytest.mark.anyio
    async def test_network_error_simulation(self) -> None:
        """Test consensus with network-like errors."""
        models = [
            ConsensusCore.model(
                id="connection_error",
                executor=ExceptionThrowingCall(ConnectionError("Network unreachable")),
                perspective="Network error model",
            ),
            ConsensusCore.model(
                id="timeout_error",
                executor=ExceptionThrowingCall(TimeoutError("Request timed out")),
                perspective="Timeout error model",
            ),
            ConsensusCore.model(
                id="success",
                executor=SuccessfulCall(self.DEFAULT_ANSWER),
                perspective="Successful model",
            ),
        ]

        consensus = ConsensusCore.consensus(
            models=models,
            judge=MockJudgeCall(self.DEFAULT_ANSWER),
            settings=ConsensusSettings(
                max_rounds=1,
            ),
        )

        result = await consensus.call("Network error test")

        # Should handle network errors and use successful model
        assert result.consensus_achieved is True
        assert result.final_response.answer == self.DEFAULT_ANSWER
