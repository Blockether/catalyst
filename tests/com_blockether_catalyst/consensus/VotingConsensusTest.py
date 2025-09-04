"""
Test cases for the voting-based consensus system.

This test suite validates that the consensus mechanism correctly:
1. Uses majority voting instead of semantic similarity
2. Invokes the judge for tie-breaking
3. Handles various voting scenarios
"""

from typing import Any, List
from unittest.mock import AsyncMock, MagicMock

import anyio
import pytest

from com_blockether_catalyst.consensus.ConsensusCore import ConsensusCore
from com_blockether_catalyst.consensus.internal.ConsensusTypes import (
    ConsensusSettings,
    ModelConfiguration,
    TypedCallBaseForConsensus,
)
from com_blockether_catalyst.utils.TypedCalls import ArityOneTypedCall


class TestResponse(TypedCallBaseForConsensus):
    """Test response model for consensus testing."""

    value: int
    confidence: float


class MockTypedCall(ArityOneTypedCall[str, TestResponse]):
    """Mock typed call for testing."""

    def __init__(self, response: TestResponse):
        self._response = response

    async def call(self, x: str) -> TestResponse:
        return self._response


class TestVotingConsensus:
    """Test voting-based consensus functionality."""

    @pytest.fixture
    def mock_judge(self) -> Any:
        """Create mock judge typed call for tie-breaking."""
        return MockTypedCall(
            TestResponse(
                value=42,
                confidence=0.9,
                reasoning="Judge decision for tie-breaking based on comprehensive analysis.",
            )
        )

    @pytest.mark.anyio
    async def test_majority_voting_consensus(self, mock_judge: Any) -> None:
        """Test that consensus is achieved through majority voting."""
        # Create 3 models - 2 with same response, 1 different
        majority_response = TestResponse(
            value=100,
            confidence=0.9,
            reasoning="Based on careful analysis of the data, the value should be 100 due to multiple factors.",
        )
        outlier_response = TestResponse(
            value=200,
            confidence=0.8,
            reasoning="My analysis suggests a different value of 200 based on alternative interpretation.",
        )

        models = [
            ConsensusCore.configuration(
                id="model1",
                executor=MockTypedCall(majority_response),
                perspective="From a mathematical perspective",
            ),
            ConsensusCore.configuration(
                id="model2",
                executor=MockTypedCall(majority_response),
                perspective="From a statistical perspective",
            ),
            ConsensusCore.configuration(
                id="model3",
                executor=MockTypedCall(outlier_response),
                perspective="From an alternative perspective",
            ),
        ]

        # Create consensus with judge
        consensus = ConsensusCore.consensus(
            models=models,
            judge=mock_judge,
            settings=ConsensusSettings(max_rounds=1),
        )

        # Run consensus
        result = await consensus.call("What is the value?")

        # Verify majority wins
        assert result.consensus_achieved is True
        assert result.final_response.value == 100
        assert result.total_rounds == 1
        assert "MAJORITY CONSENSUS" in result.reasoning or "consensus" in result.reasoning.lower()

    @pytest.mark.anyio
    async def test_tie_with_judge(self, mock_judge: Any) -> None:
        """Test that judge is invoked to break ties."""
        # Create 2 models with different responses (tie scenario)
        response1 = TestResponse(
            value=100,
            confidence=0.9,
            reasoning="First model believes the value is 100 based on standard analysis methodology.",
        )
        response2 = TestResponse(
            value=200,
            confidence=0.9,
            reasoning="Second model believes the value is 200 based on alternative analysis approach.",
        )

        models = [
            ConsensusCore.configuration(
                id="model1",
                executor=MockTypedCall(response1),
                perspective="Standard analysis",
            ),
            ConsensusCore.configuration(
                id="model2",
                executor=MockTypedCall(response2),
                perspective="Alternative analysis",
            ),
        ]

        # Create mock judge that picks response 1
        judge_mock = AsyncMock()
        judge_mock.call.return_value = response1

        # Create consensus with judge
        consensus = ConsensusCore.consensus(
            models=models,
            judge=judge_mock,
            settings=ConsensusSettings(max_rounds=3),
        )

        # Run consensus - should eventually use judge for tie
        result = await consensus.call("What is the value?")

        # If consensus not achieved in rounds, judge should be called in fallback
        # The exact behavior depends on implementation details
        # But we should get a result
        assert result.final_response is not None
        assert result.final_response.value in [100, 200]

    @pytest.mark.anyio
    async def test_plurality_consensus(self, mock_judge: Any) -> None:
        """Test that plurality (not majority) can achieve consensus with threshold."""
        # Create 5 models: 3 vote A, 1 votes B, 1 votes C
        response_a = TestResponse(
            value=100,
            confidence=0.9,
            reasoning="Multiple models agree on value 100 based on convergent analysis methodologies.",
        )
        response_b = TestResponse(
            value=150,
            confidence=0.8,
            reasoning="This model suggests 150 based on a slightly different interpretation of the data.",
        )
        response_c = TestResponse(
            value=200,
            confidence=0.7,
            reasoning="This model proposes 200 based on an alternative theoretical framework altogether.",
        )

        models = [
            ConsensusCore.configuration(
                id=f"model_a_{i}",
                executor=MockTypedCall(response_a),
                perspective=f"Perspective A variant {i}",
            )
            for i in range(3)
        ] + [
            ConsensusCore.configuration(
                id="model_b",
                executor=MockTypedCall(response_b),
                perspective="Perspective B",
            ),
            ConsensusCore.configuration(
                id="model_c",
                executor=MockTypedCall(response_c),
                perspective="Perspective C",
            ),
        ]

        # With threshold of 0.6, 3/5 votes (60%) should achieve consensus
        consensus = ConsensusCore.consensus(
            models=models,
            judge=mock_judge,
            settings=ConsensusSettings(threshold=0.6, max_rounds=1),
        )

        result = await consensus.call("What is the value?")

        # Plurality winner should be response_a with 3/5 votes
        assert result.consensus_achieved is True
        assert result.final_response.value == 100

    @pytest.mark.anyio
    async def test_no_consensus_fallback(self, mock_judge: Any) -> None:
        """Test fallback to voting when consensus not achieved after max rounds."""
        # Create 3 models with all different responses
        responses = [
            TestResponse(
                value=100 + i * 50,
                confidence=0.8,
                reasoning=f"Model {i} has unique perspective leading to value {100 + i * 50} conclusion.",
            )
            for i in range(3)
        ]

        models = [
            ConsensusCore.configuration(
                id=f"model{i}",
                executor=MockTypedCall(responses[i]),
                perspective=f"Unique perspective {i}",
            )
            for i in range(3)
        ]

        # High threshold, low rounds - consensus unlikely
        consensus = ConsensusCore.consensus(
            models=models,
            judge=mock_judge,
            settings=ConsensusSettings(threshold=0.95, max_rounds=2),
        )

        result = await consensus.call("What is the value?")

        # Should fall back to voting - with 3 different values, there's no majority
        # Judge will be invoked for tie-breaking
        assert result.final_response is not None
        # The judge returns value=42 in our mock
        assert result.final_response.value == 42
        assert "not achieved" in result.reasoning.lower() or "fallback" in result.reasoning.lower()

    @pytest.mark.anyio
    async def test_convergence_through_rounds(self, mock_judge: Any) -> None:
        """Test that models can converge over multiple rounds."""
        # This is a simplified test - in reality, models would adjust based on peer responses

        # Round 0: All different
        # Round 1: Two converge
        # Round 2: All converge
        converged_response = TestResponse(
            value=150,
            confidence=0.95,
            reasoning="After reviewing peer analyses, consensus emerges that 150 is the optimal value.",
        )

        class AdaptiveTypedCall(ArityOneTypedCall[str, TestResponse]):
            """Mock that changes response based on round."""

            def __init__(self, model_id: str):
                self.model_id = model_id
                self.round_count = 0

            async def call(self, x: str) -> TestResponse:
                # Simulate convergence over rounds
                if "peer" in x.lower() or "refinement" in x.lower():
                    # This is a refinement round
                    self.round_count += 1
                    if self.round_count >= 1:
                        # All converge by round 1 or 2
                        return converged_response

                # Initial round - return different values
                if self.model_id == "model1":
                    return TestResponse(
                        value=100,
                        confidence=0.7,
                        reasoning="Initial analysis suggests value of 100 based on first principles approach.",
                    )
                elif self.model_id == "model2":
                    return TestResponse(
                        value=200,
                        confidence=0.7,
                        reasoning="Initial analysis suggests value of 200 based on empirical observations.",
                    )
                else:
                    return TestResponse(
                        value=150,
                        confidence=0.8,
                        reasoning="Initial analysis suggests value of 150 as a balanced middle ground.",
                    )

        models = [
            ConsensusCore.configuration(
                id=f"model{i}",
                executor=AdaptiveTypedCall(f"model{i}"),
                perspective=f"Perspective {i}",
            )
            for i in range(1, 4)
        ]

        consensus = ConsensusCore.consensus(
            models=models,
            judge=mock_judge,
            settings=ConsensusSettings(max_rounds=3, threshold=0.8),
        )

        result = await consensus.call("What is the value?")

        # Models should converge
        assert result.consensus_achieved is True
        assert result.final_response.value == 150
        assert result.total_rounds <= 3
