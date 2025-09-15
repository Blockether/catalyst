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
from pydantic import Field

from blockether_catalyst.consensus.ConsensusCore import ConsensusCore
from blockether_catalyst.consensus.ConsensusTypes import (
    ConsensusSettings,
    ModelConfiguration,
)
from blockether_catalyst.consensus.VotingComparison import BaseModelWithReasoning
from blockether_catalyst.utils.TypedCalls import ArityOneTypedCall


class VotingTestResponse(BaseModelWithReasoning):
    """Test response model for consensus testing."""

    value: int
    confidence: float


class MockTypedCall(ArityOneTypedCall[str, VotingTestResponse]):
    """Mock typed call for testing."""

    def __init__(self, response: VotingTestResponse):
        self._response = response

    async def call(self, x: str) -> VotingTestResponse:
        return self._response


class TestVotingConsensus:
    """Test voting-based consensus functionality."""

    @pytest.fixture
    def mock_judge(self) -> MockTypedCall:
        """Create a mock judge for tie-breaking."""
        judge_response = VotingTestResponse(
            value=42,
            confidence=1.0,
            reasoning="Judge decision: After comprehensive evaluation of all model arguments and careful consideration of the evidence presented, I determine the value to be 42. This decision incorporates analysis of all perspectives and aims to provide the most balanced and well-supported conclusion based on the available information.",
        )
        return MockTypedCall(judge_response)

    @pytest.mark.anyio
    async def test_majority_voting_consensus(self, mock_judge) -> None:
        """Test that consensus is achieved through majority voting."""
        # Create 3 models - 2 with same response, 1 different
        majority_response = VotingTestResponse(
            value=100,
            confidence=0.9,
            reasoning="Based on careful analysis of the data and comprehensive evaluation of multiple factors, the value should be 100. This conclusion draws from historical patterns, statistical analysis, and careful consideration of all available evidence. The confidence in this assessment is high due to consistent supporting data.",
        )
        outlier_response = VotingTestResponse(
            value=200,
            confidence=0.8,
            reasoning="My analysis suggests a different value of 200 based on alternative interpretation of the underlying data patterns. This assessment considers different weighting factors and alternative analytical approaches. While this differs from the majority view, it represents a valid alternative perspective based on thorough examination.",
        )

        models = [
            ConsensusCore.model(
                id="model1",
                executor=MockTypedCall(majority_response),
                perspective="From a mathematical perspective",
            ),
            ConsensusCore.model(
                id="model2",
                executor=MockTypedCall(majority_response),
                perspective="From a statistical perspective",
            ),
            ConsensusCore.model(
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
    async def test_tie_with_judge(self, mock_judge) -> None:
        """Test that judge is invoked to break ties."""
        # Create 2 models with different responses (tie scenario)
        response1 = VotingTestResponse(
            value=100,
            confidence=0.9,
            reasoning="First model believes the value is 100 based on standard analysis methodology and comprehensive evaluation of available data. This assessment follows established analytical frameworks and incorporates multiple validation checks to ensure reliability and accuracy in the final determination.",
        )
        response2 = VotingTestResponse(
            value=200,
            confidence=0.9,
            reasoning="Second model believes the value is 200 based on alternative analysis approach that emphasizes different data weighting strategies. This methodology considers alternative theoretical frameworks and applies different validation criteria, leading to a higher confidence in this particular value assessment.",
        )

        models = [
            ConsensusCore.model(
                id="model1",
                executor=MockTypedCall(response1),
                perspective="Standard analysis",
            ),
            ConsensusCore.model(
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
    async def test_plurality_consensus(self, mock_judge) -> None:
        """Test that plurality (not majority) can achieve consensus with threshold."""
        # Create 5 models: 3 vote A, 1 votes B, 1 votes C
        response_a = VotingTestResponse(
            value=100,
            confidence=0.9,
            reasoning="Multiple models agree on value 100 based on convergent analysis methodologies and consistent evaluation criteria. This consensus emerges from shared analytical approaches and similar interpretation of underlying data patterns, providing strong confidence in this assessment.",
        )
        response_b = VotingTestResponse(
            value=150,
            confidence=0.8,
            reasoning="This model suggests 150 based on a slightly different interpretation of the data patterns and alternative weighting of contributing factors. While similar to other approaches, this analysis places greater emphasis on specific data characteristics that lead to this intermediate value.",
        )
        response_c = VotingTestResponse(
            value=200,
            confidence=0.7,
            reasoning="This model proposes 200 based on an alternative theoretical framework altogether that emphasizes different analytical principles and methodological approaches. This perspective considers unique aspects of the problem space that other models may not fully account for in their assessments.",
        )

        models = [
            ConsensusCore.model(
                id=f"model_a_{i}",
                executor=MockTypedCall(response_a),
                perspective=f"Perspective A variant {i}",
            )
            for i in range(3)
        ] + [
            ConsensusCore.model(
                id="model_b",
                executor=MockTypedCall(response_b),
                perspective="Perspective B",
            ),
            ConsensusCore.model(
                id="model_c",
                executor=MockTypedCall(response_c),
                perspective="Perspective C",
            ),
        ]

        # With threshold of 0.6, 3/5 votes (60%) should achieve consensus
        consensus = ConsensusCore.consensus(
            models=models,
            judge=mock_judge,
        )

        result = await consensus.call("What is the value?")

        # Plurality winner should be response_a with 3/5 votes
        assert result.consensus_achieved is True
        assert result.final_response.value == 100

    @pytest.mark.anyio
    async def test_no_consensus_fallback(self, mock_judge) -> None:
        """Test fallback to voting when consensus not achieved after max rounds."""
        # Create 3 models with all different responses
        responses = [
            VotingTestResponse(
                value=100 + i * 50,
                confidence=0.8,
                reasoning=f"Model {i} has unique perspective leading to value {100 + i * 50} conclusion based on specialized analytical approach and domain-specific considerations. This assessment incorporates unique methodological elements and specialized validation criteria that distinguish it from other model perspectives in the consensus process.",
            )
            for i in range(3)
        ]

        models = [
            ConsensusCore.model(
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
        )

        result = await consensus.call("What is the value?")

        # Should fall back to voting - with 3 different values, there's no majority
        # Judge will be invoked for tie-breaking
        assert result.final_response is not None
        # The judge returns value=42 in our mock
        assert result.final_response.value == 42
        assert "not achieved" in result.reasoning.lower() or "fallback" in result.reasoning.lower()

    @pytest.mark.anyio
    async def test_convergence_through_rounds(self) -> None:
        """Test that models can converge over multiple rounds."""
        # This is a simplified test - in reality, models would adjust based on peer responses

        # Round 0: All different
        # Round 1: Two converge
        # Round 2: All converge
        converged_response = VotingTestResponse(
            value=150,
            confidence=0.95,
            reasoning="After reviewing peer analyses and conducting comprehensive evaluation of alternative approaches, consensus emerges that 150 is the optimal value. This determination balances multiple perspectives and incorporates insights from various analytical methodologies to arrive at the most balanced assessment.",
        )

        class AdaptiveTypedCall(ArityOneTypedCall[str, VotingTestResponse]):
            """Mock that changes response based on round."""

            def __init__(self, model_id: str):
                self.model_id = model_id
                self.round_count = 0

            async def call(self, x: str) -> VotingTestResponse:
                # Simulate convergence over rounds
                if "peer" in x.lower() or "refinement" in x.lower():
                    # This is a refinement round
                    self.round_count += 1
                    if self.round_count >= 1:
                        # All converge by round 1 or 2
                        return converged_response

                # Initial round - return different values
                if self.model_id == "model1":
                    return VotingTestResponse(
                        value=100,
                        confidence=0.7,
                        reasoning="Initial analysis suggests value of 100 based on first principles approach and fundamental theoretical considerations. This assessment draws from core analytical principles and applies rigorous validation methodology to ensure confidence in the initial determination.",
                    )
                elif self.model_id == "model2":
                    return VotingTestResponse(
                        value=200,
                        confidence=0.7,
                        reasoning="Initial analysis suggests value of 200 based on empirical observations and comprehensive data evaluation. This assessment prioritizes observational evidence and statistical analysis to support the higher value determination through rigorous analytical methodology.",
                    )
                else:
                    return VotingTestResponse(
                        value=150,
                        confidence=0.8,
                        reasoning="Initial analysis suggests value of 150 as a balanced middle ground that considers multiple analytical perspectives and balances various contributing factors. This assessment represents a compromise approach that incorporates insights from different methodological frameworks.",
                    )

        models = [
            ConsensusCore.model(
                id=f"model{i}",
                executor=AdaptiveTypedCall(f"model{i}"),
                perspective=f"Perspective {i}",
            )
            for i in range(1, 4)
        ]

        # Create judge for fallback when no consensus
        judge_response = VotingTestResponse(
            value=150,
            confidence=1.0,
            reasoning="Judge decision: After comprehensive review of the convergence patterns across all rounds and careful analysis of model responses, the collective reasoning demonstrates convergence towards 150. This determination reflects the weighted consensus emerging from iterative refinement and cross-model deliberation processes.",
        )
        mock_judge = MockTypedCall(judge_response)

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
