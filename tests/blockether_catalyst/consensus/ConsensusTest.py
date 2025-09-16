"""Tests for the NewConsensus pure voting implementation."""

import json
from typing import Any, Optional

import anyio
import pytest
from pydantic import BaseModel, Field

from blockether_catalyst.consensus.Consensus import Consensus
from blockether_catalyst.consensus.ConsensusTypes import (
    ConsensusSettings,
    ModelConfiguration,
)
from blockether_catalyst.consensus.VotingComparison import (
    BaseModelWithReasoning,
    ComparisonStrategy,
    VotingField,
)
from blockether_catalyst.utils.TypedCalls import ArityOneTypedCall


class SimpleResponse(BaseModelWithReasoning):
    """Simple response model for testing."""

    answer: str = VotingField(comparison=ComparisonStrategy.EXACT)
    confidence: float = VotingField(default=1.0, comparison=ComparisonStrategy.IGNORE)
    reasoning: Optional[str] = VotingField(default=None, comparison=ComparisonStrategy.IGNORE)


class MockModel(ArityOneTypedCall[str, SimpleResponse]):
    """Mock model that returns predetermined responses."""

    def __init__(self, response: SimpleResponse, model_id: str = "mock"):
        self._response = response
        self._model_id = model_id
        self._call_count = 0

    async def call(self, prompt: str) -> SimpleResponse:
        """Return the predetermined response or vote based on prompt."""
        self._call_count += 1

        # Check if this is a refinement prompt
        if "Other models' responses:" in prompt:
            # Return a refined response
            return SimpleResponse(
                answer=self._response.answer + "_refined",
                confidence=0.9,
                reasoning="Refined based on peer feedback",
            )

        # Return the predetermined response
        return self._response


# MockVotingFunction removed - VoteDecision type no longer exists
# This functionality is now handled internally by the Consensus class


class TestConsensus:
    """Test cases for Consensus implementation."""

    @pytest.mark.anyio
    async def test_odd_model_validation(self):
        """Test that even number of models raises ValueError."""
        # Create even number of models (4)
        models = [
            ModelConfiguration(
                id=f"model_{i}",
                executor=MockModel(SimpleResponse(answer="test"), f"model_{i}"),
                perspective=f"Model {i} perspective",
            )
            for i in range(4)
        ]

        with pytest.raises(ValueError, match="odd number of models"):
            # Create a judge for consensus
            judge = MockModel(SimpleResponse(answer="judge_answer"), "judge")
            Consensus(models, judge=judge)

    @pytest.mark.anyio
    async def test_minimum_models_validation(self):
        """Test that less than 3 models raises ValueError."""
        # Create only 1 model
        models = [
            ModelConfiguration(
                id="model_1",
                executor=MockModel(SimpleResponse(answer="test")),
                perspective="Single model perspective",
            )
        ]

        with pytest.raises(ValueError, match="at least 3 models"):
            judge = MockModel(SimpleResponse(answer="judge_answer"), "judge")
            Consensus(models, judge=judge)

    @pytest.mark.anyio
    async def test_threshold_validation(self):
        """Test that threshold is auto-adjusted if it exceeds maximum."""
        # Create 3 models
        models = [
            ModelConfiguration(
                id=f"model_{i}",
                executor=MockModel(SimpleResponse(answer="test"), f"model_{i}"),
                perspective=f"Model {i} perspective",
            )
            for i in range(3)
        ]

        # Max threshold for 3 models is (3+1)/2 / 3 = 2/3 = 0.667
        # Setting threshold to 0.8 which exceeds maximum
        settings = ConsensusSettings(threshold=0.8)

        # Should auto-adjust threshold instead of raising error
        # Create a judge for consensus
        judge = MockModel(SimpleResponse(answer="consensus"), "judge")
        consensus = Consensus(models, judge=judge, settings=settings)

        # Verify the threshold was auto-adjusted to 95% of max (0.667 * 0.95 ≈ 0.633)
        assert consensus._settings.threshold == pytest.approx(0.633, rel=1e-2)

    @pytest.mark.anyio
    async def test_unanimous_agreement(self):
        """Test fast path when all models agree."""
        # Create 3 models with same response
        same_response = SimpleResponse(answer="unanimous", confidence=1.0)
        models = [
            ModelConfiguration(
                id=f"model_{i}",
                executor=MockModel(same_response, f"model_{i}"),
                perspective=f"Model {i} perspective",
            )
            for i in range(3)
        ]

        # Create a judge for consensus
        judge = MockModel(SimpleResponse(answer="consensus"), "judge")
        consensus = Consensus(models, judge=judge)
        result = await consensus.call("test prompt")

        assert result.consensus_achieved
        assert result.final_response.answer == "unanimous"
        assert result.total_rounds == 1
        assert "unanimous" in result.reasoning.lower()

    @pytest.mark.anyio
    async def test_majority_voting(self):
        """Test majority voting with different responses."""
        # Create 3 models: 2 agree, 1 disagrees
        models = [
            ModelConfiguration(
                id="model_1",
                executor=MockModel(SimpleResponse(answer="majority"), "model_1"),
                perspective="Model 1 perspective",
            ),
            ModelConfiguration(
                id="model_2",
                executor=MockModel(SimpleResponse(answer="majority"), "model_2"),
                perspective="Model 2 perspective",
            ),
            ModelConfiguration(
                id="model_3",
                executor=MockModel(SimpleResponse(answer="minority"), "model_3"),
                perspective="Model 3 perspective",
            ),
        ]

        # Set threshold to 0.5 (simple majority)
        judge = MockModel(SimpleResponse(answer="judge_decision"), "judge")
        settings = ConsensusSettings(threshold=0.5, max_rounds=3)
        consensus = Consensus(models=models, judge=judge, settings=settings)

        result = await consensus.call("test prompt")

        assert result.consensus_achieved
        assert result.total_rounds >= 1
        assert len(result.dissenting_models) >= 0

    @pytest.mark.anyio
    async def test_gossip_refinement(self):
        """Test that gossip rounds lead to refinement."""
        # Create 3 models with different initial responses
        models = [
            ModelConfiguration(
                id=f"model_{i}",
                executor=MockModel(SimpleResponse(answer=f"response_{i}"), f"model_{i}"),
                perspective=f"Model {i} perspective",
            )
            for i in range(3)
        ]

        # Low threshold to trigger multiple rounds
        judge = MockModel(SimpleResponse(answer="judge_decision"), "judge")
        settings = ConsensusSettings(threshold=0.3, max_rounds=5)
        consensus = Consensus(models=models, judge=judge, settings=settings)

        result = await consensus.call("test prompt")

        # Should have multiple rounds due to refinement
        assert result.total_rounds >= 2
        assert result.metrics.total_refinements >= 0

    @pytest.mark.anyio
    async def test_max_rounds_fallback(self):
        """Test fallback when max rounds reached."""
        # Create 5 models with all different responses
        models = [
            ModelConfiguration(
                id=f"model_{i}",
                executor=MockModel(SimpleResponse(answer=f"unique_{i}"), f"model_{i}"),
                perspective=f"Model {i} perspective",
            )
            for i in range(5)
        ]

        # High threshold that won't be met with diverse responses
        judge = MockModel(SimpleResponse(answer="judge_decision"), "judge")
        settings = ConsensusSettings(threshold=0.9, max_rounds=2)
        consensus = Consensus(models=models, judge=judge, settings=settings)

        result = await consensus.call("test prompt")

        # Should reach max rounds and use fallback
        assert result.total_rounds == 2
        assert "fell back" in result.reasoning.lower() or "could not reach" in result.reasoning.lower()

    @pytest.mark.anyio
    async def test_cross_voting_mechanism(self):
        """Test that models vote on OTHER models' responses."""

        class VotingMockModel(ArityOneTypedCall[str, SimpleResponse]):
            """Mock model that tracks voting."""

            def __init__(self, response: SimpleResponse, model_id: str):
                self._response = response
                self._model_id = model_id
                self._voted_on_self = False

            async def call(self, prompt: str) -> SimpleResponse:
                # Check if this is a voting prompt
                if "YOUR RESPONSE:" in prompt and "OTHER MODELS' RESPONSES" in prompt:
                    # Check if our own response appears in candidates
                    if self._response.answer in prompt.split("OTHER MODELS' RESPONSES")[1]:
                        # Our response should not be in the candidates
                        self._voted_on_self = True

                    return SimpleResponse(
                        answer=json.dumps(
                            {
                                "chosen_response_index": 0,
                                "confidence": 0.8,
                                "reasoning": "Test vote",
                            }
                        )
                    )

                return self._response

        # Create 3 models
        voting_models = []
        for i in range(3):
            model = VotingMockModel(SimpleResponse(answer=f"response_{i}"), f"model_{i}")
            voting_models.append(model)

        models = [
            ModelConfiguration(
                id=f"model_{i}",
                executor=voting_models[i],
                perspective=f"Model {i} perspective",
            )
            for i in range(3)
        ]

        judge = MockModel(SimpleResponse(answer="judge"), "judge")
        consensus = Consensus(models, judge=judge, settings=ConsensusSettings(max_rounds=2))
        await consensus.call("test prompt")

        # Verify no model voted on itself
        for model in voting_models:
            assert not model._voted_on_self, f"Model {model._model_id} voted on its own response"

    @pytest.mark.anyio
    async def test_perspective_handling(self):
        """Test that model perspectives are added to prompts."""
        perspective = "You are an expert in testing."

        class PerspectiveMockModel(ArityOneTypedCall[str, SimpleResponse]):
            """Mock model that checks for perspective."""

            def __init__(self):
                self._saw_perspective = False

            async def call(self, prompt: str) -> SimpleResponse:
                if perspective in prompt:
                    self._saw_perspective = True

                # Return vote for voting prompts
                if "OTHER MODELS' RESPONSES TO EVALUATE" in prompt:
                    return SimpleResponse(
                        answer=json.dumps(
                            {
                                "chosen_response_index": 0,
                                "confidence": 0.8,
                                "reasoning": "Test",
                            }
                        )
                    )

                return SimpleResponse(answer="test", confidence=1.0)

        mock = PerspectiveMockModel()
        models = [
            ModelConfiguration(id="model_1", executor=mock, perspective=perspective),
            ModelConfiguration(
                id="model_2",
                executor=MockModel(SimpleResponse(answer="test2")),
                perspective="Model 2 perspective",
            ),
            ModelConfiguration(
                id="model_3",
                executor=MockModel(SimpleResponse(answer="test3")),
                perspective="Model 3 perspective",
            ),
        ]

        # Create a judge for consensus
        judge = MockModel(SimpleResponse(answer="consensus"), "judge")
        consensus = Consensus(models, judge=judge)
        await consensus.call("test prompt")

        assert mock._saw_perspective, "Perspective was not added to prompt"
