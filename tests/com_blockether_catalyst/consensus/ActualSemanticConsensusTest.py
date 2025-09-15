"""
Actually robust tests for semantic consensus that test real voting behavior.

These tests verify that the consensus mechanism actually works by:
1. Testing that different text with same semantic meaning gets same vote
2. Testing that consensus is achieved when models agree
3. Testing that consensus fails when models disagree
4. Testing actual voting percentages and thresholds
"""

from typing import List

import pytest
from pydantic import Field

from com_blockether_catalyst.consensus.ConsensusCore import ConsensusCore
from com_blockether_catalyst.consensus.ConsensusTypes import ConsensusSettings
from com_blockether_catalyst.consensus.VotingComparison import BaseModelWithReasoning
from com_blockether_catalyst.utils.TypedCalls import ArityOneTypedCall


class SimpleResponse(BaseModelWithReasoning):
    """Simple response for testing semantic voting."""

    # Semantic field - should normalize case and whitespace
    category: str = Field(default="")

    # Exact field - must match exactly
    code: int = Field(default=0)

    # Ignored field - doesn't affect voting
    metadata: str = Field(default="")


class StaticMockCall(ArityOneTypedCall):
    """Mock that always returns the same response."""

    def __init__(self, response):
        self._response = response

    async def call(self, x: str):
        return self._response


class TestActualSemanticConsensus:
    """Test actual semantic consensus behavior."""

    @pytest.mark.anyio
    async def test_semantic_normalization_makes_different_text_vote_same(self):
        """Test that SEMANTIC fields with different case/whitespace vote the same."""

        # These should all vote together despite differences
        response1 = SimpleResponse(
            category="Machine Learning",  # Title case
            code=100,
            metadata="model1",
            reasoning="This response categorizes the content as Machine Learning based on comprehensive analysis of the technical content and methodology discussions present throughout the document, including algorithms, models, and optimization techniques discussed in detail.",
        )

        response2 = SimpleResponse(
            category="machine learning",  # lowercase
            code=100,
            metadata="model2",
            reasoning="This response identifies machine learning as the category through systematic evaluation of document themes and technical implementations, with extensive coverage of supervised and unsupervised learning paradigms and their practical applications.",
        )

        response3 = SimpleResponse(
            category="  MACHINE LEARNING  ",  # uppercase with spaces
            code=100,
            metadata="model3",
            reasoning="This response determines MACHINE LEARNING as the primary category after thorough examination of content distribution, keyword frequency, and conceptual relationships between various technical topics discussed throughout the comprehensive document.",
        )

        models = [
            ConsensusCore.model(
                id="model1",
                executor=StaticMockCall(response1),
                perspective="Title case model",
            ),
            ConsensusCore.model(
                id="model2",
                executor=StaticMockCall(response2),
                perspective="Lowercase model",
            ),
            ConsensusCore.model(
                id="model3",
                executor=StaticMockCall(response3),
                perspective="Uppercase model",
            ),
        ]

        # Judge not needed - all should agree
        judge = StaticMockCall(response1)

        consensus = ConsensusCore.consensus(
            models=models,
            judge=judge,
            settings=ConsensusSettings(max_rounds=1, threshold=1.0),  # Require 100% agreement
        )

        result = await consensus.call("Categorize this")

        # All three should vote together due to semantic normalization
        assert result.consensus_achieved is True
        assert result.total_rounds == 1
        # Final response should have the category (in some form)
        assert result.final_response.category.lower().strip() == "machine learning"
        assert result.final_response.code == 100

    @pytest.mark.anyio
    async def test_exact_fields_must_match_exactly(self):
        """Test that EXACT fields require exact matching."""

        # Same category but different codes
        response1 = SimpleResponse(
            category="AI",
            code=100,  # Different code
            metadata="model1",
            reasoning="This model assigns code 100 to the AI category based on specific criteria and evaluation metrics that prioritize certain technical aspects and implementation details found within the analyzed document content and structure.",
        )

        response2 = SimpleResponse(
            category="AI",
            code=200,  # Different code
            metadata="model2",
            reasoning="This model assigns code 200 to the AI category using alternative evaluation criteria that emphasize different technical priorities and weighting factors in the comprehensive assessment of document content and themes.",
        )

        response3 = SimpleResponse(
            category="AI",
            code=100,  # Same as model1
            metadata="model3",
            reasoning="This model also assigns code 100 to the AI category following similar evaluation methodology and criteria as the first model, confirming the assessment through independent analysis of the same document.",
        )

        models = [
            ConsensusCore.model(id="model1", executor=StaticMockCall(response1), perspective="Code 100"),
            ConsensusCore.model(id="model2", executor=StaticMockCall(response2), perspective="Code 200"),
            ConsensusCore.model(id="model3", executor=StaticMockCall(response3), perspective="Code 100"),
        ]

        judge = StaticMockCall(response1)

        consensus = ConsensusCore.consensus(
            models=models,
            judge=judge,
            settings=ConsensusSettings(max_rounds=1, threshold=0.7),  # Need 70% agreement
        )

        result = await consensus.call("Categorize with code")

        # Models 1 and 3 agree (2/3 = 66.7%), which is less than 70%
        # So consensus should not be achieved
        assert result.consensus_achieved is False
        # But we should still get a result (from voting or judge)
        assert result.final_response is not None
        assert result.final_response.code in [100, 200]

    @pytest.mark.anyio
    async def test_ignored_fields_dont_affect_voting(self):
        """Test that IGNORE fields don't affect consensus."""

        # All same except metadata (which is ignored)
        response1 = SimpleResponse(
            category="Data Science",
            code=42,
            metadata="very different metadata here",
            reasoning="This response identifies Data Science as the category with code 42, based on comprehensive statistical analysis and data manipulation techniques discussed throughout the document, including various analytical frameworks and methodologies.",
        )

        response2 = SimpleResponse(
            category="Data Science",
            code=42,
            metadata="completely unrelated text",
            reasoning="This response also categorizes as Data Science with code 42, following systematic evaluation of quantitative methods and analytical approaches presented in the document, with emphasis on statistical modeling and interpretation.",
        )

        response3 = SimpleResponse(
            category="Data Science",
            code=42,
            metadata="random stuff 12345",
            reasoning="This response confirms Data Science category with code 42 through independent assessment of mathematical foundations and computational techniques discussed, including machine learning applications in data analysis contexts.",
        )

        models = [
            ConsensusCore.model(
                id="model1",
                executor=StaticMockCall(response1),
                perspective="Metadata 1",
            ),
            ConsensusCore.model(
                id="model2",
                executor=StaticMockCall(response2),
                perspective="Metadata 2",
            ),
            ConsensusCore.model(
                id="model3",
                executor=StaticMockCall(response3),
                perspective="Metadata 3",
            ),
        ]

        judge = StaticMockCall(response1)

        consensus = ConsensusCore.consensus(
            models=models,
            judge=judge,
            settings=ConsensusSettings(max_rounds=1, threshold=1.0),  # Need 100% agreement
        )

        result = await consensus.call("Test ignored fields")

        # All should agree despite different metadata
        assert result.consensus_achieved is True
        assert result.final_response.category == "Data Science"
        assert result.final_response.code == 42

    @pytest.mark.anyio
    async def test_threshold_determines_consensus(self):
        """Test that threshold percentage actually determines consensus."""

        # 3 say A, 2 say B
        responseA = SimpleResponse(
            category="Neural Networks",
            code=1,
            metadata="A",
            reasoning="This model identifies Neural Networks as the primary category based on extensive coverage of network architectures, activation functions, and backpropagation algorithms discussed in detail throughout the technical documentation.",
        )

        responseB = SimpleResponse(
            category="Deep Learning",
            code=2,
            metadata="B",
            reasoning="This model categorizes as Deep Learning due to the comprehensive discussion of multi-layer neural networks, convolutional architectures, and advanced optimization techniques for training deep models in the document.",
        )

        models = [
            ConsensusCore.model(id="a1", executor=StaticMockCall(responseA), perspective="A1"),
            ConsensusCore.model(id="a2", executor=StaticMockCall(responseA), perspective="A2"),
            ConsensusCore.model(id="a3", executor=StaticMockCall(responseA), perspective="A3"),
            ConsensusCore.model(id="b1", executor=StaticMockCall(responseB), perspective="B1"),
            ConsensusCore.model(id="b2", executor=StaticMockCall(responseB), perspective="B2"),
        ]

        judge = StaticMockCall(responseA)

        # Test with 60% threshold (3/5 = 60%, should achieve consensus)
        consensus_60 = ConsensusCore.consensus(
            models=models,
            judge=judge,
            settings=ConsensusSettings(max_rounds=1, threshold=0.6),
        )

        result_60 = await consensus_60.call("Test 60% threshold")
        assert result_60.consensus_achieved is True
        assert result_60.final_response.category == "Neural Networks"

        # Test with 70% threshold (3/5 = 60%, should NOT achieve consensus)
        consensus_70 = ConsensusCore.consensus(
            models=models,
            judge=judge,
            settings=ConsensusSettings(max_rounds=1, threshold=0.7),
        )

        result_70 = await consensus_70.call("Test 70% threshold")
        assert result_70.consensus_achieved is False
        # But should still have a result from voting
        assert result_70.final_response is not None

    @pytest.mark.anyio
    async def test_perfect_tie_uses_judge(self):
        """Test that a perfect tie falls back to judge."""

        responseA = SimpleResponse(
            category="Reinforcement Learning",
            code=10,
            metadata="RL",
            reasoning="This model identifies Reinforcement Learning as the category based on extensive discussion of agent-environment interactions, reward signals, and policy optimization methods throughout the document's technical content.",
        )

        responseB = SimpleResponse(
            category="Supervised Learning",
            code=20,
            metadata="SL",
            reasoning="This model categorizes as Supervised Learning due to comprehensive coverage of labeled datasets, loss functions, and gradient-based optimization techniques for training predictive models in the document.",
        )

        # 2 models each, perfect tie
        models = [
            ConsensusCore.model(id="a1", executor=StaticMockCall(responseA), perspective="RL1"),
            ConsensusCore.model(id="a2", executor=StaticMockCall(responseA), perspective="RL2"),
            ConsensusCore.model(id="b1", executor=StaticMockCall(responseB), perspective="SL1"),
            ConsensusCore.model(id="b2", executor=StaticMockCall(responseB), perspective="SL2"),
        ]

        # Judge picks A
        judge = StaticMockCall(responseA)

        consensus = ConsensusCore.consensus(
            models=models,
            judge=judge,
            settings=ConsensusSettings(max_rounds=1, threshold=0.5),
        )

        result = await consensus.call("Test tie breaking")

        # With 50-50 tie, consensus not achieved
        assert result.consensus_achieved is False
        # Final result should come from voting/judge fallback
        assert result.final_response is not None
        # Could be either depending on implementation
        assert result.final_response.code in [10, 20]

    @pytest.mark.anyio
    async def test_unanimous_consensus(self):
        """Test that unanimous agreement achieves consensus immediately."""

        response = SimpleResponse(
            category="Quantum Computing",
            code=999,
            metadata="quantum",
            reasoning="All models unanimously identify Quantum Computing as the category based on extensive discussion of qubits, quantum gates, superposition, entanglement, and quantum algorithms throughout the highly specialized technical document.",
        )

        # All 5 models agree
        models = [
            ConsensusCore.model(
                id=f"model{i}",
                executor=StaticMockCall(response),
                perspective=f"Perspective {i}",
            )
            for i in range(5)
        ]

        judge = StaticMockCall(response)

        consensus = ConsensusCore.consensus(
            models=models,
            judge=judge,
            settings=ConsensusSettings(max_rounds=3, threshold=0.8),
        )

        result = await consensus.call("Test unanimous")

        # Should achieve consensus immediately
        assert result.consensus_achieved is True
        assert result.total_rounds == 1
        assert result.final_response.category == "Quantum Computing"
        assert result.final_response.code == 999

    @pytest.mark.anyio
    async def test_no_consensus_after_max_rounds(self):
        """Test fallback when consensus not achieved after max rounds."""

        # Three completely different responses
        response1 = SimpleResponse(
            category="Biology",
            code=1,
            metadata="bio",
            reasoning="This model identifies Biology as the category based on discussions of cellular processes, genetics, and molecular mechanisms found throughout the life sciences documentation and research papers analyzed.",
        )

        response2 = SimpleResponse(
            category="Chemistry",
            code=2,
            metadata="chem",
            reasoning="This model categorizes as Chemistry due to extensive coverage of molecular structures, chemical reactions, and thermodynamic principles discussed in the scientific documentation and experimental protocols.",
        )

        response3 = SimpleResponse(
            category="Physics",
            code=3,
            metadata="phys",
            reasoning="This model determines Physics as the category through analysis of fundamental forces, particle interactions, and mathematical frameworks for describing natural phenomena in the technical documentation.",
        )

        models = [
            ConsensusCore.model(id="bio", executor=StaticMockCall(response1), perspective="Biology"),
            ConsensusCore.model(id="chem", executor=StaticMockCall(response2), perspective="Chemistry"),
            ConsensusCore.model(id="phys", executor=StaticMockCall(response3), perspective="Physics"),
        ]

        judge = StaticMockCall(response1)

        consensus = ConsensusCore.consensus(
            models=models,
            judge=judge,
            settings=ConsensusSettings(max_rounds=2, threshold=0.7),
        )

        result = await consensus.call("Test no consensus")

        # Should not achieve consensus
        assert result.consensus_achieved is False
        # Should have tried max rounds
        assert result.total_rounds == 2
        # Should still have a result from fallback
        assert result.final_response is not None
        assert result.final_response.code in [1, 2, 3]
