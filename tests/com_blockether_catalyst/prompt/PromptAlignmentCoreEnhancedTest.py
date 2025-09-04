"""
Tests for enhanced PromptAlignmentCore features.

- Principle persistence and database
- Kudos learning from successful responses
- Ideal response training
"""

from typing import List
from unittest.mock import AsyncMock, MagicMock

import pytest

from com_blockether_catalyst.consensus import Consensus
from com_blockether_catalyst.consensus.internal.ConsensusTypes import ConsensusResult
from com_blockether_catalyst.prompt import (
    PromptAlignmentCore,
    PromptConfiguration,
)
from com_blockether_catalyst.prompt.internal import (
    AlignmentFeedback,
    AlignmentPrinciple,
    AlignmentPrincipleList,
    EvaluationResult,
    SemanticStringList,
)


class TestPromptAlignmentCoreEnhanced:
    """Test cases for enhanced PromptAlignmentCore features."""

    # Test constants
    TEST_PROMPT = "What is machine learning?"
    TEST_GOOD_PROMPT = "Explain machine learning with specific examples from healthcare"
    TEST_GOOD_RESPONSE = "Machine learning in healthcare enables predictive diagnostics..."
    TEST_IDEAL_RESPONSE = "Machine learning is a subset of AI that uses statistical techniques..."
    TEST_DOMAIN = "technical"

    @pytest.fixture
    def mock_target_consensus(self) -> MagicMock:
        """Create mock target consensus."""
        mock = MagicMock(spec=Consensus)
        evaluation_result = EvaluationResult(
            alignment_score=0.85,
            feedback="Good alignment",
            strengths=SemanticStringList(["Clear"]),
            weaknesses=SemanticStringList([]),
            suggested_improvements=SemanticStringList([]),
            reasoning="The prompt aligns well with the target behavior and produces quality responses consistently.",
        )
        mock.call = AsyncMock(
            return_value=ConsensusResult(
                final_response=evaluation_result,
                consensus_reached=True,
                model_responses=[evaluation_result],
                voting_rounds=[],
                total_rounds=1,
                disagreement_details={},
            )
        )
        return mock

    @pytest.fixture
    def mock_alignment_consensus(self) -> MagicMock:
        """Create mock alignment consensus."""
        mock = MagicMock(spec=Consensus)
        alignment_feedback = AlignmentFeedback(
            overall_assessment="Excellent prompt structure that can be learned from",
            specific_issues=SemanticStringList([]),
            improvement_suggestions=SemanticStringList([]),
            principles_to_apply=AlignmentPrincipleList(
                [
                    AlignmentPrinciple(
                        principle="Always include specific domain examples",
                        importance=0.9,
                    ),
                    AlignmentPrinciple(
                        principle="Start with a clear definition before examples",
                        importance=0.85,
                    ),
                ]
            ),
            confidence_score=0.9,
            reasoning="This interaction demonstrates effective prompt construction that should be captured as reusable principles.",
        )
        mock.call = AsyncMock(
            return_value=ConsensusResult(
                final_response=alignment_feedback,
                consensus_reached=True,
                model_responses=[alignment_feedback],
                voting_rounds=[],
                total_rounds=1,
                disagreement_details={},
            )
        )
        return mock

    @pytest.fixture
    def alignment_core_with_persistence(
        self, mock_target_consensus: MagicMock, mock_alignment_consensus: MagicMock
    ) -> PromptAlignmentCore:
        """Create PromptAlignmentCore with persistence enabled."""
        return PromptAlignmentCore(
            target_consensus=mock_target_consensus,
            alignment_consensus=mock_alignment_consensus,
        )

    @pytest.mark.anyio
    async def test_kudos_learning(
        self,
        alignment_core_with_persistence: PromptAlignmentCore,
        mock_alignment_consensus: MagicMock,
    ) -> None:
        """Test learning from successful prompt-response pairs."""
        # Learn from a successful interaction
        principles = await alignment_core_with_persistence.learn_from_success(
            self.TEST_GOOD_PROMPT, self.TEST_GOOD_RESPONSE
        )

        assert len(principles) == 2
        assert principles[0].principle == "Always include specific domain examples"
        assert principles[0].importance == 0.9

        # Verify principles were persisted
        stored_principles = alignment_core_with_persistence.get_stored_principles()
        assert len(stored_principles) == 2
        assert stored_principles[0].principle == principles[0].principle

        # Verify successful pattern was stored
        assert len(alignment_core_with_persistence._successful_patterns) == 1
        assert alignment_core_with_persistence._successful_patterns[0][0] == self.TEST_GOOD_PROMPT

    @pytest.mark.anyio
    async def test_ideal_response_training(
        self,
        alignment_core_with_persistence: PromptAlignmentCore,
        mock_alignment_model: AsyncMock,
    ) -> None:
        """Test extracting principles from ideal responses."""
        # Extract principles from ideal response
        principles = await alignment_core_with_persistence.extract_principles_from_ideal(
            self.TEST_PROMPT, self.TEST_IDEAL_RESPONSE
        )

        assert len(principles) == 2
        # Importance should be boosted for ideal response principles
        assert principles[0].importance > 0.85  # Boosted from 0.9

        # Verify persistence
        stored_principles = alignment_core_with_persistence.get_stored_principles()
        assert len(stored_principles) == 2

    def test_principle_database_operations(self, alignment_core_with_persistence: PromptAlignmentCore) -> None:
        """Test principle database storage and retrieval."""
        # Add principles to database
        test_principles = [
            AlignmentPrinciple(principle="Be specific and concrete", importance=0.9),
            AlignmentPrinciple(
                principle="Include relevant examples",
                importance=0.85,
            ),
        ]

        alignment_core_with_persistence._add_principles(test_principles)

        # Retrieve principles
        stored = alignment_core_with_persistence.get_stored_principles()
        assert len(stored) == 2
        assert stored[0].principle == "Be specific and concrete"

        # Test principle count
        count = alignment_core_with_persistence.get_principle_count()
        assert count == 2

        # Test duplicate prevention
        alignment_core_with_persistence._add_principles(test_principles)
        stored = alignment_core_with_persistence.get_stored_principles()
        assert len(stored) == 2  # Should still be 2, not 4

    def test_principle_export_import(self, alignment_core_with_persistence: PromptAlignmentCore) -> None:
        """Test exporting and importing principles as shareable resources."""
        # Add some principles
        test_principles = [AlignmentPrinciple(principle="Use clear language", importance=0.9)]
        alignment_core_with_persistence._add_principles(test_principles)

        # Export principles
        exported = alignment_core_with_persistence.export_principles()
        assert len(exported) == 1
        assert exported[0]["principle"] == "Use clear language"
        assert exported[0]["importance"] == 0.9

        # Create new core and import
        new_core = PromptAlignmentCore(
            target_consensus=alignment_core_with_persistence._target_consensus,
            alignment_consensus=alignment_core_with_persistence._alignment_consensus,
        )

        new_core.import_principles(exported)
        imported = new_core.get_stored_principles()
        assert len(imported) == 1
        assert imported[0].principle == "Use clear language"

    @pytest.mark.anyio
    async def test_stored_principles_in_alignment(
        self,
        alignment_core_with_persistence: PromptAlignmentCore,
        mock_target_consensus: MagicMock,
        mock_alignment_model: AsyncMock,
    ) -> None:
        """Test that stored principles are used during alignment."""
        # Pre-populate database with principles
        stored_principles = [AlignmentPrinciple(principle="Always be specific", importance=0.95)]
        alignment_core_with_persistence._add_principles(stored_principles)

        # Setup mock to return low score initially, then high
        mock_target_consensus.call.side_effect = [
            ConsensusResult(
                reasoning="Initial evaluation consensus",
                consensus_achieved=True,
                final_response=EvaluationResult(
                    alignment_score=0.4,
                    feedback="Needs improvement",
                    strengths=SemanticStringList([]),
                    weaknesses=SemanticStringList(["Too vague"]),
                    suggested_improvements=SemanticStringList(["Be more specific"]),
                    reasoning="The prompt lacks the specificity required for quality responses.",
                ),
                rounds=[],
                total_rounds=1,
                convergence_score=1.0,
                participating_models=["model1"],
            ),
            ConsensusResult(
                reasoning="Second evaluation consensus",
                consensus_achieved=True,
                final_response=EvaluationResult(
                    alignment_score=0.9,
                    feedback="Much better",
                    strengths=SemanticStringList(["Specific"]),
                    weaknesses=SemanticStringList([]),
                    suggested_improvements=SemanticStringList([]),
                    reasoning="The prompt now includes the specificity needed for alignment.",
                ),
                rounds=[],
                total_rounds=1,
                convergence_score=1.0,
                participating_models=["model1"],
            ),
            ConsensusResult(
                reasoning="Final evaluation consensus",
                consensus_achieved=True,
                final_response=EvaluationResult(
                    alignment_score=0.9,
                    feedback="Final check",
                    strengths=SemanticStringList(["Specific"]),
                    weaknesses=SemanticStringList([]),
                    suggested_improvements=SemanticStringList([]),
                    reasoning="The prompt successfully aligns with the specified requirements and demonstrates good structure.",
                ),
                rounds=[],
                total_rounds=1,
                convergence_score=1.0,
                participating_models=["model1"],
            ),
        ]

        config = PromptConfiguration(
            initial_prompt=self.TEST_PROMPT,
            target_behavior="Provide detailed technical explanation",
            max_iterations=3,
            score_threshold=0.85,
        )

        result = await alignment_core_with_persistence.align_prompt(config)

        # Should have used stored principles plus new ones
        assert result.final_score == 0.9
        assert result.iterations_used == 2

        # Check that stored principles are in the domain
        final_stored = alignment_core_with_persistence.get_stored_principles()
        assert len(final_stored) >= 1  # Original + any new ones

    def test_persistence_disabled(self, mock_target_consensus: MagicMock, mock_alignment_consensus: MagicMock) -> None:
        """Test that persistence can be disabled."""
        core_no_persist = PromptAlignmentCore(
            target_consensus=mock_target_consensus,
            alignment_consensus=mock_alignment_consensus,
        )

        # Add principles (should not persist)
        test_principles = [AlignmentPrinciple(principle="Test principle", importance=0.8)]

        # This should not store anything since persistence is disabled
        core_no_persist._add_principles(test_principles)

        # Database should still work but won't be used in alignment
        stored = core_no_persist.get_stored_principles()
        assert len(stored) == 1  # Still stores in memory

        # Persistence is always enabled now (no flag to disable)
        assert core_no_persist.get_principle_count() == 1

    def test_multiple_principle_management(self, alignment_core_with_persistence: PromptAlignmentCore) -> None:
        """Test managing multiple principles without domain categorization."""
        principle_texts = [
            "Be specific and clear",
            "Include relevant examples",
            "Use proper structure",
            "Ensure completeness",
        ]

        for text in principle_texts:
            principles = [
                AlignmentPrinciple(
                    principle=text,
                    importance=0.9,
                )
            ]
            alignment_core_with_persistence._add_principles(principles)

        # Check all principles are stored
        principle_count = alignment_core_with_persistence.get_principle_count()
        assert principle_count == 4

        # All principles should be retrievable
        stored = alignment_core_with_persistence.get_stored_principles()
        assert len(stored) == 4
        stored_texts = [p.principle for p in stored]
        for text in principle_texts:
            assert text in stored_texts

    @pytest.mark.anyio
    async def test_principle_importance_adjustment(
        self,
        alignment_core_with_persistence: PromptAlignmentCore,
        mock_alignment_consensus: MagicMock,
    ) -> None:
        """Test that ideal response principles get importance boost."""
        # Mock should return principles with base importance
        mock_alignment_consensus.call.return_value = ConsensusResult(
            reasoning="Consensus on ideal principles",
            consensus_achieved=True,
            final_response=AlignmentFeedback(
                overall_assessment="Analysis of ideal response",
                specific_issues=SemanticStringList([]),
                improvement_suggestions=SemanticStringList([]),
                principles_to_apply=AlignmentPrincipleList(
                    [AlignmentPrinciple(principle="Test principle", importance=0.8)]
                ),
                confidence_score=0.9,
                reasoning="Ideal response analysis shows these principles lead to quality outputs.",
            ),
            rounds=[],
            total_rounds=1,
            convergence_score=1.0,
            participating_models=["model1"],
        )

        principles = await alignment_core_with_persistence.extract_principles_from_ideal(
            self.TEST_PROMPT, self.TEST_IDEAL_RESPONSE
        )

        # Importance should be boosted by 1.2x (capped at 1.0)
        assert principles[0].importance == 0.96  # 0.8 * 1.2
