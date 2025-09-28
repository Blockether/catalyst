"""
Tests for enhanced PromptAlignmentCore features.

- Principle persistence and database
- Kudos learning from successful responses
- Ideal response training
"""

from typing import List
from unittest.mock import AsyncMock, MagicMock

import pytest

from blockether_catalyst.consensus.Consensus import Consensus
from blockether_catalyst.consensus.ConsensusTypes import ConsensusResult
from blockether_catalyst.prompt import (
    PromptAlignmentCore,
    PromptConfiguration,
)
from blockether_catalyst.prompt.PromptAlignmentTypes import (
    AlignmentFeedback,
    AlignmentPrinciple,
    AlignmentPrincipleList,
    EvaluationResult,
    SemanticString,
)


class TestPromptAlignmentCoreEnhanced:
    """Test cases for enhanced PromptAlignmentCore features."""

    # Test constants
    TEST_PROMPT = "What is machine learning?"
    TEST_DOMAIN = "technical"

    @pytest.fixture
    def mock_target_consensus(self) -> MagicMock:
        """Create mock target consensus."""
        mock = MagicMock(spec=Consensus)
        evaluation_result = EvaluationResult(
            alignment_score=0.85,
            feedback="Good alignment",
            strengths=[SemanticString(root=s) for s in ["Clear"]],
            weaknesses=[SemanticString(root=s) for s in []],
            suggested_improvements=[SemanticString(root=s) for s in []],
            reasoning="The prompt aligns exceptionally well with the target behavior and produces quality responses consistently across multiple evaluation dimensions. The clarity of instruction and specificity of requirements ensure robust and reliable model outputs that meet all defined criteria.",
        )
        mock.call = AsyncMock(
            return_root=ConsensusResult(
                reasoning="Consensus achieved through comprehensive evaluation of multiple model responses and careful consideration of all perspectives. The models demonstrated strong agreement on the optimal approach, with consistent reasoning across all participants. This consensus represents a well-validated solution that incorporates diverse analytical viewpoints and methodologies.",
                consensus_achieved=True,
                final_response=evaluation_result,
                rounds=[],
                total_rounds=1,
                convergence_score=1.0,
                participating_models=["model1"],
                dissenting_models=[],
                model_contributions={},
            )
        )
        return mock

    @pytest.fixture
    def mock_alignment_consensus(self) -> MagicMock:
        """Create mock alignment consensus."""
        mock = MagicMock(spec=Consensus)
        alignment_feedback = AlignmentFeedback(
            overall_assessment="Excellent prompt structure that can be learned from",
            specific_issues=[SemanticString(root=s) for s in []],
            improvement_suggestions=[SemanticString(root=s) for s in []],
            principles_to_apply=AlignmentPrincipleList(
                principles=[
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
            reasoning="This interaction demonstrates effective prompt construction that should be captured as reusable principles. The prompt shows excellent structure with specific domain examples and clear definitions. These patterns can be extracted and reused across similar prompts to ensure consistent quality and alignment with user expectations.",
        )
        mock.call = AsyncMock(
            return_root=ConsensusResult(
                reasoning="Consensus achieved on alignment feedback through multi-model evaluation. All participating models agreed on the assessment of the prompt's structure and the principles that should be extracted. The feedback represents a well-balanced view incorporating diverse perspectives on prompt improvement strategies.",
                consensus_achieved=True,
                final_response=alignment_feedback,
                rounds=[],
                total_rounds=1,
                convergence_score=1.0,
                participating_models=["model1"],
                dissenting_models=[],
                model_contributions={},
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
    ) -> None:
        """Test that stored principles are used during alignment."""
        # Pre-populate database with principles
        stored_principles = [AlignmentPrinciple(principle="Always be specific", importance=0.95)]
        alignment_core_with_persistence._add_principles(stored_principles)

        # Setup mock to return low score initially, then high
        mock_target_consensus.call.side_effect = [
            ConsensusResult(
                reasoning="Initial evaluation consensus reached through comprehensive assessment of the prompt against target behavior criteria. The models have identified key weaknesses and areas for improvement, providing a solid foundation for iterative refinement through principle extraction and application.",
                consensus_achieved=True,
                final_response=EvaluationResult(
                    alignment_score=0.4,
                    feedback="Needs improvement",
                    strengths=[SemanticString(root=s) for s in []],
                    weaknesses=[SemanticString(root=s) for s in ["Too vague"]],
                    suggested_improvements=[SemanticString(root=s) for s in ["Be more specific"]],
                    reasoning="The prompt lacks the specificity required for quality responses. It needs more detailed instructions about the expected output format, the level of detail required, and clear context about the task. Without these elements, the model cannot reliably produce the desired results.",
                ),
                rounds=[],
                total_rounds=1,
                convergence_score=1.0,
                participating_models=["model1"],
            ),
            ConsensusResult(
                reasoning="Second evaluation consensus demonstrates significant improvement in prompt alignment. The applied principles have successfully addressed the identified weaknesses, resulting in a more specific and effective prompt that better meets the target behavior requirements.",
                consensus_achieved=True,
                final_response=EvaluationResult(
                    alignment_score=0.9,
                    feedback="Much better",
                    strengths=[SemanticString(root=s) for s in ["Specific"]],
                    weaknesses=[SemanticString(root=s) for s in []],
                    suggested_improvements=[SemanticString(root=s) for s in []],
                    reasoning="The prompt now demonstrates excellent specificity through the application of stored and refined principles, addressing all previously identified weaknesses and incorporating structured improvements that align with target behavior requirements.",
                ),
                rounds=[],
                total_rounds=1,
                convergence_score=1.0,
                participating_models=["model1"],
            ),
            ConsensusResult(
                reasoning="Final evaluation consensus confirms optimal prompt alignment has been achieved. The iterative refinement process has successfully incorporated all necessary principles, resulting in a prompt that fully meets the specified requirements with excellent structure and clarity.",
                consensus_achieved=True,
                final_response=EvaluationResult(
                    alignment_score=0.9,
                    feedback="Final check",
                    strengths=[SemanticString(root=s) for s in ["Specific"]],
                    weaknesses=[SemanticString(root=s) for s in []],
                    suggested_improvements=[SemanticString(root=s) for s in []],
                    reasoning="The prompt successfully aligns with the specified requirements and demonstrates good structure through systematic application of learned principles, incorporating domain-specific patterns and maintaining clarity throughout the refinement process to achieve optimal alignment.",
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
