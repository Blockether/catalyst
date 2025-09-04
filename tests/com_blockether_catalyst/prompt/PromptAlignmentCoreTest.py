"""
Tests for PromptAlignmentCore functionality.

This module tests the core prompt alignment functionality using mocked TypedCalls.
"""

from typing import List
from unittest.mock import AsyncMock, MagicMock

import pytest

from com_blockether_catalyst.consensus import Consensus
from com_blockether_catalyst.consensus.internal.ConsensusTypes import ConsensusResult
from com_blockether_catalyst.prompt import (
    AlignmentResult,
    PromptAlignmentCore,
    PromptConfiguration,
)
from com_blockether_catalyst.prompt.internal.PromptAlignmentTypes import (
    AlignmentFeedback,
    AlignmentMetrics,
    AlignmentPrinciple,
    AlignmentPrincipleList,
    EvaluationResult,
    SemanticStringList,
)


class TestPromptAlignmentCore:
    """Test cases for PromptAlignmentCore."""

    # Test constants
    TEST_INITIAL_PROMPT = "What is the capital of France?"
    TEST_TARGET_BEHAVIOR = "Provide detailed, educational responses with historical context"
    TEST_ALIGNED_PROMPT = "What is the capital of France? Please provide a detailed, educational response including historical context and significance."

    @pytest.fixture
    def mock_target_consensus(self) -> MagicMock:
        """Create mock target consensus."""
        mock = MagicMock(spec=Consensus)
        return mock

    @pytest.fixture
    def mock_alignment_consensus(self) -> MagicMock:
        """Create mock alignment consensus."""
        mock = MagicMock(spec=Consensus)
        return mock

    @pytest.fixture
    def alignment_core(
        self, mock_target_consensus: MagicMock, mock_alignment_consensus: MagicMock
    ) -> PromptAlignmentCore:
        """Create PromptAlignmentCore instance with mocked consensus."""
        return PromptAlignmentCore(
            target_consensus=mock_target_consensus,
            alignment_consensus=mock_alignment_consensus,
        )

    @pytest.mark.anyio
    async def test_successful_alignment(
        self,
        alignment_core: PromptAlignmentCore,
        mock_target_consensus: MagicMock,
        mock_alignment_consensus: MagicMock,
    ) -> None:
        """Test successful prompt alignment."""
        # Setup mock responses
        mock_target_consensus.call.side_effect = [
            ConsensusResult(
                reasoning="Initial evaluation consensus",
                consensus_achieved=True,
                final_response=EvaluationResult(
                    alignment_score=0.5,
                    feedback="Prompt lacks detail and context",
                    strengths=SemanticStringList(["Clear question"]),
                    weaknesses=SemanticStringList(["No context requested", "Too brief"]),
                    suggested_improvements=SemanticStringList(
                        [
                            "Add request for historical context",
                            "Ask for detailed response",
                        ]
                    ),
                    reasoning="The prompt is clear but does not align with the target behavior of providing detailed, educational responses with historical context.",
                ),
                rounds=[],
                total_rounds=1,
                convergence_score=1.0,
                participating_models=["model1"],
            ),
            ConsensusResult(
                reasoning="Improved evaluation consensus",
                consensus_achieved=True,
                final_response=EvaluationResult(
                    alignment_score=0.9,
                    feedback="Prompt now requests detailed, contextual information",
                    strengths=SemanticStringList(["Clear question", "Requests detail", "Asks for context"]),
                    weaknesses=SemanticStringList([]),
                    suggested_improvements=SemanticStringList([]),
                    reasoning="The improved prompt successfully aligns with the target behavior by explicitly requesting detailed information with historical context.",
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
                    feedback="Final evaluation confirms alignment",
                    strengths=SemanticStringList(["Clear", "Detailed", "Contextual"]),
                    weaknesses=SemanticStringList([]),
                    suggested_improvements=SemanticStringList([]),
                    reasoning="The prompt successfully achieves the target behavior requirements.",
                ),
                rounds=[],
                total_rounds=1,
                convergence_score=1.0,
                participating_models=["model1"],
            ),
        ]

        mock_alignment_consensus.call.side_effect = [
            ConsensusResult(
                reasoning="Consensus on alignment feedback",
                consensus_achieved=True,
                final_response=AlignmentFeedback(
                    overall_assessment="Prompt needs to explicitly request detail and context",
                    specific_issues=SemanticStringList(["Missing request for historical context", "Too brief"]),
                    improvement_suggestions=SemanticStringList(
                        [
                            "Add 'Please provide a detailed response'",
                            "Include 'with historical context'",
                        ]
                    ),
                    principles_to_apply=AlignmentPrincipleList(
                        [
                            AlignmentPrinciple(
                                principle="Always request the level of detail needed",
                                importance=0.9,
                            )
                        ]
                    ),
                    revised_prompt_suggestion=self.TEST_ALIGNED_PROMPT,
                    confidence_score=0.85,
                    reasoning="The prompt can be significantly improved by explicitly stating the requirements for detail and historical context that align with the target behavior.",
                ),
                rounds=[],
                total_rounds=1,
                convergence_score=1.0,
                participating_models=["model1"],
            ),
        ]

        config = PromptConfiguration(
            initial_prompt=self.TEST_INITIAL_PROMPT,
            target_behavior=self.TEST_TARGET_BEHAVIOR,
            max_iterations=5,
            score_threshold=0.8,
        )

        result = await alignment_core.align_prompt(config)

        assert isinstance(result, AlignmentResult)
        assert result.original_prompt == self.TEST_INITIAL_PROMPT
        # The aligned prompt should contain the original and additional context/requirements
        assert self.TEST_INITIAL_PROMPT in result.aligned_prompt
        assert "detail" in result.aligned_prompt.lower() or "context" in result.aligned_prompt.lower()
        assert result.final_score == 0.9
        assert result.iterations_used == 2
        assert len(result.evolution_history) == 2

    @pytest.mark.anyio
    async def test_principle_based_alignment(
        self,
        alignment_core: PromptAlignmentCore,
        mock_target_consensus: MagicMock,
        mock_alignment_consensus: MagicMock,
    ) -> None:
        """Test principle-based alignment strategy."""
        mock_target_consensus.call.side_effect = [
            EvaluationResult(
                alignment_score=0.4,
                feedback="Needs improvement",
                strengths=SemanticStringList(["Clear"]),
                weaknesses=SemanticStringList(["Lacks detail"]),
                suggested_improvements=SemanticStringList(["Add detail"]),
                reasoning="The prompt is too simple and doesn't request the level of detail required by the target behavior.",
            ),
            EvaluationResult(
                alignment_score=0.85,
                feedback="Much better",
                strengths=SemanticStringList(["Clear", "Detailed"]),
                weaknesses=SemanticStringList([]),
                suggested_improvements=SemanticStringList([]),
                reasoning="The prompt now successfully incorporates principles that align with the target behavior.",
            ),
            EvaluationResult(
                alignment_score=0.85,
                feedback="Good alignment",
                strengths=SemanticStringList(["Clear", "Detailed"]),
                weaknesses=SemanticStringList([]),
                suggested_improvements=SemanticStringList([]),
                reasoning="The final prompt aligns well with the target behavior.",
            ),
        ]

        mock_alignment_consensus.call.side_effect = [
            ConsensusResult(
                reasoning="Consensus on alignment feedback",
                consensus_achieved=True,
                final_response=AlignmentFeedback(
                    overall_assessment="Apply principles for improvement",
                    specific_issues=SemanticStringList(["Too brief"]),
                    improvement_suggestions=SemanticStringList(["Add detail request"]),
                    principles_to_apply=AlignmentPrincipleList(
                        [
                            AlignmentPrinciple(
                                principle="Request specific detail level",
                                importance=0.95,
                            ),
                            AlignmentPrinciple(
                                principle="Include context requirements",
                                importance=0.85,
                            ),
                        ]
                    ),
                    confidence_score=0.8,
                    reasoning="Applying these principles will help align the prompt with the target behavior of providing detailed, educational responses.",
                ),
                rounds=[],
                total_rounds=1,
                convergence_score=1.0,
                participating_models=["model1"],
            ),
        ]

        config = PromptConfiguration(
            initial_prompt=self.TEST_INITIAL_PROMPT,
            target_behavior=self.TEST_TARGET_BEHAVIOR,
        )

        result = await alignment_core.align_prompt(config)

        assert result.final_score == 0.85
        assert len(result.principles_applied) == 4
        # Principles should be sorted by importance, check if max is 0.95
        assert any(p.importance == 0.95 for p in result.principles_applied)

    @pytest.mark.anyio
    async def test_max_iterations_limit(
        self,
        alignment_core: PromptAlignmentCore,
        mock_target_consensus: MagicMock,
        mock_alignment_consensus: MagicMock,
    ) -> None:
        """Test that alignment stops at max iterations."""
        # Always return low score to force max iterations
        mock_target_consensus.call.return_value = ConsensusResult(
            reasoning="Low score evaluation",
            consensus_achieved=True,
            final_response=EvaluationResult(
                alignment_score=0.3,
                feedback="Still needs work",
                strengths=[],
                weaknesses=["Many issues"],
                suggested_improvements=["Keep trying"],
                reasoning="The prompt continues to have issues that prevent it from aligning with the target behavior.",
            ),
            rounds=[],
            total_rounds=1,
            convergence_score=1.0,
            participating_models=["model1"],
        )

        mock_alignment_consensus.call.return_value = ConsensusResult(
            reasoning="Consensus on alignment feedback",
            consensus_achieved=True,
            final_response=AlignmentFeedback(
                overall_assessment="Needs more work to achieve target alignment",
                specific_issues=SemanticStringList(["Still not aligned"]),
                improvement_suggestions=SemanticStringList(["Try again"]),
                reasoning="The prompt requires additional refinement to address specific alignment issues and improve overall effectiveness.",
                principles_to_apply=AlignmentPrincipleList([]),
                confidence_score=0.5,
            ),
            rounds=[],
            total_rounds=1,
            convergence_score=1.0,
            participating_models=["model1"],
        )

        config = PromptConfiguration(
            initial_prompt=self.TEST_INITIAL_PROMPT,
            target_behavior=self.TEST_TARGET_BEHAVIOR,
            max_iterations=3,
            score_threshold=0.9,
        )

        result = await alignment_core.align_prompt(config)

        assert result.iterations_used == 3
        assert result.final_score == 0.3

    @pytest.mark.anyio
    async def test_early_termination_on_threshold(
        self,
        alignment_core: PromptAlignmentCore,
        mock_target_consensus: MagicMock,
        mock_alignment_consensus: MagicMock,
    ) -> None:
        """Test that alignment stops when threshold is reached."""
        mock_target_consensus.call.side_effect = [
            EvaluationResult(
                alignment_score=0.85,  # Already above default threshold
                feedback="Good alignment",
                strengths=["Clear", "Detailed"],
                weaknesses=[],
                suggested_improvements=[],
                reasoning="The prompt already aligns well with the target behavior, exceeding the required threshold.",
            ),
            EvaluationResult(
                alignment_score=0.85,
                feedback="Final check",
                strengths=["Clear", "Detailed"],
                weaknesses=[],
                suggested_improvements=[],
                reasoning="Final evaluation confirms the prompt meets alignment requirements.",
            ),
        ]

        config = PromptConfiguration(
            initial_prompt=self.TEST_INITIAL_PROMPT,
            target_behavior=self.TEST_TARGET_BEHAVIOR,
            score_threshold=0.8,
        )

        result = await alignment_core.align_prompt(config)

        assert result.iterations_used == 1
        assert result.final_score == 0.85

    @pytest.mark.anyio
    async def test_prompt_validation(self, alignment_core: PromptAlignmentCore) -> None:
        """Test prompt validation."""
        # Test too short prompt
        config = PromptConfiguration(initial_prompt="Hi", target_behavior=self.TEST_TARGET_BEHAVIOR)

        with pytest.raises(ValueError, match="Prompt too short"):
            await alignment_core.align_prompt(config)

        # Test too long prompt
        config = PromptConfiguration(initial_prompt="x" * 6000, target_behavior=self.TEST_TARGET_BEHAVIOR)

        with pytest.raises(ValueError, match="Prompt too long"):
            await alignment_core.align_prompt(config)

    @pytest.mark.anyio
    async def test_batch_alignment(
        self,
        alignment_core: PromptAlignmentCore,
        mock_target_consensus: MagicMock,
        mock_alignment_consensus: MagicMock,
    ) -> None:
        """Test batch alignment of multiple prompts."""
        mock_target_consensus.call.return_value = ConsensusResult(
            reasoning="Evaluation consensus for batch",
            consensus_achieved=True,
            final_response=EvaluationResult(
                alignment_score=0.85,
                feedback="Good alignment achieved",
                strengths=SemanticStringList(["Clear"]),
                weaknesses=SemanticStringList([]),
                suggested_improvements=SemanticStringList([]),
                reasoning="The prompt aligns well with the target behavior and meets the required threshold.",
            ),
            rounds=[],
            total_rounds=1,
            convergence_score=1.0,
            participating_models=["model1"],
        )

        mock_alignment_consensus.call.return_value = ConsensusResult(
            reasoning="Consensus on alignment feedback",
            consensus_achieved=True,
            final_response=AlignmentFeedback(
                overall_assessment="Good prompt with proper alignment achieved",
                specific_issues=SemanticStringList([]),
                improvement_suggestions=SemanticStringList([]),
                principles_to_apply=AlignmentPrincipleList([]),
                revised_prompt_suggestion="Improved prompt",
                confidence_score=0.8,
                reasoning="The prompt successfully achieves the alignment goals.",
            ),
            rounds=[],
            total_rounds=1,
            convergence_score=1.0,
            participating_models=["model1"],
        )

        configs = [
            PromptConfiguration(
                initial_prompt="What is Python?",
                target_behavior="Technical explanation",
                score_threshold=0.8,
            ),
            PromptConfiguration(
                initial_prompt="Explain machine learning",
                target_behavior="Beginner-friendly explanation",
                score_threshold=0.8,
            ),
        ]

        results = await alignment_core.batch_align(configs)

        assert len(results) == 2
        assert all(isinstance(r, AlignmentResult) for r in results)
        assert results[0].final_score == 0.85
        assert results[1].final_score == 0.85

    def test_cache_operations(self, alignment_core: PromptAlignmentCore) -> None:
        """Test cache functionality."""
        test_prompt = "Test prompt"

        # Initially no cache
        assert alignment_core.get_cached_evolution(test_prompt) is None

        # Clear cache (should not error on empty cache)
        alignment_core.clear_cache()

        assert alignment_core.get_cached_evolution(test_prompt) is None

    @pytest.mark.anyio
    async def test_metrics_calculation(
        self,
        alignment_core: PromptAlignmentCore,
        mock_target_consensus: MagicMock,
        mock_alignment_consensus: MagicMock,
    ) -> None:
        """Test alignment metrics calculation."""
        mock_target_consensus.call.side_effect = [
            EvaluationResult(
                alignment_score=0.3,
                feedback="Poor alignment with target",
                strengths=[],
                weaknesses=["Many"],
                suggested_improvements=["Improve"],
                reasoning="Initial prompt has significant alignment issues with the target behavior.",
            ),
            EvaluationResult(
                alignment_score=0.6,
                feedback="Better alignment but needs more work",
                strengths=["Some"],
                weaknesses=["Few"],
                suggested_improvements=["Continue"],
                reasoning="The prompt shows improvement but still needs refinement to meet target behavior.",
            ),
            EvaluationResult(
                alignment_score=0.85,
                feedback="Good alignment achieved",
                strengths=["Many"],
                weaknesses=[],
                suggested_improvements=[],
                reasoning="The prompt now successfully aligns with the target behavior requirements.",
            ),
            EvaluationResult(
                alignment_score=0.85,
                feedback="Final alignment achieved successfully",
                strengths=["Many"],
                weaknesses=[],
                suggested_improvements=[],
                reasoning="Final evaluation confirms successful alignment with target behavior.",
            ),
        ]

        mock_alignment_consensus.call.return_value = ConsensusResult(
            reasoning="Consensus on alignment feedback",
            consensus_achieved=True,
            final_response=AlignmentFeedback(
                overall_assessment="Prompt needs significant improvements to meet requirements",
                specific_issues=SemanticStringList(["Issues"]),
                improvement_suggestions=SemanticStringList(["Suggestions"]),
                principles_to_apply=AlignmentPrincipleList([]),
                revised_prompt_suggestion="Better prompt",
                confidence_score=0.75,
                reasoning="Feedback provided to iteratively improve prompt alignment.",
            ),
            rounds=[],
            total_rounds=1,
            convergence_score=1.0,
            participating_models=["model1"],
        )

        config = PromptConfiguration(
            initial_prompt=self.TEST_INITIAL_PROMPT,
            target_behavior=self.TEST_TARGET_BEHAVIOR,
            max_iterations=3,
        )

        result = await alignment_core.align_prompt(config)

        assert isinstance(result.metrics, AlignmentMetrics)
        assert result.metrics.total_iterations == 3
        assert result.metrics.final_score == 0.85
        assert result.metrics.convergence_rate == (0.85 - 0.3) / 3
        assert 0.0 <= result.metrics.stability_score <= 1.0
