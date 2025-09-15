"""
Tests for PromptAlignmentCore functionality.

This module tests the core prompt alignment functionality using mocked TypedCalls.
"""

from typing import List
from unittest.mock import AsyncMock, MagicMock

import pytest

from blockether_catalyst.consensus.Consensus import Consensus
from blockether_catalyst.consensus.ConsensusTypes import ConsensusResult
from blockether_catalyst.prompt import (
    AlignmentResult,
    PromptAlignmentCore,
    PromptConfiguration,
)
from blockether_catalyst.prompt.PromptAlignmentTypes import (
    AlignmentFeedback,
    AlignmentMetrics,
    AlignmentPrinciple,
    AlignmentPrincipleList,
    EvaluationResult,
    SemanticString,
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
                reasoning="Initial evaluation consensus reached through voting mechanism. The models have analyzed the prompt and determined its alignment with target behavior. This assessment forms the baseline for subsequent improvement iterations in the alignment process.",
                consensus_achieved=True,
                final_response=EvaluationResult(
                    alignment_score=0.5,
                    feedback="Prompt lacks detail and context",
                    strengths=[SemanticString(s) for s in ["Clear question"]],
                    weaknesses=[SemanticString(s) for s in ["No context requested", "Too brief"]],
                    suggested_improvements=[
                        SemanticString(s)
                        for s in [
                            "Add request for historical context",
                            "Ask for detailed response",
                        ]
                    ],
                    reasoning="The prompt is clear but does not align with the target behavior of providing detailed, educational responses with historical context. The current formulation lacks explicit instructions for comprehensive information delivery and fails to request the contextual depth required by the target specifications.",
                ),
                rounds=[],
                total_rounds=1,
                convergence_score=1.0,
                participating_models=["model1"],
            ),
            ConsensusResult(
                reasoning="Improved evaluation consensus after applying alignment principles. The models have re-evaluated the refined prompt and observed significant improvements in alignment with the target behavior. The consensus indicates successful application of feedback.",
                consensus_achieved=True,
                final_response=EvaluationResult(
                    alignment_score=0.9,
                    feedback="Prompt now requests detailed, contextual information",
                    strengths=[
                        SemanticString(s)
                        for s in [
                            "Clear question",
                            "Requests detail",
                            "Asks for context",
                        ]
                    ],
                    weaknesses=[SemanticString(s) for s in []],
                    suggested_improvements=[SemanticString(s) for s in []],
                    reasoning="The improved prompt successfully aligns with the target behavior by explicitly requesting detailed information with historical context. The refinements have addressed all identified weaknesses and incorporated the necessary elements to ensure comprehensive, educational responses that meet the specified requirements.",
                ),
                rounds=[],
                total_rounds=1,
                convergence_score=1.0,
                participating_models=["model1"],
            ),
            ConsensusResult(
                reasoning="Final evaluation consensus confirms that the prompt has achieved the desired alignment threshold. The models unanimously agree that the current version successfully incorporates all required elements and meets the target behavior specifications.",
                consensus_achieved=True,
                final_response=EvaluationResult(
                    alignment_score=0.9,
                    feedback="Final evaluation confirms alignment",
                    strengths=[SemanticString(s) for s in ["Clear", "Detailed", "Contextual"]],
                    weaknesses=[SemanticString(s) for s in []],
                    suggested_improvements=[SemanticString(s) for s in []],
                    reasoning="The prompt successfully achieves the target behavior requirements through effective incorporation of feedback and iterative refinement. The final version demonstrates excellent alignment with all specified criteria and provides clear guidance for generating the desired type of response.",
                ),
                rounds=[],
                total_rounds=1,
                convergence_score=1.0,
                participating_models=["model1"],
            ),
        ]

        mock_alignment_consensus.call.side_effect = [
            ConsensusResult(
                reasoning="Consensus on alignment feedback has been achieved through model voting. The feedback incorporates analysis of prompt strengths and weaknesses, providing actionable suggestions for improvement. This consensus guides the iterative refinement process.",
                consensus_achieved=True,
                final_response=AlignmentFeedback(
                    overall_assessment="Prompt needs to explicitly request detail and context",
                    specific_issues=[
                        SemanticString(s) for s in ["Missing request for historical context", "Too brief"]
                    ],
                    improvement_suggestions=[
                        SemanticString(s)
                        for s in [
                            "Add 'Please provide a detailed response'",
                            "Include 'with historical context'",
                        ]
                    ],
                    principles_to_apply=AlignmentPrincipleList(
                        principles=[
                            AlignmentPrinciple(
                                principle="Always request the level of detail needed",
                                importance=0.9,
                            )
                        ]
                    ),
                    revised_prompt_suggestion=self.TEST_ALIGNED_PROMPT,
                    confidence_score=0.85,
                    reasoning="The prompt can be significantly improved by explicitly stating the requirements for detail and historical context that align with the target behavior. The current version lacks clear instructions for comprehensive responses and would benefit from explicit guidance on the level of detail and contextual information required.",
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
            ConsensusResult(
                reasoning="Consensus on evaluation reached through multi-model assessment. The models have analyzed the prompt against the target behavior criteria and reached agreement on the alignment score. This evaluation provides the foundation for determining necessary improvements.",
                consensus_achieved=True,
                final_response=EvaluationResult(
                    alignment_score=0.4,
                    feedback="Needs improvement",
                    strengths=[SemanticString(s) for s in ["Clear"]],
                    weaknesses=[SemanticString(s) for s in ["Lacks detail"]],
                    suggested_improvements=[SemanticString(s) for s in ["Add detail"]],
                    reasoning="The prompt is too simple and doesn't request the level of detail required by the target behavior. "
                    "It lacks specific instructions about the format and structure of the expected response. "
                    "The prompt should be more explicit about requirements and provide better context for the task.",
                ),
                rounds=[],
                total_rounds=1,
                convergence_score=1.0,
                participating_models=["model1"],
            ),
            ConsensusResult(
                reasoning="Consensus on evaluation reached through multi-model assessment. The models have analyzed the prompt against the target behavior criteria and reached agreement on the alignment score. This evaluation provides the foundation for determining necessary improvements.",
                consensus_achieved=True,
                final_response=EvaluationResult(
                    alignment_score=0.85,
                    feedback="Much better",
                    strengths=[SemanticString(s) for s in ["Clear", "Detailed"]],
                    weaknesses=[SemanticString(s) for s in []],
                    suggested_improvements=[SemanticString(s) for s in []],
                    reasoning="The prompt now successfully incorporates principles that align with the target behavior. "
                    "The iterative refinement process has addressed all identified weaknesses and enhanced the prompt's effectiveness. "
                    "The alignment score reflects successful integration of feedback and principles.",
                ),
                rounds=[],
                total_rounds=1,
                convergence_score=1.0,
                participating_models=["model1"],
            ),
            ConsensusResult(
                reasoning="Consensus on evaluation reached through multi-model assessment. The models have analyzed the prompt against the target behavior criteria and reached agreement on the alignment score. This evaluation provides the foundation for determining necessary improvements.",
                consensus_achieved=True,
                final_response=EvaluationResult(
                    alignment_score=0.85,
                    feedback="Good alignment",
                    strengths=[SemanticString(s) for s in ["Clear", "Detailed"]],
                    weaknesses=[SemanticString(s) for s in []],
                    suggested_improvements=[SemanticString(s) for s in []],
                    reasoning="The final prompt aligns well with the target behavior after successful application of alignment principles. "
                    "All evaluation criteria have been met and the prompt demonstrates excellent adherence to the specified requirements. "
                    "The high alignment score confirms that the iterative refinement process has achieved its objectives.",
                ),
                rounds=[],
                total_rounds=1,
                convergence_score=1.0,
                participating_models=["model1"],
            ),
        ]

        mock_alignment_consensus.call.side_effect = [
            ConsensusResult(
                reasoning="Consensus on alignment feedback has been achieved through model voting. The feedback incorporates analysis of prompt strengths and weaknesses, providing actionable suggestions for improvement. This consensus guides the iterative refinement process.",
                consensus_achieved=True,
                final_response=AlignmentFeedback(
                    overall_assessment="Apply principles for improvement",
                    specific_issues=[SemanticString(s) for s in ["Too brief"]],
                    improvement_suggestions=[SemanticString(s) for s in ["Add detail request"]],
                    principles_to_apply=AlignmentPrincipleList(
                        principles=[
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
                    reasoning="Applying these principles will help align the prompt with the target behavior of providing detailed, educational responses. The identified principles address the core weaknesses in the current prompt and provide actionable guidance for achieving effective alignment with the specified requirements and objectives.",
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
            reasoning="Low score evaluation indicates significant misalignment with target behavior requirements. The prompt requires substantial improvements across multiple dimensions to achieve acceptable alignment levels. Continued iterations are necessary to address fundamental issues and incorporate essential elements.",
            consensus_achieved=True,
            final_response=EvaluationResult(
                alignment_score=0.3,
                feedback="Still needs work",
                strengths=[SemanticString(s) for s in []],
                weaknesses=[SemanticString(s) for s in ["Many issues"]],
                suggested_improvements=[SemanticString(s) for s in ["Keep trying"]],
                reasoning="The prompt continues to have issues that prevent it from aligning with the target behavior despite iterative refinement attempts. Multiple fundamental problems persist that require significant restructuring to achieve meaningful alignment with the specified requirements.",
            ),
            rounds=[],
            total_rounds=1,
            convergence_score=1.0,
            participating_models=["model1"],
        )

        mock_alignment_consensus.call.return_value = ConsensusResult(
            reasoning="Consensus on alignment feedback achieved through multi-model voting and analysis. The feedback identifies specific areas for improvement and provides actionable recommendations to enhance prompt alignment. This guidance supports the iterative refinement process toward achieving target behavior specifications.",
            consensus_achieved=True,
            final_response=AlignmentFeedback(
                overall_assessment="Needs more work to achieve target alignment",
                specific_issues=[SemanticString(s) for s in ["Still not aligned"]],
                improvement_suggestions=[SemanticString(s) for s in ["Try again"]],
                reasoning="The prompt requires additional refinement to address specific alignment issues and improve overall effectiveness. Multiple aspects of the current formulation fail to meet target behavior requirements, necessitating continued iterative improvements to achieve acceptable alignment levels.",
                principles_to_apply=AlignmentPrincipleList(principles=[]),
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
            ConsensusResult(
                reasoning="Consensus on evaluation reached through multi-model assessment. The models have analyzed the prompt against the target behavior criteria and reached agreement on the alignment score. This evaluation provides the foundation for determining necessary improvements.",
                consensus_achieved=True,
                final_response=EvaluationResult(
                    alignment_score=0.85,  # Already above default threshold
                    feedback="Good alignment",
                    strengths=[SemanticString(s) for s in ["Clear", "Detailed"]],
                    weaknesses=[SemanticString(s) for s in []],
                    suggested_improvements=[SemanticString(s) for s in []],
                    reasoning="The prompt already aligns well with the target behavior, exceeding the required threshold. The high initial score indicates that the prompt effectively meets the specified requirements without requiring significant refinement or additional iterations.",
                ),
                rounds=[],
                total_rounds=1,
                convergence_score=1.0,
                participating_models=["model1"],
            ),
            ConsensusResult(
                reasoning="Consensus on evaluation reached through multi-model assessment. The models have analyzed the prompt against the target behavior criteria and reached agreement on the alignment score. This evaluation provides the foundation for determining necessary improvements.",
                consensus_achieved=True,
                final_response=EvaluationResult(
                    alignment_score=0.85,
                    feedback="Final check",
                    strengths=[SemanticString(s) for s in ["Clear", "Detailed"]],
                    weaknesses=[SemanticString(s) for s in []],
                    suggested_improvements=[SemanticString(s) for s in []],
                    reasoning="Final evaluation confirms the prompt meets alignment requirements after assessment against target behavior criteria. The sustained high alignment score validates that the prompt successfully achieves the desired objectives and maintains consistency with specifications.",
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
        config = PromptConfiguration(initial_prompt="x" * 11000, target_behavior=self.TEST_TARGET_BEHAVIOR)

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
            reasoning="Evaluation consensus for batch processing achieved through systematic assessment of multiple prompts. Each prompt has been analyzed against its respective target behavior to determine alignment scores. The batch evaluation enables efficient processing while maintaining consistent quality standards.",
            consensus_achieved=True,
            final_response=EvaluationResult(
                alignment_score=0.85,
                feedback="Good alignment achieved",
                strengths=[SemanticString(s) for s in ["Clear"]],
                weaknesses=[SemanticString(s) for s in []],
                suggested_improvements=[SemanticString(s) for s in []],
                reasoning="The prompt aligns well with the target behavior and meets the required threshold. The alignment score indicates successful adherence to specified requirements. Further refinement is not necessary as the current version effectively achieves the desired goals.",
            ),
            rounds=[],
            total_rounds=1,
            convergence_score=1.0,
            participating_models=["model1"],
        )

        mock_alignment_consensus.call.return_value = ConsensusResult(
            reasoning="Consensus on alignment feedback achieved through multi-model voting and analysis. The feedback identifies specific areas for improvement and provides actionable recommendations to enhance prompt alignment. This guidance supports the iterative refinement process toward achieving target behavior specifications.",
            consensus_achieved=True,
            final_response=AlignmentFeedback(
                overall_assessment="Good prompt with proper alignment achieved",
                specific_issues=[SemanticString(s) for s in []],
                improvement_suggestions=[SemanticString(s) for s in []],
                principles_to_apply=AlignmentPrincipleList(principles=[]),
                revised_prompt_suggestion="Improved prompt",
                confidence_score=0.8,
                reasoning="The prompt successfully achieves the alignment goals through iterative refinement and application of alignment principles. The current version demonstrates effective adherence to target behavior specifications and requirements.",
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
            ConsensusResult(
                reasoning="Consensus on evaluation reached through multi-model assessment. The models have analyzed the prompt against the target behavior criteria and reached agreement on the alignment score. This evaluation provides the foundation for determining necessary improvements.",
                consensus_achieved=True,
                final_response=EvaluationResult(
                    alignment_score=0.3,
                    feedback="Poor alignment with target",
                    strengths=[SemanticString(s) for s in []],
                    weaknesses=[SemanticString(s) for s in ["Many"]],
                    suggested_improvements=[SemanticString(s) for s in ["Improve"]],
                    reasoning="Initial prompt has significant alignment issues with the target behavior that require substantial refinement. The low alignment score indicates multiple areas needing improvement to meet the specified requirements and achieve effective alignment.",
                ),
                rounds=[],
                total_rounds=1,
                convergence_score=1.0,
                participating_models=["model1"],
            ),
            ConsensusResult(
                reasoning="Consensus on evaluation reached through multi-model assessment. The models have analyzed the prompt against the target behavior criteria and reached agreement on the alignment score. This evaluation provides the foundation for determining necessary improvements.",
                consensus_achieved=True,
                final_response=EvaluationResult(
                    alignment_score=0.6,
                    feedback="Better alignment but needs more work",
                    strengths=[SemanticString(s) for s in ["Some"]],
                    weaknesses=[SemanticString(s) for s in ["Few"]],
                    suggested_improvements=[SemanticString(s) for s in ["Continue"]],
                    reasoning="The prompt shows improvement but still needs refinement to meet target behavior requirements. The moderate alignment score indicates progress has been made, however additional iterations are necessary to achieve the desired level of alignment.",
                ),
                rounds=[],
                total_rounds=1,
                convergence_score=1.0,
                participating_models=["model1"],
            ),
            ConsensusResult(
                reasoning="Consensus on evaluation reached through multi-model assessment. The models have analyzed the prompt against the target behavior criteria and reached agreement on the alignment score. This evaluation provides the foundation for determining necessary improvements.",
                consensus_achieved=True,
                final_response=EvaluationResult(
                    alignment_score=0.85,
                    feedback="Good alignment achieved",
                    strengths=[SemanticString(s) for s in ["Many"]],
                    weaknesses=[SemanticString(s) for s in []],
                    suggested_improvements=[SemanticString(s) for s in []],
                    reasoning="The prompt now successfully aligns with the target behavior requirements after iterative refinement. The high alignment score demonstrates effective incorporation of feedback and successful application of alignment principles throughout the process.",
                ),
                rounds=[],
                total_rounds=1,
                convergence_score=1.0,
                participating_models=["model1"],
            ),
            ConsensusResult(
                reasoning="Consensus on evaluation reached through multi-model assessment. The models have analyzed the prompt against the target behavior criteria and reached agreement on the alignment score. This evaluation provides the foundation for determining necessary improvements.",
                consensus_achieved=True,
                final_response=EvaluationResult(
                    alignment_score=0.85,
                    feedback="Final alignment achieved successfully",
                    strengths=[SemanticString(s) for s in ["Many"]],
                    weaknesses=[SemanticString(s) for s in []],
                    suggested_improvements=[SemanticString(s) for s in []],
                    reasoning="Final evaluation confirms successful alignment with target behavior after completing the iterative refinement process. The stable high score indicates that the prompt effectively meets all specified requirements and achieves the desired alignment objectives.",
                ),
                rounds=[],
                total_rounds=1,
                convergence_score=1.0,
                participating_models=["model1"],
            ),
        ]

        mock_alignment_consensus.call.return_value = ConsensusResult(
            reasoning="Consensus on alignment feedback achieved through multi-model voting and analysis. The feedback identifies specific areas for improvement and provides actionable recommendations to enhance prompt alignment. This guidance supports the iterative refinement process toward achieving target behavior specifications.",
            consensus_achieved=True,
            final_response=AlignmentFeedback(
                overall_assessment="Prompt needs significant improvements to meet requirements",
                specific_issues=[SemanticString(s) for s in ["Issues"]],
                improvement_suggestions=[SemanticString(s) for s in ["Suggestions"]],
                principles_to_apply=AlignmentPrincipleList(principles=[]),
                revised_prompt_suggestion="Better prompt",
                confidence_score=0.75,
                reasoning="Feedback provided to iteratively improve prompt alignment through systematic application of improvement suggestions. The process identifies specific issues and provides actionable recommendations to enhance alignment with target behavior specifications.",
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
