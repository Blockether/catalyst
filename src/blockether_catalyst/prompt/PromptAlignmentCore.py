"""
Core module for prompt alignment using Consensus for multi-model reliability.

Based on: https://colab.research.google.com/github/pair-code/model-alignment/blob/main/notebooks/Gemma_for_Model_Alignment.ipynb

Enhanced with Consensus mechanism for more reliable and consistent alignment through
multiple models working together to evaluate and refine prompts.
"""

import logging
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from ..consensus.Consensus import Consensus
from .PrincipleBasedAlignmentStrategy import PrincipleBasedAlignmentStrategy
from .PromptAlignmentTypes import (
    AlignmentFeedback,
    AlignmentMetrics,
    AlignmentPrinciple,
    EvaluationResult,
    PromptEvolution,
)

logger = logging.getLogger(__name__)


class PromptConfiguration(BaseModel):
    """Configuration for prompt alignment process."""

    initial_prompt: str = Field(description="The initial prompt to align")
    target_behavior: str = Field(description="Description of the desired behavior for the aligned prompt")
    max_iterations: int = Field(default=5, ge=1, le=20, description="Maximum alignment iterations")
    score_threshold: float = Field(default=0.8, ge=0.0, le=1.0, description="Target alignment score")
    preserve_context: bool = Field(default=True, description="Whether to preserve original context in prompts")


class AlignmentResult(BaseModel):
    """Result of prompt alignment process."""

    original_prompt: str = Field(description="The original input prompt")
    aligned_prompt: str = Field(description="The aligned output prompt")
    iterations_used: int = Field(description="Number of iterations performed")
    final_score: float = Field(description="Final alignment score achieved")
    evolution_history: List[PromptEvolution] = Field(default_factory=list, description="History of prompt evolution")
    principles_applied: List[AlignmentPrinciple] = Field(
        default_factory=list, description="Principles applied during alignment"
    )
    metrics: AlignmentMetrics = Field(description="Detailed metrics from alignment process")


class PromptAlignmentCore:
    """Core implementation for prompt alignment using TypedCalls."""

    # Static configuration constants
    MIN_PROMPT_LENGTH = 10
    MAX_PROMPT_LENGTH = 10000
    DEFAULT_TEMPERATURE = 0.7

    def __init__(
        self,
        target_consensus: Consensus[EvaluationResult],
        alignment_consensus: Consensus[AlignmentFeedback],
    ):
        """
        Initialize prompt alignment core with Consensus for multi-model reliability.

        Args:
            target_consensus: Consensus for evaluation (multiple models evaluate prompts)
            alignment_consensus: Consensus for alignment feedback (multiple models provide improvement suggestions)
        """
        self._target_consensus = target_consensus
        self._alignment_consensus = alignment_consensus

        self._principle_strategy = PrincipleBasedAlignmentStrategy()
        self._evolution_cache: Dict[str, List[PromptEvolution]] = {}

        self._principles: List[AlignmentPrinciple] = []

    async def align_prompt(self, config: PromptConfiguration) -> AlignmentResult:
        """
        Align a prompt according to the specified configuration.

        Args:
            config: Configuration for the alignment process

        Returns:
            AlignmentResult containing the aligned prompt and metrics
        """
        if len(config.initial_prompt) < self.MIN_PROMPT_LENGTH:
            raise ValueError(f"Prompt too short (min {self.MIN_PROMPT_LENGTH} chars)")
        if len(config.initial_prompt) > self.MAX_PROMPT_LENGTH:
            raise ValueError(f"Prompt too long (max {self.MAX_PROMPT_LENGTH} chars)")

        current_prompt = config.initial_prompt
        evolution_history: List[PromptEvolution] = []
        principles_applied: List[AlignmentPrinciple] = []

        for iteration in range(config.max_iterations):
            # Evaluate current prompt
            evaluation = await self._evaluate_prompt(current_prompt, config.target_behavior)

            # Record evolution
            evolution = PromptEvolution(
                iteration=iteration,
                prompt=current_prompt,
                score=evaluation.alignment_score,
                feedback=evaluation.feedback,
                improvements_made=[s.value for s in evaluation.suggested_improvements],
            )
            evolution_history.append(evolution)

            # Check if we've reached the target
            if evaluation.alignment_score >= config.score_threshold:
                logger.info(f"Alignment achieved at iteration {iteration} " f"with score {evaluation.alignment_score}")
                break

            # Get alignment feedback
            feedback = await self._get_alignment_feedback(current_prompt, evaluation, config.target_behavior)
            stored_principles = self.get_stored_principles()

            (
                aligned_prompt,
                new_principles,
            ) = await self._apply_principle_based_alignment(
                current_prompt, feedback, config.preserve_context, stored_principles
            )
            principles_applied.extend(new_principles)

            self._add_principles(new_principles)

            current_prompt = aligned_prompt

        # Final evaluation
        final_evaluation = await self._evaluate_prompt(current_prompt, config.target_behavior)

        # Calculate metrics
        metrics = self._calculate_metrics(evolution_history, final_evaluation.alignment_score)

        return AlignmentResult(
            original_prompt=config.initial_prompt,
            aligned_prompt=current_prompt,
            iterations_used=len(evolution_history),
            final_score=final_evaluation.alignment_score,
            evolution_history=evolution_history,
            principles_applied=principles_applied,
            metrics=metrics,
        )

    async def _evaluate_prompt(self, prompt: str, target_behavior: str) -> EvaluationResult:
        """Evaluate a prompt against target behavior."""
        evaluation_request = f"""
        Evaluate this prompt against the target behavior.

        Prompt: {prompt}

        Target Behavior: {target_behavior}

        Provide a detailed evaluation with alignment score and feedback.
        """

        consensus_result = await self._target_consensus.call(evaluation_request)
        return consensus_result.final_response

    async def _get_alignment_feedback(
        self,
        prompt: str,
        evaluation: EvaluationResult,
        target_behavior: str,
    ) -> AlignmentFeedback:
        """Get alignment feedback from the alignment model using critique approach."""
        feedback_request = f"""
        Critique this prompt and its evaluation to extract principles for improvement.

        Current Prompt: {prompt}

        Evaluation Score: {evaluation.alignment_score}
        Evaluation Feedback: {evaluation.feedback}

        Target Behavior: {target_behavior}

        Please critique this prompt and identify specific principles that would improve its alignment.
        Focus on extracting reusable guidelines that can be systematically applied.

        What principles should guide the improvement of this prompt to better achieve the target behavior?
        """

        consensus_result = await self._alignment_consensus.call(feedback_request)
        return consensus_result.final_response

    async def _apply_principle_based_alignment(
        self,
        prompt: str,
        feedback: AlignmentFeedback,
        preserve_context: bool,
        stored_principles: Optional[List[AlignmentPrinciple]] = None,
    ) -> Tuple[str, List[AlignmentPrinciple]]:
        """Apply principle-based alignment strategy with optional stored principles."""
        # Extract new principles from feedback
        new_principles = self._principle_strategy.extract_principles(feedback)

        # Combine with stored principles if available
        all_principles = new_principles.copy()
        if stored_principles:
            # Add stored principles with slightly lower importance
            for sp in stored_principles:
                adjusted = AlignmentPrinciple(
                    principle=sp.principle,
                    importance=sp.importance * 0.9,  # Slightly lower for stored
                )
                all_principles.append(adjusted)

        # Apply all principles
        aligned_prompt = self._principle_strategy.apply_principles(prompt, all_principles, preserve_context)
        return aligned_prompt, new_principles

    def _add_principles(self, principles: List[AlignmentPrinciple]) -> None:
        """
        Add principles to the persistent database for reuse.

        Args:
            principles: Principles to add
        """
        # Add unique principles (avoid duplicates)
        existing_texts = {p.principle for p in self._principles}
        for principle in principles:
            if principle.principle not in existing_texts:
                self._principles.append(principle)
                existing_texts.add(principle.principle)

    def get_stored_principles(self) -> List[AlignmentPrinciple]:
        """
        Retrieve all stored principles.

        Returns:
            List of all stored principles
        """
        return self._principles.copy()

    def get_principle_count(self) -> int:
        """
        Get the total number of stored principles.

        Returns:
            Number of stored principles
        """
        return len(self._principles)

    def export_principles(self) -> List[Dict]:
        """
        Export all principles as a shareable resource.

        This implements the notebook's concept of principles as "resources to share
        and collaborate on".

        Returns:
            List of principles (JSON-serializable)
        """
        return [
            {
                "principle": p.principle,
                "importance": p.importance,
            }
            for p in self._principles
        ]

    def import_principles(self, principles_data: List[Dict]) -> None:
        """
        Import principles from a shared resource.

        Args:
            principles_data: List of principle dictionaries
        """
        principles = [
            AlignmentPrinciple(
                principle=p["principle"],
                importance=p.get("importance", 0.8),
            )
            for p in principles_data
        ]
        self._add_principles(principles)

        logger.info(f"Imported {len(principles)} principles")

    def _calculate_metrics(self, history: List[PromptEvolution], final_score: float) -> AlignmentMetrics:
        """Calculate alignment process metrics."""
        if not history:
            return AlignmentMetrics(
                total_iterations=0,
                average_improvement=0.0,
                final_score=final_score,
                convergence_rate=0.0,
                stability_score=1.0,
            )

        scores = [e.score for e in history]
        improvements = [scores[i] - scores[i - 1] for i in range(1, len(scores))]

        # Calculate stability (lower variance is better)
        if len(improvements) > 1:
            variance = sum((x - sum(improvements) / len(improvements)) ** 2 for x in improvements) / len(improvements)
            stability = 1.0 / (1.0 + variance)
        else:
            stability = 1.0

        return AlignmentMetrics(
            total_iterations=len(history),
            average_improvement=(sum(improvements) / len(improvements) if improvements else 0.0),
            final_score=final_score,
            convergence_rate=((final_score - scores[0]) / len(history) if len(history) > 0 else 0.0),
            stability_score=stability,
        )

    async def batch_align(self, configs: List[PromptConfiguration]) -> List[AlignmentResult]:
        """
        Align multiple prompts in batch.

        Args:
            configs: List of prompt configurations to align

        Returns:
            List of alignment results
        """
        results = []
        for config in configs:
            try:
                result = await self.align_prompt(config)
                results.append(result)
            except Exception:
                logger.exception(f"Failed to align prompt: {config.initial_prompt[:50]}...")
                # Create a failed result
                results.append(
                    AlignmentResult(
                        original_prompt=config.initial_prompt,
                        aligned_prompt=config.initial_prompt,
                        iterations_used=0,
                        final_score=0.0,
                        evolution_history=[],
                        principles_applied=[],
                        metrics=AlignmentMetrics(
                            total_iterations=0,
                            average_improvement=0.0,
                            final_score=0.0,
                            convergence_rate=0.0,
                            stability_score=0.0,
                        ),
                    )
                )

        return results

    def get_cached_evolution(self, prompt: str) -> Optional[List[PromptEvolution]]:
        """Get cached evolution history for a prompt if available."""
        return self._evolution_cache.get(prompt)

    def clear_cache(self) -> None:
        """Clear the evolution cache."""
        self._evolution_cache.clear()
