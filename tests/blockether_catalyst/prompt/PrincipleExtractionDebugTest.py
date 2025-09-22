"""
Debug tests to understand why principles aren't being extracted from feedback.

This module contains tests that verify the entire principle extraction flow
to identify where the issue is occurring.
"""

import pytest
from pydantic import BaseModel

from blockether_catalyst.prompt.PrincipleBasedAlignmentStrategy import (
    PrincipleBasedAlignmentStrategy,
)
from blockether_catalyst.prompt.PromptAlignmentTypes import (
    AlignmentFeedback,
    AlignmentPrinciple,
    AlignmentPrincipleList,
    SemanticString,
)


class TestPrincipleExtractionDebug:
    """Debug tests for principle extraction issues."""

    # Test constants
    GOOD_FEEDBACK_TEXT = (
        "The prompt is well-structured and provides clear instructions for extracting and classifying information"
    )
    SCORE = 0.85

    @pytest.fixture
    def strategy(self) -> PrincipleBasedAlignmentStrategy:
        """Create PrincipleBasedAlignmentStrategy instance."""
        return PrincipleBasedAlignmentStrategy()

    def test_empty_feedback_extraction(self, strategy: PrincipleBasedAlignmentStrategy) -> None:
        """Test extraction from feedback with empty principle lists."""
        # This mimics what seems to be happening - feedback with no explicit principles
        feedback = AlignmentFeedback(
            overall_assessment=self.GOOD_FEEDBACK_TEXT,
            specific_issues=[],  # Empty
            improvement_suggestions=[],  # Empty
            principles_to_apply=AlignmentPrincipleList(principles=[]),  # Empty
            confidence_score=self.SCORE,
            reasoning="The prompt is already well-aligned with expectations. It demonstrates excellent structure, clarity, and completeness. The instructions are specific and actionable, making it easy for the model to understand and execute the desired task effectively.",
        )

        principles = strategy.extract_principles(feedback)

        # THIS IS THE PROBLEM - no principles are extracted from empty feedback
        assert len(principles) == 0, "No principles should be extracted from empty feedback"

    def test_feedback_with_suggestions_only(self, strategy: PrincipleBasedAlignmentStrategy) -> None:
        """Test extraction when only improvement suggestions are present."""
        feedback = AlignmentFeedback(
            overall_assessment="Needs improvement to achieve better alignment",
            specific_issues=[],
            improvement_suggestions=[
                SemanticString(value="Should add more specific instructions"),
                SemanticString(value="Must include concrete examples"),
                SemanticString(value="Ensure clarity in requirements"),
            ],
            principles_to_apply=AlignmentPrincipleList(principles=[]),
            confidence_score=0.7,
            reasoning="The prompt needs these improvements to be more effective. The current version lacks specificity and concrete guidance. By adding more specific instructions, including concrete examples, and ensuring clarity in requirements, the prompt will better guide the model to produce the desired output consistently.",
        )

        principles = strategy.extract_principles(feedback)

        # Should extract principles from suggestions
        assert len(principles) == 3
        assert all(p.importance == 0.8 for p in principles)
        assert any("should" in p.principle.lower() for p in principles)

    def test_feedback_with_issues_only(self, strategy: PrincipleBasedAlignmentStrategy) -> None:
        """Test extraction when only specific issues are present."""
        feedback = AlignmentFeedback(
            overall_assessment="Has several issues that need to be addressed",
            specific_issues=[
                SemanticString(value="Lacks clarity"),
                SemanticString(value="Missing examples"),
                SemanticString(value="Too vague"),
            ],
            improvement_suggestions=[],
            principles_to_apply=AlignmentPrincipleList(principles=[]),
            confidence_score=0.6,
            reasoning="These issues need to be addressed for the prompt to be effective. The lack of clarity makes it difficult for the model to understand the requirements. Missing examples leave the model without concrete guidance. The vague nature of the instructions leads to inconsistent and unpredictable outputs.",
        )

        principles = strategy.extract_principles(feedback)

        # Should convert issues to principles
        assert len(principles) == 3
        assert all(p.importance == 0.9 for p in principles)
        assert "should include clarity" in principles[0].principle

    def test_feedback_with_explicit_principles(self, strategy: PrincipleBasedAlignmentStrategy) -> None:
        """Test extraction when explicit principles are provided."""
        feedback = AlignmentFeedback(
            overall_assessment="Needs improvement to achieve better alignment",
            specific_issues=[],
            improvement_suggestions=[],
            principles_to_apply=AlignmentPrincipleList(
                principles=[
                    AlignmentPrinciple(
                        principle="Always provide concrete examples",
                        importance=0.95,
                    ),
                    AlignmentPrinciple(
                        principle="Ensure clarity in all instructions",
                        importance=0.9,
                    ),
                ]
            ),
            confidence_score=0.75,
            reasoning="Apply these principles for improvement. The prompt currently lacks the concrete examples and clarity needed for effective performance. By systematically applying these principles, we can transform the prompt into a more robust and reliable instruction set that will consistently produce high-quality outputs.",
        )

        principles = strategy.extract_principles(feedback)

        # Should extract explicit principles
        assert len(principles) == 2
        assert principles[0].principle == "Always provide concrete examples"
        assert principles[0].importance == 0.95

    def test_mixed_feedback_extraction(self, strategy: PrincipleBasedAlignmentStrategy) -> None:
        """Test extraction from feedback with all types of input."""
        feedback = AlignmentFeedback(
            overall_assessment="Multiple areas for improvement",
            specific_issues=[
                SemanticString(value="Lacks context"),
                SemanticString(value="Too complex"),
            ],
            improvement_suggestions=[
                SemanticString(value="Should simplify the language"),
                SemanticString(value="Must add relevant examples"),
            ],
            principles_to_apply=AlignmentPrincipleList(
                principles=[
                    AlignmentPrinciple(
                        principle="Be concise and clear",
                        importance=0.85,
                    ),
                ]
            ),
            confidence_score=0.7,
            reasoning="Multiple improvements needed across different aspects of the prompt. The lack of context makes it difficult to understand the requirements. The complexity creates confusion. The language needs simplification and relevant examples must be added. Being concise and clear will significantly improve the prompt's effectiveness.",
        )

        principles = strategy.extract_principles(feedback)

        # Should extract from all sources (limited by MAX_PRINCIPLES_PER_FEEDBACK)
        assert len(principles) > 0
        assert len(principles) <= strategy.MAX_PRINCIPLES_PER_FEEDBACK

        # Should have principles from different sources
        principle_texts = [p.principle for p in principles]
        assert any("concise" in p.lower() for p in principle_texts)  # From explicit
        assert any("should" in p.lower() or "must" in p.lower() for p in principle_texts)  # From suggestions

    def test_high_score_feedback_without_issues(self, strategy: PrincipleBasedAlignmentStrategy) -> None:
        """Test what happens with high-scoring feedback that has no issues."""
        # This simulates when a prompt is already good
        feedback = AlignmentFeedback(
            overall_assessment="Excellent prompt that meets all requirements",
            specific_issues=[],
            improvement_suggestions=[],
            principles_to_apply=AlignmentPrincipleList(principles=[]),
            confidence_score=0.95,
            reasoning="The prompt is already optimal and well-aligned with the target requirements. It demonstrates exceptional clarity, provides comprehensive context, includes relevant examples, and uses precise language. The structure is logical and the instructions are unambiguous, making it an exemplary prompt that requires no further refinement.",
        )

        principles = strategy.extract_principles(feedback)

        # No principles should be extracted when everything is good
        assert len(principles) == 0

    def test_actionable_vs_non_actionable_suggestions(self, strategy: PrincipleBasedAlignmentStrategy) -> None:
        """Test that only actionable suggestions become principles."""
        feedback = AlignmentFeedback(
            overall_assessment="Mixed feedback with both positives and areas for improvement",
            specific_issues=[],
            improvement_suggestions=[
                SemanticString(value="Should add more detail"),  # Actionable
                SemanticString(value="Must include examples"),  # Actionable
                SemanticString(value="This is interesting"),  # Not actionable
                SemanticString(value="Good attempt"),  # Not actionable
                SemanticString(value="Ensure proper formatting"),  # Actionable
            ],
            principles_to_apply=AlignmentPrincipleList(principles=[]),
            confidence_score=0.7,
            reasoning="Some improvements needed to enhance the prompt's effectiveness. While the prompt has good elements, adding more detail, including examples, and ensuring proper formatting will make it more reliable. The actionable suggestions provided will systematically address these areas and create a more robust prompt.",
        )

        principles = strategy.extract_principles(feedback)

        # Should only extract actionable suggestions
        assert len(principles) == 3  # Only the actionable ones
        for p in principles:
            assert strategy._is_actionable_principle(p.principle)

    def test_principle_length_filtering(self, strategy: PrincipleBasedAlignmentStrategy) -> None:
        """Test that principles are filtered by length constraints."""
        feedback = AlignmentFeedback(
            overall_assessment="Test length filtering",
            specific_issues=[],
            improvement_suggestions=[],
            principles_to_apply=AlignmentPrincipleList(
                principles=[
                    AlignmentPrinciple(
                        principle="Too short",  # Less than MIN_PRINCIPLE_LENGTH
                        importance=0.9,
                    ),
                    AlignmentPrinciple(
                        principle="This principle has exactly the right length to be included",
                        importance=0.85,
                    ),
                    AlignmentPrinciple(
                        principle="A" * 300,  # More than MAX_PRINCIPLE_LENGTH
                        importance=0.8,
                    ),
                ]
            ),
            confidence_score=0.75,
            reasoning="Testing length constraints for principle extraction. This test ensures that the system properly filters principles based on their length, rejecting those that are too short to be meaningful or too long to be practical. Only principles within the acceptable length range should be extracted and applied to maintain quality.",
        )

        principles = strategy.extract_principles(feedback)

        # Should only include principles within length limits
        assert len(principles) == 1
        assert "exactly the right length" in principles[0].principle

    def test_issue_to_principle_conversion_edge_cases(self, strategy: PrincipleBasedAlignmentStrategy) -> None:
        """Test edge cases in converting issues to principles."""
        test_cases = [
            ("Missing critical information", "should include critical information"),
            ("No examples", "Include relevant examples to clarify expectations"),
            ("unclear requirements", "Ensure clarity and specificity in requirements"),
            ("too vague", "Make requirements specific and concrete"),
            ("too long", "Keep prompts concise while maintaining clarity"),
            ("no context", "Provide sufficient context for understanding"),
            ("Some random issue", "Ensure: Some random issue"),  # Generic conversion
        ]

        for issue, expected_pattern in test_cases:
            principle = strategy._convert_issue_to_principle(issue)
            assert (
                expected_pattern.lower() in principle.lower() or principle.lower() in expected_pattern.lower()
            ), f"Issue '{issue}' should convert to something like '{expected_pattern}', got '{principle}'"
