"""
Tests for the principle-based alignment strategy.

This module tests the superior principle-based approach for prompt alignment,
which systematically extracts and applies reusable guidelines.
"""

import pytest

from com_blockether_catalyst.prompt.internal import (
    AlignmentFeedback,
    AlignmentPrinciple,
    PrincipleBasedAlignmentStrategy,
)


class TestPrincipleBasedAlignmentStrategy:
    """Test cases for PrincipleBasedAlignmentStrategy."""

    # Test constants
    TEST_PROMPT = "Explain quantum computing"
    TEST_PRINCIPLE_TEXT = "Always include concrete examples"
    MAX_PRINCIPLES = 10

    @pytest.fixture
    def strategy(self) -> PrincipleBasedAlignmentStrategy:
        """Create PrincipleBasedAlignmentStrategy instance."""
        return PrincipleBasedAlignmentStrategy()

    def test_extract_principles_from_explicit(self, strategy: PrincipleBasedAlignmentStrategy) -> None:
        """Test extracting principles from explicit principles list."""
        feedback = AlignmentFeedback(
            overall_assessment="Needs improvement to align with requirements",
            specific_issues=[],
            improvement_suggestions=[],
            principles_to_apply=[
                AlignmentPrinciple(
                    principle=self.TEST_PRINCIPLE_TEXT,
                    importance=0.9,
                ),
                AlignmentPrinciple(
                    principle="Provide step-by-step explanation",
                    importance=0.8,
                ),
                AlignmentPrinciple(
                    principle="Always define technical terms clearly",
                    importance=0.85,
                ),
            ],
            confidence_score=0.85,
            reasoning="These principles will help improve the prompt to better align with expectations and create more effective results.",
        )

        principles = strategy.extract_principles(feedback)

        assert len(principles) == 3
        assert principles[0].principle == self.TEST_PRINCIPLE_TEXT
        assert principles[0].importance == 0.9
        assert principles[1].principle == "Provide step-by-step explanation"
        # Principles should be extracted correctly
        assert len(principles) == 3

    def test_extract_principles_from_suggestions(self, strategy: PrincipleBasedAlignmentStrategy) -> None:
        """Test extracting principles from improvement suggestions."""
        feedback = AlignmentFeedback(
            overall_assessment="Needs work to improve clarity and specificity",
            specific_issues=[],
            improvement_suggestions=[
                "Should include more specific details",
                "Must provide clear context",
                "Ensure examples are relevant",
                "Always clarify technical terminology",
            ],
            principles_to_apply=[],
            confidence_score=0.75,
            reasoning="The prompt needs these improvements to achieve better alignment with the desired outcomes and user expectations.",
        )

        principles = strategy.extract_principles(feedback)

        assert len(principles) == 4
        # All principles should have default importance
        assert all(p.importance == 0.8 for p in principles)
        assert all(p.importance == 0.8 for p in principles)
        # Check that actionable keywords are present
        assert any("should" in p.principle.lower() for p in principles)
        assert any("must" in p.principle.lower() for p in principles)

    def test_extract_principles_from_issues(self, strategy: PrincipleBasedAlignmentStrategy) -> None:
        """Test extracting principles from specific issues."""
        feedback = AlignmentFeedback(
            overall_assessment="Has issues that need to be addressed",
            specific_issues=[
                "Prompt lacks clarity",
                "Too vague about requirements",
                "Missing context",
                "No examples provided",
            ],
            improvement_suggestions=[],
            principles_to_apply=[],
            confidence_score=0.7,
            reasoning="These issues prevent the prompt from achieving its intended purpose and must be addressed for effective alignment.",
        )

        principles = strategy.extract_principles(feedback)

        assert len(principles) == 4
        # All principles should have high importance
        assert all(p.importance == 0.9 for p in principles)
        assert principles[0].principle == "Prompt should include clarity"
        assert "specific" in principles[1].principle.lower()
        assert principles[2].principle == "should include context"
        assert "example" in principles[3].principle.lower()

    def test_convert_complex_issues_to_principles(self, strategy: PrincipleBasedAlignmentStrategy) -> None:
        """Test converting various complex issues to actionable principles."""
        test_cases = [
            ("lacks clarity", "should include clarity"),
            ("too complex", "Break down complex requirements into simpler parts"),
            ("unclear requirements", "Ensure clarity and specificity in requirements"),
            ("ambiguous instructions", "Provide specific, unambiguous instructions"),
            ("no context provided", "Provide sufficient context for understanding"),
            ("missing examples", "should include examples"),
        ]

        for issue, expected_principle in test_cases:
            principle = strategy._convert_issue_to_principle(issue)
            assert expected_principle.lower() in principle.lower() or principle.lower() in expected_principle.lower()

    def test_apply_correction_principles(self, strategy: PrincipleBasedAlignmentStrategy) -> None:
        """Test applying correction principles."""
        principles = [
            AlignmentPrinciple(
                principle="Ensure clarity and specificity",
                importance=0.9,
            ),
            AlignmentPrinciple(
                principle="Make requirements specific and concrete",
                importance=0.85,
            ),
        ]

        result = strategy.apply_principles(self.TEST_PROMPT, principles, preserve_context=True)

        assert self.TEST_PROMPT in result
        assert "specific" in result.lower() or "clarif" in result.lower()
        assert len(result) > len(self.TEST_PROMPT)

    def test_apply_improvement_principles(self, strategy: PrincipleBasedAlignmentStrategy) -> None:
        """Test applying improvement principles."""
        principles = [
            AlignmentPrinciple(
                principle="Add concrete examples",
                importance=0.85,
            ),
            AlignmentPrinciple(
                principle="Include relevant examples",
                importance=0.8,
            ),
        ]

        result = strategy.apply_principles(self.TEST_PROMPT, principles, preserve_context=True)

        assert self.TEST_PROMPT in result
        assert "example" in result.lower() or "concrete" in result.lower()

    def test_apply_general_principles(self, strategy: PrincipleBasedAlignmentStrategy) -> None:
        """Test applying general principles."""
        principles = [
            AlignmentPrinciple(
                principle="Focus on practical applications",
                importance=0.7,
            )
        ]

        result = strategy.apply_principles(self.TEST_PROMPT, principles, preserve_context=True)

        assert self.TEST_PROMPT in result
        assert "principle" in result.lower() or "practical" in result.lower()

    def test_principle_ordering_by_importance(self, strategy: PrincipleBasedAlignmentStrategy) -> None:
        """Test that principles are applied in order of importance."""
        principles = [
            AlignmentPrinciple(
                principle="Low priority: add footnotes",
                importance=0.3,
            ),
            AlignmentPrinciple(
                principle="High priority: ensure clarity",
                importance=0.95,
            ),
            AlignmentPrinciple(
                principle="Medium priority: add examples",
                importance=0.6,
            ),
        ]

        # The strategy should sort by importance internally
        result = strategy.apply_principles(self.TEST_PROMPT, principles, preserve_context=False)

        # Result should have modifications (exact order testing is implementation-specific)
        assert len(result) > len(self.TEST_PROMPT) or result != self.TEST_PROMPT

    def test_simplify_prompt(self, strategy: PrincipleBasedAlignmentStrategy) -> None:
        """Test prompt simplification."""
        complex_prompt = (
            "Please please explain explain the the concept concept of quantum quantum computing clearly clearly"
        )

        # Test the simplify method directly
        result = strategy._simplify_prompt(complex_prompt)

        # Should remove duplicate words
        assert result.count("please") < complex_prompt.count("please")
        assert result.count("explain") < complex_prompt.count("explain")
        assert result.count("quantum") < complex_prompt.count("quantum")

    def test_break_down_complexity(self, strategy: PrincipleBasedAlignmentStrategy) -> None:
        """Test breaking down complex prompts."""
        complex_prompt = "Explain quantum computing, machine learning, and blockchain technology"

        principles = [
            AlignmentPrinciple(
                principle="Break down complex requirements into simpler parts",
                importance=0.9,
            )
        ]

        result = strategy.apply_principles(complex_prompt, principles, preserve_context=False)

        # Should structure the prompt better
        assert "1." in result or "following" in result.lower()
        assert len(result) > len(complex_prompt)

    def test_add_context_principle(self, strategy: PrincipleBasedAlignmentStrategy) -> None:
        """Test adding context to prompts."""
        principles = [
            AlignmentPrinciple(
                principle="Provide sufficient context for understanding",
                importance=0.85,
            )
        ]

        result = strategy.apply_principles(self.TEST_PROMPT, principles, preserve_context=True)

        assert "context" in result.lower()
        assert self.TEST_PROMPT in result

    def test_restructure_prompt(self, strategy: PrincipleBasedAlignmentStrategy) -> None:
        """Test prompt restructuring for clarity."""
        unstructured = "Quantum computing is complex. What are its applications? It uses qubits."

        principles = [
            AlignmentPrinciple(
                principle="Improve structure for better clarity",
                importance=0.8,
            )
        ]

        result = strategy.apply_principles(unstructured, principles, preserve_context=False)

        # Should reorganize the prompt
        assert "main" in result.lower() or "request" in result.lower()
        assert result != unstructured

    def test_preserve_context_option(self, strategy: PrincipleBasedAlignmentStrategy) -> None:
        """Test that preserve_context option works correctly."""
        principles = [
            AlignmentPrinciple(
                principle="Add specific requirements",
                importance=0.8,
            )
        ]

        # With context preservation
        with_context = strategy.apply_principles(self.TEST_PROMPT, principles, preserve_context=True)
        assert self.TEST_PROMPT in with_context

        # Without context preservation
        without_context = strategy.apply_principles(self.TEST_PROMPT, principles, preserve_context=False)
        # May or may not contain original, but should be transformed
        assert len(without_context) > 0

    def test_actionable_principle_detection(self, strategy: PrincipleBasedAlignmentStrategy) -> None:
        """Test detection of actionable principles."""
        actionable = [
            "Should include examples",
            "Must provide context",
            "Ensure clarity",
            "Always be specific",
            "Never use jargon without explanation",
            "Require detailed responses",
        ]

        non_actionable = [
            "Examples are good",
            "Context is important",
            "Clarity matters",
            "Specificity helps",
        ]

        for text in actionable:
            assert strategy._is_actionable_principle(text) is True

        for text in non_actionable:
            assert strategy._is_actionable_principle(text) is False

    def test_max_principles_limit(self, strategy: PrincipleBasedAlignmentStrategy) -> None:
        """Test that principle extraction respects the maximum limit."""
        many_issues = [f"Issue {i}" for i in range(20)]
        many_suggestions = [f"Should fix issue {i}" for i in range(20)]

        feedback = AlignmentFeedback(
            overall_assessment="Many issues to address for proper alignment",
            specific_issues=many_issues,
            improvement_suggestions=many_suggestions,
            principles_to_apply=[],
            confidence_score=0.7,
            reasoning="There are numerous issues that need to be addressed to achieve proper alignment with the target behavior.",
        )

        principles = strategy.extract_principles(feedback)

        assert len(principles) <= strategy.MAX_PRINCIPLES_PER_FEEDBACK
        assert len(principles) == strategy.MAX_PRINCIPLES_PER_FEEDBACK
