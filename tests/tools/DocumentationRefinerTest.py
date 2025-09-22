"""Tests for DocumentationRefiner."""

import asyncio

# Import from tools directory
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from agno.models.openai import OpenAILike

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "tools"))
from DocumentationRefiner import (
    DocumentationConsensusRefiner,
    DocumentationEvaluation,
    PerspectiveEvaluator,
)

from blockether_catalyst.consensus.ConsensusTypes import (
    ConsensusResult,
    ConsensusSettings,
    VerbosityLevel,
)


class TestDocumentationRefiner:
    """Test suite for DocumentationRefiner."""

    @pytest.fixture
    def mock_llms(self):
        """Create mock LLMs for consensus testing."""
        llms = []
        for i in range(3):  # Need at least 3 for consensus
            llm = MagicMock(spec=OpenAILike)
            llm.run = MagicMock()
            llms.append(llm)
        return llms

    @pytest.fixture
    def mock_judge(self):
        """Create a mock judge LLM."""
        judge = MagicMock(spec=OpenAILike)
        judge.run = MagicMock()
        return judge

    @pytest.fixture
    def refiner(self, mock_llms, mock_judge):
        """Create DocumentationConsensusRefiner instance."""
        settings = ConsensusSettings(
            max_rounds=2,
            threshold=0.6,
            first_round_threshold=0.8,
            verbosity=VerbosityLevel.SILENT,
        )
        return DocumentationConsensusRefiner(mock_llms, mock_judge, settings)

    @pytest.fixture
    def sample_documentation(self):
        """Sample documentation for testing."""
        return """# Test Documentation

## Introduction
This is a test document. We will test the system. The system is for testing.

## Features  
- Feature 1: Does something
- Feature 2: Does another thing
- Feature 1: Does something (duplicate)

## Installation
Install with pip.

## Introduction Again
This is redundant with the first introduction.
"""

    def test_documentation_evaluation_creation(self):
        """Test DocumentationEvaluation model creation."""
        eval = DocumentationEvaluation(
            clarity_score=8.0,
            completeness_score=9.0,
            technical_accuracy_score=8.5,
            value_proposition_score=7.5,
            has_hello_world=True,
            has_api_docs_link=False,
            has_examples_section=True,
            navigation_matches_content=True,
            strengths=["Clear structure", "Good examples"],
            critical_issues=["Missing API docs"],
            missing_sections=["API Reference"],
            immediate_fixes=["Add API link"],
            overall_assessment="Good documentation with minor gaps",
            reasoning="The documentation demonstrates good organizational structure with clear sections and logical flow. However, it lacks a comprehensive API reference section which is critical for developers. The documentation would benefit from adding detailed API endpoints, parameters, and response formats to enable effective integration.",
        )

        assert eval.clarity_score == 8.0
        assert eval.completeness_score == 9.0
        assert eval.has_hello_world is True
        assert len(eval.strengths) == 2
        assert "Missing API docs" in eval.critical_issues

    def test_perspective_evaluator_init(self):
        """Test PerspectiveEvaluator initialization."""
        mock_llm = MagicMock(spec=OpenAILike)
        evaluator = PerspectiveEvaluator("new_user", mock_llm)

        assert evaluator.perspective == "new_user"
        assert evaluator.llm == mock_llm
        assert "BRAND NEW USER" in evaluator.prompt_template

    @pytest.mark.anyio
    async def test_perspective_evaluator_call(self):
        """Test PerspectiveEvaluator call method."""
        mock_llm = MagicMock(spec=OpenAILike)
        evaluator = PerspectiveEvaluator("new_user", mock_llm)

        # Mock the agent response - patch where it's imported
        with patch("agno.agent.Agent") as MockAgent:
            mock_agent_instance = MagicMock()
            mock_agent_instance.run.return_value = MagicMock(
                content='{"clarity_score": 7.0, "completeness_score": 8.0, "technical_accuracy_score": 9.0, "value_proposition_score": 8.5, "has_hello_world": true, "has_api_docs_link": false, "has_examples_section": true, "navigation_matches_content": true, "strengths": ["Clear"], "critical_issues": [], "missing_sections": [], "immediate_fixes": [], "overall_assessment": "Good", "reasoning": "The documentation is well structured with clear sections and good technical accuracy. The installation instructions are straightforward and the feature descriptions are comprehensive. The value proposition is clearly articulated, making it easy to understand the benefits of the system."}'
            )
            MockAgent.return_value = mock_agent_instance

            result = await evaluator.call("Test documentation")

            assert isinstance(result, DocumentationEvaluation)
            assert result.clarity_score == 7.0
            assert result.completeness_score == 8.0

    def test_generate_report(self, refiner):
        """Test report generation from ConsensusResult."""
        eval = DocumentationEvaluation(
            clarity_score=8.0,
            completeness_score=7.5,
            technical_accuracy_score=9.0,
            value_proposition_score=9.0,
            has_hello_world=False,
            has_api_docs_link=False,
            has_examples_section=False,
            navigation_matches_content=False,
            strengths=[
                "Clear structure",
                "Good technical accuracy",
                "Strong value proposition",
            ],
            critical_issues=["No Hello World", "Missing API docs"],
            missing_sections=["Examples section"],
            immediate_fixes=["Add Hello World", "Link API docs"],
            overall_assessment="Good documentation but missing key onboarding elements",
            reasoning="The analysis reveals that while the documentation has strong technical accuracy and clear value proposition, it critically lacks essential onboarding elements. The absence of a Hello World example, API documentation link, and proper examples section significantly hampers developer adoption. These issues need immediate attention.",
        )

        # Create a mock ConsensusResult
        from blockether_catalyst.consensus.ConsensusTypes import ConsensusMetrics

        mock_result = MagicMock(spec=ConsensusResult)
        mock_result.final_response = eval
        mock_result.consensus_achieved = True
        mock_result.total_rounds = 2
        mock_result.convergence_score = 0.85
        mock_result.reasoning = "Consensus achieved after 2 rounds"
        mock_result.model_contributions = {
            "new_user_evaluator": 0.9,
            "experienced_dev_evaluator": 0.95,
        }
        mock_result.metrics = ConsensusMetrics(
            dissent_rate=0.1,
            consensus_confidence=0.9,
            fallback_method=None,
            duration_ms=1500,
            rounds_to_convergence=2,
            total_model_calls=6,
            convergence_achieved=True,
            convergence_indicator=0.85,
        )

        report = refiner.generate_report(mock_result)

        assert (
            "8.2/10" in report
        )  # Overall score (8*0.3 + 7.5*0.3 + 9*0.2 + 9*0.2 = 2.4 + 2.25 + 1.8 + 1.8 = 8.25 ~= 8.2)
        assert "✅ Yes" in report  # Consensus achieved
        assert "Clear structure" in report
        assert "❌ No Hello World" in report
        assert "❌" in report  # Has checklist failures

    @pytest.mark.anyio
    async def test_evaluate_documentation(self, refiner, mock_llms, mock_judge):
        """Test full consensus evaluation workflow."""
        # Create sample documentation
        doc_content = """# Test Documentation
        
        This is a test document for evaluation.
        
        ## Installation
        Install with pip.
        
        ## Features
        - Feature 1
        - Feature 2
        """

        # Mock evaluator responses
        mock_eval_response = DocumentationEvaluation(
            clarity_score=7.0,
            completeness_score=6.0,
            technical_accuracy_score=8.0,
            value_proposition_score=7.5,
            has_hello_world=False,
            has_api_docs_link=False,
            has_examples_section=False,
            navigation_matches_content=True,
            strengths=["Clear structure"],
            critical_issues=["No examples"],
            missing_sections=["Examples"],
            immediate_fixes=["Add Hello World"],
            overall_assessment="Needs improvement",
            reasoning="The documentation is missing several key elements that are essential for onboarding new users. Specifically, there is no Hello World example to help users get started quickly, no API documentation link, and no examples section to demonstrate practical usage patterns.",
        )

        # Patch the PerspectiveEvaluator call method
        with patch.object(PerspectiveEvaluator, "call", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = mock_eval_response

            # Run evaluation
            result = await refiner.evaluate_documentation(doc_content)

            assert isinstance(result, ConsensusResult)
            assert isinstance(result.final_response, DocumentationEvaluation)

            # Verify evaluators were called
            assert mock_call.call_count >= 3  # At least 3 evaluators

    def test_consensus_settings_initialization(self):
        """Test ConsensusSettings proper initialization."""
        settings = ConsensusSettings(
            max_rounds=3,
            threshold=0.6,
            first_round_threshold=0.8,
            verbosity=VerbosityLevel.SILENT,
        )

        assert settings.max_rounds == 3
        assert settings.threshold == 0.6
        assert settings.first_round_threshold == 0.8
        assert settings.verbosity == VerbosityLevel.SILENT

    def test_refiner_initialization(self, mock_llms, mock_judge):
        """Test DocumentationConsensusRefiner initialization."""
        refiner = DocumentationConsensusRefiner(llm_models=mock_llms, judge_model=mock_judge)

        assert len(refiner.llm_models) == 3
        assert refiner.judge_model == mock_judge
        assert refiner.settings.max_rounds == 3  # Default
        assert refiner.settings.threshold == 0.6  # Default

    def test_refiner_requires_minimum_models(self, mock_judge):
        """Test that refiner requires at least 3 models."""
        with pytest.raises(ValueError, match="Need at least 3 models"):
            refiner = DocumentationConsensusRefiner(
                llm_models=[mock_judge, mock_judge],  # Only 2
                judge_model=mock_judge,
            )
            # Need to call evaluate to trigger the error
            asyncio.run(refiner.evaluate_documentation("test"))
