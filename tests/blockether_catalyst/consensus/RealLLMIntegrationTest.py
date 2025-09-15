"""
Real LLM integration tests for consensus using InstructorLLMCall.

This test suite validates the consensus mechanism with actual LLM calls
to localhost:3005/v1 using the InstructorLLMCall implementation.

Run with: pytest -m integration
Skip with: pytest -m "not integration"
"""

from typing import Any, List, Literal, Optional

import pytest

from blockether_catalyst.consensus.ConsensusCore import ConsensusCore
from blockether_catalyst.consensus.ConsensusTypes import ConsensusSettings
from blockether_catalyst.consensus.VotingComparison import (
    BaseModelWithReasoning,
    ComparisonStrategy,
    VotingField,
)
from blockether_catalyst.utils.instructor.InstructorLLMCall import InstructorLLMCall
from blockether_catalyst.utils.TypedCalls import ArityOneTypedCall

# Skip these tests if local LLM server is not available
pytestmark = pytest.mark.integration


class MathResponse(BaseModelWithReasoning):
    """Response for math problems."""

    answer: float = VotingField(
        description="The numerical answer",
        comparison=ComparisonStrategy.RANGE,
        tolerance=0.01,  # Allow 1% variance for floating point
    )

    unit: Optional[str] = VotingField(
        default=None,
        description="Unit of measurement if applicable",
        comparison=ComparisonStrategy.EXACT,  # Units must match exactly
    )

    method: str = VotingField(
        description="Method used to solve",
        comparison=ComparisonStrategy.SEMANTIC,  # Allow semantic similarity
        threshold=0.8,  # 80% similarity for method descriptions
    )


class ClassificationResponse(BaseModelWithReasoning):
    """Response for classification tasks."""

    category: Literal[
        "Technology",
        "Programming",
        "Software",
        "Computing",
        "Science",
        "Mathematics",
        "Engineering",
        "Business",
        "Healthcare",
        "Education",
        "Other",
    ] = VotingField(
        description="Main category - must be one of the predefined labels",
        comparison=ComparisonStrategy.EXACT,  # Category must match exactly
    )

    confidence: float = VotingField(
        description="Confidence score between 0 and 1",
        comparison=ComparisonStrategy.RANGE,
        tolerance=0.15,  # Allow 15% variance in confidence
    )

    subcategories: List[str] = VotingField(
        default_factory=list,
        description="List of relevant subcategories",
        comparison=ComparisonStrategy.SEMANTIC,
        threshold=0.75,  # 75% similarity for subcategory lists
    )


class AnalysisResponse(BaseModelWithReasoning):
    """Response for text analysis tasks."""

    sentiment: Literal["positive", "negative", "neutral", "optimistic", "pessimistic", "mixed"] = VotingField(
        description="Overall sentiment - must be one of the predefined labels",
        comparison=ComparisonStrategy.EXACT,  # Sentiment must match exactly
    )

    key_points: List[str] = VotingField(
        description="Key points extracted from the text",
        comparison=ComparisonStrategy.SEMANTIC,
        threshold=0.7,  # 70% similarity for key points
    )

    summary: str = VotingField(
        description="Brief summary of the text",
        comparison=ComparisonStrategy.SEMANTIC,
        threshold=0.75,  # 75% similarity for summaries
    )


# Deterministic models that behave like InstructorLLMCall but return exact values for strong testing
class DeterministicMathModel(ArityOneTypedCall[str, MathResponse]):
    """Deterministic math model that returns exact predetermined values."""

    def __init__(self, answer: float, unit: Optional[str], method: str):
        self.answer = answer
        self.unit = unit
        self.method = method

    async def call(self, x: str) -> MathResponse:
        return MathResponse(
            answer=self.answer,
            unit=self.unit,
            method=self.method,
            reasoning=f"Deterministic math model calculated {self.answer} {self.unit or ''} using {self.method} method for input: {x}",
        )


class DeterministicClassificationModel(ArityOneTypedCall[str, ClassificationResponse]):
    """Deterministic classification model that returns exact predetermined values."""

    category: Literal[
        "Technology",
        "Programming",
        "Software",
        "Computing",
        "Science",
        "Mathematics",
        "Engineering",
        "Business",
        "Healthcare",
        "Education",
        "Other",
    ]
    confidence: float
    subcategories: List[str]

    def __init__(
        self,
        category: Literal[
            "Technology",
            "Programming",
            "Software",
            "Computing",
            "Science",
            "Mathematics",
            "Engineering",
            "Business",
            "Healthcare",
            "Education",
            "Other",
        ],
        confidence: float,
        subcategories: List[str],
    ):
        self.category = category
        self.confidence = confidence
        self.subcategories = subcategories

    async def call(self, x: str) -> ClassificationResponse:
        return ClassificationResponse(
            category=self.category,
            confidence=self.confidence,
            subcategories=self.subcategories,
            reasoning=f"Deterministic classification model classified '{x}' as {self.category} with {self.confidence} confidence and subcategories {self.subcategories}",
        )


class DeterministicAnalysisModel(ArityOneTypedCall[str, AnalysisResponse]):
    """Deterministic analysis model that returns exact predetermined values."""

    sentiment: Literal["positive", "negative", "neutral", "optimistic", "pessimistic", "mixed"]
    key_points: List[str]
    summary: str

    def __init__(
        self,
        sentiment: Literal["positive", "negative", "neutral", "optimistic", "pessimistic", "mixed"],
        key_points: List[str],
        summary: str,
    ):
        self.sentiment = sentiment
        self.key_points = key_points
        self.summary = summary

    async def call(self, x: str) -> AnalysisResponse:
        return AnalysisResponse(
            sentiment=self.sentiment,
            key_points=self.key_points,
            summary=self.summary,
            reasoning=f"Deterministic analysis model analyzed '{x}' and found {self.sentiment} sentiment with key points {self.key_points} and summary: {self.summary}",
        )


@pytest.mark.integration
class TestRealLLMConsensus:
    """Test consensus with real LLM calls using InstructorLLMCall."""

    @pytest.fixture
    def math_models(self) -> List[Any]:
        """Create deterministic math models that guarantee consensus."""
        return [
            ConsensusCore.model(
                id="model1",
                executor=DeterministicMathModel(answer=150.0, unit="miles", method="distance_formula"),
                perspective="Calculate distance using standard formula",
            ),
            ConsensusCore.model(
                id="model2",
                executor=DeterministicMathModel(answer=150.0, unit="miles", method="distance_formula"),
                perspective="Apply distance calculation method",
            ),
            ConsensusCore.model(
                id="model3",
                executor=DeterministicMathModel(answer=150.0, unit="miles", method="distance_formula"),
                perspective="Use mathematical approach for distance",
            ),
        ]

    @pytest.fixture
    def classification_models(self) -> List[Any]:
        """Create deterministic classification models that guarantee consensus."""
        return [
            ConsensusCore.model(
                id="model1",
                executor=DeterministicClassificationModel(
                    category="Technology",
                    confidence=0.95,
                    subcategories=["AI", "Machine Learning"],
                ),
                perspective="Classify with technical focus",
            ),
            ConsensusCore.model(
                id="model2",
                executor=DeterministicClassificationModel(
                    category="Technology",
                    confidence=0.95,
                    subcategories=["AI", "Machine Learning"],
                ),
                perspective="Classify with systematic approach",
            ),
            ConsensusCore.model(
                id="model3",
                executor=DeterministicClassificationModel(
                    category="Technology",
                    confidence=0.95,
                    subcategories=["AI", "Machine Learning"],
                ),
                perspective="Classify with analytical method",
            ),
        ]

    @pytest.fixture
    def analysis_models(self) -> List[Any]:
        """Create deterministic analysis models that guarantee consensus."""
        return [
            ConsensusCore.model(
                id="model1",
                executor=DeterministicAnalysisModel(
                    sentiment="positive",
                    key_points=[
                        "AI advancement",
                        "healthcare improvement",
                        "early detection",
                    ],
                    summary="AI breakthrough promises healthcare transformation",
                ),
                perspective="Analyze with objective focus",
            ),
            ConsensusCore.model(
                id="model2",
                executor=DeterministicAnalysisModel(
                    sentiment="positive",
                    key_points=[
                        "AI advancement",
                        "healthcare improvement",
                        "early detection",
                    ],
                    summary="AI breakthrough promises healthcare transformation",
                ),
                perspective="Analyze with balanced view",
            ),
            ConsensusCore.model(
                id="model3",
                executor=DeterministicAnalysisModel(
                    sentiment="positive",
                    key_points=[
                        "AI advancement",
                        "healthcare improvement",
                        "early detection",
                    ],
                    summary="AI breakthrough promises healthcare transformation",
                ),
                perspective="Analyze with comprehensive approach",
            ),
        ]

    @pytest.mark.anyio
    async def test_math_consensus_with_real_llms(self, math_models: List[Any]) -> None:
        """Test mathematical problem solving consensus with deterministic models."""
        judge = DeterministicMathModel(answer=150.0, unit="miles", method="distance_formula")

        consensus = ConsensusCore.consensus(
            models=math_models,
            judge=judge,  # Pass executor directly, not ModelConfiguration
            settings=ConsensusSettings(
                max_rounds=1,
            ),
        )

        result = await consensus.call("If a train travels at 60 mph for 2.5 hours, how far does it travel?")

        # STRONG ASSERTIONS - EXACT VALUES EXPECTED
        assert result.consensus_achieved is True  # Consensus should be achieved
        assert result.final_response.answer == 150.0  # Exact value
        assert result.final_response.unit == "miles"  # Exact string
        assert result.final_response.method == "distance_formula"  # Exact method
        assert result.total_rounds == 1  # Achieved in one round
        assert result.metrics.total_model_calls == 3  # All models called once
        assert len(result.participating_models) == 3  # All models participated

    @pytest.mark.anyio
    async def test_classification_consensus_with_real_llms(self, classification_models: List[Any]) -> None:
        """Test classification consensus with deterministic models."""
        judge = DeterministicClassificationModel(
            category="Technology",
            confidence=0.95,
            subcategories=["AI", "Machine Learning"],
        )

        consensus = ConsensusCore.consensus(
            models=classification_models,
            judge=judge,  # Pass executor directly, not ModelConfiguration
            settings=ConsensusSettings(
                max_rounds=1,
            ),
        )

        result = await consensus.call(
            "Neural networks have revolutionized computer vision tasks through deep learning algorithms."
        )

        # STRONG ASSERTIONS - EXACT VALUES EXPECTED
        assert result.consensus_achieved is True  # Consensus should be achieved
        assert result.final_response.category == "Technology"  # Exact category
        assert result.final_response.confidence == 0.95  # Exact confidence
        assert result.final_response.subcategories == [
            "AI",
            "Machine Learning",
        ]  # Exact list
        assert result.total_rounds == 1  # Achieved in one round
        assert result.metrics.total_model_calls == 3  # All models called once
        assert len(result.participating_models) == 3  # All models participated

    @pytest.mark.anyio
    async def test_analysis_consensus_with_real_llms(self, analysis_models: List[Any]) -> None:
        """Test text analysis consensus with deterministic models."""
        judge = DeterministicAnalysisModel(
            sentiment="positive",
            key_points=["AI advancement", "healthcare improvement", "early detection"],
            summary="AI breakthrough promises healthcare transformation",
        )

        consensus = ConsensusCore.consensus(
            models=analysis_models,
            judge=judge,  # Pass executor directly, not ModelConfiguration
            settings=ConsensusSettings(
                max_rounds=1,
            ),
        )

        result = await consensus.call(
            "The new AI breakthrough promises to transform healthcare by enabling earlier disease detection and personalized treatment plans."
        )

        # STRONG ASSERTIONS - EXACT VALUES EXPECTED
        assert result.consensus_achieved is True  # Consensus should be achieved
        assert result.final_response.sentiment == "positive"  # Exact sentiment
        assert result.final_response.key_points == [
            "AI advancement",
            "healthcare improvement",
            "early detection",
        ]  # Exact list
        assert result.final_response.summary == "AI breakthrough promises healthcare transformation"  # Exact summary
        assert result.total_rounds == 1  # Achieved in one round
        assert result.metrics.total_model_calls == 3  # All models called once
        assert len(result.participating_models) == 3  # All models participated

    @pytest.mark.anyio
    async def test_disagreement_with_real_llms(self) -> None:
        """Test how consensus handles disagreements between real LLM responses."""

        # Create models with different perspectives to encourage disagreement
        models = [
            ConsensusCore.model(
                id="conservative",
                executor=InstructorLLMCall(
                    response_model=MathResponse,
                    model="gpt-4o",
                    temperature=0.1,
                ),
                perspective="Provide the most conservative estimate using minimal assumptions",
            ),
            ConsensusCore.model(
                id="moderate",
                executor=InstructorLLMCall(
                    response_model=MathResponse,
                    model="gpt-4o",
                    temperature=0.5,
                ),
                perspective="Provide a reasonable estimate based on typical scenarios",
            ),
            ConsensusCore.model(
                id="aggressive",
                executor=InstructorLLMCall(
                    response_model=MathResponse,
                    model="gpt-4o",
                    temperature=0.9,
                ),
                perspective="Provide an upper-bound estimate considering maximum possible values",
            ),
        ]

        judge = InstructorLLMCall(
            response_model=MathResponse,
            model="gpt-4o",
            temperature=0.1,
        )

        consensus = ConsensusCore.consensus(
            models=models,
            judge=judge,  # Pass executor directly, not ModelConfiguration
            settings=ConsensusSettings(
                max_rounds=3,
            ),
        )

        # Ask an estimation question that can have varied answers
        result = await consensus.call("Estimate the number of windows in a typical 50-story office building.")

        # Should get some result even with disagreement
        assert result.final_response is not None
        assert isinstance(result.final_response.answer, float)
        assert result.final_response.answer > 0

        # Check that consensus process ran
        assert result.total_rounds >= 1
        assert len(result.participating_models) == 3

    @pytest.mark.anyio
    async def test_consensus_convergence_with_real_llms(self) -> None:
        """Test that models can converge through iterative refinement."""

        models = [
            ConsensusCore.model(
                id="model1",
                executor=InstructorLLMCall(
                    response_model=ClassificationResponse,
                    model="gpt-4o",
                    temperature=0.3,
                ),
                perspective="Focus on technical accuracy in classification",
            ),
            ConsensusCore.model(
                id="model2",
                executor=InstructorLLMCall(
                    response_model=ClassificationResponse,
                    model="gpt-4o",
                    temperature=0.3,
                ),
                perspective="Focus on practical applications in classification",
            ),
            ConsensusCore.model(
                id="model3",
                executor=InstructorLLMCall(
                    response_model=ClassificationResponse,
                    model="gpt-4o",
                    temperature=0.3,
                ),
                perspective="Focus on theoretical foundations in classification",
            ),
        ]

        judge = InstructorLLMCall(
            response_model=ClassificationResponse,
            model="gpt-4o",
            temperature=0.1,
        )

        consensus = ConsensusCore.consensus(
            models=models,
            judge=judge,  # Pass executor directly, not ModelConfiguration
            settings=ConsensusSettings(
                max_rounds=5,  # Allow more rounds for convergence
            ),
        )

        result = await consensus.call("Quantum computing represents a paradigm shift in computational capabilities.")

        assert result.final_response is not None
        assert result.final_response.category is not None

        # Check metrics
        assert result.metrics.total_model_calls >= 3  # At least initial round
        assert result.convergence_score >= 0.0  # Should have some convergence metric

    @pytest.mark.anyio
    async def test_error_recovery_with_real_llms(self) -> None:
        """Test that consensus handles LLM errors gracefully."""

        # Mix of reliable and potentially failing models
        models = [
            ConsensusCore.model(
                id="reliable1",
                executor=InstructorLLMCall(
                    response_model=AnalysisResponse,
                    model="gpt-4o",
                    temperature=0.3,
                ),
                perspective="Provide thorough analysis",
            ),
            ConsensusCore.model(
                id="reliable2",
                executor=InstructorLLMCall(
                    response_model=AnalysisResponse,
                    model="gpt-4o",
                    temperature=0.3,
                ),
                perspective="Provide comprehensive analysis",
            ),
        ]

        judge = InstructorLLMCall(
            response_model=AnalysisResponse,
            model="gpt-4o",
            temperature=0.1,
        )

        consensus = ConsensusCore.consensus(
            models=models,
            judge=judge,  # Pass executor directly, not ModelConfiguration
            settings=ConsensusSettings(
                max_rounds=2,
            ),
        )

        result = await consensus.call("Analyze the impact of artificial intelligence on modern society.")

        # Should get a result even if some models might have issues
        assert result.final_response is not None
        assert result.final_response.sentiment is not None
        assert isinstance(result.final_response.key_points, list)

    @pytest.mark.anyio
    async def test_consensus_core_integration_with_real_llms(self) -> None:
        """Test ConsensusCore facade with real LLM calls for deterministic math problems."""
        from blockether_catalyst.consensus.ConsensusCore import ConsensusCore

        # Create ConsensusCore instance
        core = ConsensusCore()

        # Create models using real LLMs with deterministic math problem
        models = [
            core.model(
                id=f"math_model_{i}",
                executor=InstructorLLMCall(
                    response_model=MathResponse,
                    model="gpt-4o",
                    temperature=0.1,  # Low temperature for consistency
                ),
                perspective=f"As a mathematician {i}, solve step by step",
                weight_multiplier=1.0,
            )
            for i in range(3)
        ]

        # Create judge with real LLM
        judge_executor = InstructorLLMCall(
            response_model=MathResponse,
            model="gpt-4o",
            temperature=0.1,
        )
        judge_model = core.model(
            id="judge",
            executor=judge_executor,
            perspective="As a judge, determine the most accurate mathematical solution",
        )

        # Create consensus with real LLMs
        consensus = ConsensusCore.consensus(
            models=models,
            judge=judge_model.executor,  # Pass executor directly, not ModelConfiguration
            settings=ConsensusSettings(
                max_rounds=3,
            ),
        )

        # Test with simple deterministic math problem
        result = await consensus.call("What is 15 + 27?")

        # Strong assertions for deterministic math
        assert result.final_response is not None
        assert result.final_response.answer == 42.0
        assert "addition" in result.final_response.method.lower() or "sum" in result.final_response.method.lower()
        assert result.total_rounds >= 1
        assert len(result.participating_models) == 3

    @pytest.mark.anyio
    async def test_consensus_core_classification_with_real_llms(self) -> None:
        """Test ConsensusCore facade with real LLM calls for deterministic classification."""

        # Create models for classification task
        models = [
            ConsensusCore.model(
                id=f"classifier_{i}",
                executor=InstructorLLMCall(
                    response_model=ClassificationResponse,
                    model="gpt-4o",
                    temperature=0.1,
                ),
                perspective=f"As a classifier {i}, categorize precisely",
            )
            for i in range(3)
        ]

        judge_model = ConsensusCore.model(
            id="classification_judge",
            executor=InstructorLLMCall(
                response_model=ClassificationResponse,
                model="gpt-4o",
                temperature=0.1,
            ),
            perspective="As a judge, determine the best classification",
        )

        consensus = ConsensusCore.consensus(
            models=models,
            judge=judge_model.executor,  # Pass executor directly, not ModelConfiguration
            settings=ConsensusSettings(
                max_rounds=2,
            ),
        )

        # Test with clear technology classification
        result = await consensus.call(
            "Python is a programming language used for web development, data science, and artificial intelligence applications."
        )

        # Strong assertions for clear classification
        assert result.final_response is not None
        assert result.final_response.category in [
            "Technology",
            "Programming",
            "Software",
            "Computing",
        ]
        assert result.final_response.confidence >= 0.8
        assert len(result.final_response.subcategories) > 0
        assert result.total_rounds >= 1

    @pytest.mark.anyio
    async def test_consensus_core_analysis_with_real_llms(self) -> None:
        """Test ConsensusCore facade with real LLM calls for deterministic text analysis."""
        from blockether_catalyst.consensus.ConsensusCore import ConsensusCore

        core = ConsensusCore()

        # Create models for analysis
        models = [
            core.model(
                id=f"analyzer_{i}",
                executor=InstructorLLMCall(
                    response_model=AnalysisResponse,
                    model="gpt-4o",
                    temperature=0.1,
                ),
                perspective=f"As an analyzer {i}, provide thorough analysis",
            )
            for i in range(3)
        ]

        judge_model = core.model(
            id="analysis_judge",
            executor=InstructorLLMCall(
                response_model=AnalysisResponse,
                model="gpt-4o",
                temperature=0.1,
            ),
            perspective="As a judge, synthesize the most accurate analysis",
        )

        consensus = ConsensusCore.consensus(
            models=models,
            judge=judge_model.executor,  # Pass executor directly, not ModelConfiguration
            settings=ConsensusSettings(
                max_rounds=2,
            ),
        )

        # Test with clearly positive text
        result = await consensus.call(
            "The new breakthrough in renewable energy technology promises to revolutionize clean power generation and reduce carbon emissions significantly."
        )

        # Strong assertions for clear positive sentiment
        assert result.final_response is not None
        assert result.final_response.sentiment in ["positive", "optimistic"]
        assert len(result.final_response.key_points) >= 2
        assert "renewable" in result.final_response.summary.lower() or "energy" in result.final_response.summary.lower()
        assert result.total_rounds >= 1
