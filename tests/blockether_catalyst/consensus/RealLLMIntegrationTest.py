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
            ConsensusCore.model(
                id="reliable3",
                executor=InstructorLLMCall(
                    response_model=AnalysisResponse,
                    model="gpt-4o",
                    temperature=0.3,
                ),
                perspective="Provide detailed analysis",
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

    @pytest.mark.anyio
    async def test_consensus_bug_reasoning_field_ignored(self) -> None:
        """Test bug where consensus fails despite all models agreeing on voting fields.

        This reproduces the issue where:
        - All 3 models agree on ALL voting fields
        - Only the 'reasoning' field differs (which should be ignored)
        - Result shows 100% agreement but fallback is used
        - Field consistency shows 83.3% (5/6 fields) incorrectly including reasoning
        """

        # Create a simple response model with reasoning that should be ignored
        class SimpleConsensusResponse(BaseModelWithReasoning):
            """Simple response with few fields to test consensus bug."""

            term: str = VotingField(
                description="The term being analyzed",
                comparison=ComparisonStrategy.EXACT,
            )

            meaning: str = VotingField(
                description="The meaning of the term",
                comparison=ComparisonStrategy.SEMANTIC,
                threshold=0.7,  # Lower threshold to allow more semantic variation
            )

            category: Literal["acronym", "keyword", "phrase"] = VotingField(
                description="Category of the term",
                comparison=ComparisonStrategy.EXACT,
            )

            status: Literal["meaningful", "not_meaningful"] = VotingField(
                description="Whether the term is meaningful",
                comparison=ComparisonStrategy.EXACT,
            )

        # Create deterministic models that return IDENTICAL voting fields but DIFFERENT reasoning
        class DeterministicConsensusModel(ArityOneTypedCall[str, SimpleConsensusResponse]):
            """Model that returns exact same values but different reasoning."""

            def __init__(self, model_id: str):
                self.model_id = model_id

            async def call(self, x: str) -> SimpleConsensusResponse:
                # Models return SLIGHTLY different semantic meanings but should still match
                if self.model_id == "model1":
                    meaning_text = (
                        "Environmental, Social, and Governance factors for evaluating sustainability and ethical impact"
                    )
                elif self.model_id == "model2":
                    meaning_text = "Environmental, Social, Governance criteria used to assess sustainability and corporate responsibility"
                elif self.model_id == "model3":
                    meaning_text = "Environmental, Social, and Governance standards for measuring sustainability and ethical business practices"
                else:
                    meaning_text = "Environmental, Social, and Governance factors for evaluating sustainability"

                return SimpleConsensusResponse(
                    term="ESG",
                    meaning=meaning_text,  # Semantically similar but not identical
                    category="acronym",
                    status="meaningful",
                    # ONLY reasoning differs significantly between models
                    reasoning=f"Model {self.model_id} specific reasoning: This is different for each model but should be ignored for voting. {x}",
                )

        # Create 3 models with same voting values but different reasoning
        models = [
            ConsensusCore.model(
                id="model1",
                executor=DeterministicConsensusModel("model1"),
                perspective="Conservative perspective",
            ),
            ConsensusCore.model(
                id="model2",
                executor=DeterministicConsensusModel("model2"),
                perspective="Balanced perspective",
            ),
            ConsensusCore.model(
                id="model3",
                executor=DeterministicConsensusModel("model3"),
                perspective="Liberal perspective",
            ),
        ]

        # Judge also returns same values
        judge = DeterministicConsensusModel("judge")

        # Create consensus with settings that match the user's scenario
        consensus = ConsensusCore.consensus(
            models=models,
            judge=judge,
            settings=ConsensusSettings(
                max_rounds=3,  # Match the user's max_rounds
                threshold=0.65,  # 65% threshold means 2/3 models = 66.7% should pass
                first_round_threshold=0.65,  # Same as regular threshold
            ),
        )

        result = await consensus.call("What does ESG stand for?")

        # CRITICAL ASSERTIONS - This is where the bug manifests

        # Detailed debugging output
        print("\n=== CONSENSUS RESULT DEBUG ===")
        print(f"Consensus achieved: {result.consensus_achieved}")
        print(f"Total rounds: {result.total_rounds}")
        print(f"Convergence score: {result.convergence_score}")
        print(f"Agreement percentage: {result.metrics.consensus_confidence * 100:.1f}%")
        print(f"Fallback method: {result.metrics.fallback_method}")
        print(f"Dissenting models: {result.dissenting_models}")

        # Print round details
        for i, round_data in enumerate(result.rounds):
            print(f"\n--- Round {i + 1} ---")
            print(f"  Responses: {len(round_data.responses)}")
            print(f"  Consensus achieved: {round_data.consensus_achieved}")
            if round_data.disagreement_analysis:
                print(f"  Disagreement fields: {list(round_data.disagreement_analysis.disagreement_fields.keys())}")
                print(f"  Consensus fields: {round_data.disagreement_analysis.consensus_fields}")

        print("\n=== END DEBUG ===\n")

        # The bug: All models agree on voting fields, but consensus fails
        assert result.consensus_achieved is True, (
            f"BUG: Consensus should be achieved when all models agree on voting fields! "
            f"Got consensus={result.consensus_achieved}, fallback={result.metrics.fallback_method}"
        )

        # All models should have voted the same (no dissenting models)
        assert len(result.dissenting_models) == 0, (
            f"BUG: No models should be dissenting when all voting fields match! "
            f"Got dissenting: {result.dissenting_models}"
        )

        # Convergence score should be 1.0 for perfect agreement
        assert result.convergence_score == 1.0, (
            f"BUG: Convergence should be 1.0 when all models agree! " f"Got: {result.convergence_score}"
        )

        # MUST achieve consensus in round 1 when all models agree
        assert result.total_rounds == 1, (
            f"BUG: Consensus MUST be achieved in round 1 when all models agree on voting fields! "
            f"Got {result.total_rounds} rounds"
        )

        # Verify the response values are correct
        assert result.final_response.term == "ESG"
        assert result.final_response.category == "acronym"
        assert result.final_response.status == "meaningful"

        # No fallback method should be used
        assert result.metrics.fallback_method is None, (
            f"BUG: No fallback should be used when consensus is achieved! "
            f"Got fallback: {result.metrics.fallback_method}"
        )

        # COMPREHENSIVE METRICS VERIFICATION
        # Verify all consensus metrics are internally consistent

        # Duration should be positive
        assert result.metrics.duration_ms > 0, "Duration must be positive"

        # Rounds metrics must match
        assert result.metrics.rounds_to_convergence == 1, "Should converge in 1 round"
        assert result.metrics.rounds_to_convergence == result.total_rounds, "Rounds metrics must match"

        # Model calls should be 3 (one per model, one round)
        assert (
            result.metrics.total_model_calls == 3
        ), f"Should have exactly 3 model calls (1 per model) in round 1, got {result.metrics.total_model_calls}"

        # Convergence metrics must be consistent
        assert result.metrics.convergence_achieved is True, "Convergence should be achieved"
        assert result.metrics.convergence_achieved == result.consensus_achieved, "Convergence flags must match"

        # Dissent rate should be 0 when all agree
        assert result.metrics.dissent_rate == 0.0, "No dissent when all models agree"

        # Consensus confidence should be 1.0 for perfect agreement
        assert (
            result.metrics.consensus_confidence == 1.0
        ), f"Consensus confidence should be 1.0 for perfect agreement, got {result.metrics.consensus_confidence}"

        # Convergence indicator should match convergence score
        assert result.metrics.convergence_indicator == result.convergence_score, "Convergence indicators must match"
        assert result.metrics.convergence_indicator == 1.0, "Perfect convergence indicator expected"

        # Model contributions should all be high (1.0) when all agree
        assert len(result.metrics.model_contributions) == 3, "Should have contributions from all 3 models"
        assert all(
            score == 1.0 for score in result.metrics.model_contributions.values()
        ), f"All models should have 1.0 contribution when agreeing, got {result.metrics.model_contributions}"

        # Refinement metrics (should be 0 for round 1)
        assert result.metrics.total_refinements == 0, "No refinements in first round"
        assert result.metrics.avg_refinements_per_round == 0.0, "No avg refinements in first round"

        # Information flow should be empty for round 1
        assert len(result.metrics.information_flows) == 1, "Should have 1 round of information flow"
        assert result.metrics.information_flows[0] == {}, "First round has no information flow"

        # Participating models check
        assert len(result.participating_models) == 3, "All 3 models should participate"
        assert set(result.participating_models) == {"model1", "model2", "model3"}, "Correct model IDs"

        # Round data verification
        assert len(result.rounds) == 1, "Should have exactly 1 round"
        round1 = result.rounds[0]
        assert round1.round_number == 0, "First round is numbered 0"
        assert len(round1.responses) == 3, "All 3 models responded"
        assert round1.consensus_achieved is True, "Consensus achieved in round 1"
        assert round1.consensus_response is not None, "Consensus response must be set"

        # Verify consensus response matches final response
        assert round1.consensus_response == result.final_response, "Round consensus response must match final response"

        # Disagreement analysis for round 1
        assert round1.disagreement_analysis is not None, "Should have disagreement analysis"
        assert len(round1.disagreement_analysis.disagreement_fields) == 0, "No disagreements"
        assert (
            "reasoning" not in round1.disagreement_analysis.consensus_fields
        ), "Reasoning should NOT be in consensus fields (it's ignored)"
        assert set(round1.disagreement_analysis.consensus_fields) == {
            "term",
            "meaning",
            "category",
            "status",
        }, f"Should have exactly the voting fields in consensus, got {round1.disagreement_analysis.consensus_fields}"

        # Verify reasoning is properly formatted
        assert result.reasoning is not None, "Should have consensus reasoning"
        assert (
            "Unanimous consensus" in result.reasoning or "All 3 models" in result.reasoning
        ), f"Reasoning should mention unanimous agreement, got: {result.reasoning}"

        # Vote groups verification - MUST be present when consensus is checked
        assert hasattr(round1, "vote_groups"), "Vote groups must be set after consensus check"
        assert round1.vote_groups is not None, "Vote groups cannot be None"
        assert len(round1.vote_groups) == 1, "Should have exactly 1 vote group when all agree"
        vote_group = list(round1.vote_groups.values())[0]
        assert len(vote_group) == 3, "All 3 models in same vote group"
