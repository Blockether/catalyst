"""
Test for VotingField warnings when descriptions are used on model references.

This tests the OpenAI JSON Schema restriction where $ref fields cannot have descriptions.
"""

import logging
from typing import List

from pydantic import BaseModel

from blockether_catalyst.consensus.ConsensusTypes import BaseModelWithReasoning
from blockether_catalyst.consensus.VotingComparison import (
    ComparisonStrategy,
    VotingField,
)


class TestVotingFieldModelReferenceWarnings:
    """Test that VotingField properly warns about descriptions on model references."""

    def test_primitive_field_with_description_no_warning(self, caplog):
        """Primitive types should NOT generate warnings when they have descriptions."""

        class TestModel(BaseModelWithReasoning):
            # Primitive types CAN have descriptions - no warning expected
            name: str = VotingField(
                description="This is allowed for primitive types",
                comparison=ComparisonStrategy.EXACT,
            )
            score: float = VotingField(
                description="Score between 0 and 1",
                comparison=ComparisonStrategy.RANGE,
                tolerance=0.1,
            )

        # No warning should be logged for primitive types
        assert not any("OpenAI JSON Schema" in record.message for record in caplog.records)

    def test_model_reference_with_description_warns(self, caplog):
        """Model references should generate warnings when they have descriptions."""

        class InnerModel(BaseModel):
            value: str

        with caplog.at_level(logging.WARNING):

            class TestModel(BaseModelWithReasoning):
                # This SHOULD generate a warning
                inner_field: InnerModel = VotingField(
                    description="This causes OpenAI API errors!",
                    comparison=ComparisonStrategy.DERIVED,
                )

        # Check that warning was logged
        assert any(
            "VotingField with description used for InnerModel field 'inner_field'" in record.message
            for record in caplog.records
        )
        assert any(
            "OpenAI JSON Schema does not allow descriptions on $ref fields" in record.message
            for record in caplog.records
        )

    def test_list_of_models_with_description_warns(self, caplog):
        """List[Model] should generate warnings when they have descriptions."""

        class ItemModel(BaseModel):
            name: str

        with caplog.at_level(logging.WARNING):

            class TestModel(BaseModelWithReasoning):
                # This SHOULD generate a warning
                items: List[ItemModel] = VotingField(
                    description="List of items - this causes errors!",
                    comparison=ComparisonStrategy.IGNORE,
                )

        # Check that warning was logged
        assert any(
            "VotingField with description used for List[ItemModel]" in record.message for record in caplog.records
        )
        assert any(
            "OpenAI JSON Schema does not allow descriptions on $ref fields" in record.message
            for record in caplog.records
        )

    def test_list_of_primitives_with_description_no_warning(self, caplog):
        """List[primitive] should NOT generate warnings."""

        class TestModel(BaseModelWithReasoning):
            # List of primitives CAN have descriptions - no warning expected
            tags: List[str] = VotingField(
                description="List of tags",
                comparison=ComparisonStrategy.IGNORE,
            )
            scores: List[float] = VotingField(
                description="List of scores",
                comparison=ComparisonStrategy.IGNORE,
            )

        # No warning should be logged for lists of primitives
        assert not any("OpenAI JSON Schema" in record.message for record in caplog.records)

    def test_evaluation_factor_example(self, caplog):
        """Test the actual EvaluationFactor case that caused the original issue."""

        class EvaluationFactor(BaseModel):
            """Individual evaluation factor."""

            score: float = VotingField(
                ge=0.0,
                le=1.0,
                description="Score between 0 and 1",  # This is OK - primitive type
                comparison=ComparisonStrategy.RANGE,
            )
            reasoning: str = VotingField(
                min_length=100,
                description="Explanation for the score (minimum 100 characters)",  # This is OK - primitive type
                comparison=ComparisonStrategy.SEMANTIC,
            )

        with caplog.at_level(logging.WARNING):

            class AnswerOutput(BaseModelWithReasoning):
                # This SHOULD generate a warning
                contradiction_presence: EvaluationFactor = VotingField(
                    description="Evaluate contradiction presence",  # This causes the error!
                    comparison=ComparisonStrategy.DERIVED,
                    threshold=0.8,
                )

        # Check that warning was logged
        assert any(
            "VotingField with description used for EvaluationFactor field 'contradiction_presence'" in record.message
            for record in caplog.records
        )
        assert any(
            "Move the description to the EvaluationFactor model's fields or docstring" in record.message
            for record in caplog.records
        )

    def test_correct_pattern_no_warning(self, caplog):
        """Test the correct pattern with no descriptions on model references."""

        class EvaluationFactor(BaseModel):
            """
            Individual evaluation factor.

            This docstring documents what the factor represents,
            rather than putting descriptions on references to this model.
            """

            score: float = VotingField(
                ge=0.0,
                le=1.0,
                description="Score between 0 and 1",  # OK on primitive
                comparison=ComparisonStrategy.RANGE,
            )
            reasoning: str = VotingField(
                min_length=100,
                description="Explanation for the score (minimum 100 characters)",  # OK on primitive
                comparison=ComparisonStrategy.SEMANTIC,
            )

        class AnswerOutput(BaseModelWithReasoning):
            # Correct pattern - NO description on model reference
            contradiction_presence: EvaluationFactor = VotingField(
                comparison=ComparisonStrategy.DERIVED,
                threshold=0.8,
                # Note: NO description parameter here!
            )

            # Descriptions are OK on primitive fields
            confidence: float = VotingField(
                description="Overall confidence score",
                comparison=ComparisonStrategy.RANGE,
                tolerance=0.1,
            )

        # No warnings should be generated for the correct pattern
        assert not any("OpenAI JSON Schema" in record.message for record in caplog.records)
