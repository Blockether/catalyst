"""
Tests for ConsensusCore - the main facade for consensus operations.

This test suite verifies the ConsensusCore class which provides the primary
interface for creating and managing consensus instances.
"""

from typing import Any, Type
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel, Field

from com_blockether_catalyst.consensus import (
    Consensus,
    ConsensusCore,
    ConsensusSettings,
    ModelConfiguration,
)
from com_blockether_catalyst.consensus.internal.ConsensusTypes import (
    TypedCallBaseForConsensus,
)
from com_blockether_catalyst.utils.TypedCalls import ArityOneTypedCall


class ConsensusTestResponse(TypedCallBaseForConsensus):
    """Test response model that extends TypedCallBaseForConsensus."""

    answer: str = Field(description="The answer")
    confidence: float = Field(default=0.9, description="Confidence level")


class MockTypedCall(ArityOneTypedCall[str, ConsensusTestResponse]):
    """Mock typed call for testing."""

    def __init__(self) -> None:
        """Initialize mock with test response."""
        self._response: ConsensusTestResponse = ConsensusTestResponse(
            answer="test",
            confidence=0.9,
            reasoning="This is a test response with mock reasoning for consensus validation.",
        )

    async def call(self, x: str) -> ConsensusTestResponse:
        """Return mock response."""
        return self._response


class TestConsensusCore:
    """Test suite for ConsensusCore facade."""

    @pytest.fixture
    def core(self) -> Any:
        """Create ConsensusCore instance."""
        return ConsensusCore()

    @pytest.fixture
    def mock_executor(self) -> Any:
        """Create mock typed call."""
        return MockTypedCall()

    @pytest.fixture
    def mock_judge(self) -> Any:
        """Create mock judge typed call for tie-breaking."""
        return MockTypedCall()

    def test_initialization(self, core: Any) -> None:
        """Test ConsensusCore initialization."""
        assert isinstance(core, ConsensusCore)

    def test_consensus_default(self, core: Any, mock_executor: Any, mock_judge: Any) -> None:
        """Test creating Consensus with defaults."""
        models = [
            core.configuration(
                id="test-model",
                executor=mock_executor,
                perspective="As a test model",
            )
        ]
        consensus = core.consensus(models=models, judge=mock_judge)

        assert isinstance(consensus, Consensus)
        assert consensus._settings is not None

    def test_consensus_with_settings(self, core: Any, mock_executor: Any, mock_judge: Any) -> None:
        """Test creating Consensus with custom settings."""
        models = [
            core.configuration(
                id="test-model",
                executor=mock_executor,
                perspective="As a test model",
            )
        ]
        settings = ConsensusSettings(
            max_concurrent_calls=5,
        )

        consensus = core.consensus(models=models, judge=mock_judge, settings=settings)

        assert isinstance(consensus, Consensus)
        assert consensus._settings.max_concurrent_calls == 5

    def test_configuration(self, core: Any, mock_executor: Any) -> None:
        """Test creating model configuration."""
        config = core.configuration(
            id="test-model",
            executor=mock_executor,
            perspective="As a test model",
            weight_multiplier=0.8,
        )

        assert isinstance(config, ModelConfiguration)
        assert config.id == "test-model"
        assert config.executor == mock_executor
        assert config.weight_multiplier == 0.8

    def test_configuration_with_weight(self, core: Any, mock_executor: Any) -> None:
        """Test creating model configuration with custom weight."""
        config = core.configuration(
            id="expert-model",
            executor=mock_executor,
            perspective="As an expert",
            weight_multiplier=2.0,
        )

        assert config.id == "expert-model"
        assert config.weight_multiplier == 2.0

    def test_create_consensus_request_removed(self, core: Any, mock_executor: Any, mock_judge: Any) -> None:
        """Test that consensus request creation was simplified."""
        models = [
            core.configuration(id=f"model-{i}", executor=mock_executor, perspective=f"As model {i}") for i in range(3)
        ]

        # ConsensusRequest was removed - consensus is now directly created
        from com_blockether_catalyst.consensus.internal.ConsensusTypes import (
            ConsensusSettings,
        )

        settings = ConsensusSettings(max_rounds=3)
        consensus = core.consensus(models=models, judge=mock_judge, settings=settings)

        assert isinstance(consensus, Consensus)
        assert len(consensus._models) == 3

    @pytest.mark.asyncio
    async def test_full_consensus_workflow(self, core: Any, mock_executor: Any, mock_judge: Any) -> None:
        """Test complete consensus workflow using ConsensusCore."""
        # Create model configurations
        models = [
            core.configuration(
                id=f"model-{i}",
                executor=mock_executor,
                perspective=f"As model {i}",
                weight_multiplier=1.0 - (i * 0.1),
            )
            for i in range(3)
        ]

        # Create and execute consensus directly
        from com_blockether_catalyst.consensus.internal.ConsensusTypes import (
            ConsensusSettings,
        )

        settings = ConsensusSettings(max_rounds=3)
        consensus = core.consensus(models=models, judge=mock_judge, settings=settings)
        result = await consensus.call("What is the meaning of life?")

        # Verify result
        assert result.total_rounds >= 1
        assert result.total_rounds <= 3
        assert len(result.participating_models) == 3
        assert isinstance(result.final_response, ConsensusTestResponse)

    def test_static_methods_are_static(self) -> None:
        """Test that static methods work without instance."""
        # Create mock typed call
        mock_call = MockTypedCall()

        # Test static method without instance
        config = ConsensusCore.configuration(id="static-test", executor=mock_call, perspective="As a static test")
        assert config.id == "static-test"

        # Test hashgraph consensus creation
        mock_judge = MockTypedCall()
        consensus = ConsensusCore.consensus(models=[config], judge=mock_judge)
        assert isinstance(consensus, Consensus)

    def test_type_annotations(self, core: Any) -> None:
        """Test that all methods have proper type annotations."""
        # This test verifies that the methods are properly typed
        # by checking their signatures
        import inspect

        # Check consensus
        sig = inspect.signature(core.consensus)
        assert sig.return_annotation == Consensus

        # Check configuration
        sig = inspect.signature(core.configuration)
        assert sig.return_annotation == ModelConfiguration

        # create_consensus_request was removed - check that it doesn't exist
        assert not hasattr(core, "create_consensus_request")
