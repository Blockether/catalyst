"""
Core module for consensus functionality.

This module provides the main entry point for consensus operations,
including -inspired multi-model consensus mechanisms.
"""

from typing import Generic, List, Optional, TypeVar

from ..utils.TypedCalls import ArityOneTypedCall
from .Consensus import Consensus
from .ConsensusTypes import (
    ConsensusSettings,
    ModelConfiguration,
    TypedCallBaseForConsensus,
)

# Type variable bound to TypedCallBaseForConsensus
T = TypeVar("T", bound=TypedCallBaseForConsensus)


class ConsensusCore(Generic[T]):
    @staticmethod
    def consensus(
        models: List[ModelConfiguration[T]],
        judge: ArityOneTypedCall[str, T],
        settings: Optional[ConsensusSettings] = None,
    ) -> Consensus[T]:
        """Create a consensus instance with majority voting.

        Args:
            models: Model configurations for consensus (all must return type T)
            judge: REQUIRED judge TypedCall[str, T] for tie-breaking. Will be used
                  to resolve ties when models have equal votes after all rounds.
                  Must return the same type T as the models' executors
            settings: Consensus settings (optional)

        Returns:
            Consensus[T] instance configured with majority voting and judge-based tie-breaking
        """
        return Consensus[T](
            models=models,
            judge=judge,
            settings=settings,
        )

    @staticmethod
    def model(
        id: str,
        executor: ArityOneTypedCall[str, T],
        perspective: str,
        weight_multiplier: float = 1.0,
    ) -> ModelConfiguration[T]:
        """Create a model configuration - simplified without capabilities.

        Args:
            id: Unique identifier for the model
            executor: The typed call implementation for this model returning type T
            perspective: REQUIRED - The perspective/role the model should take
                        (e.g., 'As a mathematician', 'From a security perspective')
            weight_multiplier: Weight multiplier for this model's vote (default: 1.0)

        Returns:
            ModelConfiguration[T] properly typed
        """
        return ModelConfiguration[T](
            id=id,
            executor=executor,
            perspective=perspective,
            weight_multiplier=weight_multiplier,
        )
