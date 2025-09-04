"""
Core module for consensus functionality.

This module provides the main entry point for consensus operations,
including -inspired multi-model consensus mechanisms.
"""

from typing import List, Optional

from ..utils.TypedCalls import ArityOneTypedCall
from .internal.Consensus import Consensus
from .internal.ConsensusTypes import (
    ConsensusSettings,
    ModelConfiguration,
)


class ConsensusCore:
    @staticmethod
    def consensus(
        models: List[ModelConfiguration],
        judge: ArityOneTypedCall,
        settings: Optional[ConsensusSettings] = None,
    ) -> Consensus:
        """Create a consensus instance with majority voting.

        Args:
            models: Model configurations for consensus
            judge: REQUIRED judge TypedCall[str, T] for tie-breaking. Will be used
                  to resolve ties when models have equal votes after all rounds.
                  Must return the same type T as the models' executors
            settings: Consensus settings (optional)

        Returns:
            Consensus instance configured with majority voting and judge-based tie-breaking
        """
        return Consensus(
            models=models,
            judge=judge,
            settings=settings,
        )

    @staticmethod
    def configuration(
        id: str,
        executor: ArityOneTypedCall,
        perspective: str,
        weight_multiplier: float = 1.0,
    ) -> ModelConfiguration:
        """Create a model configuration - simplified without capabilities.

        Args:
            id: Unique identifier for the model
            executor: The typed call implementation for this model
            perspective: REQUIRED - The perspective/role the model should take
                        (e.g., 'As a mathematician', 'From a security perspective')
            weight_multiplier: Weight multiplier for this model's vote (default: 1.0)

        Returns:
            ModelConfiguration properly configured
        """
        return ModelConfiguration(
            id=id,
            executor=executor,
            perspective=perspective,
            weight_multiplier=weight_multiplier,
        )
