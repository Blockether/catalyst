"""
Consensus module - -inspired consensus mechanism for multi-model reasoning.

This module provides tools for achieving consensus among multiple AI models
using techniques inspired by the  distributed consensus algorithm.
"""

from .ConsensusCore import ConsensusCore
from .internal.Consensus import (
    Consensus,
    ModelMetrics,
)
from .internal.ConsensusTypes import (
    ComparisonStrategy,
    ConsensusQualityMetrics,
    ConsensusResult,
    ConsensusRound,
    ConsensusSettings,
    GossipHistory,
    ModelConfiguration,
    ModelResponse,
    TypedCallBaseForConsensus,
    VotingField,
)

__all__ = [
    # Main Components
    "ConsensusCore",
    "Consensus",
    # Base Classes
    "TypedCallBaseForConsensus",
    # Core Types from Consensus
    "GossipHistory",
    "ModelResponse",
    "ConsensusRound",
    "ModelMetrics",
    "ConsensusResult",
    # Request/Response Types
    # Configuration Types
    "ModelConfiguration",
    "ConsensusSettings",
    # Field Comparison
    "VotingField",
    "ComparisonStrategy",
    # Enums
    # Metrics
    "ConsensusQualityMetrics",
]
