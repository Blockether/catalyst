"""
Core encoder module for text embeddings using model2vec.

This module provides a singleton encoder that's initialized once and reused
across the application for generating text embeddings.
"""

import logging
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from model2vec import StaticModel

logger = logging.getLogger(__name__)


class EncoderCore:
    """
    Singleton encoder for generating text embeddings.

    This class provides static methods for encoding text into embeddings
    using the model2vec StaticModel. The model is loaded once on first use
    and cached for subsequent calls.
    """

    # Class-level attributes for singleton pattern
    _model: Optional[StaticModel] = None
    _initialized: bool = False

    @classmethod
    def _initialize(cls) -> None:
        """Initialize the model if not already loaded."""
        if not cls._initialized:
            try:
                # Get the path to the static model - same approach as templates
                module_path = Path(__file__).parent.parent
                local_model_path = module_path / "static_models" / "model2vec" / "potion-8M-base"

                logger.info(f"Loading encoder model from: {local_model_path}")
                cls._model = StaticModel.from_pretrained(str(local_model_path))
                cls._initialized = True
                logger.info("Encoder model loaded successfully")
            except Exception as e:
                logger.exception("Failed to load encoder model")
                raise RuntimeError(f"Could not initialize encoder model: {e}")

    @classmethod
    def encode(cls, text: Union[str, List[str]]) -> np.ndarray:
        """
        Encode text into embeddings.

        Args:
            text: Single text string or list of texts to encode

        Returns:
            Numpy array of embeddings. Shape (1, embedding_dim) for single text,
            or (n_texts, embedding_dim) for multiple texts.

        Raises:
            RuntimeError: If model initialization fails
        """
        # Ensure model is initialized
        cls._initialize()

        if not cls._model:
            raise RuntimeError("Encoder model is not initialized")

        if isinstance(text, str):
            return cls._model.encode([text])

        return cls._model.encode(text)

    @classmethod
    def encode_single(cls, text: str) -> np.ndarray:
        """
        Encode a single text into an embedding vector.

        Args:
            text: Single text string to encode

        Returns:
            1D numpy array of the embedding vector

        Raises:
            RuntimeError: If model initialization fails
            ValueError: If text is not a string
        """
        if not isinstance(text, str):
            raise ValueError(f"Expected string, got {type(text)}")

        # Use encode method and return first (only) embedding
        embeddings = cls.encode(text)
        return embeddings[0]  # type: ignore[no-any-return]

    @classmethod
    def cosine_similarity(
        cls,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
    ) -> float:
        """
        Calculate cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Cosine similarity score between -1 and 1

        Raises:
            ValueError: If embeddings have different shapes
        """
        if embedding1.shape != embedding2.shape:
            raise ValueError(f"Embedding shapes must match: {embedding1.shape} != {embedding2.shape}")

        # Normalize embeddings
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        # Handle zero vectors
        if norm1 == 0 or norm2 == 0:
            return 0.0

        # Calculate cosine similarity
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        return float(similarity)

    @classmethod
    def reset(cls) -> None:
        """
        Reset the encoder by clearing the cached model.

        This is primarily useful for testing or when switching models.
        """
        cls._model = None
        cls._initialized = False
        logger.info("Encoder reset - model will be reloaded on next use")
