"""Tests for PotionEightEncoder module."""

import numpy as np
import pytest

from blockether_catalyst.encoder import PotionEightEncoder


class TestPotionEightEncoder:
    """Test cases for PotionEightEncoder."""

    def test_encode_single_text(self) -> None:
        """Test encoding a single text string."""
        text = "Hello, this is a test sentence."

        # Test regular encode (returns 2D array)
        embedding_2d = PotionEightEncoder.encode(text)
        assert isinstance(embedding_2d, np.ndarray)
        assert embedding_2d.ndim == 2
        assert embedding_2d.shape[0] == 1

        # Test encode_single (returns 1D array)
        embedding_1d = PotionEightEncoder.encode_single(text)
        assert isinstance(embedding_1d, np.ndarray)
        assert embedding_1d.ndim == 1

        # Both should have the same values
        assert np.array_equal(embedding_2d[0], embedding_1d)

        # Check it has content (not zero vector)
        norm = np.linalg.norm(embedding_1d)
        assert norm > 0  # Has magnitude

    def test_encode_multiple_texts(self) -> None:
        """Test encoding multiple texts."""
        texts = [
            "First sentence for testing.",
            "Second sentence is different.",
            "Third one is also unique.",
        ]
        embeddings = PotionEightEncoder.encode(texts)

        # Check we get a 2D array
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.ndim == 2
        assert embeddings.shape[0] == len(texts)

        # Check all have content (not zero vectors)
        for embedding in embeddings:
            norm = np.linalg.norm(embedding)
            assert norm > 0

    def test_cosine_similarity(self) -> None:
        """Test cosine similarity calculation."""
        # Similar texts should have high similarity
        text1 = "The weather is nice today."
        text2 = "Today the weather is pleasant."
        text3 = "Python is a programming language."

        emb1 = PotionEightEncoder.encode_single(text1)
        emb2 = PotionEightEncoder.encode_single(text2)
        emb3 = PotionEightEncoder.encode_single(text3)

        # Similar texts should have higher similarity
        sim_12 = PotionEightEncoder.cosine_similarity(emb1, emb2)
        sim_13 = PotionEightEncoder.cosine_similarity(emb1, emb3)

        assert sim_12 > sim_13  # Weather texts more similar than weather vs programming
        assert -1.0 <= sim_12 <= 1.0
        assert -1.0 <= sim_13 <= 1.0

    def test_semantic_hash_consistency(self) -> None:
        """Test that semantic hashing is consistent - removed as _semantic_hash no longer exists."""
        # The _semantic_hash method was removed from Consensus
        # This test is kept as a placeholder to document that the functionality was removed
        pass  # Normalization should make them identical

    def test_semantic_hash_similarity(self) -> None:
        """Test that similar texts get similar hashes with low threshold - removed as _semantic_hash no longer exists."""
        # The _semantic_hash method was removed from Consensus
        # This test is kept as a placeholder to document that the functionality was removed
        pass

        # With high threshold (0.95), even similar texts should differ
        # (unless they're nearly identical)
        # This depends on the actual similarity of the embeddings

    def test_empty_text(self) -> None:
        """Test encoding empty text."""
        # The simplified encoder doesn't validate empty text
        # It just passes it to the model
        result = PotionEightEncoder.encode("")
        assert result is not None  # Should return something, even for empty text

    def test_initialization(self) -> None:
        """Test that the encoder initializes properly on first use."""
        # The encoder should auto-initialize on first use
        embedding = PotionEightEncoder.encode("Initialize the model")
        assert embedding is not None
        assert PotionEightEncoder._initialized is True
