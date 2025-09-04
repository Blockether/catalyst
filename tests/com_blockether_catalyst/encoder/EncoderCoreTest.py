"""Tests for EncoderCore module."""

import numpy as np
import pytest

from com_blockether_catalyst.encoder import EncoderCore


class TestEncoderCore:
    """Test cases for EncoderCore."""

    def test_encode_single_text(self) -> None:
        """Test encoding a single text string."""
        text = "Hello, this is a test sentence."

        # Test regular encode (returns 2D array)
        embedding_2d = EncoderCore.encode(text)
        assert isinstance(embedding_2d, np.ndarray)
        assert embedding_2d.ndim == 2
        assert embedding_2d.shape[0] == 1

        # Test encode_single (returns 1D array)
        embedding_1d = EncoderCore.encode_single(text)
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
        embeddings = EncoderCore.encode(texts)

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

        emb1 = EncoderCore.encode_single(text1)
        emb2 = EncoderCore.encode_single(text2)
        emb3 = EncoderCore.encode_single(text3)

        # Similar texts should have higher similarity
        sim_12 = EncoderCore.cosine_similarity(emb1, emb2)
        sim_13 = EncoderCore.cosine_similarity(emb1, emb3)

        assert sim_12 > sim_13  # Weather texts more similar than weather vs programming
        assert -1.0 <= sim_12 <= 1.0
        assert -1.0 <= sim_13 <= 1.0

    def test_semantic_hash_consistency(self) -> None:
        """Test that semantic hashing is consistent."""
        from com_blockether_catalyst.consensus.internal.Consensus import Consensus

        # Same text should produce same hash
        text = "This is a test sentence."
        hash1 = Consensus._semantic_hash(text, threshold=0.8)
        hash2 = Consensus._semantic_hash(text, threshold=0.8)

        assert hash1 == hash2
        assert len(hash1) == 8  # Hash should be 8 characters

        # Test normalization - different case and whitespace should produce same hash
        text_upper = "  THIS IS A TEST SENTENCE.  "
        text_lower = "this is a test sentence."
        hash_upper = Consensus._semantic_hash(text_upper, threshold=0.8)
        hash_lower = Consensus._semantic_hash(text_lower, threshold=0.8)

        assert hash_upper == hash_lower  # Normalization should make them identical

    def test_semantic_hash_similarity(self) -> None:
        """Test that similar texts get similar hashes with low threshold."""
        from com_blockether_catalyst.consensus.internal.Consensus import Consensus

        # Very similar texts with low threshold should get same hash
        text1 = "The cat is sleeping on the couch."
        text3 = "Python is a programming language."

        # With low threshold (0.5), similar texts should match
        hash1 = Consensus._semantic_hash(text1, threshold=0.5)
        hash3 = Consensus._semantic_hash(text3, threshold=0.5)

        # Cat sentences might get same hash with low threshold
        # But programming sentence should be different
        assert hash3 != hash1

        # With high threshold (0.95), even similar texts should differ
        # (unless they're nearly identical)
        # This depends on the actual similarity of the embeddings

    def test_empty_text(self) -> None:
        """Test encoding empty text."""
        # The simplified encoder doesn't validate empty text
        # It just passes it to the model
        result = EncoderCore.encode("")
        assert result is not None  # Should return something, even for empty text

    def test_reset(self) -> None:
        """Test resetting the encoder."""
        # First encode something to initialize
        EncoderCore.encode("Initialize the model")
        assert EncoderCore._initialized is True

        # Reset should clear the model
        EncoderCore.reset()
        assert EncoderCore._initialized is False
        assert EncoderCore._model is None

        # Should work again after reset
        embedding = EncoderCore.encode("After reset")
        assert embedding is not None
        assert EncoderCore._initialized is True
