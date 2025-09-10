"""
Integration tests with real LLM connections.

This module tests the consensus system with actual LLM calls to verify
end-to-end functionality with real language models.
"""

import os
from typing import Any

import pytest

from com_blockether_catalyst.consensus.ConsensusCore import ConsensusCore
from com_blockether_catalyst.consensus.ConsensusTypes import (
    ConsensusSettings,
    TypedCallBaseForConsensus,
)
from com_blockether_catalyst.consensus.VotingComparison import (
    ComparisonStrategy,
    VotingField,
)
from com_blockether_catalyst.knowledge.KnowledgeExtractionTypes import (
    ChunkingDecision,
    ChunkOutput,
)
from com_blockether_catalyst.utils.instructor.InstructorLLMCall import (
    InstructorLLMCall,
)


class TestRealLLMIntegration:
    """Test consensus with real LLM connections."""

    @pytest.mark.anyio
    async def test_chunking_consensus_with_real_llms(self) -> None:
        """Test chunking consensus using real LLMs with ordered derived comparison."""

        # Set environment variables for the LLM connections
        os.environ["INSTRUCTOR_API_BASE_URL"] = "http://localhost:3005/v1"
        os.environ["INSTRUCTOR_API_KEY"] = "test-key"

        # Create real LLM calls for chunking models
        model1_call = InstructorLLMCall(
            response_model=ChunkingDecision,
            model="gpt-4o",  # Using gpt-4o as specified
            temperature=0.3,
        )

        model2_call = InstructorLLMCall(
            response_model=ChunkingDecision,
            model="gpt-4o",  # Using same model with different temperature for variety
            temperature=0.7,
        )

        # Judge model for tie-breaking
        judge_call = InstructorLLMCall(
            response_model=ChunkingDecision,
            model="gpt-4o",
            temperature=0.1,  # Low temperature for consistent judging
        )

        # Create consensus with real LLMs
        settings = ConsensusSettings(
            max_rounds=3,
            threshold=0.7,  # Lower threshold for real LLM variability
        )

        consensus = ConsensusCore.consensus(
            models=[
                ConsensusCore.model(
                    id="chunking_model1",
                    executor=model1_call,
                    perspective="You are a document structure expert. Focus on logical document organization and clear section boundaries.",
                    weight_multiplier=1.0,
                ),
                ConsensusCore.model(
                    id="chunking_model2",
                    executor=model2_call,
                    perspective="You are a semantic coherence specialist. Focus on maintaining meaning and context within chunks.",
                    weight_multiplier=1.0,
                ),
            ],
            judge=judge_call,
            settings=settings,
        )

        # Test prompt for chunking
        prompt = """
        Please chunk the following document into logical sections:

        # Introduction to REST APIs
        REST (Representational State Transfer) is an architectural style for designing networked applications.
        It relies on a stateless, client-server communication protocol. REST APIs use HTTP requests to perform
        CRUD (Create, Read, Update, Delete) operations on resources.

        ## HTTP Methods
        The primary HTTP methods used in REST APIs are:
        - GET: Retrieve data from the server
        - POST: Send data to create a new resource
        - PUT: Update an existing resource
        - DELETE: Remove a resource

        ## Status Codes
        REST APIs use standard HTTP status codes to indicate the success or failure of requests:
        - 200 OK: Successful request
        - 201 Created: Resource successfully created
        - 404 Not Found: Resource doesn't exist
        - 500 Internal Server Error: Server-side error

        ## Conclusion
        REST APIs provide a standardized way for different systems to communicate over HTTP.
        They are widely adopted due to their simplicity, scalability, and stateless nature.
        """

        result = await consensus.call(prompt)

        # Verify basic consensus achievement
        assert result.final_response is not None, "Should have a final response"
        assert isinstance(result.final_response, ChunkingDecision), "Response should be ChunkingDecision"

        # Check that chunks were created
        chunks = result.final_response.chunks
        assert len(chunks) > 0, "Should have at least one chunk"

        # Verify chunk structure
        for chunk in chunks:
            assert isinstance(chunk, ChunkOutput), "Each chunk should be ChunkOutput"
            assert chunk.root, "Chunk should have text"

        # Log results for debugging
        print(f"Consensus achieved: {result.consensus_achieved}")
        print(f"Convergence score: {result.convergence_score}")
        print(f"Number of rounds: {result.total_rounds}")
        print(f"Number of chunks: {len(chunks)}")

        assert result.convergence_score >= 0.7, "Should have good convergence"

    @pytest.mark.anyio
    async def test_simple_consensus_with_real_llms(self) -> None:
        """Test a simple consensus case with real LLMs."""

        # Set environment variables
        os.environ["INSTRUCTOR_API_BASE_URL"] = "http://localhost:3005/v1"
        os.environ["INSTRUCTOR_API_KEY"] = "test-key"

        class SimpleResponse(TypedCallBaseForConsensus):
            answer: str = VotingField(
                comparison=ComparisonStrategy.SEMANTIC,
                threshold=0.8,
                description="The answer to the question",
            )
            confidence: float = VotingField(
                comparison=ComparisonStrategy.RANGE,
                tolerance=0.2,
                description="Confidence score between 0 and 1",
            )

        # Create real LLM calls
        model1 = InstructorLLMCall(
            response_model=SimpleResponse,
            model="gpt-4o",
            temperature=0.5,
        )

        model2 = InstructorLLMCall(
            response_model=SimpleResponse,
            model="gpt-4o",
            temperature=0.5,
        )

        # Judge model for tie-breaking
        judge = InstructorLLMCall(
            response_model=SimpleResponse,
            model="gpt-4o",
            temperature=0.1,  # Low temperature for consistent judging
        )

        # Create consensus
        settings = ConsensusSettings(max_rounds=2, threshold=0.75)
        consensus = ConsensusCore.consensus(
            models=[
                ConsensusCore.model(
                    id="model1",
                    executor=model1,
                    perspective="You are a helpful assistant focused on accuracy.",
                    weight_multiplier=1.0,
                ),
                ConsensusCore.model(
                    id="model2",
                    executor=model2,
                    perspective="You are a helpful assistant focused on clarity.",
                    weight_multiplier=1.0,
                ),
            ],
            judge=judge,
            settings=settings,
        )

        prompt = "What is the capital of France? Provide a brief answer."

        try:
            result = await consensus.call(prompt)

            assert result.final_response is not None
            assert result.final_response.answer
            assert "Paris" in result.final_response.answer
            assert 0 <= result.final_response.confidence <= 1

            print(f"Answer: {result.final_response.answer}")
            print(f"Confidence: {result.final_response.confidence}")
            print(f"Consensus achieved: {result.consensus_achieved}")

        except Exception as e:
            if "Connection" in str(e) or "refused" in str(e):
                pytest.skip(f"LLM server not available: {e}")
            else:
                raise
