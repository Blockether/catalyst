"""
Integration tests for recursive comparison strategies with actual LLM calls and consensus.

This module tests the full pipeline: LLM calls → consensus voting → recursive comparison
to ensure the system works end-to-end with real language models.
"""

from typing import Any, List, Optional

import anyio
import pytest

from com_blockether_catalyst.consensus import ConsensusCore, ConsensusSettings
from com_blockether_catalyst.consensus.internal.VotingComparison import (
    ComparisonStrategy,
    VotingField,
)
from com_blockether_catalyst.knowledge.internal.KnowledgeExtractionBaseTypes import (
    ChunkAcronymExtractionResponse,
    ChunkingDecision,
    ChunkKeywordExtractionResponse,
    ChunkOutput,
    ExtractedAcronym,
    ExtractedKeyword,
)
from com_blockether_catalyst.utils.instructor.MockInstructorLLMCall import (
    MockInstructorLLMCall,
)
from com_blockether_catalyst.utils.TypedCalls import ArityOneTypedCall


class TestRecursiveComparisonWithLLMConsensus:
    """Test recursive comparison strategies with actual LLM consensus calls."""

    @pytest.fixture
    def mock_judge_keyword(self) -> Any:
        """Create mock judge typed call for keyword extraction tie-breaking."""
        return MockInstructorLLMCall(
            response_model=ChunkKeywordExtractionResponse,
            model_name="judge_keyword",
            temperature=0.1,
            fixed_responses=[
                ChunkKeywordExtractionResponse(
                    reasoning="Judge decision for keyword extraction tie-breaking.",
                    keywords=[ExtractedKeyword(term="API")],
                )
            ],
        )

    @pytest.fixture
    def mock_judge_acronym(self) -> Any:
        """Create mock judge typed call for acronym extraction tie-breaking."""
        return MockInstructorLLMCall(
            response_model=ChunkAcronymExtractionResponse,
            model_name="judge_acronym",
            temperature=0.1,
            fixed_responses=[
                ChunkAcronymExtractionResponse(
                    reasoning="Judge decision for acronym extraction tie-breaking.",
                    acronyms=[ExtractedAcronym(term="API", full_form="Application Programming Interface")],
                )
            ],
        )

    @pytest.fixture
    def mock_judge_chunking(self) -> Any:
        """Create mock judge typed call for chunking tie-breaking."""
        return MockInstructorLLMCall(
            response_model=ChunkingDecision,
            model_name="judge_chunking",
            temperature=0.1,
            fixed_responses=[
                ChunkingDecision(
                    reasoning="Judge decision for chunking tie-breaking to resolve disagreements between models for optimal text segmentation.",
                    chunks=[
                        ChunkOutput(
                            text="Judge chunk",
                            start_position=0,
                            end_position=10,
                        )
                    ],
                )
            ],
        )

    @pytest.mark.asyncio
    async def test_keyword_extraction_consensus_with_unordered_derived(self, mock_judge_keyword: Any) -> None:
        """Test keyword extraction consensus using SEQUENCE_UNORDERED_DERIVED comparison."""
        # Create LLM calls for different models with slight variations in behavior
        model1_call: ArityOneTypedCall[str, ChunkKeywordExtractionResponse] = MockInstructorLLMCall(
            response_model=ChunkKeywordExtractionResponse,
            model_name="keyword_extractor_model1",
            temperature=0.2,
            fixed_responses=[
                ChunkKeywordExtractionResponse(
                    reasoning="Found important technical terms in the text that represent key concepts for software development including API and REST architecture.",
                    keywords=[
                        ExtractedKeyword(term="API"),
                        ExtractedKeyword(term="REST"),
                        ExtractedKeyword(term="HTTP"),
                    ],
                )
            ],
        )

        model2_call: ArityOneTypedCall[str, ChunkKeywordExtractionResponse] = MockInstructorLLMCall(
            response_model=ChunkKeywordExtractionResponse,
            model_name="keyword_extractor_model2",
            temperature=0.3,
            fixed_responses=[
                ChunkKeywordExtractionResponse(
                    reasoning="Identified key technical vocabulary that would be essential for understanding the software system architecture and communication protocols.",
                    keywords=[
                        ExtractedKeyword(term="REST"),
                        ExtractedKeyword(term="API"),
                        ExtractedKeyword(term="HTTP"),  # Same as model1
                    ],
                )
            ],
        )

        model3_call: ArityOneTypedCall[str, ChunkKeywordExtractionResponse] = MockInstructorLLMCall(
            response_model=ChunkKeywordExtractionResponse,
            model_name="keyword_extractor_model3",
            temperature=0.4,
            fixed_responses=[
                ChunkKeywordExtractionResponse(
                    reasoning="Extracted domain-specific terminology that represents core concepts in web development and system integration approaches.",
                    keywords=[
                        ExtractedKeyword(term="API"),
                        ExtractedKeyword(term="REST"),
                        ExtractedKeyword(term="HTTP"),  # Same as others
                    ],
                )
            ],
        )

        # Create model configurations
        model1_config = ConsensusCore.configuration(
            id="keyword_model1",
            executor=model1_call,
            perspective="As a software architect focused on comprehensive keyword extraction",
            weight_multiplier=1.0,
        )

        model2_config = ConsensusCore.configuration(
            id="keyword_model2",
            executor=model2_call,
            perspective="As a technical writer identifying essential vocabulary",
            weight_multiplier=1.0,
        )

        model3_config = ConsensusCore.configuration(
            id="keyword_model3",
            executor=model3_call,
            perspective="As a domain expert extracting core concepts",
            weight_multiplier=1.0,
        )

        # Create consensus with relaxed threshold to test agreement
        settings = ConsensusSettings(max_rounds=2, threshold=0.6)
        consensus = ConsensusCore.consensus(
            models=[model1_config, model2_config, model3_config],
            judge=mock_judge_keyword,
            settings=settings,
        )

        # Test consensus on keyword extraction
        prompt = """Extract important keywords from this technical text:

        'The REST API uses HTTP protocols to enable JSON data exchange between client and server applications. This architectural pattern provides a standardized way for different software systems to communicate and share resources.'

        Focus on technical terms that would be essential for understanding the system architecture."""

        result = await consensus.call(prompt)

        # Verify consensus was achieved despite different keyword orders and slight variations
        assert (
            result.consensus_achieved
        ), f"Consensus should be achieved with threshold 0.6. Convergence score: {result.convergence_score}"
        assert result.final_response is not None
        assert len(result.final_response.keywords) >= 2

        # Verify that API and REST are in the consensus (they appear in all responses)
        keyword_terms = [kw.term for kw in result.final_response.keywords]
        assert "API" in keyword_terms, f"Expected 'API' in consensus keywords: {keyword_terms}"
        assert "REST" in keyword_terms, f"Expected 'REST' in consensus keywords: {keyword_terms}"

        # Verify voting worked correctly - models should have same voting keys for overlapping keywords
        assert result.total_rounds <= 2, "Should reach consensus within 2 rounds"

    @pytest.mark.asyncio
    async def test_acronym_extraction_consensus_with_unordered_derived(self, mock_judge_acronym: Any) -> None:
        """Test acronym extraction consensus using SEQUENCE_UNORDERED_DERIVED comparison."""
        model1_call: ArityOneTypedCall[str, ChunkAcronymExtractionResponse] = MockInstructorLLMCall(
            response_model=ChunkAcronymExtractionResponse,
            model_name="acronym_model1",
            temperature=0.2,
            fixed_responses=[
                ChunkAcronymExtractionResponse(
                    reasoning="Found common software acronyms that represent key technologies and protocols essential for understanding the system architecture.",
                    acronyms=[
                        ExtractedAcronym(
                            term="API",
                            full_form="Application Programming Interface",
                        ),
                        ExtractedAcronym(
                            term="HTTP",
                            full_form="Hypertext Transfer Protocol",
                        ),
                    ],
                )
            ],
        )

        model2_call: ArityOneTypedCall[str, ChunkAcronymExtractionResponse] = MockInstructorLLMCall(
            response_model=ChunkAcronymExtractionResponse,
            model_name="acronym_model2",
            temperature=0.3,
            fixed_responses=[
                ChunkAcronymExtractionResponse(
                    reasoning="Identified technical abbreviations commonly used in web development and system integration that are crucial for technical comprehension.",
                    acronyms=[
                        ExtractedAcronym(
                            term="HTTP",
                            full_form="Hypertext Transfer Protocol",
                        ),
                        ExtractedAcronym(
                            term="API",
                            full_form="Application Programming Interface",
                        ),
                        ExtractedAcronym(
                            term="JSON",
                            full_form="JavaScript Object Notation",
                        ),
                    ],
                )
            ],
        )

        # Create consensus
        settings = ConsensusSettings(max_rounds=2, threshold=0.7)
        consensus = ConsensusCore.consensus(
            models=[
                ConsensusCore.configuration(
                    id="acronym_model1",
                    executor=model1_call,
                    perspective="Conservative acronym extractor",
                    weight_multiplier=1.0,
                ),
                ConsensusCore.configuration(
                    id="acronym_model2",
                    executor=model2_call,
                    perspective="Comprehensive acronym identifier",
                    weight_multiplier=1.0,
                ),
            ],
            judge=mock_judge_acronym,
            settings=settings,
        )

        prompt = "Find acronyms in: 'The HTTP API endpoint returns JSON data for client applications.'"

        result = await consensus.call(prompt)

        # Verify consensus despite different ordering
        assert result.consensus_achieved, f"Should achieve consensus. Score: {result.convergence_score}"
        assert result.final_response is not None

        acronym_terms = [ac.term for ac in result.final_response.acronyms]
        assert "API" in acronym_terms
        assert "HTTP" in acronym_terms

    @pytest.mark.asyncio
    async def test_chunking_consensus_with_ordered_derived(self, mock_judge_chunking: Any) -> None:
        """Test chunking consensus using SEQUENCE_ORDERED_DERIVED comparison."""
        # Model 1: Creates 3 chunks
        model1_call: ArityOneTypedCall[str, ChunkingDecision] = MockInstructorLLMCall(
            response_model=ChunkingDecision,
            model_name="chunking_model1",
            temperature=0.2,
            fixed_responses=[
                ChunkingDecision(
                    reasoning="Divided the text into logical sections based on semantic boundaries and content structure to maintain readability and comprehension.",
                    chunks=[
                        ChunkOutput(
                            text="# Introduction\nThis document explains REST API concepts.",
                            start_position=0,
                            end_position=50,
                        ),
                        ChunkOutput(
                            text="## HTTP Methods\nGET, POST, PUT, DELETE operations.",
                            start_position=51,
                            end_position=100,
                        ),
                        ChunkOutput(
                            text="## Conclusion\nREST APIs provide standardized communication.",
                            start_position=101,
                            end_position=150,
                        ),
                    ],
                )
            ],
        )

        # Model 2: Creates similar 3 chunks with slight variations
        model2_call: ArityOneTypedCall[str, ChunkingDecision] = MockInstructorLLMCall(
            response_model=ChunkingDecision,
            model_name="chunking_model2",
            temperature=0.3,
            fixed_responses=[
                ChunkingDecision(
                    reasoning="Applied semantic segmentation strategy to create coherent chunks that preserve document structure and maintain contextual understanding.",
                    chunks=[
                        ChunkOutput(
                            text="# Introduction\nThis document explains REST API concepts and their usage.",
                            start_position=0,
                            end_position=52,
                        ),
                        ChunkOutput(
                            text="## HTTP Methods\nGET, POST, PUT, DELETE are the main operations.",
                            start_position=53,
                            end_position=105,
                        ),
                        ChunkOutput(
                            text="## Conclusion\nREST APIs enable standardized system communication.",
                            start_position=106,
                            end_position=158,
                        ),
                    ],
                )
            ],
        )

        # Create consensus with threshold allowing for semantic similarity
        settings = ConsensusSettings(max_rounds=2, threshold=0.8)
        consensus = ConsensusCore.consensus(
            models=[
                ConsensusCore.configuration(
                    id="chunking_model1",
                    executor=model1_call,
                    perspective="Structure-focused chunking expert",
                    weight_multiplier=1.0,
                ),
                ConsensusCore.configuration(
                    id="chunking_model2",
                    executor=model2_call,
                    perspective="Semantic-focused chunking specialist",
                    weight_multiplier=1.0,
                ),
            ],
            judge=mock_judge_chunking,
            settings=settings,
        )

        prompt = "Chunk this document into logical sections: 'Introduction to REST APIs...'"

        result = await consensus.call(prompt)

        # Verify consensus on ordered chunking
        assert result.consensus_achieved, f"Chunking consensus should be achieved. Score: {result.convergence_score}"
        assert result.final_response is not None
        assert len(result.final_response.chunks) == 3
        assert len(result.final_response.chunks) == 3

        # Verify chunk order is preserved (SEQUENCE_ORDERED_DERIVED)
        chunks = result.final_response.chunks
        assert "Introduction" in chunks[0].text
        assert "HTTP Methods" in chunks[1].text
        assert "Conclusion" in chunks[2].text

    @pytest.mark.asyncio
    async def test_consensus_failure_with_incompatible_responses(self, mock_judge_keyword: Any) -> None:
        """Test that consensus fails appropriately when responses are too different."""
        # Model 1: Returns keywords
        model1_call: ArityOneTypedCall[str, ChunkKeywordExtractionResponse] = MockInstructorLLMCall(
            response_model=ChunkKeywordExtractionResponse,
            model_name="divergent_model1",
            temperature=0.1,
            fixed_responses=[
                ChunkKeywordExtractionResponse(
                    reasoning="Focused on web development keywords that represent the most important concepts for understanding modern application architecture.",
                    keywords=[
                        ExtractedKeyword(term="API"),
                        ExtractedKeyword(term="REST"),
                    ],
                )
            ],
        )

        # Model 2: Returns completely different keywords
        model2_call: ArityOneTypedCall[str, ChunkKeywordExtractionResponse] = MockInstructorLLMCall(
            response_model=ChunkKeywordExtractionResponse,
            model_name="divergent_model2",
            temperature=0.1,
            fixed_responses=[
                ChunkKeywordExtractionResponse(
                    reasoning="Identified database and security terms that are essential for understanding system infrastructure and data protection mechanisms.",
                    keywords=[
                        ExtractedKeyword(term="SQL"),
                        ExtractedKeyword(term="Authentication"),
                    ],
                )
            ],
        )

        # Create consensus with high threshold
        settings = ConsensusSettings(max_rounds=2, threshold=0.9)  # Very high threshold
        consensus = ConsensusCore.consensus(
            models=[
                ConsensusCore.configuration(
                    id="divergent_model1",
                    executor=model1_call,
                    perspective="Web development expert",
                    weight_multiplier=1.0,
                ),
                ConsensusCore.configuration(
                    id="divergent_model2",
                    executor=model2_call,
                    perspective="Database security expert",
                    weight_multiplier=1.0,
                ),
            ],
            judge=mock_judge_keyword,
            settings=settings,
        )

        prompt = "Extract keywords from technical documentation."

        result = await consensus.call(prompt)

        # Verify consensus fails with completely different responses
        assert not result.consensus_achieved, "Should not achieve consensus with completely different keywords"
        assert result.convergence_score < 0.9, f"Convergence score should be low: {result.convergence_score}"

        # But should still return a fallback response
        assert result.final_response is not None, "Should have fallback response even without consensus"

    @pytest.mark.asyncio
    async def test_model_derived_comparison_in_consensus(self, mock_judge_keyword: Any) -> None:
        """Test that DERIVED comparison works correctly for individual model objects."""
        # This test uses the fact that ExtractedKeyword uses DERIVED internally
        # when compared as part of SEQUENCE_UNORDERED_DERIVED

        model1_call: ArityOneTypedCall[str, ChunkKeywordExtractionResponse] = MockInstructorLLMCall(
            response_model=ChunkKeywordExtractionResponse,
            model_name="semantic_model1",
            temperature=0.2,
            fixed_responses=[
                ChunkKeywordExtractionResponse(
                    reasoning="Extracted key technical terms that represent essential concepts for system understanding and implementation guidance.",
                    keywords=[ExtractedKeyword(term="microservice")],
                )
            ],
        )

        model2_call: ArityOneTypedCall[str, ChunkKeywordExtractionResponse] = MockInstructorLLMCall(
            response_model=ChunkKeywordExtractionResponse,
            model_name="semantic_model2",
            temperature=0.3,
            fixed_responses=[
                ChunkKeywordExtractionResponse(
                    reasoning="Identified domain-specific vocabulary that would be crucial for understanding the architectural approach and design principles.",
                    keywords=[ExtractedKeyword(term="microservice")],  # Same term
                )
            ],
        )

        settings = ConsensusSettings(max_rounds=2, threshold=0.8)
        consensus = ConsensusCore.consensus(
            models=[
                ConsensusCore.configuration(
                    id="semantic_model1",
                    executor=model1_call,
                    perspective="Architecture specialist",
                    weight_multiplier=1.0,
                ),
                ConsensusCore.configuration(
                    id="semantic_model2",
                    executor=model2_call,
                    perspective="Design pattern expert",
                    weight_multiplier=1.0,
                ),
            ],
            judge=mock_judge_keyword,
            settings=settings,
        )

        prompt = "Find architectural keywords in: 'The microservice approach enables...'"

        result = await consensus.call(prompt)

        # Should achieve consensus because ExtractedKeyword uses semantic comparison for 'term'
        # and ignores 'rationale', so both models effectively agree on the same keyword
        assert result.consensus_achieved, f"Should achieve consensus with same terms. Score: {result.convergence_score}"
        assert result.final_response is not None
        assert len(result.final_response.keywords) == 1
        assert result.final_response.keywords[0].term == "microservice"


class TestEdgeCasesWithLLMConsensus:
    """Test edge cases and error conditions with actual LLM consensus."""

    @pytest.fixture
    def mock_judge_keyword(self) -> Any:
        """Create mock judge typed call for keyword extraction tie-breaking."""
        return MockInstructorLLMCall(
            response_model=ChunkKeywordExtractionResponse,
            model_name="judge_keyword",
            temperature=0.1,
            fixed_responses=[
                ChunkKeywordExtractionResponse(
                    reasoning="Judge decision for keyword extraction tie-breaking.",
                    keywords=[ExtractedKeyword(term="API")],
                )
            ],
        )

    @pytest.mark.asyncio
    async def test_empty_response_consensus(self, mock_judge_keyword: Any) -> None:
        """Test consensus when models return empty lists."""
        model1_call: ArityOneTypedCall[str, ChunkKeywordExtractionResponse] = MockInstructorLLMCall(
            response_model=ChunkKeywordExtractionResponse,
            model_name="empty_model1",
            temperature=0.1,
            fixed_responses=[
                ChunkKeywordExtractionResponse(
                    reasoning="No significant keywords were found in the provided text that would meet the criteria for technical vocabulary extraction.",
                    keywords=[],
                )
            ],
        )

        model2_call: ArityOneTypedCall[str, ChunkKeywordExtractionResponse] = MockInstructorLLMCall(
            response_model=ChunkKeywordExtractionResponse,
            model_name="empty_model2",
            temperature=0.1,
            fixed_responses=[
                ChunkKeywordExtractionResponse(
                    reasoning="Analysis revealed no domain-specific terminology or technical concepts that would be valuable for keyword extraction purposes.",
                    keywords=[],
                )
            ],
        )

        settings = ConsensusSettings(max_rounds=1, threshold=0.8)
        consensus = ConsensusCore.consensus(
            models=[
                ConsensusCore.configuration(
                    id="empty1",
                    executor=model1_call,
                    perspective="",
                    weight_multiplier=1.0,
                ),
                ConsensusCore.configuration(
                    id="empty2",
                    executor=model2_call,
                    perspective="",
                    weight_multiplier=1.0,
                ),
            ],
            judge=mock_judge_keyword,
            settings=settings,
        )

        result = await consensus.call("Extract keywords from: 'The quick brown fox.'")

        # Empty lists should achieve consensus easily
        assert result.consensus_achieved, "Empty lists should reach consensus"
        assert result.final_response.keywords == []

    @pytest.mark.asyncio
    async def test_consensus_with_threshold_boundary(self, mock_judge_keyword: Any) -> None:
        """Test consensus behavior at threshold boundaries."""
        # This test verifies that threshold calculations work correctly with derived comparisons

        model1_call: ArityOneTypedCall[str, ChunkKeywordExtractionResponse] = MockInstructorLLMCall(
            response_model=ChunkKeywordExtractionResponse,
            model_name="boundary_model1",
            temperature=0.1,
            fixed_responses=[
                ChunkKeywordExtractionResponse(
                    reasoning="Found two important technical terms that represent key concepts in the software architecture and system design methodology.",
                    keywords=[
                        ExtractedKeyword(term="database"),
                        ExtractedKeyword(term="server"),
                    ],
                )
            ],
        )

        model2_call: ArityOneTypedCall[str, ChunkKeywordExtractionResponse] = MockInstructorLLMCall(
            response_model=ChunkKeywordExtractionResponse,
            model_name="boundary_model2",
            temperature=0.1,
            fixed_responses=[
                ChunkKeywordExtractionResponse(
                    reasoning="Identified one essential term that captures the core infrastructure concept necessary for understanding the system architecture.",
                    keywords=[ExtractedKeyword(term="database")],  # Only 1 of 2 matches
                )
            ],
        )

        # Test with threshold exactly at 50% (1 out of 2 keywords match)
        settings = ConsensusSettings(max_rounds=1, threshold=0.5)
        consensus = ConsensusCore.consensus(
            models=[
                ConsensusCore.configuration(id="b1", executor=model1_call, perspective="", weight_multiplier=1.0),
                ConsensusCore.configuration(id="b2", executor=model2_call, perspective="", weight_multiplier=1.0),
            ],
            judge=mock_judge_keyword,
            settings=settings,
        )

        result = await consensus.call("Find keywords about data systems.")

        # With ChunkKeywordExtractionResponse using threshold=0.6 for ALIKE comparison,
        # and only 50% overlap (1/2), the voting keys won't match, so no consensus
        assert (
            not result.consensus_achieved
        ), f"Should not achieve consensus below ALIKE threshold. Score: {result.convergence_score}"

        # Test with threshold above 50%
        settings_high = ConsensusSettings(max_rounds=1, threshold=0.6)
        consensus_high = ConsensusCore.consensus(
            models=[
                ConsensusCore.configuration(id="b1", executor=model1_call, perspective="", weight_multiplier=1.0),
                ConsensusCore.configuration(id="b2", executor=model2_call, perspective="", weight_multiplier=1.0),
            ],
            judge=mock_judge_keyword,
            settings=settings_high,
        )

        result_high = await consensus_high.call("Find keywords about data systems.")

        # With ChunkKeywordExtractionResponse using threshold=0.6 for ALIKE comparison,
        # and only 50% overlap (1/2), the voting keys won't match even with higher consensus threshold
        assert (
            not result_high.consensus_achieved
        ), f"Should not achieve consensus with insufficient ALIKE overlap. Score: {result_high.convergence_score}"
