"""
Comprehensive test for semantic consensus with lists of semantic strings.

This test suite validates:
1. Simple lists of semantic strings
2. Models with List[SemanticString] attributes
3. Document chunks with semantic content
4. Deeply nested models with semantic lists
"""

from typing import List, Optional

import pytest

from com_blockether_catalyst.consensus.ConsensusCore import ConsensusCore
from com_blockether_catalyst.consensus.ConsensusTypes import ConsensusSettings
from com_blockether_catalyst.consensus.VotingComparison import (
    BaseModelWithReasoning,
    ComparisonStrategy,
    VotingField,
)
from com_blockether_catalyst.utils.TypedCalls import ArityOneTypedCall


# Define SemanticString as a proper model with voting support
class SemanticString(BaseModelWithReasoning):
    """A semantic string with proper voting support."""

    value: str = VotingField(
        comparison=ComparisonStrategy.SEMANTIC,
        threshold=0.8,
        description="Semantic string value",
    )


# Response model with list of semantic strings
class TopicsResponse(BaseModelWithReasoning):
    """Response containing a list of semantic topics."""

    topics: List[SemanticString] = VotingField(
        comparison=ComparisonStrategy.DERIVED,
        threshold=0.7,
        description="List of semantic topics",
    )


class MockTypedCall(ArityOneTypedCall):
    """Mock typed call for testing."""

    def __init__(self, response):
        self._response = response

    async def call(self, x: str):
        return self._response


# Test 2: Model with semantic document chunks
class DocumentChunk(BaseModelWithReasoning):
    """A document chunk with semantic content."""

    content: str = VotingField(
        comparison=ComparisonStrategy.SEMANTIC,
        threshold=0.85,
        description="Document content",
    )
    page_number: int = VotingField(comparison=ComparisonStrategy.EXACT, description="Page number")


class ResponseWithChunks(BaseModelWithReasoning):
    """Response containing document chunks."""

    chunks: List[DocumentChunk] = VotingField(
        comparison=ComparisonStrategy.DERIVED,
        threshold=0.75,
        description="List of document chunks",
    )
    total_pages: int = VotingField(comparison=ComparisonStrategy.EXACT, description="Total number of pages")


# Test 3: Nested model with semantic lists
class SemanticKeyword(BaseModelWithReasoning):
    """A semantic keyword with voting support."""

    keyword: str = VotingField(
        comparison=ComparisonStrategy.SEMANTIC,
        threshold=0.8,
        description="Semantic keyword",
    )


class SemanticSection(BaseModelWithReasoning):
    """A section with semantic keywords."""

    title: str = VotingField(
        comparison=ComparisonStrategy.SEMANTIC,
        threshold=0.85,
        description="Section title",
    )
    keywords: List[SemanticKeyword] = VotingField(
        comparison=ComparisonStrategy.DERIVED,
        threshold=0.7,
        description="List of semantic keywords",
    )


class Chapter(BaseModelWithReasoning):
    """A chapter containing semantic sections."""

    chapter_name: str = VotingField(
        comparison=ComparisonStrategy.SEMANTIC,
        threshold=0.85,
        description="Chapter name",
    )
    sections: List[SemanticSection] = VotingField(
        comparison=ComparisonStrategy.DERIVED,
        threshold=0.7,
        description="List of sections",
    )


class Book(BaseModelWithReasoning):
    """A book with nested chapters and semantic sections."""

    title: str = VotingField(comparison=ComparisonStrategy.SEMANTIC, threshold=0.9, description="Book title")
    chapters: List[Chapter] = VotingField(
        comparison=ComparisonStrategy.DERIVED,
        threshold=0.7,
        description="List of chapters",
    )
    summary: str = VotingField(
        comparison=ComparisonStrategy.SEMANTIC,
        threshold=0.75,
        description="Book summary",
    )


class TestSemanticConsensus:
    """Test semantic consensus with various configurations."""

    @pytest.mark.anyio
    async def test_semantic_list_achieves_consensus(self):
        """Test that semantic strings with different cases achieve consensus."""

        # Create topics with same content but different case
        topics1 = [
            SemanticString(
                value="machine learning",
                reasoning="Identified as key topic through comprehensive analysis of document content and thematic patterns throughout the text structure with high confidence based on statistical analysis.",
            ),
            SemanticString(
                value="artificial intelligence",
                reasoning="Core concept identified with high confidence based on frequency analysis and contextual importance in the document requiring comprehensive evaluation of semantic patterns.",
            ),
        ]

        topics2 = [
            SemanticString(
                value="Machine Learning",  # Different case
                reasoning="Identified as key topic through comprehensive analysis of document content and thematic patterns throughout the text structure with high confidence based on statistical analysis.",
            ),
            SemanticString(
                value="Artificial Intelligence",  # Different case
                reasoning="Core concept identified with high confidence based on frequency analysis and contextual importance in the document requiring comprehensive evaluation of semantic patterns.",
            ),
        ]

        topics3 = [
            SemanticString(
                value="machine learning",  # Same as topics1
                reasoning="Identified as key topic through comprehensive analysis of document content and thematic patterns throughout the text structure with high confidence based on statistical analysis.",
            ),
            SemanticString(
                value="artificial intelligence",  # Same as topics1
                reasoning="Core concept identified with high confidence based on frequency analysis and contextual importance in the document requiring comprehensive evaluation of semantic patterns.",
            ),
        ]

        # Create responses
        response1 = TopicsResponse(
            topics=topics1,
            reasoning="Analysis completed with high confidence identifying core machine learning and artificial intelligence topics throughout the comprehensive document review process.",
        )

        response2 = TopicsResponse(
            topics=topics2,
            reasoning="Document evaluation reveals strong presence of machine learning and artificial intelligence themes based on detailed semantic analysis and extraction.",
        )

        response3 = TopicsResponse(
            topics=topics3,
            reasoning="Comprehensive review confirms machine learning and artificial intelligence as primary topics through systematic content analysis and validation with multiple verification steps.",
        )

        # Create models
        models = [
            ConsensusCore.model(id="model1", executor=MockTypedCall(response1), perspective="Analysis A"),
            ConsensusCore.model(id="model2", executor=MockTypedCall(response2), perspective="Analysis B"),
            ConsensusCore.model(id="model3", executor=MockTypedCall(response3), perspective="Analysis C"),
        ]

        # Create judge
        judge = MockTypedCall(response1)

        # Run consensus with low threshold to ensure we get consensus
        consensus = ConsensusCore.consensus(
            models=models,
            judge=judge,
            settings=ConsensusSettings(max_rounds=1, threshold=0.6),
        )

        result = await consensus.call("Extract topics")

        # Should achieve consensus since semantic comparison normalizes case
        assert result.consensus_achieved is True
        assert len(result.final_response.topics) == 2
        assert result.total_rounds == 1

    @pytest.mark.anyio
    async def test_empty_semantic_lists(self):
        """Test consensus with empty semantic lists."""

        # Create responses with empty lists
        response = TopicsResponse(
            topics=[],
            reasoning="Document analysis found no clearly identifiable topics meeting the threshold criteria for inclusion in the final semantic categorization after comprehensive evaluation process.",
        )

        # Create models
        models = [
            ConsensusCore.model(
                id=f"model{i}",
                executor=MockTypedCall(response),
                perspective=f"Analysis {i}",
            )
            for i in range(3)
        ]

        # Create judge
        judge = MockTypedCall(response)

        # Run consensus
        consensus = ConsensusCore.consensus(models=models, judge=judge, settings=ConsensusSettings(max_rounds=1))

        result = await consensus.call("Extract topics from empty doc")

        # Should achieve consensus on empty lists
        assert result.consensus_achieved is True
        assert len(result.final_response.topics) == 0
        assert result.total_rounds == 1

    @pytest.mark.anyio
    async def test_model_with_semantic_chunks(self):
        """Test consensus with model containing semantic document chunks."""

        # Create responses with document chunks
        chunk1 = DocumentChunk(
            content="2.7 Pre-Applied (Fast Track) Limits To reduce the workflow",
            page_number=5,
            reasoning="Extracted verbatim from document page 5 with accurate preservation of original text formatting and structure for consistency and completeness in the extraction process, ensuring all relevant information is captured without modification.",
        )

        chunk1_variant = DocumentChunk(
            content="2.7 Pre-Applied (Fast Track) Limits\nTo reduce the workflow",  # Newline difference
            page_number=5,
            reasoning="Extracted from document with preserved formatting including line breaks and whitespace to maintain original document structure and readability throughout the extraction and processing pipeline for accurate representation.",
        )

        chunk2 = DocumentChunk(
            content="The eligibility and amounts of such limits follow",
            page_number=6,
            reasoning="Continuation of limits section from page 6 providing additional details on eligibility criteria and specific amount thresholds for policy application, including all relevant terms and conditions for complete understanding.",
        )

        response1 = ResponseWithChunks(
            chunks=[chunk1, chunk2],
            total_pages=10,
            reasoning="Document analysis extracted key policy sections related to pre-applied limits and eligibility criteria from the comprehensive policy framework document with careful attention to accuracy.",
        )

        response2 = ResponseWithChunks(
            chunks=[
                chunk1_variant,
                chunk2,
            ],  # Same chunks but with formatting difference
            total_pages=10,
            reasoning="Systematic extraction of relevant policy chunks maintaining original document structure and formatting for accurate representation and compliance verification purposes in review process.",
        )

        response3 = ResponseWithChunks(
            chunks=[chunk1, chunk2],
            total_pages=10,
            reasoning="Careful extraction of policy segments ensuring complete coverage of pre-applied limits and associated eligibility requirements for comprehensive understanding of policy framework.",
        )

        # Create models
        models = [
            ConsensusCore.model(
                id="model1",
                executor=MockTypedCall(response1),
                perspective="Extraction method A",
            ),
            ConsensusCore.model(
                id="model2",
                executor=MockTypedCall(response2),
                perspective="Extraction method B",
            ),
            ConsensusCore.model(
                id="model3",
                executor=MockTypedCall(response3),
                perspective="Extraction method C",
            ),
        ]

        # Create judge
        judge = MockTypedCall(response1)

        # Run consensus
        consensus = ConsensusCore.consensus(
            models=models,
            judge=judge,
            settings=ConsensusSettings(max_rounds=1, threshold=0.6),
        )

        result = await consensus.call("Extract key document chunks")

        # Should achieve consensus despite minor formatting differences
        assert result.consensus_achieved is True
        assert len(result.final_response.chunks) == 2
        assert result.final_response.total_pages == 10

    @pytest.mark.anyio
    async def test_deeply_nested_semantic_lists(self):
        """Test consensus with deeply nested models containing semantic lists."""

        # Create a book with chapters and semantic sections
        section1 = SemanticSection(
            title="Introduction to ML",
            keywords=[
                SemanticKeyword(
                    keyword="supervised learning",
                    reasoning="Core ML paradigm involving labeled training data for predictive modeling and classification tasks in various domains, including image recognition, natural language processing, and time series forecasting applications.",
                ),
                SemanticKeyword(
                    keyword="unsupervised learning",
                    reasoning="ML approach for discovering patterns in unlabeled data through clustering and dimensionality reduction techniques, enabling exploratory data analysis and feature learning without explicit supervision or labeled examples.",
                ),
                SemanticKeyword(
                    keyword="reinforcement learning",
                    reasoning="Learning paradigm based on agent-environment interaction with reward signals for optimal decision-making strategies, widely used in robotics, game playing, and autonomous systems for sequential decision problems.",
                ),
            ],
            reasoning="Core ML paradigms identified through comprehensive analysis of introductory chapter content and fundamental concept coverage in textbook material, representing the three primary learning approaches in artificial intelligence education.",
        )

        section1_variant = SemanticSection(
            title="Introduction to ML",
            keywords=[
                SemanticKeyword(
                    keyword="Supervised Learning",
                    reasoning="Core ML paradigm involving labeled training data for predictive modeling and classification tasks in various domains, including image recognition, natural language processing, and time series forecasting applications.",
                ),  # Different case
                SemanticKeyword(
                    keyword="Unsupervised Learning",
                    reasoning="ML approach for discovering patterns in unlabeled data through clustering and dimensionality reduction techniques, enabling exploratory data analysis and feature learning without explicit supervision or labeled examples.",
                ),
                SemanticKeyword(
                    keyword="Reinforcement Learning",
                    reasoning="Learning paradigm based on agent-environment interaction with reward signals for optimal decision-making strategies, widely used in robotics, game playing, and autonomous systems for sequential decision problems.",
                ),
            ],
            reasoning="Main ML approaches extracted from comprehensive review of introductory material covering fundamental machine learning concepts and methodologies, establishing the foundation for understanding advanced artificial intelligence techniques.",
        )

        section2 = SemanticSection(
            title="Neural Networks",
            keywords=[
                SemanticKeyword(
                    keyword="perceptron",
                    reasoning="Fundamental neural network building block representing single-layer linear classifier for binary classification problems, serving as the historical foundation for modern deep learning architectures and multi-layer neural networks.",
                ),
                SemanticKeyword(
                    keyword="backpropagation",
                    reasoning="Essential algorithm for training multi-layer neural networks through gradient computation and weight adjustment, enabling efficient learning in deep architectures by propagating error signals backwards through network layers.",
                ),
                SemanticKeyword(
                    keyword="gradient descent",
                    reasoning="Optimization algorithm for minimizing loss functions through iterative parameter updates in neural network training, with variants including stochastic, mini-batch, and adaptive methods for improved convergence properties.",
                ),
            ],
            reasoning="Neural network fundamentals identified as key concepts for understanding deep learning architectures and training methodologies in modern AI systems, providing essential knowledge for implementing artificial neural network solutions.",
        )

        chapter1 = Chapter(
            chapter_name="Machine Learning Basics",
            sections=[section1, section2],
            reasoning="Foundational ML concepts chapter covering essential paradigms and neural network fundamentals for comprehensive understanding of machine learning principles, structured to provide progressive learning from basic to advanced concepts.",
        )

        chapter1_variant = Chapter(
            chapter_name="Machine Learning Basics",
            sections=[section1_variant, section2],
            reasoning="Core ML principles chapter presenting fundamental concepts and techniques essential for understanding modern artificial intelligence applications, organized to facilitate learning progression through theoretical and practical perspectives.",
        )

        book1 = Book(
            title="AI Fundamentals",
            chapters=[chapter1],
            summary="A comprehensive guide to artificial intelligence and machine learning fundamentals",
            reasoning="Complete analysis of AI textbook structure reveals systematic coverage of ML basics with emphasis on neural networks and learning paradigms for educational purposes.",
        )

        book2 = Book(
            title="AI Fundamentals",
            chapters=[chapter1_variant],
            summary="A comprehensive guide to artificial intelligence and machine learning fundamentals.",  # Added period
            reasoning="Thorough examination of textbook content shows structured approach to teaching AI concepts from basic to advanced topics with practical applications, utilizing progressive difficulty levels and comprehensive examples throughout.",
        )

        book3 = Book(
            title="AI Fundamentals",
            chapters=[chapter1],
            summary="a comprehensive guide to artificial intelligence and machine learning fundamentals",  # Lowercase
            reasoning="Detailed review of educational material demonstrates well-organized presentation of AI and ML foundational knowledge for students and practitioners alike.",
        )

        # Create models
        models = [
            ConsensusCore.model(
                id="model1",
                executor=MockTypedCall(book1),
                perspective="Academic review",
            ),
            ConsensusCore.model(
                id="model2",
                executor=MockTypedCall(book2),
                perspective="Technical analysis",
            ),
            ConsensusCore.model(
                id="model3",
                executor=MockTypedCall(book3),
                perspective="Content evaluation",
            ),
        ]

        # Create judge
        judge = MockTypedCall(book1)

        # Run consensus
        consensus = ConsensusCore.consensus(
            models=models,
            judge=judge,
            settings=ConsensusSettings(max_rounds=1, threshold=0.6),
        )

        result = await consensus.call("Analyze the book structure")

        # Should have a valid result even if consensus not fully achieved
        assert result.final_response is not None
        assert result.final_response.title == "AI Fundamentals"
        assert len(result.final_response.chapters) == 1
        assert len(result.final_response.chapters[0].sections) == 2
        assert len(result.final_response.chapters[0].sections[0].keywords) == 3

    @pytest.mark.anyio
    async def test_semantic_list_with_different_items(self):
        """Test consensus when semantic lists have different items."""

        response1 = TopicsResponse(
            topics=[
                SemanticString(
                    value="machine learning",
                    reasoning="Topic identified through comprehensive analysis of document content and patterns throughout the text structure, with high frequency of related terms and concepts indicating central importance to the document's subject matter.",
                ),
                SemanticString(
                    value="artificial intelligence",
                    reasoning="Core AI concept identified as central theme based on frequency analysis and contextual importance, appearing consistently across multiple sections and serving as the overarching framework for related technical discussions.",
                ),
                SemanticString(
                    value="deep learning",
                    reasoning="Specialized ML technique prominently featured with detailed coverage of architectures and applications, including convolutional networks, transformers, and other advanced neural network designs for complex pattern recognition tasks.",
                ),
                SemanticString(
                    value="neural networks",
                    reasoning="Fundamental component of modern AI systems with extensive discussion of various network architectures, training methodologies, and practical implementations across diverse application domains in artificial intelligence research.",
                ),  # Extra item
            ],
            reasoning="Comprehensive analysis identifies four key AI topics including neural networks as a fundamental component of modern artificial intelligence systems and applications.",
        )

        response2 = TopicsResponse(
            topics=[
                SemanticString(
                    value="machine learning",
                    reasoning="Topic identified through comprehensive analysis of document content and patterns throughout the text structure, with high frequency of related terms and concepts indicating central importance to the document's subject matter.",
                ),
                SemanticString(
                    value="artificial intelligence",
                    reasoning="Core AI concept identified as central theme based on frequency analysis and contextual importance, appearing consistently across multiple sections and serving as the overarching framework for related technical discussions.",
                ),
                SemanticString(
                    value="deep learning",
                    reasoning="Specialized ML technique prominently featured with detailed coverage of architectures and applications, including convolutional networks, transformers, and other advanced neural network designs for complex pattern recognition tasks.",
                ),
            ],
            reasoning="Core analysis focuses on three primary AI domains that represent the essential foundations of artificial intelligence technology and its practical implementations.",
        )

        response3 = TopicsResponse(
            topics=[
                SemanticString(
                    value="machine learning",
                    reasoning="Topic identified through comprehensive analysis of document content and patterns throughout the text structure, with high frequency of related terms and concepts indicating central importance to the document's subject matter.",
                ),
                SemanticString(
                    value="artificial intelligence",
                    reasoning="Core AI concept identified as central theme based on frequency analysis and contextual importance, appearing consistently across multiple sections and serving as the overarching framework for related technical discussions.",
                ),
                SemanticString(
                    value="computer vision",
                    reasoning="Key application area of AI focusing on image understanding and visual perception tasks in various domains, encompassing object detection, image segmentation, facial recognition, and scene understanding for practical applications.",
                ),  # Different item
            ],
            reasoning="Analysis reveals AI topics with particular emphasis on computer vision as a key application area alongside core machine learning concepts and methodologies.",
        )

        # Create models
        models = [
            ConsensusCore.model(
                id="model1",
                executor=MockTypedCall(response1),
                perspective="Comprehensive analysis",
            ),
            ConsensusCore.model(
                id="model2",
                executor=MockTypedCall(response2),
                perspective="Core concepts",
            ),
            ConsensusCore.model(
                id="model3",
                executor=MockTypedCall(response3),
                perspective="Applied perspective",
            ),
        ]

        # Judge will be needed for different lists
        judge_response = TopicsResponse(
            topics=[
                SemanticString(
                    value="machine learning",
                    reasoning="Most commonly identified topic across all model responses with consistent semantic interpretation and importance, representing the foundational technology for modern artificial intelligence systems and applications.",
                ),
                SemanticString(
                    value="artificial intelligence",
                    reasoning="Universal agreement on AI as central theme with high frequency and contextual relevance in document, serving as the overarching framework encompassing all related technologies and methodologies discussed.",
                ),
                SemanticString(
                    value="deep learning",
                    reasoning="Majority consensus on deep learning as key topic with significant coverage in document analysis results, highlighting its importance as a breakthrough technology in modern artificial intelligence research.",
                ),
            ],
            reasoning="Judge determination based on majority consensus identifies the three most commonly cited topics across all model responses after comprehensive evaluation.",
        )
        judge = MockTypedCall(judge_response)

        # Run consensus
        consensus = ConsensusCore.consensus(
            models=models,
            judge=judge,
            settings=ConsensusSettings(max_rounds=2, threshold=0.6),
        )

        result = await consensus.call("Extract main topics")

        # May not achieve immediate consensus due to different items
        assert result.final_response is not None
        assert isinstance(result.final_response.topics, list)
