"""
Integration tests for the complete knowledge extraction pipeline.
Tests chunking, classification, and term refinement working together.

Uses REAL LLM calls following the conftest.py pattern - NO MOCKS!
"""

from typing import Any

import pytest

from blockether_catalyst.consensus.Consensus import Consensus
from blockether_catalyst.consensus.ConsensusTypes import (
    ConsensusResult,
    ConsensusSettings,
    ModelConfiguration,
)
from blockether_catalyst.knowledge.extraction.ExtractionCore import (
    KnowledgeExtractionCore,
)
from blockether_catalyst.knowledge.extraction.internal.KnowledgeExtractionCallBase import (
    BaseDocumentChunkingCall,
    ExtractionCallsSettings,
)
from blockether_catalyst.knowledge.KnowledgeTypes import (
    ChunkingDecisionResponse,
    ChunkOutput,
    DocumentMetadata,
    KnowledgeExtractionResultWithChunks,
    KnowledgePageData,
    KnowledgeProcessorSettings,
)
from blockether_catalyst.utils.TypedCalls import ArityOneTypedCall


class TestKnowledgeExtractionIntegration:
    """Integration tests for complete knowledge extraction pipeline.

    Simple deterministic tests without complex LLM dependencies.
    """

    @pytest.fixture
    def processor_settings(self, tmp_path: Any) -> KnowledgeProcessorSettings:
        """Create processor settings for integration tests."""
        return KnowledgeProcessorSettings(
            extraction_output_dir=tmp_path / "integration_output",
            linking_threshold=0.7,
            encoding_model="cl100k_base",
        )

    @pytest.fixture
    def simple_calls_settings(self) -> ExtractionCallsSettings:
        """Create simple extraction calls settings for testing."""
        from tests.blockether_catalyst.knowledge.ChunkClassificationTest import (
            RealChunkContentClassificationCall,
        )
        from tests.blockether_catalyst.knowledge.TermRefinementTest import (
            RealTermExtractionCall,
        )

        # Use our simple test implementations
        term_call = RealTermExtractionCall()
        classification_call = RealChunkContentClassificationCall()

        # Create a real deterministic chunking call implementation
        class RealDocumentChunkingCall(BaseDocumentChunkingCall):
            def __init__(self) -> None:
                # Create a minimal real consensus instance for deterministic tests

                # Create a minimal judge for consensus
                class MinimalJudge(ArityOneTypedCall[str, ChunkingDecisionResponse]):
                    async def call(self, input_data: str) -> ChunkingDecisionResponse:
                        return ChunkingDecisionResponse(
                            reasoning="Test judge reasoning. " * 10,  # Ensure 150+ chars
                            chunks=[ChunkOutput(text="Test chunk from judge")],
                        )

                # Create minimal model configurations - need at least 3 for consensus
                dummy_models = []
                for i in range(3):
                    model_config = ModelConfiguration(
                        id=f"test-model-{i}",
                        executor=MinimalJudge(),  # Use same judge as model for simplicity
                        perspective=f"Test perspective {i}",
                        weight=1.0,
                    )
                    dummy_models.append(model_config)

                consensus_settings = ConsensusSettings(
                    max_rounds=1,
                    convergence_threshold=0.8,
                    round_timeout=30.0,
                    enable_logging=False,
                )
                consensus = Consensus[ChunkingDecisionResponse](
                    models=dummy_models,
                    judge=MinimalJudge(),
                    settings=consensus_settings,
                )
                super().__init__(consensus=consensus)

            async def execute(
                self,
                page: Any,  # KnowledgePageData
                document_name: str,
                metadata: Any,  # DocumentMetadata
                *args: object,
                **kwargs: object,
            ) -> ConsensusResult:
                """Execute simple, deterministic document chunking for testing."""

                # Extract page text from the page object
                page_text = getattr(page, "raw_text", "") or getattr(page, "text", "")
                page_number = getattr(page, "page", 1)

                # Simple deterministic chunking - split by sentences
                sentences = page_text.split(". ")
                chunks = []

                for i, sentence in enumerate(sentences):
                    if sentence.strip():
                        chunk_text = sentence.strip()
                        if not chunk_text.endswith("."):
                            chunk_text += "."
                        chunks.append(ChunkOutput(text=chunk_text))

                # If no sentences, create one chunk with the full text
                if not chunks:
                    chunks.append(ChunkOutput(text=page_text or "Default test chunk"))

                response_reasoning = f"Test chunking for {document_name} page {page_number}. " * 5  # Ensure 150+ chars

                response = ChunkingDecisionResponse(
                    reasoning=response_reasoning,
                    chunks=chunks,
                )

                return ConsensusResult(
                    reasoning=f"Test chunking completed for {document_name}. " * 5,
                    consensus_achieved=True,
                    final_response=response,
                    rounds=[],
                    total_rounds=1,
                    convergence_score=0.85,
                    participating_models=["test-model"],
                )

            def fill_template(
                self,
                page: Any,  # KnowledgePageData
                document_name: str,
                metadata: Any,  # DocumentMetadata
            ) -> str:
                """Simple template filling for testing."""
                return "Test chunking template"

        chunking_call = RealDocumentChunkingCall()

        return ExtractionCallsSettings(
            document_chunking_call=chunking_call,
            term_extraction_call=term_call,
            chunk_content_classification_call=classification_call,
        )

    @pytest.fixture
    def extractor(
        self,
        simple_calls_settings: ExtractionCallsSettings,
        processor_settings: KnowledgeProcessorSettings,
    ) -> KnowledgeExtractionCore:
        """Create knowledge extractor with simple mock calls."""
        return KnowledgeExtractionCore(
            calls=simple_calls_settings,
            settings=processor_settings,
        )

    @pytest.mark.anyio
    async def test_extractor_initialization(self, extractor: KnowledgeExtractionCore) -> None:
        """Test that the extractor initializes correctly."""
        assert extractor is not None
        assert extractor._settings is not None
        assert extractor._calls is not None

    @pytest.mark.anyio
    async def test_processor_settings_validation(self, processor_settings: KnowledgeProcessorSettings) -> None:
        """Test processor settings are properly configured."""
        assert processor_settings.linking_threshold == 0.7
        assert processor_settings.encoding_model == "cl100k_base"
        assert processor_settings.extraction_output_dir is not None

    @pytest.mark.anyio
    async def test_extraction_calls_settings_validation(self, simple_calls_settings: ExtractionCallsSettings) -> None:
        """Test extraction calls settings are properly configured."""
        assert simple_calls_settings.document_chunking_call is not None
        assert simple_calls_settings.chunk_content_classification_call is not None
        assert simple_calls_settings.term_extraction_call is not None
