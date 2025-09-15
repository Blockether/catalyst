"""
Base classes for knowledge extraction LLM calls.

Users inherit from these base classes to implement their own LLM providers.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, List, TypeVar

from pydantic import BaseModel, Field

from blockether_catalyst.consensus.Consensus import Consensus
from blockether_catalyst.consensus.ConsensusTypes import (
    ConsensusResult,
)
from blockether_catalyst.consensus.VotingComparison import BaseModelWithReasoning

from .KnowledgeTypes import (
    ChunkingDecisionResponse,
    DocumentMetadata,
    KnowledgePageDataWithRawText,
    TermMeaningExtractionResponse,
)

# Type variables for the response types
TResponse = TypeVar("TResponse", bound=BaseModelWithReasoning)


class BaseConsensusCall(ABC, Generic[TResponse]):
    """
    Base class for all consensus-based LLM calls.

    This class handles the consensus logic while allowing subclasses to define
    their own fill_template implementations with different signatures.
    """

    def __init__(self, consensus: Consensus[TResponse]):
        """
        Initialize with a consensus instance.

        Args:
            consensus: Consensus instance configured with models and settings
        """
        self._consensus = consensus

    @property
    def consensus(self) -> Consensus[TResponse]:
        """Get the consensus instance."""
        return self._consensus

    @abstractmethod
    def fill_template(self, *args: Any, **kwargs: Any) -> str:
        """
        Fill the prompt for the specific extraction type.

        This method is abstract and must be implemented by subclasses with
        their specific signature requirements.
        """
        pass

    async def perform_llm_call(self, prompt: str) -> ConsensusResult[TResponse]:
        """
        Perform the LLM call using consensus.

        This method is implemented and calls the consensus with the provided prompt.

        Args:
            prompt: The filled prompt string

        Returns:
            ConsensusResult containing the consensus response
        """
        return await self._consensus.call(prompt)

    async def execute(self, *args: Any, **kwargs: Any) -> ConsensusResult[TResponse]:
        """
        Execute the full extraction pipeline with post-processing.

        This method fills the prompt with the provided arguments, performs the LLM call,
        and applies post-processing to potentially add calculated fields.

        Args:
            *args: Arguments to pass to fill_template and post_process
            **kwargs: Keyword arguments to pass to fill_template and post_process

        Returns:
            ConsensusResult containing the extraction response,
            potentially enhanced with calculated fields
        """
        prompt = self.fill_template(*args, **kwargs)
        result = await self.perform_llm_call(prompt)
        return result


class BaseTermExtractionCall(BaseConsensusCall[TermMeaningExtractionResponse]):
    """
    Base class for term extraction calls.

    Users inherit from this to implement term extraction with their LLM.
    All methods have typed signatures that MUST be followed.
    This class automatically adds full_form (equal to the term) in post-processing.
    """

    @abstractmethod
    def fill_template(
        self,
        term: str,
        type: str,
        occurrences_contexts: List[str],
        cooccurring_terms: Dict[str, List[str]],
    ) -> str:
        pass

    def post_process(
        self,
        result: ConsensusResult[TermMeaningExtractionResponse],
        term: str,
        **kwargs: Any,
    ) -> ConsensusResult[TermMeaningExtractionResponse]:
        """
        Post-process to add full_form field (which equals the term for keywords).

        Args:
            result: The consensus result from the LLM call
            term: The keyword term
            **kwargs: Other arguments passed to fill_template

        Returns:
            Enhanced consensus result with full_form set to the term
        """
        # Simply set the full_form on the final response
        result.final_response.full_form = term

        # Also set it on all responses in rounds for consistency
        for round_data in result.rounds:
            for response in round_data.responses:
                response.content.full_form = term
            if round_data.consensus_response:
                round_data.consensus_response.full_form = term

        return result


class BaseDocumentChunkingCall(BaseConsensusCall[ChunkingDecisionResponse]):
    """
    Base class for document chunking calls.

    Users inherit from this to implement document chunking with their LLM.
    This class automatically calculates total_chunks in post-processing.
    """

    @abstractmethod
    def fill_template(
        self,
        page: KnowledgePageDataWithRawText,
        document_name: str,
        metadata: DocumentMetadata,
    ) -> str:
        """
        Fill the prompt for document chunking.

        Users MUST implement this method with EXACTLY these parameters.

        Args:
            page: Page data to chunk
            document_name: Name of the document being processed
            metadata: Document metadata

        Returns:
            Filled prompt string ready for LLM
        """
        pass


class ExtractionCallsSettings(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    document_chunking_call: BaseDocumentChunkingCall = Field(
        description="User-implemented call for document chunking",
    )

    term_extraction_call: BaseTermExtractionCall = Field(
        description="User-implemented call for initial term discovery in chunks (MANDATORY)",
    )
