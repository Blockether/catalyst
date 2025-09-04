"""
Base classes for knowledge extraction LLM calls.

Users inherit from these base classes to implement their own LLM providers.
"""

from abc import ABC, abstractmethod
from typing import Any, Generic, List, Optional, Tuple, TypeVar, Union

from com_blockether_catalyst.consensus import (
    Consensus,
    ConsensusResult,
    TypedCallBaseForConsensus,
)

from .KnowledgeExtractionBaseTypes import (
    AcronymMeaningExtractionResponse,
    ChunkAcronymExtractionResponse,
    ChunkingDecision,
    ChunkKeywordExtractionResponse,
    KeywordMeaningExtractionResponse,
    KnowledgeMetadata,
    KnowledgePageData,
    TermCooccurrence,
)

# Type variables for the response types
TResponse = TypeVar("TResponse", bound=TypedCallBaseForConsensus)


class BaseConsensusCall(ABC, Generic[TResponse]):
    """
    Base class for all consensus-based LLM calls.

    This class handles the consensus logic while allowing subclasses to define
    their own fill_prompt implementations with different signatures.
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
    def fill_prompt(self, *args: Any, **kwargs: Any) -> str:
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
            *args: Arguments to pass to fill_prompt and post_process
            **kwargs: Keyword arguments to pass to fill_prompt and post_process

        Returns:
            ConsensusResult containing the extraction response,
            potentially enhanced with calculated fields
        """
        prompt = self.fill_prompt(*args, **kwargs)
        result = await self.perform_llm_call(prompt)
        return result


class BaseAcronymExtractionCall(BaseConsensusCall[AcronymMeaningExtractionResponse]):
    """
    Base class for acronym extraction calls.

    Users inherit from this to implement acronym extraction with their LLM.
    All methods have typed signatures that MUST be followed.
    """

    @abstractmethod
    def fill_prompt(
        self,
        acronym: str,
        contexts: List[str],
        cooccurrences_with_contexts: Optional[List[Tuple[TermCooccurrence, List[str]]]] = None,
        max_display_occurrences: int = 15,
        max_display_cooccurrences: int = 10,
    ) -> str:
        """
        Fill the prompt for acronym extraction.

        Users MUST implement this method with EXACTLY these parameters.
        You can use the provided parameters to build your custom prompt.

        Args:
            acronym: The acronym to extract meaning for
            contexts: List of contexts where the acronym appears
            cooccurrences_with_contexts: List of tuples (TermCooccurrence, contexts for that term)
            max_display_occurrences: Maximum number of occurrences to show in prompt
            max_display_cooccurrences: Maximum number of cooccurrences to show in prompt

        Returns:
            Filled prompt string ready for LLM
        """
        pass


class BaseKeywordExtractionCall(BaseConsensusCall[KeywordMeaningExtractionResponse]):
    """
    Base class for keyword extraction calls.

    Users inherit from this to implement keyword extraction with their LLM.
    All methods have typed signatures that MUST be followed.
    This class automatically adds full_form (equal to the term) in post-processing.
    """

    @abstractmethod
    def fill_prompt(
        self,
        term: str,
        contexts: List[str],
        cooccurrences_with_contexts: Optional[List[Tuple[TermCooccurrence, List[str]]]] = None,
        max_display_occurrences: int = 15,
        max_display_cooccurrences: int = 10,
    ) -> str:
        """
        Fill the prompt for keyword extraction.

        Users MUST implement this method with EXACTLY these parameters.
        You can use the provided parameters to build your custom prompt.

        Args:
            term: The keyword to extract meaning for
            contexts: List of contexts where the keyword appears
            cooccurrences_with_contexts: List of tuples (TermCooccurrence, contexts for that term)
            max_display_occurrences: Maximum number of occurrences to show in prompt
            max_display_cooccurrences: Maximum number of cooccurrences to show in prompt

        Returns:
            Filled prompt string ready for LLM
        """
        pass

    def post_process(
        self,
        result: ConsensusResult[KeywordMeaningExtractionResponse],
        term: str,
        **kwargs: Any,
    ) -> ConsensusResult[KeywordMeaningExtractionResponse]:
        """
        Post-process to add full_form field (which equals the term for keywords).

        Args:
            result: The consensus result from the LLM call
            term: The keyword term
            **kwargs: Other arguments passed to fill_prompt

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


class BaseChunkingCall(BaseConsensusCall[ChunkingDecision]):
    """
    Base class for document chunking calls.

    Users inherit from this to implement document chunking with their LLM.
    This class automatically calculates total_chunks in post-processing.
    """

    @abstractmethod
    def fill_prompt(self, page: KnowledgePageData, document_name: str, metadata: KnowledgeMetadata) -> str:
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


class BaseChunkAcronymExtractionCall(BaseConsensusCall[ChunkAcronymExtractionResponse]):
    """
    Base class for extracting acronyms from text chunks.

    Users inherit from this to implement chunk-level acronym extraction.
    """

    @abstractmethod
    def fill_prompt(
        self,
        chunk_text: str,
        document_name: str,
        page_number: int,
        chunk_index: int,
    ) -> str:
        """
        Fill the prompt for chunk acronym extraction.

        Users MUST implement this method with EXACTLY these parameters.

        Args:
            chunk_text: Text content of the chunk
            document_name: Name of the document
            page_number: Page number where chunk appears
            chunk_index: Index of the chunk within the page

        Returns:
            Filled prompt string ready for LLM
        """
        pass


class BaseChunkKeywordExtractionCall(BaseConsensusCall[ChunkKeywordExtractionResponse]):
    """
    Base class for extracting keywords from text chunks.

    Users inherit from this to implement chunk-level keyword extraction.
    """

    @abstractmethod
    def fill_prompt(
        self,
        chunk_text: str,
        document_name: str,
        page_number: int,
        chunk_index: int,
    ) -> str:
        """
        Fill the prompt for chunk keyword extraction.

        Users MUST implement this method with EXACTLY these parameters.

        Args:
            chunk_text: Text content of the chunk
            document_name: Name of the document
            page_number: Page number where chunk appears
            chunk_index: Index of the chunk within the page

        Returns:
            Filled prompt string ready for LLM
        """
        pass
