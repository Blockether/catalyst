"""
Mock LLM implementation for testing consensus with fixed responses.

This module provides a mock implementation of ArityOneTypedCall that returns
predefined responses for testing consensus behavior without making real API calls.
"""

from typing import List, Type, TypeVar

from pydantic import BaseModel

from ..TypedCalls import ArityOneTypedCall

# Type variable for response types
T = TypeVar("T", bound=BaseModel)


class MockInstructorLLMCall(ArityOneTypedCall[str, T]):
    """
    Mock implementation of ArityOneTypedCall that returns fixed responses.

    This class is designed for testing consensus behavior by providing
    predictable responses without making actual LLM API calls.
    """

    def __init__(
        self,
        response_model: Type[T],
        model_name: str,
        fixed_responses: List[T],
        temperature: float = 0.7,
    ):
        """
        Initialize the mock LLM call.

        Args:
            response_model: The Pydantic model class for structured responses
            model_name: Identifier for this mock model (for debugging)
            fixed_responses: List of responses to cycle through
            temperature: Temperature parameter (ignored in mock)
        """
        self.response_model = response_model
        self.model_name = model_name
        self.fixed_responses = fixed_responses
        self.temperature = temperature
        self._call_count = 0

    async def call(self, x: str) -> T:
        """
        Return the next fixed response in the cycle.

        Args:
            x: The input prompt (ignored in mock)

        Returns:
            The next fixed response of type T
        """
        # Cycle through fixed responses
        response_index = self._call_count % len(self.fixed_responses)
        response = self.fixed_responses[response_index]
        self._call_count += 1

        return response

    def reset_call_count(self) -> None:
        """Reset the call counter to start from the first response again."""
        self._call_count = 0
