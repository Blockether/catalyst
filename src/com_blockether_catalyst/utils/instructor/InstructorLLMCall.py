"""
Real LLM implementation using Instructor with localhost:3005/v1.

This module provides a production-ready implementation of ArityOneTypedCall
that uses the Instructor library to make structured LLM calls to a local API.
"""

import os
from typing import Any, Optional, Type, TypeVar

import instructor
from openai import AsyncOpenAI
from pydantic import BaseModel

from ..TypedCalls import ArityOneTypedCall

# Type variable for response types
T = TypeVar("T", bound=BaseModel)


class InstructorLLMCall(ArityOneTypedCall[str, T]):
    """
    Production implementation of ArityOneTypedCall using Instructor.

    This class makes real API calls to localhost:3005/v1 for structured LLM responses.
    It's configured to work with local LLM servers that are OpenAI-compatible.
    """

    def __init__(
        self,
        response_model: Type[T],
        completion: Optional[Any] = instructor.from_openai(
            AsyncOpenAI(
                base_url=os.environ.get("INSTRUCTOR_API_BASE_URL", "http://localhost:3005/v1"),
                api_key=os.environ.get("INSTRUCTOR_API_KEY", "nothing"),
            )
        ),
        model: str = "gpt-4o",
        temperature: float = 0.7,
    ):
        """
        Initialize the Instructor LLM call.

        Args:
            response_model: The Pydantic model class for structured responses
            model: The model to use (default: gpt-4o)
            temperature: Temperature for generation (default: 0.7)
            base_url: The base URL for the API (default: http://localhost:3005/v1)
            api_key: API key (uses environment variable if not provided)
        """
        self.response_model = response_model
        self.model = model
        self.temperature = temperature
        self._client = completion

    async def call(self, x: str) -> T:
        """
        Make a structured LLM call to localhost:3005/v1.

        Args:
            prompt: The input prompt

        Returns:
            Structured response of type T
        """
        if not self._client:
            raise ValueError("Instructor client is not initialized")

        response = await self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": x}],
            response_model=self.response_model,
            temperature=self.temperature,
        )
        return response  # type: ignore[no-any-return]
