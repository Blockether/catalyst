"""
Real LLM implementation using Instructor with localhost:3005/v1.

This module provides a production-ready implementation of ArityOneTypedCall
that uses the Instructor library to make structured LLM calls to a local API.
"""

import os
from typing import Literal, Optional, Type, TypeVar, cast

import instructor
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
        model: Optional[str] = None,
        completion_type: Optional[Literal["openai", "litellm"]] = None,
        temperature: float = 0.7,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """
        Initialize the Instructor LLM call.

        Args:
            response_model: The Pydantic model class for structured responses
            model: The model to use (default: gpt-4o)
            temperature: Temperature for generation (default: 0.7)
            base_url: The base URL for the API (default from env or http://localhost:3005/v1)
            api_key: API key (default from env or "nothing")
        """
        # Use provided values or fall back to environment variables or defaults
        actual_base_url = base_url or os.environ.get("INSTRUCTOR_API_BASE_URL", "http://localhost:3005/v1")
        actual_api_key = api_key or os.environ.get("INSTRUCTOR_API_KEY", "nothing")
        actual_model = model or os.environ.get("INSTRUCTOR_MODEL", "gpt-4o")
        actual_completion_type = completion_type or os.environ.get("INSTRUCTOR_COMPLETION_TYPE", "litellm")

        print(f"Using Instructor LLM at {actual_base_url} with model {actual_model}")

        if actual_completion_type not in ("openai", "litellm"):
            raise ValueError("completion_type must be 'openai' or 'litellm'")

        # Type narrowing - after the check above, we know it's one of the literals
        actual_completion_type = cast(Literal["openai", "litellm"], actual_completion_type)

        completion = None
        if actual_completion_type == "litellm":
            from litellm import acompletion

            completion = instructor.from_litellm(
                acompletion,
                base_url=actual_base_url,
                api_key=actual_api_key,
            )

        if actual_completion_type == "openai":
            from openai import AsyncOpenAI

            completion = instructor.from_openai(
                AsyncOpenAI(
                    base_url=actual_base_url,
                    api_key=actual_api_key,
                )
            )

        completion = cast(instructor.AsyncInstructor, completion)

        self.response_model = response_model
        self.model = actual_model
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
            max_retries=5,
        )

        return response  # type: ignore[no-any-return]
