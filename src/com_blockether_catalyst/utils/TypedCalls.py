"""
Arity-Specified Typed Calls Protocol - Generic interface for typed calls with specific arity.

This defines the protocol that any implementation must follow for typed calls,
allowing your code to be implementation-agnostic. Currently supports arity-one calls
with plans for future expansion to other arities.
"""

from abc import abstractmethod
from typing import (
    Generic,
    Protocol,
    TypeVar,
    Union,
    runtime_checkable,
)

from pydantic import BaseModel, RootModel

T = TypeVar("T", bound=BaseModel, covariant=True)
X = TypeVar("X", bound=Union[str, BaseModel, RootModel], contravariant=True)


@runtime_checkable
class ArityOneTypedCall(Protocol, Generic[X, T]):
    """
    Protocol for arity-one typed calls that return structured Pydantic models.

    This is the interface that any implementation must follow.
    Users can implement this protocol for LLMs (BAML, OpenAI, Anthropic, etc.)
    or any other service that takes input and returns typed output.

    The input type X can be:
    - A string (for simple prompts)
    - A BaseModel (for structured requests)
    - A RootModel (for wrapped primitive types)
    """

    @abstractmethod
    async def call(
        self,
        x: X,
    ) -> T:
        """
        Make a typed call and return a structured response.

        Args:
            x: The input which can be a string, BaseModel, or RootModel.

        Returns:
            Structured response as type T (Pydantic model)
        """
        ...
