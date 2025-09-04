"""
Arity-Specified Typed Calls Protocol - Generic interface for typed calls with specific arity.

This defines the protocol that any implementation must follow for typed calls,
allowing your code to be implementation-agnostic. Currently supports arity-one calls
with plans for future expansion to other arities.
"""

from abc import abstractmethod
from typing import (
    Any,
    Callable,
    Coroutine,
    Generic,
    Iterable,
    List,
    Optional,
    Protocol,
    TypeVar,
    Union,
    runtime_checkable,
)

import anyio
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


# Type variables for generic async operations
TItem = TypeVar("TItem")
TResult = TypeVar("TResult")


class AsyncBatchProcessor:
    """Helper class for efficient async batch processing operations."""

    @staticmethod
    async def map_concurrent(
        items: Iterable[TItem],
        async_func: Callable[[TItem], Coroutine[Any, Any, TResult]],
        batch_size: Optional[int] = None,
    ) -> List[TResult]:
        """
        Apply an async function to all items concurrently with optional batching.

        Args:
            items: Items to process
            async_func: Async function to apply to each item
            batch_size: Optional batch size to limit concurrent operations

        Returns:
            List of results in the same order as input items
        """
        items_list = list(items)

        if not items_list:
            return []

        if batch_size is None:
            # Process all items concurrently
            results = []

            async def collect_result(item: TItem) -> None:
                result = await async_func(item)
                results.append(result)

            async with anyio.create_task_group() as tg:
                for item in items_list:
                    tg.start_soon(collect_result, item)

            return results

        # Process in batches
        results = []
        for i in range(0, len(items_list), batch_size):
            batch = items_list[i : i + batch_size]
            batch_results = []

            async def collect_batch_result(item: TItem) -> None:
                result = await async_func(item)
                batch_results.append(result)

            async with anyio.create_task_group() as tg:
                for item in batch:
                    tg.start_soon(collect_batch_result, item)
            results.extend(batch_results)

        return results

    @staticmethod
    async def filter_concurrent(
        items: Iterable[TItem],
        predicate: Callable[[TItem], Coroutine[Any, Any, bool]],
        batch_size: Optional[int] = None,
    ) -> List[TItem]:
        """
        Filter items using an async predicate function.

        Args:
            items: Items to filter
            predicate: Async function that returns True for items to keep
            batch_size: Optional batch size to limit concurrent operations

        Returns:
            List of items that passed the predicate
        """
        items_list = list(items)

        if not items_list:
            return []

        # Get predicate results for all items
        predicate_results = await AsyncBatchProcessor.map_concurrent(items_list, predicate, batch_size)

        # Filter based on results
        return [item for item, keep in zip(items_list, predicate_results) if keep]

    @staticmethod
    async def flatten_concurrent(
        items: Iterable[TItem],
        async_func: Callable[[TItem], Coroutine[Any, Any, List[TResult]]],
        batch_size: Optional[int] = None,
    ) -> List[TResult]:
        """
        Apply an async function that returns lists and flatten the results.

        Args:
            items: Items to process
            async_func: Async function that returns a list for each item
            batch_size: Optional batch size to limit concurrent operations

        Returns:
            Flattened list of all results
        """
        nested_results = await AsyncBatchProcessor.map_concurrent(items, async_func, batch_size)

        # Flatten the results
        flattened = []
        for result_list in nested_results:
            flattened.extend(result_list)

        return flattened

    @staticmethod
    async def process_with_semaphore(
        items: Iterable[TItem],
        async_func: Callable[[TItem], Coroutine[Any, Any, TResult]],
        max_concurrent: int = 10,
    ) -> List[TResult]:
        """
        Process items with a semaphore to limit concurrent operations.

        Args:
            items: Items to process
            async_func: Async function to apply to each item
            max_concurrent: Maximum number of concurrent operations

        Returns:
            List of results in the same order as input items
        """
        semaphore = anyio.Semaphore(max_concurrent)

        async def process_with_limit(item: TItem) -> TResult:
            async with semaphore:
                return await async_func(item)

        items_list = list(items)
        results = []

        async def collect_result(item: TItem) -> None:
            result = await process_with_limit(item)
            results.append(result)

        async with anyio.create_task_group() as tg:
            for item in items_list:
                tg.start_soon(collect_result, item)

        return results

    @staticmethod
    async def aggregate_concurrent(
        items: Iterable[TItem],
        async_func: Callable[[TItem], Coroutine[Any, Any, TResult]],
        aggregator: Callable[[List[TResult]], Any],
        batch_size: Optional[int] = None,
    ) -> Any:
        """
        Apply an async function to items and aggregate the results.

        Args:
            items: Items to process
            async_func: Async function to apply to each item
            aggregator: Function to aggregate results
            batch_size: Optional batch size to limit concurrent operations

        Returns:
            Aggregated result
        """
        results = await AsyncBatchProcessor.map_concurrent(items, async_func, batch_size)
        return aggregator(results)
