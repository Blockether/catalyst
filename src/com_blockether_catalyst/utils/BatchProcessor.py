"""
Generic batch processor with retry logic and concurrent execution.
"""

import logging
from typing import Any, Callable, Coroutine, Generic, Optional, Sequence, TypeVar

import anyio
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

# Type variables for generic input and output
TInput = TypeVar("TInput")
TOutput = TypeVar("TOutput")


class BatchProcessor(Generic[TInput, TOutput]):
    """
    Generic batch processor with concurrent execution and retry logic.

    Processes items in batches with controlled concurrency and automatic
    retry on failures. Results are automatically flattened if the processor
    returns lists.

    GUARANTEES:
    - Order preservation: Results are always returned in the same order as inputs
    - Atomic processing: All items succeed or the batch fails
    - Configurable retries: Exponential backoff with customizable parameters
    - Type safety: Full generic type support for inputs and outputs
    """

    # Default configuration constants
    DEFAULT_BATCH_SIZE = 5
    DEFAULT_MAX_RETRIES = 3
    DEFAULT_RETRY_MIN_WAIT = 1000  # milliseconds
    DEFAULT_RETRY_MAX_WAIT = 10000  # milliseconds

    def __init__(
        self,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_min_wait: int = DEFAULT_RETRY_MIN_WAIT,
        retry_max_wait: int = DEFAULT_RETRY_MAX_WAIT,
        retry_exceptions: Optional[tuple[type[Exception], ...]] = None,
    ):
        """
        Initialize the batch processor.

        Args:
            batch_size: Number of items to process concurrently
            max_retries: Maximum number of retry attempts
            retry_min_wait: Minimum wait time between retries (milliseconds)
            retry_max_wait: Maximum wait time between retries (milliseconds)
            retry_exceptions: Tuple of exception types to retry on (default: all exceptions)
        """
        self._batch_size = batch_size
        self._max_retries = max_retries
        self._retry_min_wait = retry_min_wait
        self._retry_max_wait = retry_max_wait
        self._retry_exceptions = retry_exceptions or (Exception,)

    async def process_batch(
        self,
        items: Sequence[TInput],
        processor_func: Callable[[TInput], Coroutine[Any, Any, list[TOutput]]],
        flatten_results: bool = True,
    ) -> list[TOutput]:
        """
        Process a batch of items concurrently with retry logic.

        IMPORTANT: Order is preserved - results are returned in the same order
        as input items, regardless of processing order or retries.

        Args:
            items: Sequence of items to process
            processor_func: Async function to process each item
            flatten_results: Whether to flatten nested list results

        Returns:
            List of processed results in the same order as inputs (flattened if requested)
        """
        if not items:
            return []

        # Create retry decorator with configured settings
        retry_decorator = retry(
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential(min=self._retry_min_wait / 1000, max=self._retry_max_wait / 1000),
            retry=retry_if_exception_type(self._retry_exceptions),
            before_sleep=before_sleep_log(logger, logging.WARNING),
        )

        # Wrap processor function with retry logic
        @retry_decorator
        async def process_with_retry(item: TInput) -> list[TOutput]:
            try:
                result = await processor_func(item)
                # Ensure result is a list
                if not isinstance(result, list):
                    return [result]
                return result
            except Exception as e:
                logger.error(f"Error processing item: {e}")
                raise

        # Process items with controlled concurrency
        limiter = anyio.CapacityLimiter(self._batch_size)

        # Use a dictionary to maintain order
        results_dict: dict[int, list[TOutput]] = {}

        async def process_with_limiter(idx: int, item: TInput) -> None:
            async with limiter:
                result = await process_with_retry(item)
                results_dict[idx] = result

        # Process all items concurrently
        async with anyio.create_task_group() as tg:
            for idx, item in enumerate(items):
                tg.start_soon(process_with_limiter, idx, item)

        # Collect results in original order
        results = [results_dict[i] for i in range(len(items))]

        # Flatten results if requested
        if flatten_results:
            flattened: list[TOutput] = []
            for result_list in results:
                flattened.extend(result_list)
            return flattened

        # Return the nested structure as-is for non-flattened results
        return results  # type: ignore[return-value]

    async def process_in_chunks(
        self,
        items: Sequence[TInput],
        processor_func: Callable[[TInput], Coroutine[Any, Any, list[TOutput]]],
        chunk_size: Optional[int] = None,
        flatten_results: bool = True,
    ) -> list[TOutput]:
        """
        Process items in chunks with configurable chunk size.

        Useful for very large datasets where you want to process
        in smaller chunks to control memory usage.

        Args:
            items: Sequence of all items to process
            processor_func: Async function to process each item
            chunk_size: Size of each chunk (defaults to batch_size)
            flatten_results: Whether to flatten nested list results

        Returns:
            List of all processed results
        """
        chunk_size = chunk_size or self._batch_size
        all_results: list[TOutput] = []

        # Process items in chunks
        for i in range(0, len(items), chunk_size):
            chunk = items[i : i + chunk_size]
            chunk_results = await self.process_batch(chunk, processor_func, flatten_results=flatten_results)
            all_results.extend(chunk_results)

        return all_results


class BatchProcessorWithFallback(BatchProcessor[TInput, TOutput]):
    """
    Extended batch processor with fallback logic for failed items.
    """

    def __init__(
        self,
        fallback_func: Optional[Callable[[TInput, Exception], Coroutine[Any, Any, list[TOutput]]]] = None,
        **kwargs: Any,
    ):
        """
        Initialize with optional fallback function.

        Args:
            fallback_func: Async function to call when all retries fail
            **kwargs: Arguments for parent BatchProcessor
        """
        super().__init__(**kwargs)
        self._fallback_func = fallback_func

    async def process_batch(
        self,
        items: Sequence[TInput],
        processor_func: Callable[[TInput], Coroutine[Any, Any, list[TOutput]]],
        flatten_results: bool = True,
    ) -> list[TOutput]:
        """
        Process batch with fallback for failed items.
        """
        if not self._fallback_func:
            return await super().process_batch(items, processor_func, flatten_results)

        limiter = anyio.CapacityLimiter(self._batch_size)

        # Use a dictionary to maintain order
        results_dict: dict[int, Sequence[TOutput]] = {}

        async def process_with_fallback(idx: int, item: TInput) -> None:
            async with limiter:
                try:
                    # Try normal processing with retries
                    result = await self._retry_with_backoff(processor_func, item)
                    # Ensure result is a list
                    if not isinstance(result, list):
                        result = [result]
                    results_dict[idx] = result
                except Exception as e:
                    # Use fallback if all retries failed
                    logger.warning(f"All retries failed for item, using fallback: {e}")
                    # Pass the original exception, not RetryError
                    original_error: Exception = e
                    if hasattr(e, "__cause__") and isinstance(e.__cause__, Exception):
                        original_error = e.__cause__
                    if self._fallback_func is not None:
                        fallback_result = await self._fallback_func(item, original_error)
                        # Ensure fallback result is a list
                        if not isinstance(fallback_result, list):
                            fallback_result = [fallback_result]
                        results_dict[idx] = fallback_result
                    else:
                        raise

        # Process all items
        async with anyio.create_task_group() as tg:
            for idx, item in enumerate(items):
                tg.start_soon(process_with_fallback, idx, item)

        # Collect results in original order
        results = [results_dict[i] for i in range(len(items))]

        # Flatten if requested
        if flatten_results:
            flattened: list[TOutput] = []
            for result_list in results:
                flattened.extend(result_list)
            return flattened

        # Return the nested structure as-is for non-flattened results
        return results  # type: ignore[return-value]

    async def _retry_with_backoff(
        self,
        func: Callable[[TInput], Coroutine[Any, Any, list[TOutput]]],
        item: TInput,
    ) -> list[TOutput]:
        """Helper method to retry with exponential backoff."""
        retry_decorator = retry(
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential(min=self._retry_min_wait / 1000, max=self._retry_max_wait / 1000),
            retry=retry_if_exception_type(self._retry_exceptions),
            before_sleep=before_sleep_log(logger, logging.WARNING),
        )

        @retry_decorator
        async def wrapped() -> list[TOutput]:
            result = await func(item)
            # Ensure result is a list
            if not isinstance(result, list):
                return [result]
            return result

        return await wrapped()
