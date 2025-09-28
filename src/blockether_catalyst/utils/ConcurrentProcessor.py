"""
Generic batch processor with retry logic and concurrent execution.
"""

import logging
from typing import Any, Callable, Coroutine, Generic, Optional, Sequence, TypeVar, cast

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


class ConcurrentProcessor(Generic[TInput, TOutput]):
    """
    Generic concurrent processor with controlled parallelism and retry logic.

    Processes items concurrently with configurable parallelism limits and automatic
    retry on failures. Results are automatically flattened if the processor
    returns lists.

    GUARANTEES:
    - Order preservation: Results are always returned in the same order as inputs
    - Atomic processing: All items succeed or all fail
    - Configurable retries: Exponential backoff with customizable parameters
    - Type safety: Full generic type support for inputs and outputs
    """

    # Default configuration constants
    DEFAULT_CONCURRENCY = 5
    DEFAULT_MAX_RETRIES = 3
    DEFAULT_RETRY_MIN_WAIT = 3500  # milliseconds
    DEFAULT_RETRY_MAX_WAIT = 15000  # milliseconds

    def __init__(
        self,
        concurrency: int = DEFAULT_CONCURRENCY,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_min_wait: int = DEFAULT_RETRY_MIN_WAIT,
        retry_max_wait: int = DEFAULT_RETRY_MAX_WAIT,
        retry_exceptions: Optional[tuple[type[Exception], ...]] = None,
    ):
        """
        Initialize the concurrent processor.

        Args:
            concurrency: Maximum number of items to process concurrently
            max_retries: Maximum number of retry attempts
            retry_min_wait: Minimum wait time between retries (milliseconds)
            retry_max_wait: Maximum wait time between retries (milliseconds)
            retry_exceptions: Tuple of exception types to retry on (default: all exceptions)
        """
        self._concurrency = concurrency
        self._max_retries = max_retries
        self._retry_min_wait = retry_min_wait
        self._retry_max_wait = retry_max_wait
        self._retry_exceptions = retry_exceptions or (Exception,)

    async def _process_concurrently(
        self,
        items: Sequence[TInput],
        processor_fn: Callable[[TInput], Coroutine[Any, Any, Sequence[TOutput | None]]],
    ) -> Sequence[TOutput]:
        """
        Process items concurrently with retry logic and controlled parallelism.

        IMPORTANT: Each item is processed individually, but multiple items are
        processed in parallel (up to batch_size limit). Order is preserved.

        Args:
            items: Sequence of items to process
            processor_fn: Async function to process each item individually
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
        async def process_with_retry(item: TInput) -> Sequence[TOutput | None]:
            try:
                return await processor_fn(item)
            except Exception as e:
                logger.error(f"Error processing item: {e}")
                raise

        # Process items with controlled concurrency
        limiter = anyio.CapacityLimiter(self._concurrency)

        # Use a dictionary to maintain order
        results_dict: dict[int, Sequence[TOutput | None]] = {}

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

        flattened: list[TOutput | None] = []
        for result_list in results:
            flattened.extend(result_list)

        filtered: list[TOutput] = [r for r in flattened if r is not None]

        return filtered

    async def process(
        self,
        items: Sequence[TInput],
        processor_fn: Callable[[TInput], Coroutine[Any, Any, Sequence[TOutput | None] | TOutput | None]],
    ) -> Sequence[TOutput]:
        """
        Process items with configurable concurrency.

        Useful for very large datasets where you want to process
        with specific concurrency limits to control resource usage.

        Args:
            items: Sequence of all items to process
            processor_fn: Async function to process each item

        Returns:
            List of all processed results
        """

        async def _wrapped_call(x: TInput) -> Sequence[TOutput | None]:
            result = await processor_fn(x)
            if result is None:
                return []
            # Check for string specifically since str is also a Sequence
            if isinstance(result, str):
                return cast(Sequence[TOutput | None], [result])
            # Check if it's already a sequence (list, tuple, etc) but not a BaseModel
            if isinstance(result, (list, tuple)):
                return cast(Sequence[TOutput | None], result)
            # Single non-sequence item - wrap in a list
            return cast(Sequence[TOutput | None], [result])

        results = await self._process_concurrently(
            items,
            _wrapped_call,
        )

        return results
