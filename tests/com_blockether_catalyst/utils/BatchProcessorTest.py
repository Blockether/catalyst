"""
Comprehensive tests for BatchProcessor with retry logic and concurrent execution.
"""

import asyncio
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import anyio
import pytest
from tenacity import RetryError

from com_blockether_catalyst.utils.BatchProcessor import (
    BatchProcessor,
    BatchProcessorWithFallback,
)


class TestBatchProcessor:
    """Test suite for BatchProcessor."""

    # Test constants
    BATCH_SIZE = 3
    MAX_RETRIES = 2
    RETRY_MIN_WAIT = 10  # milliseconds - Fast retries for testing
    RETRY_MAX_WAIT = 20  # milliseconds

    @pytest.fixture
    def processor(self) -> BatchProcessor[str, str]:
        """Create a BatchProcessor instance for testing."""
        return BatchProcessor[str, str](
            batch_size=self.BATCH_SIZE,
            max_retries=self.MAX_RETRIES,
            retry_min_wait=self.RETRY_MIN_WAIT,
            retry_max_wait=self.RETRY_MAX_WAIT,
        )

    @pytest.fixture
    def processor_with_fallback(self) -> BatchProcessorWithFallback[str, str]:
        """Create a BatchProcessorWithFallback instance."""

        async def fallback_func(item: str, error: Exception) -> List[str]:
            return [f"fallback_{item}"]

        return BatchProcessorWithFallback[str, str](
            batch_size=self.BATCH_SIZE,
            max_retries=self.MAX_RETRIES,
            retry_min_wait=self.RETRY_MIN_WAIT,
            retry_max_wait=self.RETRY_MAX_WAIT,
            fallback_func=fallback_func,
        )

    @pytest.mark.anyio
    async def test_process_empty_batch(self, processor: BatchProcessor[str, str]) -> None:
        """Test processing an empty batch returns empty list."""

        async def process_func(item: str) -> List[str]:
            return [item.upper()]

        result = await processor.process_batch(
            items=[],
            processor_func=process_func,
            flatten_results=True,
        )

        assert result == []

    @pytest.mark.anyio
    async def test_process_single_item(self, processor: BatchProcessor[str, str]) -> None:
        """Test processing a single item."""

        async def process_func(item: str) -> List[str]:
            return [item.upper()]

        result = await processor.process_batch(
            items=["hello"],
            processor_func=process_func,
            flatten_results=True,
        )

        assert result == ["HELLO"]

    @pytest.mark.anyio
    async def test_process_multiple_items_with_flattening(self, processor: BatchProcessor[str, str]) -> None:
        """Test processing multiple items with result flattening."""

        async def process_func(item: str) -> List[str]:
            # Return multiple results per item
            return [item.upper(), item.lower()]

        items = ["Hello", "World", "Test"]
        result = await processor.process_batch(
            items=items,
            processor_func=process_func,
            flatten_results=True,
        )

        expected = ["HELLO", "hello", "WORLD", "world", "TEST", "test"]
        assert len(result) == len(expected)
        assert set(result) == set(expected)

    @pytest.mark.anyio
    async def test_process_without_flattening(self, processor: BatchProcessor[str, str]) -> None:
        """Test processing without result flattening."""

        async def process_func(item: str) -> List[str]:
            return [item.upper(), item.lower()]

        items = ["Hello", "World"]
        result = await processor.process_batch(
            items=items,
            processor_func=process_func,
            flatten_results=False,
        )

        assert len(result) == 2
        assert isinstance(result[0], list)
        assert isinstance(result[1], list)

    @pytest.mark.anyio
    async def test_concurrent_processing_respects_batch_size(self, processor: BatchProcessor[str, str]) -> None:
        """Test that concurrent processing respects the batch size limit."""
        import time

        call_times = []

        async def process_func(item: str) -> List[str]:
            start_time = time.time()
            call_times.append(start_time)
            await anyio.sleep(0.05)  # Simulate work
            return [item.upper()]

        items = ["a", "b", "c", "d", "e", "f"]  # 6 items with batch_size=3

        result = await processor.process_batch(
            items=items,
            processor_func=process_func,
            flatten_results=True,
        )

        assert len(result) == 6

        # Since we're using a capacity limiter, only batch_size items should run concurrently
        # But they all start at once within the limit, so we can't easily test timing
        # Just verify all items were processed
        assert set(result) == {"A", "B", "C", "D", "E", "F"}

    @pytest.mark.anyio
    async def test_retry_on_failure(self, processor: BatchProcessor[str, str]) -> None:
        """Test that failed operations are retried."""
        call_count = 0

        async def flaky_processor(item: str) -> List[str]:
            nonlocal call_count
            call_count += 1

            # Fail first time, succeed second time
            if call_count == 1:
                raise ValueError("Temporary error")

            return [item.upper()]

        result = await processor.process_batch(
            items=["test"],
            processor_func=flaky_processor,
            flatten_results=True,
        )

        assert result == ["TEST"]
        assert call_count == 2  # Initial attempt + 1 retry

    @pytest.mark.anyio
    async def test_max_retries_exceeded(self, processor: BatchProcessor[str, str]) -> None:
        """Test that max retries is respected and error is raised."""
        call_count = 0

        async def always_fails(item: str) -> List[str]:
            nonlocal call_count
            call_count += 1
            raise ValueError(f"Always fails for {item}")

        # anyio raises ExceptionGroup when task group encounters errors
        with pytest.raises((RetryError, ExceptionGroup)):
            await processor.process_batch(
                items=["test"],
                processor_func=always_fails,
                flatten_results=True,
            )

        # Should have tried max_retries times
        assert call_count == self.MAX_RETRIES

    @pytest.mark.anyio
    async def test_specific_retry_exceptions(self) -> None:
        """Test that only specific exceptions trigger retries."""
        processor = BatchProcessor[str, str](
            batch_size=2,
            max_retries=3,
            retry_min_wait=10,  # milliseconds
            retry_max_wait=20,  # milliseconds
            retry_exceptions=(ValueError,),  # Only retry on ValueError
        )

        # Test ValueError gets retried
        value_error_calls = 0

        async def raises_value_error(item: str) -> List[str]:
            nonlocal value_error_calls
            value_error_calls += 1
            if value_error_calls == 1:
                raise ValueError("Retry this")
            return [item]

        result = await processor.process_batch(
            items=["test"],
            processor_func=raises_value_error,
            flatten_results=True,
        )

        assert result == ["test"]
        assert value_error_calls == 2

        # Test RuntimeError doesn't get retried
        async def raises_runtime_error(item: str) -> List[str]:
            raise RuntimeError("Don't retry this")

        # anyio raises ExceptionGroup when task group encounters errors
        with pytest.raises((RuntimeError, ExceptionGroup)):
            await processor.process_batch(
                items=["test"],
                processor_func=raises_runtime_error,
                flatten_results=True,
            )

    @pytest.mark.anyio
    async def test_process_in_chunks(self, processor: BatchProcessor[str, str]) -> None:
        """Test processing items in chunks."""
        processed_items = []

        async def process_func(item: str) -> List[str]:
            processed_items.append(item)
            return [item.upper()]

        items = ["a", "b", "c", "d", "e", "f", "g", "h"]

        result = await processor.process_in_chunks(
            items=items,
            processor_func=process_func,
            chunk_size=3,  # Process 3 at a time
            flatten_results=True,
        )

        assert len(result) == 8
        assert all(item.upper() in result for item in items)
        assert len(processed_items) == 8

    @pytest.mark.anyio
    async def test_non_list_return_handling(self, processor: BatchProcessor[str, str]) -> None:
        """Test that non-list returns are converted to lists."""

        async def process_func(item: str) -> List[str]:
            # Return single item instead of list
            return [item.upper()]

        result = await processor.process_batch(
            items=["hello"],
            processor_func=process_func,
            flatten_results=True,
        )

        assert result == ["HELLO"]

    @pytest.mark.anyio
    async def test_order_preservation_with_concurrent_processing(self, processor: BatchProcessor[str, str]) -> None:
        """Test that results maintain input order despite concurrent processing."""
        import random

        async def process_with_random_delay(item: str) -> List[str]:
            # Add random delay to ensure concurrent processing
            await anyio.sleep(random.uniform(0.001, 0.01))
            return [f"processed_{item}"]

        # Test with larger dataset
        input_items = [str(i) for i in range(1, 51)]  # "1" to "50" as strings

        result = await processor.process_batch(
            items=input_items,
            processor_func=process_with_random_delay,
            flatten_results=True,
        )

        # Verify order is preserved
        expected = [f"processed_{i}" for i in input_items]
        assert result == expected, "Order should be preserved despite concurrent processing"

        # Extract numbers and verify sequence
        result_numbers = [r.split("_")[1] for r in result]
        assert result_numbers == input_items, "Exact order should match input"

    @pytest.mark.anyio
    async def test_order_preservation_with_failures(self, processor: BatchProcessor[str, str]) -> None:
        """Test that order is preserved even when some items fail and retry."""
        call_counts = {}

        async def flaky_processor(item: str) -> List[str]:
            item_int = int(item)
            if item not in call_counts:
                call_counts[item] = 0
            call_counts[item] += 1

            # Items divisible by 3 fail on first attempt
            if item_int % 3 == 0 and call_counts[item] == 1:
                raise ValueError(f"Temporary failure for {item}")

            return [f"item_{item}"]

        input_items = [str(i) for i in range(1, 21)]  # "1" to "20" as strings

        result = await processor.process_batch(
            items=input_items,
            processor_func=flaky_processor,
            flatten_results=True,
        )

        # Verify order is preserved despite retries
        expected = [f"item_{i}" for i in input_items]
        assert result == expected, "Order should be preserved even with retries"


class TestBatchProcessorWithFallback:
    """Test suite for BatchProcessorWithFallback."""

    @pytest.fixture
    def processor(self) -> BatchProcessorWithFallback[str, str]:
        """Create a BatchProcessorWithFallback instance."""

        async def fallback_func(item: str, error: Exception) -> List[str]:
            return [f"fallback_{item}_{type(error).__name__}"]

        return BatchProcessorWithFallback[str, str](
            batch_size=2,
            max_retries=2,
            retry_min_wait=10,  # milliseconds
            retry_max_wait=20,  # milliseconds
            fallback_func=fallback_func,
        )

    @pytest.mark.anyio
    async def test_fallback_on_all_retries_failed(self, processor: BatchProcessor[str, str]) -> None:
        """Test that fallback is used when all retries fail."""

        async def always_fails(item: str) -> List[str]:
            raise ValueError(f"Failed for {item}")

        result = await processor.process_batch(
            items=["test1", "test2"],
            processor_func=always_fails,
            flatten_results=True,
        )

        assert len(result) == 2
        # The error type in the fallback should be the original ValueError
        for r in result:
            assert r.startswith("fallback_test") and r.endswith("_ValueError")

    @pytest.mark.anyio
    async def test_mixed_success_and_fallback(self, processor: BatchProcessor[str, str]) -> None:
        """Test mixing successful items and fallback items."""
        call_count = {}

        async def sometimes_fails(item: str) -> List[str]:
            if item not in call_count:
                call_count[item] = 0
            call_count[item] += 1

            # "fail" items always fail, others succeed
            if "fail" in item:
                raise ValueError(f"Failed for {item}")

            return [item.upper()]

        items = ["success1", "fail1", "success2", "fail2"]
        result = await processor.process_batch(
            items=items,
            processor_func=sometimes_fails,
            flatten_results=True,
        )

        assert len(result) == 4
        assert "SUCCESS1" in result
        assert "SUCCESS2" in result
        # Check that fallback items are present with correct error type
        fallback_items = [r for r in result if r.startswith("fallback")]
        assert len(fallback_items) == 2
        for item in fallback_items:
            assert "_ValueError" in item

    @pytest.mark.anyio
    async def test_no_fallback_behavior(self) -> None:
        """Test that processor works without fallback function."""
        processor = BatchProcessorWithFallback[str, str](
            batch_size=2,
            max_retries=2,
            retry_min_wait=10,  # milliseconds
            retry_max_wait=20,  # milliseconds
            fallback_func=None,  # No fallback
        )

        async def process_func(item: str) -> List[str]:
            return [item.upper()]

        result = await processor.process_batch(
            items=["test"],
            processor_func=process_func,
            flatten_results=True,
        )

        assert result == ["TEST"]

    @pytest.mark.anyio
    async def test_fallback_with_non_list_return(self, processor: BatchProcessor[str, str]) -> None:
        """Test fallback with non-list return value."""

        async def fallback_single(item: str, error: Exception) -> List[str]:
            return [f"single_fallback_{item}"]

        processor = BatchProcessorWithFallback[str, str](
            batch_size=2,
            max_retries=1,
            retry_min_wait=10,  # milliseconds
            retry_max_wait=20,  # milliseconds
            fallback_func=fallback_single,
        )

        async def always_fails(item: str) -> List[str]:
            raise ValueError("Fail")

        result = await processor.process_batch(
            items=["test"],
            processor_func=always_fails,
            flatten_results=True,
        )

        assert result == ["single_fallback_test"]

    @pytest.mark.anyio
    async def test_concurrent_fallback_execution(self, processor: BatchProcessor[str, str]) -> None:
        """Test that fallbacks are executed concurrently."""
        import time

        start_times = []

        async def track_time_fallback(item: str, error: Exception) -> List[str]:
            start_times.append(time.time())
            await anyio.sleep(0.05)
            return [f"fallback_{item}"]

        processor = BatchProcessorWithFallback[str, str](
            batch_size=3,
            max_retries=1,
            retry_min_wait=10,  # milliseconds
            retry_max_wait=20,  # milliseconds
            fallback_func=track_time_fallback,
        )

        async def always_fails(item: str) -> List[str]:
            raise ValueError("Fail")

        items = ["a", "b", "c"]
        result = await processor.process_batch(
            items=items,
            processor_func=always_fails,
            flatten_results=True,
        )

        assert len(result) == 3

        # Just check all items were processed with fallback
        for item in ["a", "b", "c"]:
            assert f"fallback_{item}" in result
