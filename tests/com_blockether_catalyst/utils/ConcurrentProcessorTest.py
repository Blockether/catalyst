"""
Comprehensive tests for ConcurrentProcessor with retry logic and concurrent execution.
"""

from typing import List, Optional, Sequence
from unittest.mock import AsyncMock, MagicMock, patch

import anyio
import pytest
from tenacity import RetryError

from com_blockether_catalyst.utils.ConcurrentProcessor import ConcurrentProcessor


class TestConcurrentProcessor:
    """Test suite for ConcurrentProcessor."""

    # Test constants
    CONCURRENCY = 3
    MAX_RETRIES = 2
    RETRY_MIN_WAIT = 10  # milliseconds - Fast retries for testing
    RETRY_MAX_WAIT = 20  # milliseconds

    @pytest.fixture
    def processor(self) -> ConcurrentProcessor[str, str]:
        """Create a ConcurrentProcessor instance for testing."""
        return ConcurrentProcessor[str, str](
            concurrency=self.CONCURRENCY,
            max_retries=self.MAX_RETRIES,
            retry_min_wait=self.RETRY_MIN_WAIT,
            retry_max_wait=self.RETRY_MAX_WAIT,
        )

    @pytest.mark.anyio
    async def test_process_empty_list(self, processor: ConcurrentProcessor[str, str]) -> None:
        """Test processing an empty list returns empty list."""

        async def process_func(item: str) -> List[str]:
            return [item.upper()]

        result = await processor.process(
            items=[],
            processor_func=process_func,
        )

        assert result == []

    @pytest.mark.anyio
    async def test_process_single_item(self, processor: ConcurrentProcessor[str, str]) -> None:
        """Test processing a single item."""

        async def process_func(item: str) -> List[str]:
            return [item.upper()]

        result = await processor.process(
            items=["hello"],
            processor_func=process_func,
        )

        assert result == ["HELLO"]

    @pytest.mark.anyio
    async def test_process_multiple_items(self, processor: ConcurrentProcessor[str, str]) -> None:
        """Test processing multiple items with result flattening."""

        async def process_func(item: str) -> List[str]:
            # Return multiple results per item
            return [item.upper(), item.lower()]

        items = ["Hello", "World", "Test"]
        result = await processor.process(
            items=items,
            processor_func=process_func,
        )

        expected = ["HELLO", "hello", "WORLD", "world", "TEST", "test"]
        assert result == expected

    @pytest.mark.anyio
    async def test_process_with_none_filtering(self, processor: ConcurrentProcessor[str, str]) -> None:
        """Test that None values are filtered out."""

        async def process_func(item: str) -> List[Optional[str]]:
            if item == "skip":
                return [None]
            return [item.upper(), None, item.lower()]

        items = ["Hello", "skip", "World"]
        result = await processor.process(
            items=items,
            processor_func=process_func,
        )

        # None values should be filtered out
        expected = ["HELLO", "hello", "WORLD", "world"]
        assert result == expected

    @pytest.mark.anyio
    async def test_concurrent_processing_respects_concurrency_limit(
        self, processor: ConcurrentProcessor[str, str]
    ) -> None:
        """Test that concurrent processing respects the concurrency limit."""
        import time

        call_times = []

        async def process_func(item: str) -> List[str]:
            start_time = time.time()
            call_times.append(start_time)
            await anyio.sleep(0.05)  # Simulate work
            return [item.upper()]

        items = ["a", "b", "c", "d", "e", "f"]  # 6 items with concurrency=3

        result = await processor.process(
            items=items,
            processor_func=process_func,
        )

        assert len(result) == 6
        assert set(result) == {"A", "B", "C", "D", "E", "F"}

    @pytest.mark.anyio
    async def test_retry_on_failure(self, processor: ConcurrentProcessor[str, str]) -> None:
        """Test that failed operations are retried."""
        call_count = 0

        async def flaky_processor(item: str) -> List[str]:
            nonlocal call_count
            call_count += 1

            # Fail first time, succeed second time
            if call_count == 1:
                raise ValueError("Temporary error")

            return [item.upper()]

        result = await processor.process(
            items=["test"],
            processor_func=flaky_processor,
        )

        assert result == ["TEST"]
        assert call_count == 2  # Initial attempt + 1 retry

    @pytest.mark.anyio
    async def test_max_retries_exceeded(self, processor: ConcurrentProcessor[str, str]) -> None:
        """Test that max retries is respected and error is raised."""
        call_count = 0

        async def always_fails(item: str) -> List[str]:
            nonlocal call_count
            call_count += 1
            raise ValueError(f"Always fails for {item}")

        # anyio raises ExceptionGroup when task group encounters errors
        with pytest.raises((RetryError, ExceptionGroup)):
            await processor.process(
                items=["test"],
                processor_func=always_fails,
            )

        # Should have tried max_retries times
        assert call_count == self.MAX_RETRIES

    @pytest.mark.anyio
    async def test_specific_retry_exceptions(self) -> None:
        """Test that only specific exceptions trigger retries."""
        processor = ConcurrentProcessor[str, str](
            concurrency=2,
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

        result = await processor.process(
            items=["test"],
            processor_func=raises_value_error,
        )

        assert result == ["test"]
        assert value_error_calls == 2

        # Test RuntimeError doesn't get retried
        async def raises_runtime_error(item: str) -> List[str]:
            raise RuntimeError("Don't retry this")

        # anyio raises ExceptionGroup when task group encounters errors
        with pytest.raises((RuntimeError, ExceptionGroup)):
            await processor.process(
                items=["test"],
                processor_func=raises_runtime_error,
            )

    @pytest.mark.anyio
    async def test_process_with_chunks(self, processor: ConcurrentProcessor[str, str]) -> None:
        """Test processing items in chunks."""
        processed_items = []

        async def process_func(item: str) -> List[str]:
            processed_items.append(item)
            return [item.upper()]

        items = ["a", "b", "c", "d", "e", "f", "g", "h"]

        result = await processor.process(items=items, processor_func=process_func)

        assert len(result) == 8
        assert all(item.upper() in result for item in items)
        assert len(processed_items) == 8

    @pytest.mark.anyio
    async def test_non_list_return_handling(self, processor: ConcurrentProcessor[str, str]) -> None:
        """Test that non-list returns are converted to lists."""

        async def process_func(item: str) -> str:  # Returns single string, not list
            return item.upper()

        result = await processor.process(
            items=["hello"],
            processor_func=process_func,
        )

        assert result == ["HELLO"]

    @pytest.mark.anyio
    async def test_order_preservation_with_concurrent_processing(
        self, processor: ConcurrentProcessor[str, str]
    ) -> None:
        """Test that results maintain input order despite concurrent processing."""
        import random

        async def process_with_random_delay(item: str) -> List[str]:
            # Add random delay to ensure concurrent processing
            await anyio.sleep(random.uniform(0.001, 0.01))
            return [f"processed_{item}"]

        # Test with larger dataset
        input_items = [str(i) for i in range(1, 21)]  # "1" to "20" as strings

        result = await processor.process(
            items=input_items,
            processor_func=process_with_random_delay,
        )

        # Verify order is preserved (flattened results)
        expected = [f"processed_{i}" for i in input_items]
        assert result == expected

    @pytest.mark.anyio
    async def test_order_preservation_with_failures(self, processor: ConcurrentProcessor[str, str]) -> None:
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

        result = await processor.process(
            items=input_items,
            processor_func=flaky_processor,
        )

        # Verify order is preserved despite retries
        expected = [f"item_{i}" for i in input_items]
        assert result == expected, "Order should be preserved even with retries"

    @pytest.mark.anyio
    async def test_empty_return_handling(self, processor: ConcurrentProcessor[str, str]) -> None:
        """Test that empty lists and None returns are handled correctly."""

        async def process_func(item: str) -> Optional[List[str]]:
            if item == "empty":
                return []
            elif item == "none":
                return None
            else:
                return [item.upper()]

        items = ["hello", "empty", "none", "world"]
        result = await processor.process(
            items=items,
            processor_func=process_func,
        )

        # Empty lists and None should result in no output for those items
        assert result == ["HELLO", "WORLD"]

    @pytest.mark.anyio
    async def test_mixed_return_types(self, processor: ConcurrentProcessor[str, str]) -> None:
        """Test that mixed return types (single item, list, None) are handled correctly."""

        async def process_func(item: str) -> Optional[List[str] | str]:
            if item == "single":
                return "SINGLE"  # Single item
            elif item == "list":
                return ["LIST1", "LIST2"]  # List
            elif item == "none":
                return None  # None
            else:
                return []  # Empty list

        items = ["single", "list", "none", "empty"]
        result = await processor.process(
            items=items,
            processor_func=process_func,
        )

        # Should handle all types correctly
        assert result == ["SINGLE", "LIST1", "LIST2"]

    @pytest.mark.anyio
    async def test_comprehensive_none_handling(self, processor: ConcurrentProcessor[str, str]) -> None:
        """Comprehensive test to verify None values are properly filtered in all scenarios."""

        async def process_item(item: str) -> Optional[List[Optional[str]]]:
            if item == "skip":
                return None  # Return None for this item
            elif item == "empty":
                return []  # Return empty list
            elif item == "mixed":
                return ["valid", None, "another"]  # Mix of valid and None
            else:
                return [item.upper()]  # Normal processing

        items = ["hello", "skip", "world", "empty", "mixed", "test"]

        result = await processor.process(items=items, processor_func=process_item)

        # Verify the result
        expected = ["HELLO", "WORLD", "valid", "another", "TEST"]
        assert result == expected, f"Result {result} doesn't match expected {expected}"
