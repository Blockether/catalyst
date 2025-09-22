"""
Tests for PromptAlignmentCLIBase class.
"""

from pathlib import Path
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from blockether_catalyst.prompt.PromptAlignmentCLIBase import PromptAlignmentCLIBase


class TestPromptAlignmentCLIBase:
    """Test suite for PromptAlignmentCLIBase."""

    class ConcreteImplementation(PromptAlignmentCLIBase):
        """Concrete implementation for testing."""

        def _init_components(self):
            """Initialize test components."""
            self.prompt_aligner = MagicMock()

        def _get_default_prompt(self) -> str:
            """Return test default prompt."""
            return "Test prompt with {placeholder1} and {placeholder2}"

        async def _test_prompt(self, prompt: str) -> Dict[str, Any]:
            """Test implementation."""
            # Use the validation method as instructed
            filled = self._fill_template(prompt, {"placeholder1": "value1", "placeholder2": "value2"})
            return {"result": filled}

        def _display_test_results(self, results: Dict[str, Any]):
            """Display test results."""
            pass

    @pytest.fixture
    def cli_instance(self, tmp_path):
        """Create a test CLI instance."""
        # Create the prompts directory and test file
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir(exist_ok=True)
        test_file = prompts_dir / "test.txt"
        test_file.write_text("Test prompt with {placeholder1} and {placeholder2}")

        # Use tmp_path for output directory to avoid creating real directories
        output_dir = tmp_path / "output" / "test_responses"
        return self.ConcreteImplementation(prompt_name="test", prompt_dir=prompts_dir, output_dir=output_dir)

    def test_get_prompt_placeholders_extraction(self, cli_instance):
        """Test that placeholders are correctly extracted from prompt."""
        cli_instance.prompt_template = "Hello {name}, your {item} is ready"
        placeholders = cli_instance._get_prompt_placeholders()

        assert len(placeholders) == 2
        assert "name" in placeholders
        assert "item" in placeholders

    def test_get_prompt_placeholders_no_duplicates(self, cli_instance):
        """Test that duplicate placeholders are not returned."""
        cli_instance.prompt_template = "{user} said hello to {user} about {topic}"
        placeholders = cli_instance._get_prompt_placeholders()

        assert len(placeholders) == 2
        assert "user" in placeholders
        assert "topic" in placeholders

    def test_get_prompt_placeholders_empty(self, cli_instance):
        """Test extraction from prompt without placeholders."""
        cli_instance.prompt_template = "Simple prompt without placeholders"
        placeholders = cli_instance._get_prompt_placeholders()

        assert len(placeholders) == 0

    def test_validate_placeholders_all_present(self, cli_instance):
        """Test validation when all placeholders have values."""
        prompt = "Hello {name}, your {order} is ready"
        values = {"name": "Alice", "order": "pizza"}

        is_valid, missing = cli_instance._validate_placeholders(prompt, values)

        assert is_valid is True
        assert len(missing) == 0

    def test_validate_placeholders_some_missing(self, cli_instance):
        """Test validation when some placeholders are missing."""
        prompt = "Hello {name}, your {order} costs {price}"
        values = {"name": "Bob"}

        is_valid, missing = cli_instance._validate_placeholders(prompt, values)

        assert is_valid is False
        assert len(missing) == 2
        assert "order" in missing
        assert "price" in missing

    def test_validate_placeholders_no_values_provided(self, cli_instance):
        """Test validation when no values are provided."""
        prompt = "Hello {name}"

        is_valid, missing = cli_instance._validate_placeholders(prompt, None)

        assert is_valid is False
        assert len(missing) == 1
        assert "name" in missing

    def test_validate_placeholders_no_placeholders_in_prompt(self, cli_instance):
        """Test validation for prompt without placeholders."""
        prompt = "Simple static prompt"

        is_valid, missing = cli_instance._validate_placeholders(prompt, None)

        assert is_valid is True
        assert len(missing) == 0

    def test_validate_placeholders_extra_values_ignored(self, cli_instance):
        """Test that extra values don't affect validation."""
        prompt = "Hello {name}"
        values = {"name": "Charlie", "extra": "ignored", "another": "also ignored"}

        is_valid, missing = cli_instance._validate_placeholders(prompt, values)

        assert is_valid is True
        assert len(missing) == 0

    def test_fill_template_success(self, cli_instance):
        """Test successful prompt filling."""
        prompt = "User {username} has {count} items"
        values = {"username": "alice123", "count": 5}

        result = cli_instance._fill_template(prompt, values)

        assert result == "User alice123 has 5 items"

    def test_fill_template_missing_placeholder_raises_error(self, cli_instance):
        """Test that missing placeholders raise ValueError."""
        prompt = "User {username} has {count} items in {location}"
        values = {"username": "bob456"}

        with pytest.raises(ValueError) as exc_info:
            cli_instance._fill_template(prompt, values)

        error_msg = str(exc_info.value)
        assert "Missing required placeholders" in error_msg
        assert "count" in error_msg
        assert "location" in error_msg
        assert "Available values: username" in error_msg

    def test_fill_template_no_placeholders(self, cli_instance):
        """Test filling prompt without placeholders."""
        prompt = "Static prompt text"
        values = {"unused": "value"}

        result = cli_instance._fill_template(prompt, values)

        assert result == "Static prompt text"

    def test_fill_template_empty_values_dict(self, cli_instance):
        """Test error message when no values provided."""
        prompt = "Hello {name}"
        values = {}

        with pytest.raises(ValueError) as exc_info:
            cli_instance._fill_template(prompt, values)

        error_msg = str(exc_info.value)
        assert "Missing required placeholders: name" in error_msg
        assert "Available values: none" in error_msg

    def test_fill_template_complex_types(self, cli_instance):
        """Test filling prompt with non-string values."""
        prompt = "Count: {count}, List: {items}, Flag: {enabled}"
        values = {"count": 42, "items": [1, 2, 3], "enabled": True}

        result = cli_instance._fill_template(prompt, values)

        assert result == "Count: 42, List: [1, 2, 3], Flag: True"

    @pytest.mark.anyio
    async def test_test_prompt_uses_validation(self, cli_instance):
        """Test that _test_prompt implementation uses validation."""
        prompt = "Test prompt with {placeholder1} and {placeholder2}"

        # This should succeed because ConcreteImplementation provides required values
        result = await cli_instance._test_prompt(prompt)

        assert "result" in result
        assert "value1" in result["result"]
        assert "value2" in result["result"]

    def test_load_default_prompt_when_file_missing(self, cli_instance):
        """Test loading default prompt when saved file doesn't exist."""
        prompt = cli_instance._load_prompt_template()

        assert prompt == "Test prompt with {placeholder1} and {placeholder2}"

    def test_save_and_load_prompt_template(self, cli_instance):
        """Test saving and loading prompt template."""
        custom_prompt = "Custom prompt with {custom_field}"
        cli_instance._save_prompt_template(custom_prompt)

        # Create new instance to test loading
        new_instance = self.ConcreteImplementation(
            prompt_name="test",
            prompt_dir=cli_instance.prompt_dir,
            output_dir=cli_instance.output_dir,
        )

        loaded_prompt = new_instance._load_prompt_template()
        assert loaded_prompt == custom_prompt

    def test_placeholder_regex_pattern(self, cli_instance):
        """Test that placeholder regex only matches valid patterns."""
        # Test various edge cases
        test_cases = [
            ("Simple {placeholder}", ["placeholder"]),
            ("{start} middle {end}", ["start", "end"]),
            (
                "{{not_placeholder}}",
                ["not_placeholder"],
            ),  # Double braces in format string, but regex still matches
            ("{valid_123}", ["valid_123"]),  # Numbers allowed
            ("{CamelCase}", ["CamelCase"]),  # Mixed case
            ("{ spaces }", []),  # Spaces not allowed
            ("{hyphen-not-ok}", []),  # Hyphens not allowed
            ("{}", []),  # Empty placeholder not allowed
        ]

        for prompt, expected in test_cases:
            cli_instance.prompt_template = prompt
            placeholders = cli_instance._get_prompt_placeholders()
            assert sorted(placeholders) == sorted(expected), f"Failed for prompt: {prompt}"

    def test_display_raw_json(self, cli_instance):
        """Test that _display_raw_json properly formats and displays JSON."""
        from typing import List

        from pydantic import BaseModel

        class TestSection(BaseModel):
            title: str
            page: int

        class TestData(BaseModel):
            sections: List[TestSection]
            keywords: List[str]
            count: int
            nested: Dict[str, Any]

        test_data = TestData(
            sections=[
                TestSection(title="Introduction", page=1),
                TestSection(title="Methods", page=5),
            ],
            keywords=["machine learning", "AI"],
            count=42,
            nested={"deep": {"value": "test"}},
        )

        # Mock the console to capture output
        with patch.object(cli_instance.console, "print") as mock_print:
            cli_instance._display_raw_json(test_data, "Test Results")

            # Verify print was called
            assert mock_print.called

            # Get the panel that was printed
            panel_arg = mock_print.call_args[0][0]

            # Verify it's a Panel
            from rich.panel import Panel

            assert isinstance(panel_arg, Panel)

            # Verify the title contains our custom title (handle None case)
            assert panel_arg.title and "Test Results" in panel_arg.title

    def test_display_raw_json_with_non_serializable(self, cli_instance):
        """Test that _display_raw_json handles non-serializable objects."""
        from datetime import datetime

        from pydantic import BaseModel, Field

        class TestDataWithTime(BaseModel):
            timestamp: datetime = Field(default_factory=datetime.now)
            path: Path
            data: str

        test_data = TestDataWithTime(path=Path("/test/path"), data="test data")

        # Should not raise an error
        with patch.object(cli_instance.console, "print"):
            cli_instance._display_raw_json(test_data)
