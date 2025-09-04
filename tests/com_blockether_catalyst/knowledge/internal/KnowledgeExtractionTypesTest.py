"""Tests for KnowledgeExtractionTypes, specifically table formatting."""

import pytest

from com_blockether_catalyst.knowledge.internal.KnowledgeExtractionBaseTypes import (
    KnowledgeTableData,
)


class TestKnowledgeTableData:
    """Test suite for KnowledgeTableData table formatting methods."""

    def test_to_markdown_basic(self) -> None:
        """Test basic markdown table generation."""
        table_data = KnowledgeTableData(
            page=1,
            data=[
                ["Header 1", "Header 2", "Header 3"],
                ["Cell 1", "Cell 2", "Cell 3"],
                ["Cell 4", "Cell 5", "Cell 6"],
            ],
            rows=3,
            columns=3,
        )

        result = table_data.to_markdown()
        expected = (
            "| Header 1 | Header 2 | Header 3 |\n"
            "| --- | --- | --- |\n"
            "| Cell 1 | Cell 2 | Cell 3 |\n"
            "| Cell 4 | Cell 5 | Cell 6 |"
        )

        assert result == expected

    def test_to_markdown_with_none_values(self) -> None:
        """Test markdown table generation with None values."""
        table_data = KnowledgeTableData(
            page=1,
            data=[
                ["Name", "Age", "City"],
                ["John", None, "NYC"],
                [None, "25", None],
            ],
            rows=3,
            columns=3,
        )

        result = table_data.to_markdown()
        expected = "| Name | Age | City |\n" "| --- | --- | --- |\n" "| John |  | NYC |\n" "|  | 25 |  |"

        assert result == expected

    def test_to_markdown_empty(self) -> None:
        """Test markdown table generation with empty data."""
        table_data = KnowledgeTableData(
            page=1,
            data=[],
            rows=0,
            columns=0,
        )

        result = table_data.to_markdown()
        assert result == ""

    def test_to_markdown_uneven_columns(self) -> None:
        """Test markdown table with uneven column counts."""
        table_data = KnowledgeTableData(
            page=1,
            data=[
                ["A", "B", "C"],
                ["1", "2"],  # Missing one column
                ["X", "Y", "Z", "Extra"],  # Extra column
            ],
            rows=3,
            columns=3,
        )

        result = table_data.to_markdown()
        expected = "| A | B | C |\n| --- | --- | --- |\n| 1 | 2 |  |\n| X | Y | Z |"

        assert result == expected

    def test_to_ascii_table_basic(self) -> None:
        """Test basic ASCII table generation with clean whitespace format."""
        table_data = KnowledgeTableData(
            page=1,
            data=[
                ["Name", "Age", "City"],
                ["Alice", "30", "London"],
                ["Bob", "25", "NYC"],
            ],
            rows=3,
            columns=3,
        )

        result = table_data.to_ascii_table()
        lines = result.split("\n")

        # Check clean whitespace structure
        assert lines[0] == "Name   Age  City  "  # Header
        assert lines[1] == ""  # Blank line after header
        assert lines[2] == "Alice  30   London"  # Data row 1
        assert lines[3] == "Bob    25   NYC   "  # Data row 2

    def test_to_ascii_table_with_none_values(self) -> None:
        """Test ASCII table generation with None values."""
        table_data = KnowledgeTableData(
            page=1,
            data=[
                ["Col1", "Col2"],
                [None, "Value"],
                ["Data", None],
            ],
            rows=3,
            columns=2,
        )

        result = table_data.to_ascii_table()
        lines = result.split("\n")

        # Check clean format with None becoming empty strings
        assert lines[0] == "Col1  Col2 "  # Header
        assert lines[1] == ""  # Blank line
        assert lines[2] == "      Value"  # None becomes empty spaces
        assert lines[3] == "Data       "  # None becomes empty spaces

    def test_both_formats_consistency(self) -> None:
        """Test that both formats handle the same data consistently."""
        table_data = KnowledgeTableData(
            page=1,
            data=[
                ["Product", "Price", "Stock"],
                ["Apple", "1.99", "50"],
                ["Banana", "0.99", None],
                ["Orange", None, "30"],
            ],
            rows=4,
            columns=3,
        )

        markdown = table_data.to_markdown()
        ascii_table = table_data.to_ascii_table()

        # Both should handle the data (not be empty)
        assert len(markdown) > 0
        assert len(ascii_table) > 0

        # Both should have the same number of data rows
        markdown_rows = markdown.count("\n") + 1
        ascii_rows = ascii_table.count("\n") + 1

        # Markdown has header + separator + 3 data rows = 5 lines
        assert markdown_rows == 5
        # Clean ASCII has header + blank line + 3 data rows = 5 lines
        assert ascii_rows == 5

    def test_to_html_table_basic(self) -> None:
        """Test basic HTML table generation."""
        table_data = KnowledgeTableData(
            page=1,
            data=[
                ["Header 1", "Header 2", "Header 3"],
                ["Cell 1", "Cell 2", "Cell 3"],
                ["Cell 4", "Cell 5", "Cell 6"],
            ],
            rows=3,
            columns=3,
        )

        result = table_data.to_html_table()

        # Check for table tags
        assert '<table border="1"' in result
        assert "</table>" in result
        assert "<thead>" in result
        assert "</thead>" in result
        assert "<tbody>" in result
        assert "</tbody>" in result

        # Check headers
        assert "<th>Header 1</th>" in result
        assert "<th>Header 2</th>" in result
        assert "<th>Header 3</th>" in result

        # Check data cells
        assert "<td>Cell 1</td>" in result
        assert "<td>Cell 2</td>" in result
        assert "<td>Cell 6</td>" in result

    def test_to_html_table_with_none_values(self) -> None:
        """Test HTML table generation with None values."""
        table_data = KnowledgeTableData(
            page=1,
            data=[
                ["Name", "Age", "City"],
                ["John", None, "NYC"],
                [None, "25", None],
            ],
            rows=3,
            columns=3,
        )

        result = table_data.to_html_table()

        # Check that None values become empty strings
        assert "<td>John</td>" in result
        assert "<td></td>" in result  # None becomes empty
        assert "<td>NYC</td>" in result
        assert "<td>25</td>" in result

    def test_to_html_table_with_html_entities(self) -> None:
        """Test HTML table properly escapes HTML entities."""
        table_data = KnowledgeTableData(
            page=1,
            data=[
                ["Column", "Value"],
                ["<script>", "alert('xss')"],
                ["A & B", "C > D"],
            ],
            rows=3,
            columns=2,
        )

        result = table_data.to_html_table()

        # Check that HTML entities are properly escaped
        assert "&lt;script&gt;" in result
        assert "alert(&#39;xss&#39;)" not in result  # Should be escaped
        assert "A &amp; B" in result
        assert "C &gt; D" in result

    def test_to_html_table_with_nested_table(self) -> None:
        """Test HTML table with nested table in cell."""
        # Create a nested table HTML string
        nested_table = "<table><tr><td>Nested 1</td><td>Nested 2</td></tr></table>"

        table_data = KnowledgeTableData(
            page=1,
            data=[
                ["Parent Column", "Nested Table Column"],
                ["Regular Cell", nested_table],
                ["Another Cell", "Regular Value"],
            ],
            rows=3,
            columns=2,
        )

        result = table_data.to_html_table()

        # Check that nested table is preserved (not escaped)
        assert "<td>Regular Cell</td>" in result
        assert f"<td>{nested_table}</td>" in result
        assert "<td>Another Cell</td>" in result

    def test_to_html_table_empty(self) -> None:
        """Test HTML table generation with empty data."""
        table_data = KnowledgeTableData(
            page=1,
            data=[],
            rows=0,
            columns=0,
        )

        result = table_data.to_html_table()
        assert result == ""

    def test_to_html_table_uneven_columns(self) -> None:
        """Test HTML table with uneven column counts."""
        table_data = KnowledgeTableData(
            page=1,
            data=[
                ["A", "B", "C"],
                ["1", "2"],  # Missing one column
                ["X", "Y", "Z", "Extra"],  # Extra column (will be trimmed)
            ],
            rows=3,
            columns=3,
        )

        result = table_data.to_html_table()

        # Check structure
        assert "<th>A</th>" in result
        assert "<th>B</th>" in result
        assert "<th>C</th>" in result

        # Second row should have empty cell for missing column
        assert "<td>1</td>" in result
        assert "<td>2</td>" in result
        # Check that there's an empty cell in the row with "1"
        assert "<td>1</td>\n      <td>2</td>\n      <td></td>" in result

        # Third row extra column should be trimmed
        assert "<td>X</td>" in result
        assert "<td>Y</td>" in result
        assert "<td>Z</td>" in result
        assert "<td>Extra</td>" not in result
