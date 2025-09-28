"""
Citation formatter for programmatic citation style transformation.

This module handles the transformation of citation markers in text
according to different citation styles, ensuring consistency and
avoiding reliance on LLM interpretation.
"""

import re
from typing import Dict, List, Optional, Tuple

from blockether_catalyst.knowledge.answering.AnswerProviderAgent import Citation


class CitationFormatter:
    """Handles programmatic transformation of citation markers according to style."""

    STYLE_INLINE_NUMERIC = "inline_numeric"
    STYLE_FOOTNOTE = "footnote"
    STYLE_AUTHOR_DATE = "author_date"
    STYLE_SUPERSCRIPT = "superscript"

    @staticmethod
    def transform_citations_in_text(
        text: str,
        citations: List[Citation],
        from_style: str = "inline_numeric",
        to_style: str = "inline_numeric"
    ) -> Tuple[str, List[Citation]]:
        """
        Transform citation markers in text from one style to another.

        Args:
            text: The text containing citation markers
            citations: List of Citation objects
            from_style: Current citation style in the text
            to_style: Target citation style

        Returns:
            Tuple of (transformed_text, reordered_citations)
        """
        if from_style == to_style:
            return text, citations

        # Create citation lookup by index
        citation_map = {i + 1: cite for i, cite in enumerate(citations)}

        # Find all citation markers based on source style
        pattern = CitationFormatter._get_pattern_for_style(from_style)
        matches = list(re.finditer(pattern, text))

        if not matches:
            return text, citations

        # Transform each citation marker
        transformed_text = text
        offset = 0

        for match in matches:
            citation_num = CitationFormatter._extract_citation_number(match, from_style)
            if citation_num and citation_num in citation_map:
                citation = citation_map[citation_num]

                # Generate replacement based on target style
                replacement = CitationFormatter._format_citation_marker(
                    citation, citation_num, to_style
                )

                # Apply replacement with offset adjustment
                start = match.start() + offset
                end = match.end() + offset
                transformed_text = (
                    transformed_text[:start] +
                    replacement +
                    transformed_text[end:]
                )
                offset += len(replacement) - (match.end() - match.start())

        return transformed_text, citations

    @staticmethod
    def _get_pattern_for_style(style: str) -> str:
        """Get regex pattern for finding citations in the given style."""
        patterns = {
            CitationFormatter.STYLE_INLINE_NUMERIC: r'\[(\d+)\]',
            CitationFormatter.STYLE_FOOTNOTE: r'\[\^(\d+)\]',  # Fixed to match [^1] format
            CitationFormatter.STYLE_SUPERSCRIPT: r'<sup>(\d+)</sup>',
            CitationFormatter.STYLE_AUTHOR_DATE: r'\(([^,]+),\s*(\d{4})\)',
        }
        return patterns.get(style, r'\[(\d+)\]')

    @staticmethod
    def _extract_citation_number(match: re.Match, style: str) -> Optional[int]:
        """Extract citation number from regex match based on style."""
        try:
            if style in [CitationFormatter.STYLE_INLINE_NUMERIC,
                        CitationFormatter.STYLE_FOOTNOTE,
                        CitationFormatter.STYLE_SUPERSCRIPT]:
                return int(match.group(1))
            elif style == CitationFormatter.STYLE_AUTHOR_DATE:
                # For author-date, we'd need to map back to citation index
                # This is more complex and would require additional tracking
                return None
        except (ValueError, IndexError):
            return None
        return None

    @staticmethod
    def _format_citation_marker(citation: Citation, number: int, style: str) -> str:
        """Format a citation marker according to the target style."""
        if style == CitationFormatter.STYLE_INLINE_NUMERIC:
            return f"[{number}]"
        elif style == CitationFormatter.STYLE_FOOTNOTE:
            return f"[^{number}]"
        elif style == CitationFormatter.STYLE_SUPERSCRIPT:
            return f"<sup>{number}</sup>"
        elif style == CitationFormatter.STYLE_AUTHOR_DATE:
            author = citation.author or "Unknown"
            # Handle "Last, First" format and extract last name
            if "," in author:
                last_name = author.split(",")[0].strip()
            else:
                # Extract last name if full name provided
                author_parts = author.split()
                last_name = author_parts[-1] if author_parts else "Unknown"
            year = CitationFormatter._extract_year(citation.publication_date)
            return f"({last_name}, {year})"
        else:
            return f"[{number}]"

    @staticmethod
    def _extract_year(date_str: Optional[str]) -> str:
        """Extract year from date string."""
        if not date_str:
            return "n.d."

        # Try to find a 4-digit year
        year_match = re.search(r'\b(19|20)\d{2}\b', date_str)
        if year_match:
            return year_match.group(0)
        return "n.d."

    @staticmethod
    def get_style_instructions(style: str) -> str:
        """
        Get detailed instructions for a citation style.

        This replaces the unused get_citation_style_description() function
        with a properly integrated version.
        """
        instructions = {
            CitationFormatter.STYLE_INLINE_NUMERIC: """
                Use inline numeric citations in square brackets [1], [2], etc.
                Multiple citations should be separated: [1, 2, 3] or ranges: [1-3].
                Place citations immediately after the relevant statement.
                Example: "The system processes data efficiently [1]."
            """,
            CitationFormatter.STYLE_FOOTNOTE: """
                Use footnote-style citations with [^1], [^2], etc.
                Citations appear as clickable footnotes at the bottom of sections.
                Example: "This finding is significant[^1] for the field."
            """,
            CitationFormatter.STYLE_SUPERSCRIPT: """
                Use superscript numbers for citations: <sup>1</sup>, <sup>2</sup>, etc.
                Citations appear as small raised numbers.
                Example: "Recent studies<sup>1</sup> show improvement."
            """,
            CitationFormatter.STYLE_AUTHOR_DATE: """
                Use author-date format: (Author, Year).
                Include author's last name and publication year in parentheses.
                Example: "As noted in previous work (Smith, 2023)."
            """
        }
        return instructions.get(style, instructions[CitationFormatter.STYLE_INLINE_NUMERIC])

    @staticmethod
    def validate_citation_consistency(text: str, style: str) -> Dict[str, any]:
        """
        Validate that all citations in text follow the specified style.

        Returns:
            Dict with 'valid' (bool), 'issues' (list of issues found)
        """
        pattern = CitationFormatter._get_pattern_for_style(style)
        citations_found = re.findall(pattern, text)

        # Check for other style patterns that shouldn't be present
        other_styles = [
            s for s in [
                CitationFormatter.STYLE_INLINE_NUMERIC,
                CitationFormatter.STYLE_FOOTNOTE,
                CitationFormatter.STYLE_SUPERSCRIPT,
                CitationFormatter.STYLE_AUTHOR_DATE
            ] if s != style
        ]

        issues = []
        for other_style in other_styles:
            other_pattern = CitationFormatter._get_pattern_for_style(other_style)
            if re.search(other_pattern, text):
                issues.append(f"Found {other_style} citations when expecting {style}")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "citations_found": len(citations_found)
        }