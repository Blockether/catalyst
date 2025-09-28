"""
Markdown generation utilities for beautiful knowledge extraction reports.

This module provides rich markdown generation with tables, summaries,
and comprehensive information display using the python-markdown library.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence
from urllib.parse import quote

from rich.console import Console
from rich.markdown import Markdown as RichMarkdown
from rich.table import Table

from blockether_catalyst.knowledge.KnowledgeTypes import LinkedKnowledge


class MarkdownGenerator:
    """Generate beautiful markdown reports for knowledge extraction."""

    @staticmethod
    def create_table(headers: List[str], rows: List[List[Any]], alignment: Optional[List[str]] = None) -> str:
        """Create a markdown table with proper formatting.

        Args:
            headers: List of column headers
            rows: List of row data (each row is a list)
            alignment: Optional list of alignments ('left', 'center', 'right')

        Returns:
            Formatted markdown table string
        """
        if not headers or not rows:
            return ""

        # Convert all values to strings and handle None
        str_rows = []
        for row in rows:
            str_row = []
            for val in row:
                if val is None or val == "N/A":
                    str_row.append("-")
                elif isinstance(val, (int, float)):
                    if isinstance(val, float):
                        str_row.append(f"{val:.2f}" if val % 1 else str(int(val)))
                    else:
                        str_row.append(f"{val:,}")
                else:
                    str_row.append(str(val))
            str_rows.append(str_row)

        # Calculate column widths
        col_widths = []
        for i, header in enumerate(headers):
            width = len(header)
            for row in str_rows:
                if i < len(row):
                    width = max(width, len(row[i]))
            col_widths.append(width)

        # Build header row
        header_row = "|"
        for i, header in enumerate(headers):
            header_row += f" {header.ljust(col_widths[i])} |"

        # Build separator row with alignment
        separator_row = "|"
        for i, width in enumerate(col_widths):
            if alignment and i < len(alignment):
                if alignment[i] == "center":
                    separator_row += f":{'-' * width}:|"
                elif alignment[i] == "right":
                    separator_row += f"{'-' * (width + 1)}:|"
                else:  # left or default
                    separator_row += f":{'-' * (width + 1)}|"
            else:
                separator_row += f"{'-' * (width + 2)}|"

        # Build data rows
        data_rows = []
        for row in str_rows:
            row_str = "|"
            for i, cell in enumerate(row):
                if i < len(col_widths):
                    row_str += f" {cell.ljust(col_widths[i])} |"
            data_rows.append(row_str)

        # Combine all parts
        table_lines = [header_row, separator_row] + data_rows
        return "\n".join(table_lines)

    @staticmethod
    def create_statistics_card(title: str, stats: Dict[str, Any]) -> str:
        """Create a formatted statistics card.

        Args:
            title: Card title
            stats: Dictionary of statistic name to value

        Returns:
            Formatted markdown card
        """
        lines = [f"### {title}", ""]

        for key, value in stats.items():
            if isinstance(value, (int, float)):
                if isinstance(value, float):
                    formatted = f"{value:.2f}" if value % 1 else str(int(value))
                else:
                    formatted = f"{value:,}"
                lines.append(f"- **{key}:** {formatted}")
            elif isinstance(value, list):
                lines.append(f"- **{key}:** {len(value):,} items")
            else:
                lines.append(f"- **{key}:** {value}")

        lines.append("")
        return "\n".join(lines)

    @staticmethod
    def create_term_details_section(
        terms: Dict[str, Any], max_terms: int = 20, max_display_cooccurrences: int = 5
    ) -> str:
        """Create a detailed section for terms with rich information.

        Args:
            terms: Dictionary of term data
            max_terms: Maximum number of terms to display
            max_display_cooccurrences: Maximum number of co-occurrences to display per term

        Returns:
            Formatted markdown section with term details
        """
        lines = ["## 📚 All Terms and Concepts", ""]

        # Combine all terms and sort by frequency
        all_terms = [(k, v) for k, v in terms.items()]
        all_terms.sort(key=lambda x: x[1].total, reverse=True)

        if all_terms:
            # Create comprehensive table with all information
            headers = [
                "Term",
                "Type",
                "Count",
                "Docs",
                "Definition/Meaning",
                "Related Terms",
            ]
            rows = []

            for term_name, term_data in all_terms[:max_terms]:
                # Determine type and full form
                term_type = term_data.type
                display_name = term_name

                # For acronyms, include full form in the name
                if term_type == "acronym" and term_data.full_form and term_data.full_form != term_name:
                    display_name = f"{term_name} ({term_data.full_form})"

                # Document count
                doc_count = (
                    len(set(occ.document_id for occ in term_data.occurrences))
                    if hasattr(term_data, "occurrences")
                    else 0
                )

                # Format meaning with smart truncation - break at sentence end
                meaning = ""
                if hasattr(term_data, "meaning") and term_data.meaning:
                    if len(term_data.meaning) > 150:
                        # Find the last sentence boundary before 150 chars
                        cutoff = 150
                        for punct in [". ", "! ", "? ", "; "]:
                            last_punct = term_data.meaning[:200].rfind(punct)
                            if last_punct > 100:  # At least show 100 chars
                                cutoff = last_punct + 1  # Include the punctuation
                                break

                        preview = term_data.meaning[:cutoff]
                        rest = term_data.meaning[cutoff:].strip()

                        if rest:  # Only add details if there's remaining content
                            meaning = f"{preview} <details><summary>Show more</summary>{rest}</details>"
                        else:
                            meaning = preview
                    else:
                        meaning = term_data.meaning

                # Combine related terms (cooccurrences and links)
                related = []

                # Add co-occurring terms (using configurable limit)
                if hasattr(term_data, "cooccurrences") and term_data.cooccurrences:
                    top_cooccur = term_data.cooccurrences[:max_display_cooccurrences]
                    for c in top_cooccur:
                        related.append(f"{c.term}")

                # Add linked terms
                if hasattr(term_data, "links") and term_data.links:
                    top_links = term_data.links[:3]
                    for link in top_links:
                        if link.link_to not in [
                            c.term
                            for c in (
                                term_data.cooccurrences[:max_display_cooccurrences]
                                if hasattr(term_data, "cooccurrences")
                                else []
                            )
                        ]:
                            related.append(f"{link.link_to}")

                # Format related terms - only truncate if more than 5 with at least 3 more hidden
                related_str = "-"
                if related:
                    if len(related) <= 5:
                        related_str = ", ".join(related)
                    elif len(related) > 8:  # Show 5, hide rest only if 3+ more (5+3=8)
                        visible = related[:5]
                        hidden = related[5:]
                        related_str = ", ".join(visible)
                        related_str += f" <details><summary>+{len(hidden)} more</summary>{', '.join(hidden)}</details>"
                    else:
                        # If 6-8 terms total, just show them all
                        related_str = ", ".join(related)

                rows.append(
                    [
                        display_name,
                        term_type.capitalize(),
                        term_data.total,
                        doc_count,
                        meaning or "-",
                        related_str,
                    ]
                )

            lines.append(MarkdownGenerator.create_table(headers, rows))
            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def create_document_analysis_section(documents: Dict[str, Any], include_chunks: bool = True) -> str:
        """Create comprehensive document analysis section.

        Args:
            documents: Dictionary of document data
            include_chunks: Whether to include chunk analysis

        Returns:
            Formatted markdown section
        """
        lines = ["## 📄 Document Analysis", ""]

        if not documents:
            lines.append("*No documents processed*")
            return "\n".join(lines)

        # Overall document statistics
        total_pages = sum(doc.total_pages for doc in documents.values())
        total_chunks = sum(doc.total_chunks for doc in documents.values())
        total_images = sum(doc.total_images for doc in documents.values())
        total_tables = sum(doc.total_tables for doc in documents.values())

        # Direct statistics without extra heading
        lines.append(f"- **Total Documents**: {len(documents):,}")
        lines.append(f"- **Total Pages**: {total_pages:,}")
        lines.append(f"- **Total Chunks**: {total_chunks:,}")
        lines.append(f"- **Total Images**: {total_images:,}")
        lines.append(f"- **Total Tables**: {total_tables:,}")
        lines.append(f"- **Avg Pages/Document**: {total_pages / len(documents):.1f}")
        lines.append(f"- **Avg Chunks/Document**: {total_chunks / len(documents):.1f}")
        lines.append("")

        # Detailed document table
        lines.append("### 📋 Document Details")
        lines.append("")

        headers = [
            "Document",
            "Title",
            "Author",
            "Date",
            "Pages",
            "Chunks",
            "Terms",
            "Images",
            "Tables",
        ]
        rows = []

        for doc in sorted(documents.values(), key=lambda d: d.document_filename):
            total_terms = getattr(doc, "total_keywords", 0) + getattr(doc, "total_acronyms", 0)

            # Don't truncate document names or titles - they're important
            filename = doc.document_filename
            title = getattr(doc, "title", "-")

            author = getattr(doc, "author", "-")
            if len(author) > 15 and author != "-":
                author = f"{author[:15]}... <details><summary>Full author</summary>{author}</details>"

            rows.append(
                [
                    filename,
                    title,
                    author,
                    getattr(doc, "publication_date", "-")[:10],
                    doc.total_pages,
                    doc.total_chunks,
                    total_terms,
                    doc.total_images,
                    doc.total_tables,
                ]
            )

        lines.append(MarkdownGenerator.create_table(headers, rows))
        lines.append("")

        # Chunk analysis section
        if include_chunks and total_chunks > 0:
            lines.append("### 📝 Chunk Content Analysis")
            lines.append("")

            # Collect chunk statistics across all documents
            content_type_counts: Dict[str, int] = {}
            semantic_type_counts: Dict[str, int] = {}

            # Access chunks directly from LinkedKnowledge.chunks
            for chunk in linked_knowledge.chunks.values():
                # Content types (text, image, table)
                for ct in chunk.content_types:
                    content_type_counts[ct] = content_type_counts.get(ct, 0) + 1

                # Semantic types (summary, rule, example, etc.)
                for st in chunk.semantic_types:
                    semantic_type_counts[st] = semantic_type_counts.get(st, 0) + 1

            # Create content distribution table
            if content_type_counts:
                lines.append("**Content Type Distribution:**")
                lines.append("")

                headers = ["Content Type", "Count", "Percentage"]
                rows = []
                for ct, count in sorted(content_type_counts.items(), key=lambda x: x[1], reverse=True):
                    percentage = (count / total_chunks) * 100
                    rows.append([ct.capitalize(), count, f"{percentage:.1f}%"])

                lines.append(MarkdownGenerator.create_table(headers, rows))
                lines.append("")

            # Create semantic type distribution
            if semantic_type_counts:
                lines.append("**Semantic Classification Distribution:**")
                lines.append("")

                headers = ["Semantic Type", "Count", "Percentage"]
                rows = []
                for st, count in sorted(semantic_type_counts.items(), key=lambda x: x[1], reverse=True):
                    percentage = (count / total_chunks) * 100
                    rows.append([st.replace("_", " ").title(), count, f"{percentage:.1f}%"])

                lines.append(MarkdownGenerator.create_table(headers, rows))
                lines.append("")

        return "\n".join(lines)

    @staticmethod
    def create_knowledge_graph_section(terms: Dict[str, Any], max_display_cooccurrences: int = 3) -> str:
        """Create a section describing term relationships and knowledge graph.

        Args:
            terms: Dictionary of term data with links and cooccurrences
            max_display_cooccurrences: Maximum number of co-occurrences to consider for clustering

        Returns:
            Formatted markdown section
        """
        lines = ["## 🕸️ Knowledge Graph Insights", ""]

        # Calculate graph statistics
        total_links = 0
        total_cooccurrences = 0
        most_connected_terms = []

        for term_name, term_data in terms.items():
            link_count = len(term_data.links) if hasattr(term_data, "links") else 0
            cooccur_count = len(term_data.cooccurrences) if hasattr(term_data, "cooccurrences") else 0

            total_links += link_count
            total_cooccurrences += cooccur_count

            connectivity = link_count + cooccur_count
            if connectivity > 0:
                most_connected_terms.append((term_name, connectivity, link_count, cooccur_count))

        # Sort by connectivity
        most_connected_terms.sort(key=lambda x: x[1], reverse=True)

        # Graph overview
        lines.append("### 🌐 Graph Statistics")
        lines.append("")
        lines.append(f"- **Total Terms (Nodes):** {len(terms):,}")
        lines.append(f"- **Total Links (Edges):** {total_links:,}")
        lines.append(f"- **Total Co-occurrences:** {total_cooccurrences:,}")
        lines.append(f"- **Average Connections per Term:** {(total_links + total_cooccurrences) / len(terms):.2f}")
        lines.append("")

        # Most connected terms
        if most_connected_terms:
            lines.append("### 🔗 Most Connected Terms")
            lines.append("")
            lines.append("Terms with the highest number of relationships:")
            lines.append("")

            headers = ["Term", "Total Connections", "Direct Links", "Co-occurrences"]
            rows = []

            for term, total, links, cooccur in most_connected_terms[:15]:
                rows.append([term, total, links, cooccur])

            lines.append(MarkdownGenerator.create_table(headers, rows))
            lines.append("")

        # Term clusters (groups of highly connected terms)
        lines.append("### 🎯 Term Clusters")
        lines.append("")
        lines.append("*Identifying groups of related terms based on co-occurrence patterns:*")
        lines.append("")

        # Find clusters (simplified - just group by high co-occurrence)
        clusters: Dict[str, List[str]] = {}
        for term_name, term_data in terms.items():
            if hasattr(term_data, "cooccurrences") and term_data.cooccurrences:
                # Get top co-occurring terms (using parameter instead of hard-coded value)
                top_cooccur = [c.term for c in term_data.cooccurrences[:max_display_cooccurrences] if c.score > 0.5]
                if top_cooccur:
                    cluster_key = "-".join(sorted(top_cooccur[:2]))  # Use first 2 as cluster key (as string)
                    if cluster_key not in clusters:
                        clusters[cluster_key] = []
                    clusters[cluster_key].append(term_name)

        # Display top clusters
        cluster_list = [(k, v) for k, v in clusters.items() if len(v) >= 3]
        cluster_list.sort(key=lambda x: len(x[1]), reverse=True)

        for i, cluster_data in enumerate(cluster_list[:5], 1):
            _, cluster_terms = cluster_data
            lines.append(f"**Cluster {i}:** {', '.join(cluster_terms[:10])}")
            if len(cluster_terms) > 10:
                lines.append(f"  *... and {len(cluster_terms) - 10} more terms*")

        lines.append("")

        return "\n".join(lines)

    @staticmethod
    def create_extraction_report(linked_knowledge: LinkedKnowledge, include_all_sections: bool = True) -> str:
        """Create a comprehensive extraction report with all available information.

        Args:
            linked_knowledge: LinkedKnowledge object with extraction data
            include_all_sections: Whether to include all detailed sections

        Returns:
            Complete formatted markdown report
        """
        lines = []

        # Header
        lines.append("---")
        lines.append("## 🚀 Knowledge Extraction Report")
        lines.append("")
        lines.append(f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
        lines.append("")
        lines.append("---")
        lines.append("")

        # Executive Summary
        lines.append("### 📌 Executive Summary")
        lines.append("")

        summary_stats = {
            "Documents Processed": len(linked_knowledge.documents),
            "Total Pages Analyzed": sum(doc.total_pages for doc in linked_knowledge.documents.values()),
            "Knowledge Chunks Created": linked_knowledge.total_chunks,
            "Terms Extracted": len(linked_knowledge.terms),
            "Acronyms Identified": linked_knowledge.total_acronyms,
            "Keywords Discovered": linked_knowledge.total_keywords,
            "Images Processed": linked_knowledge.total_images,
            "Tables Extracted": linked_knowledge.total_tables,
        }

        for key, value in summary_stats.items():
            lines.append(f"- **{key}:** {value:,}")

        lines.append("")
        lines.append("---")
        lines.append("")

        # Document Analysis
        if include_all_sections and linked_knowledge.documents:
            doc_section = MarkdownGenerator.create_document_analysis_section(
                linked_knowledge.documents, include_chunks=True
            )
            lines.append(doc_section)
            lines.append("---")
            lines.append("")

        # Term Analysis
        if include_all_sections and linked_knowledge.terms:
            term_section = MarkdownGenerator.create_term_details_section(linked_knowledge.terms, max_terms=30)
            lines.append(term_section)
            lines.append("---")
            lines.append("")

        # Knowledge Graph
        if include_all_sections and linked_knowledge.terms:
            graph_section = MarkdownGenerator.create_knowledge_graph_section(linked_knowledge.terms)
            lines.append(graph_section)
            lines.append("---")
            lines.append("")

        # Processing Information
        lines.append("### ⏱️ Processing Information")
        lines.append("")

        if hasattr(linked_knowledge, "extraction_timestamp") and linked_knowledge.extraction_timestamp:
            # Convert Unix timestamp to Europe/Vienna timezone
            try:
                from zoneinfo import ZoneInfo

                dt = datetime.fromtimestamp(linked_knowledge.extraction_timestamp, tz=timezone.utc)
                vienna_dt = dt.astimezone(ZoneInfo("Europe/Vienna"))
                formatted_timestamp = vienna_dt.strftime("%Y-%m-%d %H:%M:%S %Z")
            except ImportError:
                # Fallback for Python < 3.9
                dt = datetime.fromtimestamp(linked_knowledge.extraction_timestamp, tz=timezone.utc)
                formatted_timestamp = dt.strftime("%Y-%m-%d %H:%M:%S UTC")
            lines.append(f"- **Extraction Completed:** {formatted_timestamp}")

        if hasattr(linked_knowledge, "processing_duration") and linked_knowledge.processing_duration:
            # Format duration as hours/minutes/seconds
            duration = linked_knowledge.processing_duration
            hours = duration // 3600
            minutes = (duration % 3600) // 60
            seconds = duration % 60

            if hours > 0:
                duration_str = f"{hours}h {minutes}m {seconds}s"
            elif minutes > 0:
                duration_str = f"{minutes}m {seconds}s"
            else:
                duration_str = f"{seconds}s"

            lines.append(f"- **Processing Duration:** {duration_str}")

        lines.append("")
        lines.append("---")
        lines.append("")

        # Footer
        lines.append("##### 📝 Notes")
        lines.append("")
        lines.append("*All statistics and relationships are automatically derived from the source documents.*")
        lines.append("")
        lines.append("---")
        lines.append("")

        return "\n".join(lines)

    @staticmethod
    def render_to_console(markdown_text: str) -> None:
        """Render markdown to console using rich.

        Args:
            markdown_text: Markdown text to render
        """
        console = Console()
        md = RichMarkdown(markdown_text)
        console.print(md)

    @staticmethod
    def save_report(markdown_text: str, output_path: str, also_render: bool = False) -> None:
        """Save markdown report to file.

        Args:
            markdown_text: Markdown text to save
            output_path: Path to save the report
            also_render: Whether to also render to console
        """
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(markdown_text)

        if also_render:
            MarkdownGenerator.render_to_console(markdown_text)
