"""
Markdown generation utilities for beautiful knowledge extraction reports.

This module provides rich markdown generation with tables, summaries,
and comprehensive information display using the python-markdown library.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence
from urllib.parse import quote

from rich.console import Console
from rich.markdown import Markdown as RichMarkdown
from rich.table import Table


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
    def create_term_details_section(terms: Dict[str, Any], max_terms: int = 20) -> str:
        """Create a detailed section for terms with rich information.

        Args:
            terms: Dictionary of term data
            max_terms: Maximum number of terms to display

        Returns:
            Formatted markdown section with term details
        """
        lines = ["## 📚 Term Analysis", ""]

        # Separate acronyms and keywords
        acronyms = [(k, v) for k, v in terms.items() if v.type == "acronym"]
        keywords = [(k, v) for k, v in terms.items() if v.type == "keyword"]

        # Sort by frequency
        acronyms.sort(key=lambda x: x[1].total, reverse=True)
        keywords.sort(key=lambda x: x[1].total, reverse=True)

        # Acronyms section
        if acronyms:
            lines.append("### 🔤 Acronyms and Abbreviations")
            lines.append("")

            # Create detailed acronym entries
            for term_name, term_data in acronyms[:max_terms]:
                lines.append(f"#### {term_name}")

                # Full form
                if hasattr(term_data, "full_form") and term_data.full_form != term_name:
                    lines.append(f"**Full Form:** {term_data.full_form}")

                # Meaning
                if hasattr(term_data, "meaning") and term_data.meaning:
                    lines.append(f"**Definition:** {term_data.meaning}")

                # Statistics
                stats = []
                stats.append(f"Occurrences: {term_data.total}")

                if hasattr(term_data, "occurrences"):
                    doc_count = len(set(occ.document_id for occ in term_data.occurrences))
                    stats.append(f"Documents: {doc_count}")

                    # Page distribution
                    pages = [occ.page for occ in term_data.occurrences]
                    if pages:
                        stats.append(f"Pages: {min(pages)}-{max(pages)}")

                lines.append(f"**Statistics:** {' | '.join(stats)}")

                # Co-occurring terms
                if hasattr(term_data, "cooccurrences") and term_data.cooccurrences:
                    top_cooccur = term_data.cooccurrences[:5]
                    cooccur_list = [f"{c.term} ({c.score:.2f})" for c in top_cooccur]
                    lines.append(f"**Related Terms:** {', '.join(cooccur_list)}")

                # Links to other terms
                if hasattr(term_data, "links") and term_data.links:
                    top_links = term_data.links[:3]
                    link_list = [f"{link.link_to} ({link.score:.2f})" for link in top_links]
                    lines.append(f"**Linked Terms:** {', '.join(link_list)}")

                lines.append("")

        # Keywords section
        if keywords:
            lines.append("### 🔑 Key Terms and Concepts")
            lines.append("")

            # Create keyword table
            headers = ["Term", "Occurrences", "Documents", "Meaning Preview"]
            rows = []

            for term_name, term_data in keywords[:max_terms]:
                doc_count = (
                    len(set(occ.document_id for occ in term_data.occurrences))
                    if hasattr(term_data, "occurrences")
                    else 0
                )

                # Truncate meaning for preview
                meaning = ""
                if hasattr(term_data, "meaning") and term_data.meaning:
                    meaning = term_data.meaning[:100]
                    if len(term_data.meaning) > 100:
                        meaning += "..."

                rows.append([term_name, term_data.total, doc_count, meaning])

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

        stats_card = MarkdownGenerator.create_statistics_card(
            "📊 Document Collection Overview",
            {
                "Total Documents": len(documents),
                "Total Pages": total_pages,
                "Total Chunks": total_chunks,
                "Total Images": total_images,
                "Total Tables": total_tables,
                "Avg Pages/Document": total_pages / len(documents),
                "Avg Chunks/Document": total_chunks / len(documents),
            },
        )
        lines.append(stats_card)

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

            rows.append(
                [
                    doc.document_filename[:30],  # Truncate long names
                    getattr(doc, "title", "-")[:20],
                    getattr(doc, "author", "-")[:15],
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

            for doc in documents.values():
                if hasattr(doc, "chunks"):
                    for chunk in doc.chunks:
                        # Content types (text, image, table)
                        if hasattr(chunk, "content_types"):
                            for ct in chunk.content_types:
                                content_type_counts[ct] = content_type_counts.get(ct, 0) + 1

                        # Semantic types (summary, rule, example, etc.)
                        if hasattr(chunk, "semantic_types"):
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
    def create_knowledge_graph_section(terms: Dict[str, Any]) -> str:
        """Create a section describing term relationships and knowledge graph.

        Args:
            terms: Dictionary of term data with links and cooccurrences

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
                # Get top co-occurring terms
                top_cooccur = [c.term for c in term_data.cooccurrences[:3] if c.score > 0.5]
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
    def create_extraction_report(linked_knowledge: Any, include_all_sections: bool = True) -> str:
        """Create a comprehensive extraction report with all available information.

        Args:
            linked_knowledge: LinkedKnowledge object with extraction data
            include_all_sections: Whether to include all detailed sections

        Returns:
            Complete formatted markdown report
        """
        lines = []

        # Header
        lines.append("# 🚀 Catalyst Knowledge Extraction Report")
        lines.append("")
        lines.append(f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
        lines.append("")
        lines.append("---")
        lines.append("")

        # Executive Summary
        lines.append("## 📌 Executive Summary")
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
        lines.append("## ⏱️ Processing Information")
        lines.append("")

        if hasattr(linked_knowledge, "extraction_timestamp"):
            lines.append(f"- **Extraction Completed:** {linked_knowledge.extraction_timestamp}")

        if hasattr(linked_knowledge, "processing_duration"):
            lines.append(f"- **Processing Duration:** {linked_knowledge.processing_duration}")

        if hasattr(linked_knowledge, "settings") and linked_knowledge.settings:
            lines.append("- **Extraction Settings:**")
            for key, value in linked_knowledge.settings.items():
                lines.append(f"  - {key}: {value}")

        lines.append("")
        lines.append("---")
        lines.append("")

        # Footer
        lines.append("## 📝 Notes")
        lines.append("")
        lines.append("This report provides a comprehensive overview of the knowledge extraction results.")
        lines.append("All statistics and relationships are automatically derived from the source documents.")
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("*Generated by Catalyst Knowledge Extraction System v1.0*")
        lines.append("*Powered by Advanced NLP and Knowledge Graph Technology*")

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
