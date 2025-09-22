#!/usr/bin/env python3
"""Test script to verify markdown generation with rich information."""

from datetime import datetime
from pathlib import Path

from blockether_catalyst.knowledge.KnowledgeTypes import (
    DocumentMetadata,
    ImageMetadata,
    KnowledgeChunkWithTerms,
    KnowledgeTableData,
    LinkedKnowledge,
    NormalizedDocumentMetadata,
    TermCooccurrence,
    TermLink,
    TermOccurrence,
    TermWithLinks,
)
from blockether_catalyst.knowledge.MarkdownGenerator import MarkdownGenerator


def create_test_data() -> LinkedKnowledge:
    """Create comprehensive test data for markdown generation."""
    # Create test documents
    doc1 = NormalizedDocumentMetadata(
        document_id="doc1_hash_abc123",
        document_filename="technical_manual.pdf",
        document_path="/docs/technical_manual.pdf",
        title="Technical Manual for Advanced Systems",
        subject="Systems Engineering",
        author="Dr. Jane Smith",
        modification_date="2024-01-15T14:30:00Z",
        publication_date="2024-01-15",
        total_pages=150,
        total_chunks=45,
        total_terms=157,  # keywords + acronyms
        total_keywords=125,
        total_acronyms=32,
        total_images=18,
        total_tables=12,
    )

    doc2 = NormalizedDocumentMetadata(
        document_id="doc2_hash_def456",
        document_filename="api_reference.pdf",
        document_path="/docs/api_reference.pdf",
        title="API Reference Guide v2.0",
        subject="API Documentation",
        author="Engineering Team",
        modification_date="2024-02-20T10:00:00Z",
        publication_date="2024-02-20",
        total_pages=85,
        total_chunks=28,
        total_terms=113,  # keywords + acronyms
        total_keywords=95,
        total_acronyms=18,
        total_images=5,
        total_tables=22,
    )

    # Create terms with rich information
    term1 = TermWithLinks(
        term="API",
        type="acronym",
        full_form="Application Programming Interface",
        meaning="An Application Programming Interface is a set of rules and protocols that allows different software applications to communicate with each other. APIs define the methods and data structures that developers can use to interact with external software components, services, or libraries.",
        total=156,
        occurrences=[
            TermOccurrence(
                document_id="doc1_hash_abc123",
                document_name="technical_manual.pdf",
                page=12,
                chunk_index=5,
                total=8,
            ),
            TermOccurrence(
                document_id="doc2_hash_def456",
                document_name="api_reference.pdf",
                page=1,
                chunk_index=0,
                total=25,
            ),
        ],
        cooccurrences=[
            TermCooccurrence(term="REST", score=0.95),
            TermCooccurrence(term="HTTP", score=0.88),
            TermCooccurrence(term="JSON", score=0.82),
        ],
        links=[
            TermLink(link_from="API", link_to="REST", score=0.95),
            TermLink(link_from="API", link_to="authentication", score=0.75),
        ],
    )

    term2 = TermWithLinks(
        term="REST",
        type="acronym",
        full_form="Representational State Transfer",
        meaning="REST is an architectural style for designing networked applications. It relies on stateless, client-server communication protocols, typically HTTP. RESTful systems are characterized by how they are organized into resources, which are accessed using standard HTTP methods.",
        total=89,
        occurrences=[
            TermOccurrence(
                document_id="doc2_hash_def456",
                document_name="api_reference.pdf",
                page=3,
                chunk_index=2,
                total=15,
            ),
        ],
        cooccurrences=[
            TermCooccurrence(term="API", score=0.95),
            TermCooccurrence(term="HTTP", score=0.92),
        ],
        links=[
            TermLink(link_from="REST", link_to="API", score=0.95),
        ],
    )

    term3 = TermWithLinks(
        term="authentication",
        type="keyword",
        full_form="authentication",
        meaning="Authentication is the process of verifying the identity of a user, device, or system. It ensures that the entity attempting to access resources is who or what it claims to be, typically through credentials like passwords, tokens, or biometric data.",
        total=67,
        occurrences=[
            TermOccurrence(
                document_id="doc1_hash_abc123",
                document_name="technical_manual.pdf",
                page=45,
                chunk_index=18,
                total=12,
            ),
            TermOccurrence(
                document_id="doc2_hash_def456",
                document_name="api_reference.pdf",
                page=8,
                chunk_index=4,
                total=20,
            ),
        ],
        cooccurrences=[
            TermCooccurrence(term="authorization", score=0.88),
            TermCooccurrence(term="security", score=0.85),
            TermCooccurrence(term="token", score=0.79),
        ],
        links=[],
    )

    # Create chunks with terms
    chunk1 = KnowledgeChunkWithTerms(
        document_id="doc1_hash_abc123",
        document_name="technical_manual.pdf",
        doc_id="doc1_hash_abc123_p12_c5",
        index=5,
        text="The API provides comprehensive authentication mechanisms including OAuth 2.0 and JWT tokens.",
        page=12,
        content_types=["text"],
        semantic_types=["explanation"],
        terms={"API": 2, "authentication": 1},
    )

    chunk2 = KnowledgeChunkWithTerms(
        document_id="doc2_hash_def456",
        document_name="api_reference.pdf",
        doc_id="doc2_hash_def456_p3_c2",
        index=2,
        text="RESTful API design principles emphasize stateless communication and resource-based URLs.",
        page=3,
        content_types=["text"],
        semantic_types=["rule", "explanation"],
        terms={"REST": 1, "API": 1},
    )

    # Create linked knowledge with all data
    linked_knowledge = LinkedKnowledge(
        documents={"doc1_hash_abc123": doc1, "doc2_hash_def456": doc2},
        terms={
            "API": term1,
            "REST": term2,
            "authentication": term3,
        },
        chunks={
            "doc1_hash_abc123_p12_c5": chunk1,
            "doc2_hash_def456_p3_c2": chunk2,
        },
        total_chunks=73,
        total_keywords=220,
        total_acronyms=50,
        total_images=23,
        total_tables=34,
        extraction_timestamp=datetime.now().isoformat(),
        processing_duration="2m 35s",
    )

    return linked_knowledge


def main():
    """Test markdown generation with comprehensive data."""
    print("Creating test data...")
    linked_knowledge = create_test_data()

    print("\n" + "=" * 80)
    print("GENERATING DETAILED MARKDOWN REPORT")
    print("=" * 80 + "\n")

    # Generate detailed report using MarkdownGenerator
    detailed_report = MarkdownGenerator.create_extraction_report(
        linked_knowledge=linked_knowledge, include_all_sections=True
    )

    print(detailed_report)

    print("\n" + "=" * 80)
    print("GENERATING SIMPLE SUMMARY")
    print("=" * 80 + "\n")

    # Generate simple summary
    simple_summary = linked_knowledge.get_extraction_summary(detailed=False)

    print(simple_summary)

    # Save reports to files for inspection
    output_dir = Path("/tmp/markdown_test")
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / "detailed_report.md", "w") as f:
        f.write(detailed_report)

    with open(output_dir / "simple_summary.md", "w") as f:
        f.write(simple_summary)

    print("\n" + "=" * 80)
    print("Reports saved to /tmp/markdown_test/")
    print("=" * 80)


if __name__ == "__main__":
    main()
