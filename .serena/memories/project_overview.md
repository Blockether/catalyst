# Catalyst Project Overview

## Purpose
Catalyst is an enterprise-grade toolkit for building LLM-powered document processing applications. It turns complex documents into queryable knowledge systems designed specifically for regulated industries where accuracy and source attribution are critical.

## Key Features
- **Hybrid Intelligence**: Combines vector search + keyword extraction + relationship mapping
- **Knowledge Graphs**: Extracts acronyms and builds relationships between terms
- **Structure Preservation**: Tables, images, and document hierarchy stay intact
- **Source Attribution**: Every answer includes exact page numbers and document sections
- **Zero Dependencies**: Fully self-contained with embedded models (~32MB)
- **Offline Capable**: No external API calls required
- **Async Processing**: Built on ASGI for high-performance document pipelines

## Target Use Cases
- Regulatory compliance document analysis
- Legal due diligence across contracts
- Financial analysis with risk metrics
- Technical documentation understanding
- Audit support with complete evidence chains

## Tech Stack
- **Language**: Python 3.12+
- **Package Management**: uv (NOT pip)
- **Core Dependencies**:
  - langchain-core>=0.3.75
  - model2vec>=0.6.0 (embedded model)
  - numpy, tenacity, trio, sqlalchemy
- **Optional Dependencies**:
  - API: fastapi, fastmcp, agno, rapidfuzz
  - Extraction: pdfplumber, scikit-learn, pypdf, pyoxipng
- **Testing**: pytest, anyio (preferred over asyncio)
- **Development**: mypy, ruff, black, isort, poethepoet
