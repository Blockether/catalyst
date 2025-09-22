<h2 align="center">
  <img width="35%" alt="Catalyst logo" src="docs/assets/logo.png"><br/>
</h2>

<div align="center">
Turn complex documents into queryable knowledge systems for regulated industries.
No hallucinations - just accurate answers with full source attribution
</div>

<div align="center">
  <h2>
    <a href="https://pypi.org/project/blockether-catalyst/"><img src="https://img.shields.io/pypi/v/blockether-catalyst?color=%23007ec6&label=pypi%20package" alt="Package version"></a>
    <a href="https://pypi.org/project/blockether-catalyst/"><img src="https://img.shields.io/pypi/pyversions/blockether-catalyst" alt="Supported Python versions"></a>
    <a href="https://github.com/Blockether/catalyst/blob/main/LICENSE">
      <img src="https://img.shields.io/badge/license-MIT-green" alt="License - MIT">
    </a>
  </h2>
</div>

<div align="center">
<h3>

[Why Catalyst?](#why-catalyst) • [Quick Start](#quick-start) • [Features](#features) • [Examples](#examples)

</h3>
</div>

## Why Catalyst?

### The Problem

**Enterprises are drowning in unstructured documents.** Legal contracts, compliance policies, technical specifications, audit reports, financial statements, research papers - massive document repositories containing critical business knowledge that remains locked away and unsearchable. This knowledge is scattered across departments, creating silos where finance can't access legal precedents, engineering can't find compliance requirements, and executives can't get a unified view of organizational commitments.

**Current solutions fail at enterprise scale.** Simple keyword search misses context. Vector search without proper preprocessing fails on real documents - tables lose their structure, acronyms aren't linked to their definitions, and cross-page references are lost. Generic AI tools hallucinate when precision matters most. And when auditors or regulators ask for evidence, you need the exact source with full context - not an AI's interpretation or an isolated paragraph missing the conditions and requirements around it.

### Why Vector Search Alone Isn't Enough

**Acronym Hell**: Financial and regulated documents are packed with acronyms (SLA, KPI, GDPR, SOX). Vector embeddings can't connect "Service Level Agreement" with "SLA" appearing 50 pages later. Your search for "service levels" returns nothing because the document only uses "SLA".

**Corporate Jargon**: Industry-specific terms that don't exist in general training data. "Counterparty risk", "regulatory capital", "compliance framework" - these need domain understanding, not just semantic similarity.

**Missing Context**: Vector search finds individual paragraphs but misses the bigger picture. You ask about "payment terms" and get a random paragraph mentioning "30 days" without the surrounding context about penalties, conditions, or exceptions.

**Evidence Requirements**: In regulated environments, answers need the full context - not just text citations. When compliance metrics are in a table, you need that table in the response. When a process diagram explains the workflow, you need that image. Legal and financial work requires complete evidence: the text, the tables, the charts - everything relevant to support the answer.

### Our Approach

**Hybrid Intelligence**: Vector search for semantic understanding + keyword extraction for precise terminology + relationship mapping to connect concepts.

**Knowledge Graphs**: We extract acronyms, validate their meanings with LLMs, and build relationships between terms. Now "SLA" searches also find "Service Level Agreement" content.

**Structure Preservation**: Tables, images, and document hierarchy stay intact. You get the actual compliance table, not a text description of it.

**Source Attribution**: Every answer includes exact page numbers and document sections. No hallucinations - if we don't know, we say so.

## Quick Start

Install Catalyst directly from GitHub (PyPI release coming soon):

```bash
# Using uv (recommended)
uv add "blockether-catalyst[extraction,api] @ git+https://github.com/Blockether/catalyst.git"

# Or using pip
pip install "blockether-catalyst[extraction,api] @ git+https://github.com/Blockether/catalyst.git"
```

## Features

### Core Capabilities

- **🔍 In-Memory Hybrid Search**: Platform-independent vector + keyword search that runs anywhere (Lambda, containers, edge) - no external dependencies
- **📦 Embedded Model**: Ships with a lightweight embedder (~32MB) directly in the library - no API calls, no latency, works offline
- **📄 PDF Intelligence**: Extracts text, tables, images, and maintains document structure
- **🧠 LLM Consensus**: Multiple validation passes ensure extraction accuracy
- **🔗 Knowledge Graphs**: Automatically links acronyms, terms, and concepts across documents
- **📊 Structure Preservation**: Tables and charts remain intact, not converted to text
- **🎯 Source Attribution**: Every answer includes exact page numbers and document sections
- **🚀 Async Processing**: Built on ASGI for high-performance document pipelines
- **🔧 Zero Dependencies**: Fully self-contained - no vector DBs, no external services, deploy anywhere

### Integrations

- **Web UI**: Ready-to-deploy document Q&A interface with HTMX
- **Workflow Engine**: Agno integration for complex document processing pipelines
- **Visualization**: Knowledge graph and chunk relationship visualizations
- **MCP Server**: Model Context Protocol support for AI assistants

## 🎯 When to Use Catalyst

Think of Catalyst as a forensic document investigator rather than a simple search engine. While traditional RAG systems are great for general Q&A, Catalyst excels when you need deep understanding, complete evidence trails, and the ability to connect concepts across massive document repositories.

### Catalyst vs Traditional RAG

| **Aspect** | **Traditional RAG** | **Catalyst** |
|-----------|---------------------|--------------|
| **📥 Document Processing** | • Simple chunking<br>• Basic text extraction<br>• Vector embeddings only | 📄 Extract text, tables, images<br>🧠 Build knowledge graphs<br>🔗 Map relationships & acronyms<br>✅ LLM consensus validation |
| **🔍 Search Capabilities** | • Semantic similarity only<br>• Isolated chunks<br>• No acronym understanding | ⚡ Hybrid search (semantic + keyword)<br>🔗 Cross-document linking<br>📍 Exact source attribution<br>🎯 Understands domain jargon |
| **⚙️ Infrastructure** | • Requires vector database<br>• API dependencies<br>• Cloud-first design | 🚫 No vector DB needed<br>🚫 No external services<br>🔌 Runs offline anywhere<br>💨 Minimal footprint |
| **⏱️ Performance** | • Fast indexing (seconds)<br>• Query: 100-500ms | 🐢 Slow extraction (2-5 min/doc)<br>⚡ Query: milliseconds<br>💾 One-time processing |
| **📊 Best For** | • General Q&A<br>• Frequently changing docs<br>• Simple retrieval | 🏛️ Regulated industries<br>⚖️ Legal/compliance<br>📋 Audit trails<br>🔍 Deep understanding |

The key difference? Catalyst spends time upfront to deeply understand your documents - extracting not just text but relationships, definitions, and context. This investment pays off when you need answers that would require a human expert to manually read through hundreds of pages.

### Real-World Use Cases

| **Scenario** | **Example Query** | **Use Catalyst?** | **Why** |
|--------------|-------------------|-------------------|---------|
| **📋 Compliance Audits** | "Show me all GDPR requirements with source pages" | ✅ **YES** | Full attribution, regulatory-grade accuracy |
| **⚖️ Legal Discovery** | "Find all liability clauses across 50 contracts" | ✅ **YES** | Cross-document linking, relationship mapping |
| **💰 Financial Analysis** | "Trace this risk metric to its definition" | ✅ **YES** | Connects terms, definitions, calculations |
| **🔧 Technical Docs** | "Map all API dependencies in the system" | ✅ **YES** | Understands system relationships |
| **💬 Chat Support** | "How do I reset my password?" | ❌ **NO** | Use simple RAG - overkill for FAQs |
| **📱 Mobile Apps** | "Real-time in-app search" | ❌ **NO** | Extraction not suitable for mobile |
| **🔄 Dynamic Content** | "Latest news updates" | ❌ **NO** | Re-extraction takes minutes |
| **🌍 General Knowledge** | "Who won the World Cup?" | ❌ **NO** | Just use ChatGPT directly |

## Knowledge Extraction

1. Deploy an OpenAI‑compatible multimodal model server (for image+text extraction).
   - Examples: MiniCPM-o-2_6 (https://huggingface.co/openbmb/MiniCPM-o-2_6) or QwenVL (https://huggingface.co/unsloth/gpt-4o-GGUF).
   - Why: a local/remote OpenAI-compatible endpoint enables accurate multimodal extraction (images, tables, captions).
   - Compatibility note: see OpenAI-compatible deployment guidance — https://www.alibabacloud.com/help/en/model-studio/qwen-vl-compatible-with-openai

2. Clone this repository:
```bash
git clone https://github.com/Blockether/catalyst.git
cd catalyst
```

3. Run the knowledge extraction tool:
```bash
python tools/KnowledgeExtraction.py "*.pdf"
```
- Ensure your model server endpoint and credentials (if any) are configured before running the extractor.

## License

MIT License - see [LICENSE](LICENSE) for details.
