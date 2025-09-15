# Project Structure

## Main Directories

```
com_blockether_catalyst/
├── src/com_blockether_catalyst/           # Main source code
│   ├── asgi/                              # ASGI web application modules
│   ├── consensus/                         # LLM consensus and voting systems
│   ├── encoder/                           # Text encoding and embeddings
│   ├── knowledge/                         # Core knowledge extraction & search
│   ├── prompt/                            # Prompt engineering and alignment
│   ├── integrations/                      # External system integrations
│   │   └── agno/                         # Agno workflow engine integration  
│   ├── utils/                            # Shared utilities
│   └── assets/                           # Static assets (models, etc.)
├── tests/                                # Test suite (mirrors src structure)
├── tools/                                # Development and utility scripts
├── docs/                                 # Documentation
└── verification/                         # Deployment verification scripts (Python only)
```

## Key Modules

### Knowledge System (`knowledge/`)
- **KnowledgeSearchCore.py**: Main search engine with hybrid vector+keyword search
- **KnowledgeExtractionCore.py**: Document processing and term extraction
- **PDFKnowledgeExtractor.py**: PDF-specific extraction logic
- **KnowledgeVisualizationASGIModule.py**: Web UI for knowledge exploration

### Consensus System (`consensus/`)  
- **Consensus.py**: Multi-LLM consensus mechanism
- **ConsensusCore.py**: Core consensus algorithms
- **VotingComparison.py**: Voting strategies for LLM outputs

### Prompt System (`prompt/`)
- **PromptAlignmentCore.py**: Prompt optimization and alignment
- **PrincipleBasedAlignmentStrategy.py**: Alignment strategies

### Utils (`utils/`)
- **TypedCalls.py**: Type-safe LLM API calls
- **ConcurrentProcessor.py**: Async processing utilities
- **instructor/**: Structured LLM output handling

## Naming Conventions
- Core implementation files end with `Core` (e.g., `KnowledgeSearchCore.py`)
- Test files end with `Test` (e.g., `KnowledgeSearchCoreTest.py`)
- Type definition files end with `Types` (e.g., `KnowledgeExtractionTypes.py`)
- ASGI modules end with `ASGIModule` for web components

## File Organization Pattern
Each major module typically has:
- `{Module}Core.py` - Main implementation
- `{Module}Types.py` - Type definitions and data models  
- `{Module}Test.py` - Comprehensive test suite
- `internal/` subfolder for implementation details (where applicable)