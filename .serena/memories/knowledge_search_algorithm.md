# Knowledge Search Algorithm - Hybrid Approach

## Overview
The KnowledgeSearchCore uses a hybrid search algorithm that combines:
1. **Direct term-based retrieval** using the term_to_chunks_index
2. **Vector similarity search** for semantic matching
3. **Smart filtering and boosting** based on query terms

## Key Components

### 1. Term Extraction
- Extracts acronyms and keywords from query using TF-IDF
- Prioritizes n-grams (longer phrases) over single words
- Acronyms are identified using pattern matching (uppercase, short words)

### 2. Two-Phase Retrieval

**Phase 1: Direct Term Lookup**
- Uses `term_to_chunks_index` for O(1) lookup of chunks containing query terms
- Directly retrieves chunks that contain identified acronyms/keywords
- Bypasses vector similarity entirely for exact matches

**Phase 2: Vector Similarity Search**
- Performs semantic search using embeddings
- Uses lower threshold (min 0.1) when query contains acronyms/keywords
- Retrieves more candidates (k*3) for better ranking

### 3. Intelligent Filtering

Results are included if:
1. **Found via term index** - Direct term match (always included)
2. **Contains query terms** - Acronyms/keywords in text (always included)
3. **Meets threshold** - Similarity score >= threshold (for other results)

### 4. Score Boosting
- Results containing acronyms get +0.2 boost
- Results containing keywords get +0.1 boost
- Final score is capped at 1.0
- Sorting uses boosted final_score

## Important Indices

### term_to_chunks_index
- Maps normalized terms to (document_id, chunk_index) tuples
- Enables O(1) lookup of chunks containing specific terms
- Built during knowledge extraction
- Stored in LinkedKnowledge

### Why This Works
- **Acronyms bypass threshold**: Direct term lookup and term-based inclusion
- **Keywords get preference**: Boosting and lower effective threshold
- **Semantic search preserved**: Vector similarity still works for conceptual queries
- **Performance optimized**: Term index provides fast direct access

## Configuration Constants
- MAX_TOP_KEYWORDS_FROM_QUERY = 20
- MAX_TOP_ACRONYMS_FROM_QUERY = 10
- Acronym boost = 0.2
- Keyword boost = 0.1
- Minimum effective threshold for term queries = 0.1