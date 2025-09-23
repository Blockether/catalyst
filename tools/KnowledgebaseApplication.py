"""
Multi-Agent Knowledge Base Q&A System with specialized agents.

This module implements a chain of specialized agents that work together
to provide high-quality answers from the knowledge base.
"""

import asyncio
import json
import logging
from datetime import datetime
import os
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, List, Literal, Optional, Tuple, cast
import uuid

from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.memory.manager import MemoryManager
from agno.models.openai import OpenAILike
from agno.workflow import Workflow
from agno.session.workflow import WorkflowSession
from agno.workflow.step import Step
from agno.run.agent import RunOutput
from agno.run.workflow import WorkflowRunOutput
from agno.workflow.types import StepInput, StepOutput, WorkflowExecutionInput
from blockether_catalyst.asgi.ASGICoreApplication import ASGICoreApplication
from blockether_catalyst.asgi.ASGICoreModule import StaticMount
from blockether_catalyst.encoder import EncoderCore
from blockether_catalyst.integrations.agno.AgnoOsASGIModule import (
    AgnoOsASGIModule,
    AssistantConfig,
    ChatConfig,
    MCPConfig,
)
from blockether_catalyst.knowledge.KnowledgeSearchCore import KnowledgeSearchCore
from blockether_catalyst.knowledge.KnowledgeTypes import OptimizedSearchResponse
from fastmcp.tools import Tool
from mangum import Mangum
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DOMAIN = "finance, banking, limits, risk management, compliance, regulations"
APPLICATION = "OLA KnowledgeProvider"

# Separate base URL and prefix for proper path construction
BASE_URL = "http://localhost:8002"
PREFIX = "/os"
RESOURCES_URL = f"{BASE_URL}{PREFIX}"

# Initialized once to prevent the parallelism transformers issues.
EncoderCore._initialize()

# Load search module
search_module = KnowledgeSearchCore.from_pickle(
    "public/knowledge_extraction/knowledge_search.pkl",
    resources_base_url=RESOURCES_URL
)

# Model configuration
KnowledgeProviderGPT4oModel = OpenAILike(
    api_key="dummy",
    base_url="http://localhost:3005/v1",
    id="gpt-4o"
)

# Database
db = SqliteDb()

# =====================================
# Pydantic Models for Agent Communication
# =====================================

os.environ["TOKENIZERS_PARALLELISM"] = "true"

class ImageAttachment(BaseModel):
    """An image attachment from search results."""
    caption: str = Field(description="Image caption or description")
    href: str = Field(description="URL to the image")
    page: int = Field(description="Page number where image appears")
    document_name: str = Field(description="Document containing the image")
    relevance_to_query: float = Field(default=0.0, description="Relevance score to the query (0-1)")


class TableAttachment(BaseModel):
    """A table attachment from search results."""
    markdown_content: str = Field(description="Table content in markdown format")
    page: int = Field(description="Page number where table appears")
    document_name: str = Field(description="Document containing the table")
    relevance_to_query: float = Field(default=0.0, description="Relevance score to the query (0-1)")
    summary: Optional[str] = Field(default=None, description="Brief summary of table contents")


class SearchResults(BaseModel):
    query: str = Field(description="The original search query")
    response: OptimizedSearchResponse = Field(description="Raw search results from the knowledge base")


class ExtractedFact(BaseModel):
    """A single extracted fact from the knowledge base."""
    content: str = Field(description="The fact content")
    document_name: str = Field(description="Name of source document")
    document_id: str = Field(description="Document identifier SHA-256 hash")
    page: int = Field(description="Page number where fact was found")
    confidence: float = Field(description="Confidence level for this fact (0-1)")
    category: str = Field(description="Category or topic of the fact")
    chunk_index: Optional[int] = Field(default=None, description="Chunk index where fact appears")


class FactRelationship(BaseModel):
    """Relationship between two facts or concepts."""
    source_fact: str = Field(description="The source fact or concept")
    target_fact: str = Field(description="The target fact or concept")
    relationship_type: str = Field(description="Type of relationship (e.g., 'causes', 'relates_to', 'contradicts')")
    strength: float = Field(description="Strength of the relationship (0-1)")
    explanation: str = Field(description="Explanation of the relationship")


class Contradiction(BaseModel):
    """A contradiction found in the information."""
    fact_a: str = Field(description="First conflicting fact")
    fact_b: str = Field(description="Second conflicting fact")
    source_a_document: str = Field(description="Document name of first fact")
    source_a_page: int = Field(description="Page number of first fact")
    source_b_document: str = Field(description="Document name of second fact")
    source_b_page: int = Field(description="Page number of second fact")
    severity: Literal["minor", "moderate", "major"] = Field(description="Severity level of contradiction")
    explanation: str = Field(description="Explanation of the contradiction")


class AnalysisOutput(BaseModel):
    """Output from AnalysisAgent."""
    extracted_facts: List[ExtractedFact] = Field(description="Extracted facts from search results")
    relationships: List[FactRelationship] = Field(description="Relationships between facts")
    contradictions: List[Contradiction] = Field(description="Identified contradictions")
    knowledge_gaps: List[str] = Field(description="Identified missing information")
    key_findings: List[str] = Field(description="Main findings from the analysis")
    relevant_images: List[ImageAttachment] = Field(default_factory=list, description="Images relevant to the analysis")
    relevant_tables: List[TableAttachment] = Field(default_factory=list, description="Tables relevant to the analysis")


class TermDefinition(BaseModel):
    """Definition of a technical term."""
    term: str = Field(description="The term being defined")
    definition: str = Field(description="Clear definition of the term")
    context: str = Field(description="Domain-specific context")
    examples: Optional[List[str]] = Field(default=None, description="Example usages")
    related_terms: Optional[List[str]] = Field(default=None, description="Related terminology")


class TermRelationship(BaseModel):
    """Relationship between terms."""
    term_a: str = Field(description="First term")
    term_b: str = Field(description="Second term")
    relationship_type: str = Field(description="Type of relationship")
    description: str = Field(description="Description of how terms relate")


class TermExpertOutput(BaseModel):
    """Output from TermExpertAgent."""
    terms: List[TermDefinition] = Field(description="Terms with explanations")
    acronyms: Dict[str, str] = Field(description="Acronyms with expansions")
    domain_context: Dict[str, str] = Field(description="Domain-specific context")
    term_relationships: List[TermRelationship] = Field(description="Relationships between terms")


class Citation(BaseModel):
    """A single citation."""
    document_name: str = Field(description="Name of the source document")
    document_id: str = Field(description="Document identifier SHA-256 hash")
    title: str = Field(description="Title of the document")
    author: Optional[str] = Field(default=None, description="Document author")
    publication_date: Optional[str] = Field(default=None, description="Publication date")
    page: int = Field(description="Page number where content appears")
    href: str = Field(description="URL to the source document")
    quote: Optional[str] = Field(default=None, description="Relevant quote from source")
    relevance_score: float = Field(description="Relevance to the query (0-1)")


class CitationOutput(BaseModel):
    """Output from CitationAgent."""
    citations: List[Citation] = Field(description="Formatted citations")
    source_credibility: Dict[str, float] = Field(description="Credibility scores for sources")

class ConfidenceInput(BaseModel):
    """Input to ConfidenceAgent."""
    analysis: AnalysisOutput = Field(description="Output from AnalysisAgent")
    search_results: SearchResults = Field(description="Original search results")
    citations: CitationOutput = Field(description="Output from CitationAgent")
    terms: TermExpertOutput = Field(description="Output from TermExpertAgent")


class ConfidenceOutput(BaseModel):
    """Output from ConfidenceAgent."""
    confidence_factors: Dict[str, float] = Field(description="Individual confidence factors")
    reasoning: str = Field(description="Explanation of confidence score")

# SynthesisInput removed - we use a simple formatting function instead

# =====================================
# Knowledge Retrieval Function
# =====================================


def knowledge_retrieval_mcp_tool(
    query: str = Field(..., title="query", description="The search query to find relevant documents."),
    max_documents: Optional[int] = Field(title="max_documents", description="The number of documents to retrieve.")
) -> dict:
    """Execute knowledge base search."""
    optimized_response: OptimizedSearchResponse = search_module.search(
        query=query,
        k=max_documents or 10,
        threshold=0.5,
        max_depth=2,
        max_cooccurrences=3
    )
    return optimized_response.model_dump()


# 2. AnalysisAgent - Deep analysis of search results
AnalysisAgent = Agent(
    id="AnalysisAgent",
    model=KnowledgeProviderGPT4oModel,
    name="AnalysisAgent",
    description="Analyzes search results to extract facts and relationships",
    telemetry=False,
    input_schema=SearchResults,
    output_schema=AnalysisOutput,
    debug_mode=False,
    instructions=[
        dedent("""
        You are an information analysis expert. Analyze the search results to:
        1. Extract concrete facts and data points
        2. Identify relationships between information pieces
        3. Detect any contradictions or conflicts
        4. Identify knowledge gaps
        5. Summarize key findings
        6. Collect relevant images and tables from results

        Focus on:
        - Factual accuracy
        - Logical relationships
        - Information completeness
        - Contradiction detection
        - Visual content relevance

        IMPORTANT: Extract images and tables from search_results.response.results
        Each result contains 'images' and 'tables' lists that must be collected.

        Return AnalysisOutput with:
        - extracted_facts: List of ExtractedFact objects with content, source, confidence, category
        - relationships: List of FactRelationship objects showing how facts connect
        - contradictions: List of Contradiction objects with conflicting facts and severity
        - knowledge_gaps: List of strings describing missing information
        - key_findings: List of strings with main takeaways
        - relevant_images: List of ImageAttachment objects from search results
        - relevant_tables: List of TableAttachment objects from search results
        """)
    ],
    reasoning=False
)

# 3. TermExpertAgent - Handles terms and acronyms
TermExpertAgent = Agent(
    id="TermExpertAgent",
    model=KnowledgeProviderGPT4oModel,
    name="TermExpertAgent",
    description="Expert in domain terminology and acronyms",
    telemetry=False,
    input_schema=SearchResults,
    output_schema=TermExpertOutput,
    store_events=True,
    debug_mode=False,
    instructions=[
        dedent(f"""
        You are a {DOMAIN} terminology expert. Your role is to:
        1. Identify and explain technical terms
        2. Expand and explain acronyms
        3. Provide domain-specific context
        4. Explain term relationships

        Focus on:
        - Clear, concise definitions
        - Domain-specific meanings
        - Acronym expansions
        - Term interconnections

        Return TermExpertOutput with:
        - terms: List of TermDefinition objects with term, definition, context, examples
        - acronyms: Dict mapping acronyms to their full expansions
        - domain_context: Dict of domain-specific contextual information
        - term_relationships: List of TermRelationship objects
        """)
    ],
    reasoning=False
)

# 4. CitationAgent - Manages citations and sources
CitationAgent = Agent(
    id="CitationAgent",
    model=KnowledgeProviderGPT4oModel,
    input_schema=SearchResults,
    output_schema=CitationOutput,
    name="CitationAgent",
    description="Formats citations and evaluates source credibility",
    telemetry=False,
    store_events=True,
    debug_mode=False,
    instructions=[
        dedent("""
        You are a citation specialist. Your responsibilities:
        1. Track all information sources used
        2. Format citations properly
        3. Evaluate source credibility
        4. Create citation markdown

        Citation format:
        - [document_name](link), author, date, pages, (relevance: X)
        - Include exact quotes when relevant
        - Group by topic when appropriate

        Return CitationOutput with:
        - citations: List of Citation objects with title, authors, date, url, relevance_score
        - source_credibility: Dict mapping source IDs to credibility scores (0-1)
        - citation_markdown: Formatted markdown string with all citations
        """)
    ],
    reasoning=False
)

# 5. ConfidenceAgent - Evaluates answer confidence
ConfidenceAgent = Agent(
    id="ConfidenceAgent",
    model=KnowledgeProviderGPT4oModel,
    name="ConfidenceAgent",
    description="Evaluates confidence in the answer quality",
    input_schema=ConfidenceInput,
    output_schema=ConfidenceOutput,
    telemetry=False,
    store_events=True,
    debug_mode=False,
    instructions=[
        dedent("""
        You are a confidence evaluation specialist. Evaluate answer confidence based on:

        IMPORTANT: Confidence is NOT the same as relevance!
        - Relevance = how well search results match the query
        - Confidence = how certain we are the answer is CORRECT and COMPLETE

        Evaluate these factors:
        1. Information completeness (0-1): Do we have all parts of the answer?
        2. Source corroboration (0-1): Do multiple sources agree?
        3. Temporal validity (0-1): Is the information current?
        4. Contradiction presence (0-1): Any conflicts? (1 = no conflicts)
        5. Factual density (0-1): Concrete facts vs vague statements
        6. Coverage quality (0-1): How well does the answer address the question?

        Calculate overall confidence as weighted average.

        Also identify:
        - Missing information that would increase confidence
        - Recommendations for improving answer quality

        Return ConfidenceOutput with:
        - overall_confidence: Float score between 0 and 1
        - confidence_factors: Dict mapping factor names to scores
        - confidence_reasoning: String explaining the confidence evaluation
        - missing_information: List of strings describing what's missing
        - recommendations: List of strings with improvement suggestions
        """)
    ],
    reasoning=False
)

def format_synthesis_response(
    analysis: AnalysisOutput,
    terms: TermExpertOutput,
    citations: CitationOutput,
    confidence: ConfidenceOutput
) -> str:
    """Format the final response from all agent outputs."""

    response_parts = []

    # Answer section - use key findings as the main answer
    response_parts.append("# Answer")
    if analysis.key_findings:
        for finding in analysis.key_findings[:3]:  # Top 3 key findings
            response_parts.append(f"- {finding}")
    response_parts.append("")

    # Key Findings section
    response_parts.append("## Key Findings")
    for finding in analysis.key_findings:
        response_parts.append(f"- {finding}")
    response_parts.append("")

    # Detailed Information - extracted facts
    response_parts.append("## Detailed Information")
    if analysis.extracted_facts:
        for fact in analysis.extracted_facts[:10]:  # Limit to 10 facts
            response_parts.append(f"- **{fact.category}**: {fact.content}")
            response_parts.append(f"  - Source: {fact.document_name}, Page {fact.page}")
    response_parts.append("")

    # Terms and Acronyms
    response_parts.append("## Terms and Acronyms")

    response_parts.append("### Primary Terms")
    if terms.terms:
        for term_def in terms.terms[:5]:  # Top 5 terms
            response_parts.append(f"- **{term_def.term}**: {term_def.definition}")
            if term_def.examples:
                response_parts.append(f"  - Example: {term_def.examples[0]}")
    response_parts.append("")

    response_parts.append("### Acronyms")
    if terms.acronyms:
        for acronym, expansion in list(terms.acronyms.items())[:10]:  # Top 10 acronyms
            response_parts.append(f"- **{acronym}**: {expansion}")
    response_parts.append("")

    # Citations
    response_parts.append("## Citations")
    response_parts.append(citations)
    response_parts.append("")

    # Attachments
    response_parts.append("## Attachments")

    # Images
    if analysis.relevant_images:
        response_parts.append("### Images")
        for img in analysis.relevant_images[:5]:  # Limit to 5 images
            response_parts.append(f"![{img.caption}]({img.href})")
            response_parts.append(f"*Document: {img.document_name}, Page {img.page}*")
            response_parts.append("")

    # Tables
    if analysis.relevant_tables:
        response_parts.append("### Tables")
        for idx, table in enumerate(analysis.relevant_tables[:3], 1):  # Limit to 3 tables
            response_parts.append(f"**Table {idx}** (from {table.document_name}, Page {table.page})")
            response_parts.append(table.markdown_content)
            if table.summary:
                response_parts.append(f"*Summary: {table.summary}*")
            response_parts.append("")

    response_parts.append("-----")

    # Confidence Evaluation
    response_parts.append("######## Confidence Evaluation")


    return "\n".join(response_parts)


async def custom_execution_function(
    workflow: Workflow,
    execution_input: WorkflowExecutionInput,
) -> str:
    """Execute the multi-agent workflow to answer knowledge base queries."""
    user_query = str(execution_input.input)
    logger.info("=== Starting Multi-Agent Workflow ===")
    logger.info(f"Query: {user_query}")

    try:
        # 1. Search Phase - Execute knowledge base search
        logger.info("[PHASE 1] Executing knowledge base search...")
        logger.debug("Search parameters: k=10, threshold=0.5, max_depth=2, max_cooccurrences=3")

        search_response = search_module.search(
            query=user_query,
            k=10,
            threshold=0.5,
            max_depth=2,
            max_cooccurrences=3
        )

        logger.info(f"[PHASE 1 COMPLETE] Search returned {search_response.total_results} results")
        logger.debug(f"Terms in response: {len(search_response.terms)}")
        logger.debug(f"First result score: {search_response.results[0].score if search_response.results else 'No results'}")

        # Parse the agent's response and create structured output
        search_results = SearchResults(
            query=user_query,
            response=search_response
        )

        # 2. Parallel Analysis Phase - Run these agents concurrently
        logger.info("[PHASE 2] Starting parallel agent execution...")
        logger.info("  - AnalysisAgent: Extracting facts and relationships")
        logger.info("  - TermExpertAgent: Processing terminology")
        logger.info("  - CitationAgent: Formatting citations")

        # Run analysis, terms, and citations agents in parallel
        analysis_results, terms_results, citations_results = await asyncio.gather(
            AnalysisAgent.arun(search_results),
            TermExpertAgent.arun(search_results),
            CitationAgent.arun(search_results)
        )

        # Log raw result types before casting
        logger.debug(f"Raw analysis result type: {type(analysis_results)}")
        logger.debug(f"Raw terms result type: {type(terms_results)}")
        logger.debug(f"Raw citations result type: {type(citations_results)}")
        logger.debug(f"Analysis content type: {type(analysis_results.content)}")
        logger.debug(f"Terms content type: {type(terms_results.content)}")
        logger.debug(f"Citations content type: {type(citations_results.content)}")

        # Cast results to proper types
        analysis_results = cast(AnalysisOutput, analysis_results.content)
        terms_results = cast(TermExpertOutput, terms_results.content)
        citations_results = cast(CitationOutput, citations_results.content)

        # Log extracted data statistics
        logger.info("[PHASE 2 COMPLETE] Agent results summary:")
        logger.info("  AnalysisAgent:")
        logger.info(f"    - Extracted facts: {len(analysis_results.extracted_facts)}")
        logger.info(f"    - Relationships found: {len(analysis_results.relationships)}")
        logger.info(f"    - Contradictions identified: {len(analysis_results.contradictions)}")
        logger.info(f"    - Key findings: {len(analysis_results.key_findings)}")
        logger.info(f"    - Knowledge gaps: {len(analysis_results.knowledge_gaps)}")
        logger.info(f"    - Relevant images: {len(analysis_results.relevant_images)}")
        logger.info(f"    - Relevant tables: {len(analysis_results.relevant_tables)}")

        logger.info("  TermExpertAgent:")
        logger.info(f"    - Terms defined: {len(terms_results.terms)}")
        logger.info(f"    - Acronyms identified: {len(terms_results.acronyms)}")
        logger.info(f"    - Term relationships: {len(terms_results.term_relationships)}")

        logger.info("  CitationAgent:")
        logger.info(f"    - Citations created: {len(citations_results.citations)}")
        logger.info(f"    - Sources evaluated: {len(citations_results.source_credibility)}")

        # 3. Confidence Evaluation - Needs analysis results
        logger.info("[PHASE 3] Evaluating confidence...")

        confidence_input = ConfidenceInput(
            analysis=analysis_results,
            search_results=search_results,
            citations=citations_results,
            terms=terms_results
        )
        logger.debug("Confidence input created successfully")

        confidence_results = await ConfidenceAgent.arun(confidence_input)

        logger.debug(f"Raw confidence result type: {type(confidence_results)}")
        logger.debug(f"Confidence content type: {type(confidence_results.content)}")

        confidence_results = cast(ConfidenceOutput, confidence_results.content)

        logger.info("[PHASE 3 COMPLETE] Confidence evaluation:")
        logger.info(f"  - Confidence factors: {len(confidence_results.confidence_factors)}")

        if confidence_results.confidence_factors:
            for factor, score in confidence_results.confidence_factors.items():
                logger.debug(f"    {factor}: {score:.2f}")

        # 4. Format the final response
        logger.info("[PHASE 4] Formatting final response...")

        final_response = format_synthesis_response(
            analysis=analysis_results,
            terms=terms_results,
            citations=citations_results,
            confidence=confidence_results
        )

        response_lines = final_response.count('\n')
        response_length = len(final_response)
        logger.info("[PHASE 4 COMPLETE] Response formatted:")
        logger.info(f"  - Total characters: {response_length}")
        logger.info(f"  - Total lines: {response_lines}")

        logger.info("=== Multi-Agent Workflow Completed Successfully ===")
        return final_response

    except Exception as e:
        logger.error(f"ERROR in multi-agent workflow: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.exception("Full traceback:")

        # Return error response with details
        return f"""#### Error Processing Query

An error occurred while processing your query.

**Error Type:** {type(e).__name__}
**Error Message:** {str(e)}

Please try rephrasing your query or contact support if the issue persists.
"""

# =====================================
# Multi-Agent Workflow
# =====================================

MultiAgentKnowledgeWorkflow = Workflow(
    id="MultiAgentKnowledgeWorkflow",
    name="Multi-Agent Knowledge Base Q&A Workflow",
    description="Orchestrated multi-agent system for high-quality knowledge base responses",
    db=db,
    telemetry=False,
    debug_mode=False,
    steps=custom_execution_function,
    store_events=True,
    store_executor_outputs=True,
    cache_session=True
)

# =====================================
# MCP Tool
# =====================================

knowledge_query_mcp_tool = Tool.from_function(
    fn=knowledge_retrieval_mcp_tool,
    name="search_knowledge",
    description="Search the knowledge base for relevant documents and information.",
    enabled=True,
)

# =====================================
# ASGI Application
# =====================================

asgi_app = ASGICoreApplication(
    title="OLA Multi-Agent ASGI",
    description="Multi-agent knowledge base Q&A system",
    version="2.0.0",
    prefix="/",
    debug=False,
)

agno_asgi_module = AgnoOsASGIModule(
    title="OLA Multi-Agent OS Module",
    description="Multi-agent system for OLA knowledge base",
    workflows=[MultiAgentKnowledgeWorkflow],
    teams=[],
    chat=ChatConfig(
        assistant=AssistantConfig(
            name="OLA Multi-Agent Assistant",
            short="O",
            runner=MultiAgentKnowledgeWorkflow
        ),
        base_url=BASE_URL
    ),
    mcp=MCPConfig(
        name="OLA Knowledge MCP",
        tools=[knowledge_query_mcp_tool],
    ),
    statics=[
        StaticMount(
            mount="/public",
            directory=Path("public"),
            name="public",
        )
    ]
)

asgi_app.mount_module(agno_asgi_module)

handler = Mangum(asgi_app.app)

if __name__ == "__main__":
    logger.info("Starting Multi-Agent Knowledge System...")
    asgi_app.run(host="0.0.0.0", port=8002)
