"""
Multi-Agent Knowledge Base Q&A System with specialized agents.

This module implements a chain of specialized agents that work together
to provide high-quality answers from the knowledge base.
"""

import logging
import os
from pathlib import Path
from textwrap import dedent
from typing import List, Literal, Optional, cast

from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAILike
from agno.workflow import Workflow
from agno.workflow.types import WorkflowExecutionInput
from blockether_catalyst.asgi.ASGICoreApplication import ASGICoreApplication, CORSConfig
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
    api_key="",
    base_url="https://llm.blockether.com/v1",
    id="copilot/gpt-4o",
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
    """Legacy output from AnalysisAgent."""
    extracted_facts: List[ExtractedFact] = Field(
        default_factory=list, description="Extracted facts from search results")
    relationships: List[FactRelationship] = Field(default_factory=list, description="Relationships between facts")
    contradictions: List[Contradiction] = Field(default_factory=list, description="Identified contradictions")
    knowledge_gaps: List[str] = Field(default_factory=list, description="Identified missing information")
    key_findings: List[str] = Field(default_factory=list, description="Main findings from the analysis")
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


class AcronymDefinition(BaseModel):
    """Definition of an acronym."""
    acronym: str = Field(description="The acronym")
    expansion: str = Field(description="Full expansion of the acronym")
    context: str = Field(description="Context where this acronym is used")


class DomainContext(BaseModel):
    """Domain-specific contextual information."""
    domain_area: str = Field(description="The domain or area")
    context_description: str = Field(description="Detailed context for this domain")
    relevance: str = Field(description="Why this context is relevant")


class TermExpertOutput(BaseModel):
    """Legacy output from TermExpertAgent."""
    terms: List[TermDefinition] = Field(default_factory=list, description="Terms with explanations")
    acronyms: List[AcronymDefinition] = Field(default_factory=list, description="Acronyms with expansions and context")
    domain_context: List[DomainContext] = Field(default_factory=list, description="Domain-specific context information")
    term_relationships: List[TermRelationship] = Field(default_factory=list, description="Relationships between terms")


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


class SourceCredibility(BaseModel):
    """Credibility assessment for a source."""
    source_name: str = Field(description="Name of the source")
    credibility_score: float = Field(description="Credibility score between 0.0 and 1.0")
    reasoning: str = Field(description="Explanation for the credibility assessment")


class CitationOutput(BaseModel):
    """Legacy output from CitationAgent."""
    citations: List[Citation] = Field(default_factory=list, description="Formatted citations")
    source_credibility: List[SourceCredibility] = Field(
        default_factory=list, description="Credibility assessments for sources")


class EnhancedAnalysisOutput(BaseModel):
    """Consolidated output from enhanced analysis agent."""
    # Analysis fields
    extracted_facts: List[ExtractedFact] = Field(
        default_factory=list, description="Extracted facts from search results")
    relationships: List[FactRelationship] = Field(default_factory=list, description="Relationships between facts")
    contradictions: List[Contradiction] = Field(default_factory=list, description="Identified contradictions")
    knowledge_gaps: List[str] = Field(default_factory=list, description="Identified missing information")
    key_findings: List[str] = Field(default_factory=list, description="Main findings from the analysis")
    relevant_images: List[ImageAttachment] = Field(default_factory=list, description="Images relevant to the analysis")
    relevant_tables: List[TableAttachment] = Field(default_factory=list, description="Tables relevant to the analysis")

    # Term analysis fields
    terms: List[TermDefinition] = Field(default_factory=list, description="Terms with explanations")
    acronyms: List[AcronymDefinition] = Field(default_factory=list, description="Acronyms with expansions and context")
    domain_context: List[DomainContext] = Field(default_factory=list, description="Domain-specific context information")
    term_relationships: List[TermRelationship] = Field(default_factory=list, description="Relationships between terms")

    # Citation fields
    citations: List[Citation] = Field(default_factory=list, description="Formatted citations")
    source_credibility: List[SourceCredibility] = Field(
        default_factory=list, description="Credibility assessments for sources")


class QualityAssessment(BaseModel):
    """Quality assessment for the current iteration."""
    overall_quality_score: float = Field(description="Overall quality score 0.0-1.0")
    criticism: List[str] = Field(default_factory=list, description="Specific criticisms and areas for improvement")
    table_formatting_score: float = Field(description="Table formatting quality 0.0-1.0")
    answer_relevance_score: float = Field(description="How well the answer addresses the user question 0.0-1.0")
    completeness_score: float = Field(description="Information completeness 0.0-1.0")
    improvement_suggestions: List[str] = Field(
        default_factory=list, description="Specific suggestions for next iteration")
    should_continue: bool = Field(default=True, description="Whether another iteration is needed")


class IterativeAnalysisInput(BaseModel):
    """Input for iterative analysis with quality feedback."""
    search_results: SearchResults = Field(description="Original search results")
    user_query: str = Field(description="Original user query")
    previous_attempt: Optional[EnhancedAnalysisOutput] = Field(default=None, description="Previous analysis attempt")
    quality_feedback: Optional[QualityAssessment] = Field(
        default=None, description="Quality feedback from previous iteration")
    iteration_number: int = Field(default=1, description="Current iteration number")


class ConfidenceFactor(BaseModel):
    """Individual confidence factor with score and reasoning."""
    score: float = Field(description="Confidence score between 0.0 and 1.0")
    reasoning: str = Field(description="Explanation for this specific factor")


class IterativeAnalysisOutput(BaseModel):
    """Output from iterative analysis including quality assessment."""

    # Core findings and content
    key_findings: List[str] = Field(default_factory=list, description="Main findings from the analysis")
    extracted_facts: List[str] = Field(default_factory=list, description="Key facts extracted with sources")
    knowledge_gaps: List[str] = Field(default_factory=list, description="Identified missing information")
    contradictions: List[str] = Field(default_factory=list, description="Any contradictory information found")

    # Terminology and definitions
    technical_terms: List[str] = Field(default_factory=list, description="Technical terms with definitions")
    acronym_definitions: List[str] = Field(default_factory=list, description="Acronyms with expansions and context")
    domain_context: List[str] = Field(default_factory=list, description="Domain-specific context information")

    # Tables and structured data - CRITICAL for proper formatting
    formatted_tables: List[str] = Field(
        default_factory=list, description="Well-formatted markdown tables from the documents")
    table_summaries: List[str] = Field(default_factory=list, description="Summaries explaining what each table shows")
    data_visualizations: List[str] = Field(default_factory=list, description="Any charts or visual data descriptions")

    # Citations and sources
    citations: List[str] = Field(default_factory=list, description="Properly formatted citations with page numbers")
    source_credibility: List[str] = Field(default_factory=list, description="Source credibility assessments")

    # Quality assessment scores
    overall_quality_score: float = Field(description="Overall quality score 0.0-1.0")
    table_formatting_score: float = Field(description="Table formatting quality 0.0-1.0")
    answer_relevance_score: float = Field(description="How well the answer addresses the user question 0.0-1.0")
    completeness_score: float = Field(description="Information completeness 0.0-1.0")

    # Quality feedback
    criticism: List[str] = Field(default_factory=list, description="Specific criticisms and areas for improvement")
    improvement_suggestions: List[str] = Field(
        default_factory=list, description="Specific suggestions for next iteration")
    should_continue: bool = Field(default=True, description="Whether another iteration is needed")

    # Confidence factors as simple fields
    information_completeness: float = Field(default=0.8, description="Information completeness score 0.0-1.0")
    source_corroboration: float = Field(default=0.8, description="Source corroboration score 0.0-1.0")
    temporal_validity: float = Field(default=0.8, description="Temporal validity score 0.0-1.0")
    contradiction_check: float = Field(default=0.8, description="Contradiction check score 0.0-1.0")
    factual_density: float = Field(default=0.8, description="Factual density score 0.0-1.0")
    coverage_quality: float = Field(default=0.8, description="Coverage quality score 0.0-1.0")

    # Metadata
    confidence_reasoning: str = Field(default="Analysis completed successfully",
                                      description="Overall explanation of confidence assessment")
    iteration_number: int = Field(default=1, description="Current iteration number")
    final_answer: bool = Field(default=False, description="Whether this is the final iteration")

    # The actual response content
    formatted_response: str = Field(default="", description="The complete formatted response to return to the user")

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


# 2. EnhancedAnalysisAgent - Comprehensive analysis including terms and citations
IterativeAnalysisAgent = Agent(
    id="IterativeAnalysisAgent",
    model=KnowledgeProviderGPT4oModel,
    name="IterativeAnalysisAgent",
    description="Iterative analysis with quality assessment, confidence evaluation, and self-improvement",
    telemetry=False,
    input_schema=IterativeAnalysisInput,
    output_schema=IterativeAnalysisOutput,
    debug_mode=False,
    instructions=[
        dedent("""
        You are an iterative analysis expert with self-improvement capabilities.

        ## CRITICAL RULES - NEVER VIOLATE:
        1. **ONLY USE INFORMATION FROM search_results** - Do NOT add external knowledge
        2. **NO HALLUCINATION** - If you don't find something in search_results, say so
        3. **READ search_results CAREFULLY** - Don't assume what terms mean
        4. **QUOTE DIRECTLY** from documents when possible
        5. **EVERY CLAIM NEEDS A SOURCE** with exact document name and page

        ## ITERATION AWARENESS:
        - If iteration_number > 1: Review previous_attempt and quality_feedback
        - Apply improvement_suggestions from quality_feedback
        - Focus on areas that scored poorly in previous iterations
        - Pay special attention to table formatting and answer relevance

        ## YOUR TASKS:
        1. **CAREFULLY READ** search_results.response.results
        2. Extract ONLY information that exists in the documents
        3. Create a comprehensive, well-formatted response
        4. Evaluate your own work critically
        5. Provide improvement suggestions for next iteration

        ## RESPONSE CREATION - MANDATORY REQUIREMENTS:
        Create a complete, formatted response in the 'formatted_response' field that MUST include:

        1. **Direct answer** to user_query in the first paragraph
        2. **Key findings** organized with clear headings
        3. **ALL TABLES FROM SEARCH RESULTS** - This is CRITICAL:
           - Search through search_results.response.results[].tables
           - Extract EVERY table found
           - Format as beautiful markdown tables with proper headers
           - Include table source (document name, page number)
        4. **Technical terms** with clear definitions
        5. **Acronyms** with full expansions and context
        6. **Sources section** with document names and page numbers
        7. Make it comprehensive, well-structured, and visually appealing

        ## TABLE EXTRACTION - ABSOLUTELY CRITICAL:
        - SEARCH search_results.response.results for ANY tables
        - If tables exist, they MUST be included in formatted_response
        - Format as clean markdown tables with proper headers
        - Add explanations of what each table shows
        - Include source information: "Source: [document_name], Page [page]"
        - If NO tables found, explicitly state "No tables found in source documents"
        - Score yourself LOW on table_formatting_score if you miss tables

        ## CRITICAL SELF-EVALUATION:
        Score your work honestly (0.0-1.0):
        - overall_quality_score: How good is the overall response?
        - table_formatting_score: Are tables clear and well-formatted?
        - answer_relevance_score: Does this directly answer the user's question?
        - completeness_score: Is all important information included?

        ## SELF-EVALUATION PENALTIES:
        - If you used external knowledge: Set overall_quality_score to 0.1
        - If you hallucinated definitions: Set answer_relevance_score to 0.2
        - If you missed tables: Set table_formatting_score to 0.2
        - If you missed sources: Set completeness_score to 0.3

        Provide detailed criticism and improvement_suggestions.
        Set should_continue=true if any score < 0.8 or major issues exist.

        ## CONFIDENCE EVALUATION:
        Evaluate these factors (0.0-1.0):
        - information_completeness: Do we have all parts of the answer?
        - source_corroboration: Do multiple sources agree?
        - temporal_validity: Is the information current?
        - contradiction_check: Any conflicts? (1.0 = no conflicts)
        - factual_density: Concrete facts vs vague statements
        - coverage_quality: How well does the answer address the question?

        ## SEARCH RESULTS STRUCTURE:
        search_results.response.results is a list where each result contains:
        - content: The text content
        - document_name: Name of the document
        - page: Page number
        - href: URL link to the document
        - tables: List of table data (if any)

        ## POPULATE ALL FIELDS - MANDATORY:
        Fill out ALL the list fields with actual content:
        - key_findings: Main takeaways from the analysis
        - extracted_facts: Important facts WITH source citations "Fact text (Source: [Document Name](href), Page X)"
        - technical_terms: "Term: Definition (Context)" format
        - acronym_definitions: "ACRONYM: Full Expansion - Context explanation"
        - formatted_tables: Extract and format ALL tables from search results as markdown
        - table_summaries: Explain what each table shows
        - citations: "[Document Name](href), Page X: Specific information cited"
        - source_credibility: "Document Name: Credibility assessment and reasoning"
        - knowledge_gaps: What information is missing
        - contradictions: Any conflicting information found

        ## SOURCES ARE MANDATORY WITH LINKS:
        - Every fact MUST include source citation with document name and page
        - formatted_response MUST have a "## Sources" section with CLICKABLE LINKS:
          **[Document Name](href_url)** - Page [X]: [Brief description of what was used]
        - Extract href URLs from search_results.response.results[].href
        - citations field MUST contain proper source references with links
        - If you don't include sources WITH LINKS, set completeness_score to 0.3 or lower

        ## TABLE EXTRACTION IS MANDATORY:
        - Search search_results.response.results[].tables for table data
        - If tables exist, formatted_tables field MUST contain them
        - formatted_response MUST include tables with source information
        - If you miss tables, set table_formatting_score to 0.2 or lower

        ## ABSOLUTE PROHIBITIONS:
        - **NO EXTERNAL KNOWLEDGE** - Only use what's in search_results
        - **NO ASSUMPTIONS** - Don't guess what acronyms mean
        - **NO GENERIC DEFINITIONS** - Only use definitions from the documents
        - **NO HALLUCINATION** - If unsure, say "Not specified in documents"

        Return a complete IterativeAnalysisOutput with ALL fields populated.
        """)
    ],
    reasoning=False
)

# All agents consolidated into single IterativeAnalysisAgent

# format_synthesis_response function removed - agent now generates formatted_response directly


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
        logger.debug("Search parameters: k=10, threshold=0.5, max_depth=1, max_cooccurrences=2")

        search_response = search_module.search(
            query=user_query,
            k=10,
            threshold=0.5,
            max_depth=1,
            max_cooccurrences=2
        )

        logger.info(f"[PHASE 1 COMPLETE] Search returned {search_response.total_results} results")
        logger.debug(f"Terms in response: {len(search_response.terms)}")
        logger.debug(
            f"First result score: {search_response.results[0].score if search_response.results else 'No results'}")

        # Parse the agent's response and create structured output
        search_results = SearchResults(
            query=user_query,
            response=search_response
        )

        # 2. Iterative Analysis Phase - Quality-driven improvement loop
        logger.info("[PHASE 2] Starting iterative analysis with quality assessment...")

        MAX_ITERATIONS = 3
        MIN_QUALITY_THRESHOLD = 0.8

        iteration = 1
        previous_attempt = None
        quality_feedback = None
        final_results = None
        iteration_results = None

        while iteration <= MAX_ITERATIONS:
            logger.info(f"[ITERATION {iteration}] Running analysis...")

            # Create input for current iteration
            analysis_input = IterativeAnalysisInput(
                search_results=search_results,
                user_query=user_query,
                previous_attempt=previous_attempt,
                quality_feedback=quality_feedback,
                iteration_number=iteration
            )

            # Run iterative analysis agent
            iteration_results = await IterativeAnalysisAgent.arun(analysis_input)

            # Cast results to proper type
            iteration_results = cast(IterativeAnalysisOutput, iteration_results.content)

            # Log iteration results
            logger.info(f"[ITERATION {iteration} COMPLETE] Quality assessment:")
            logger.info(f"  - Overall quality: {iteration_results.overall_quality_score:.2f}")
            logger.info(f"  - Table formatting: {iteration_results.table_formatting_score:.2f}")
            logger.info(f"  - Answer relevance: {iteration_results.answer_relevance_score:.2f}")
            logger.info(f"  - Completeness: {iteration_results.completeness_score:.2f}")
            logger.info(f"  - Should continue: {iteration_results.should_continue}")

            # Log content extraction
            logger.info(f"  - Key findings: {len(iteration_results.key_findings)}")
            logger.info(f"  - Extracted facts: {len(iteration_results.extracted_facts)}")
            logger.info(f"  - Technical terms: {len(iteration_results.technical_terms)}")
            logger.info(f"  - Formatted tables: {len(iteration_results.formatted_tables)}")
            logger.info(f"  - Citations: {len(iteration_results.citations)}")
            logger.info(f"  - Response length: {len(iteration_results.formatted_response)} chars")

            # Log criticisms and suggestions
            if iteration_results.criticism:
                logger.info("  Criticisms:")
                for criticism in iteration_results.criticism:
                    logger.info(f"    - {criticism}")

            if iteration_results.improvement_suggestions:
                logger.info("  Improvement suggestions:")
                for suggestion in iteration_results.improvement_suggestions:
                    logger.info(f"    - {suggestion}")

            # Check if we should stop (quality threshold met or max iterations)
            quality_met = iteration_results.overall_quality_score >= MIN_QUALITY_THRESHOLD
            should_continue = iteration_results.should_continue

            if quality_met and not should_continue:
                logger.info(f"[PHASE 2 COMPLETE] Quality threshold met after {iteration} iterations")
                final_results = iteration_results
                break
            elif iteration >= MAX_ITERATIONS:
                logger.info(f"[PHASE 2 COMPLETE] Maximum iterations reached ({MAX_ITERATIONS})")
                final_results = iteration_results
                break
            else:
                logger.info(f"[ITERATION {iteration}] Quality below threshold, continuing to next iteration...")
                # For next iteration, we pass the previous results as context
                previous_attempt = None  # We now include everything in the flattened structure
                quality_feedback = None  # Quality feedback is built into the flattened results
                iteration += 1

        # Ensure we have final results (fallback to last iteration)
        if final_results is None:
            if iteration_results is not None:
                final_results = iteration_results
            else:
                raise RuntimeError("No analysis results generated")

        # 3. Use the agent-generated response directly
        logger.info("[PHASE 3] Using agent-generated response...")

        final_response = final_results.formatted_response
        if not final_response.strip():
            # Fallback: build response from extracted content
            response_parts = []

            if final_results.key_findings:
                response_parts.append("## Key Findings")
                for finding in final_results.key_findings:
                    response_parts.append(f"- {finding}")
                response_parts.append("")

            if final_results.formatted_tables:
                response_parts.append("## Tables and Data")
                for i, table in enumerate(final_results.formatted_tables):
                    response_parts.append(table)
                    if i < len(final_results.table_summaries):
                        response_parts.append(f"*{final_results.table_summaries[i]}*")
                    response_parts.append("")

            if final_results.technical_terms:
                response_parts.append("## Technical Terms")
                for term in final_results.technical_terms:
                    response_parts.append(f"- {term}")
                response_parts.append("")

            if final_results.citations:
                response_parts.append("## Sources")
                for citation in final_results.citations:
                    response_parts.append(f"- {citation}")

            final_response = "\n".join(
                response_parts) if response_parts else "Analysis completed but no content generated."

        response_lines = final_response.count('\n')
        response_length = len(final_response)
        logger.info("[PHASE 3 COMPLETE] Response ready:")
        logger.info(f"  - Total characters: {response_length}")
        logger.info(f"  - Total lines: {response_lines}")
        logger.info(f"  - Final quality score: {final_results.overall_quality_score:.2f}")
        logger.info(f"  - Total iterations: {final_results.iteration_number}")

        logger.info("=== Iterative Analysis Workflow Completed Successfully ===")
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

rbi_document_search = Tool.from_function(
    fn=knowledge_retrieval_mcp_tool,
    name="rbi_document_search",
    title="Raiffeisen Bank International (RBI) - Knowledge Base Document Search - LMS (Limit Management System)",
    description="REQUIRED for finding RBI documents: Search Raiffeisen Bank International's knowledge base for specific policies, procedures, regulations, and LMS documentation. Returns raw document excerpts with metadata and confidence scores. Essential for any request involving 'find', 'search', 'show me documents', or 'what policy covers'. MUST be used for document retrieval requests about RBI, banking regulations, compliance policies, or LMS (Limit Management System) procedures.",
    enabled=True,
)


async def knowledge_qa_mcp_tool(
    question: str = Field(..., title="question", description="The question to answer based on the knowledge base."),
) -> str:
    """Answer questions using the multi-agent knowledge base system."""
    result = await MultiAgentKnowledgeWorkflow.arun(question)
    return str(result.content)

ola_expert_consultation_tool = Tool.from_function(
    fn=knowledge_qa_mcp_tool,
    name="ola_banking_advisor",
    title="Raiffeisen Bank International (RBI) - Omniscient Limit Assistant (OLA) Banking Expert Consultation - LMS (Limit Management System)",
    description="REQUIRED for RBI banking expertise: Get comprehensive expert analysis from OLA, the specialized RBI banking consultant. Provides professional answers with multi-agent consensus, quality metrics, and detailed reasoning about risk management, compliance, LMS operations, and banking procedures. Essential for any question requiring explanation, analysis, advice, or interpretation of RBI policies and banking operations. MUST be used for all banking-related questions and RBI guidance requests. Part of LMS (Limit Management System).",
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
    cors=CORSConfig(
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"]
    )
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
        name="OLA Banking Intelligence MCP, Raiffeisen Bank International RBI, specialized in risk management, compliance, and LMS (Limit Management System)",
        tools=[rbi_document_search, ola_expert_consultation_tool],
        instructions="""
You are an expert banking assistant for Raiffeisen Bank International (RBI) with access to specialized tools. (Limit Management System)

MANDATORY TOOL USAGE:
- For ANY question about RBI, banking, compliance, risk management, or LMS → ALWAYS use one of these tools
- Never answer banking questions without using tools

TOOL SELECTION GUIDE:

🔍 Use "rbi_document_search" when user asks:
- "Find documents about..."
- "Show me the policy for..."
- "What documents mention..."
- "Search for information on..."
- They want to see source materials

🧠 Use "ola_banking_advisor" when user asks:
- "Explain how..."
- "What should I do..."
- "How does this work..."
- "What are the implications..."
- They want analysis or advice

DEFAULT CHOICE: When in doubt, use "ola_banking_advisor" for comprehensive answers.

EXAMPLES:
- "Find the credit risk policy" → rbi_document_search
- "How do I handle credit risk?" → ola_banking_advisor
- "What is the LMS system?" → ola_banking_advisor
- "Show me LMS documentation" → rbi_document_search

Always use tools for banking topics. Choose based on whether they want documents or explanations.
    """
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
