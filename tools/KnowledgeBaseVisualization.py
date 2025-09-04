"""
Knowledge Base Q&A MCP Server Example using TypedLLMCall with Instructor.

This example demonstrates how to create a knowledge base Q&A system using:
- TypedLLMCall protocol for structured LLM responses
- Instructor library for Pydantic model validation
- Knowledge search core for document retrieval
- MCP server for tool hosting and external client integration
"""

import logging
import time
from pathlib import Path
from typing import List, Optional

import instructor
from openai import OpenAI
from pydantic import BaseModel, Field
from agno.run.response import RunResponse
from agno.storage.in_memory import InMemoryStorage
from mangum import Mangum

from com_blockether_catalyst.asgi import ASGICoreApplication, ASGIConfig
from com_blockether_catalyst.asgi.ASGITypes import CORSConfig as ASGICORSConfig
from com_blockether_catalyst.integrations.agno import (
    AgnoWorkflowAPIModule,
    MCPConfig,
    WorkflowApiASGIModule,
    WorkflowConfig,
)
from com_blockether_catalyst.knowledge import KnowledgeSearchCore
from com_blockether_catalyst.knowledge.KnowledgeVisualizationASGIModule import KnowledgeVisualizationASGIModule
from com_blockether_catalyst.utils.TypedCalls import ArityOneTypedCall

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


knowledge_search_core = KnowledgeSearchCore.from_pickle("public/knowledge_search.pkl")

# ============================================================================
# Knowledge Q&A Response Models
# ============================================================================

class SourceCitation(BaseModel):
    """Represents a source citation with document and page information."""
    document_name: str = Field(description="Name of the source document")
    page_number: Optional[int] = Field(description="Page number where information was found")
    relevance_score: float = Field(description="Relevance score of this source (0-1)")

class KnowledgeSection(BaseModel):
    """Represents a section of the knowledge response."""
    title: str = Field(description="Section title")
    content: str = Field(description="Section content in markdown format")
    key_terms: List[str] = Field(default_factory=list, description="Important terms mentioned in this section")

class KnowledgeResponse(BaseModel):
    """Structured response for knowledge base queries."""
    summary: str = Field(description="Brief summary of the answer")
    sections: List[KnowledgeSection] = Field(description="Detailed sections of the response")
    sources: List[SourceCitation] = Field(description="Source citations")
    related_topics: List[str] = Field(default_factory=list, description="Related topics for further exploration")
    confidence: float = Field(description="Confidence level of the response (0-1)")

class KnowledgeQuery(BaseModel):
    """Input model for knowledge base queries."""
    question: str = Field(description="The user's question")
    search_context: str = Field(description="Search results and context from the knowledge base")

# ============================================================================
# TypedLLMCall Implementation using Instructor
# ============================================================================

class KnowledgeTypedLLMCall(ArityOneTypedCall[KnowledgeQuery, KnowledgeResponse]):
    """TypedLLMCall implementation for knowledge base Q&A using instructor."""

    def __init__(self, base_url: str = "http://localhost:3005/v1", model_id: str = "gpt-4o"):
        """Initialize the TypedLLMCall with OpenAI-compatible endpoint."""
        self._client = instructor.from_openai(
            OpenAI(
                base_url=base_url,
                api_key="dummy"  # Dummy key for local server
            )
        )
        self._model_id = model_id

    async def call(self, x: KnowledgeQuery) -> KnowledgeResponse:
        """Make a typed call to generate a knowledge response."""

        system_prompt = """You are an intelligent knowledge base assistant that helps users find and understand information from processed document collections.

CORE CAPABILITIES:
- Analyze search results from knowledge base
- Explain concepts, terms, and acronyms found in documents
- Provide context and relationships between different pieces of information
- Present information in clear, structured, and meaningful way

WORKFLOW for answering questions:
1. ANALYZE: Examine provided search results to understand context and relationships
2. SYNTHESIZE: Combine information from multiple sources when appropriate
3. STRUCTURE: Organize response into logical sections
4. CITE: Include source document references and page numbers
5. RELATE: Suggest related topics for further exploration

RESPONSE GUIDELINES:
- Base all answers on the actual content provided in search results
- Use clear, structured sections for complex topics
- Include confidence level based on available information
- Highlight key definitions and important points
- If information is limited, acknowledge gaps honestly
- Suggest related search terms when appropriate

Always be helpful, thorough, and accurate. Focus on presenting actual content from documents."""

        user_prompt = f"""Question: {x.question}

Search Results from Knowledge Base:
{x.search_context}

Please analyze the search results and provide a comprehensive, well-structured answer to the question. Include proper citations and organize the response into logical sections."""

        # Use instructor to get structured response
        start_time = time.time()
        logger.info(f"🤖 Starting HTTP request to LLM model: {self._model_id}")

        response = self._client.chat.completions.create(
            model=self._model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_model=KnowledgeResponse
        )

        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"🤖 HTTP request to LLM completed in {duration:.3f}s - Confidence: {response.confidence:.1%}, Sections: {len(response.sections)}, Sources: {len(response.sources)}")

        return response

# Create the typed LLM call instance
knowledge_llm_call = KnowledgeTypedLLMCall()


# ============================================================================
# Main Workflow with MCP Server Integration
# ============================================================================

async def knowledge_workflow(_workflow_instance, **kwargs) -> RunResponse:
    """
    Knowledge base Q&A workflow that can be called via REST API or MCP tools.

    This workflow:
    1. First searches the knowledge base for relevant information
    2. Then uses the TypedLLMCall to provide a comprehensive structured answer
    3. Includes proper citations and source references
    """
    total_start_time = time.time()
    message = kwargs.get("message", "")
    logger.info(f"🚀 Starting knowledge workflow for query: '{message}'")

    # First, search the knowledge base for relevant information
    search_start_time = time.time()
    logger.info("📊 Beginning knowledge base search...")
    search_results = knowledge_search_core.search_enhanced(
        query=message,
        k=10,
        threshold=0.1,
        max_depth=2,
        max_cooccurrences=5
    )
    search_time = time.time() - search_start_time
    logger.info(f"📊 Knowledge search completed in {search_time:.3f}s, found {len(search_results)} results")

    # Prepare context for the TypedLLMCall with search results
    context_start_time = time.time()
    search_results_text = ""
    if search_results:
        search_results_text = "\n".join([
            f"Result {i+1} (Score: {result.score:.3f}):\n"
            f"Document: {result.document_name}\n"
            f"Page: {result.page or 'Unknown'}\n"
            f"Text: {result.text}\n"
            f"Primary Terms: {[term.term for term in result.primary_terms]}\n"
            f"Related Terms: {[term.term for term in result.related_terms]}\n"
            for i, result in enumerate(search_results)
        ])
    else:
        search_results_text = "No results found in the knowledge base for this query."

    # Create query object for TypedLLMCall
    query = KnowledgeQuery(
        question=message,
        search_context=search_results_text
    )
    context_time = time.time() - context_start_time
    logger.info(f"🔄 Context preparation completed in {context_time:.3f}s")

    # Use TypedLLMCall to get structured response
    llm_start_time = time.time()
    logger.info("🤖 Starting LLM call to generate structured response...")
    structured_response = await knowledge_llm_call.call(query)
    llm_time = time.time() - llm_start_time
    logger.info(f"🤖 LLM call completed in {llm_time:.3f}s")

    # Format the structured response as markdown
    markdown_content = f"# {structured_response.summary}\n\n"

    for section in structured_response.sections:
        markdown_content += f"## {section.title}\n\n{section.content}\n\n"

        if section.key_terms:
            markdown_content += f"**Key Terms:** {', '.join(section.key_terms)}\n\n"

    if structured_response.sources:
        markdown_content += "## Sources\n\n"
        for i, source in enumerate(structured_response.sources, 1):
            page_info = f", Page {source.page_number}" if source.page_number else ""
            markdown_content += f"{i}. {source.document_name}{page_info} (Relevance: {source.relevance_score:.2f})\n"
        markdown_content += "\n"

    if structured_response.related_topics:
        markdown_content += "## Related Topics\n\n"
        markdown_content += "\n".join([f"- {topic}" for topic in structured_response.related_topics])
        markdown_content += "\n\n"

    markdown_content += f"*Confidence Level: {structured_response.confidence:.1%}*"

    # Calculate and log total workflow time
    total_time = time.time() - total_start_time
    logger.info(f"✅ Knowledge workflow completed in {total_time:.3f}s (search: {search_time:.3f}s, context: {context_time:.3f}s, llm: {llm_time:.3f}s)")

    return RunResponse(
        content=markdown_content,
    )



# ============================================================================
# Application Setup
# ============================================================================

def create_app() -> ASGICoreApplication:
    """
    Create the knowledge base Q&A MCP server application.

    This creates a single ASGI application that:
    1. Uses TypedLLMCall with instructor for structured LLM responses
    2. Hosts knowledge base tools as custom MCP tools
    3. Provides MCP server endpoint for external clients to connect
    4. Includes REST API endpoints for workflow execution
    5. Ready for external MCP clients (like Claude Desktop) to connect and use tools
    """
    # Configure workflow
    workflow_config = WorkflowConfig(
        run_callback=knowledge_workflow,
        alias="Knowledge Base Q&A",  # Display name shown in chat UI
        description="Knowledge base question answering with MCP server hosting knowledge tools",
        storage=InMemoryStorage(mode="workflow"),
        debug_mode=True,
        telemetry=False,
    )

    # Configure MCP with knowledge base tools
    mcp_config = MCPConfig(
        server_name="knowledge-base-mcp",
        custom_tools={},  # Add custom tools here if needed
    )

    # Configure API
    api_config = AgnoWorkflowAPIModule(
        name="Knowledge Base Q&A API",
        description="Knowledge base Q&A with MCP server",
        prefix="/api",
        mcp=mcp_config
    )

    # Create the workflow module
    workflow_module = WorkflowApiASGIModule(
        workflow=workflow_config,
        api=api_config,
    )

    # Create the ASGI application
    app_config = ASGIConfig(
        title="Knowledge Base Q&A with MCP",
        description="Knowledge base Q&A hosting MCP server with knowledge tools",
        version="1.0.0",
        prefix="/kb",
        debug=True,
        cors_config=ASGICORSConfig(
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        ),
    )

    app = ASGICoreApplication(config=app_config)
    app.mount_module(workflow_module)

    # Create and mount the visualization module
    visualization_module = KnowledgeVisualizationASGIModule(
        output_dir=Path("output/"),
        prefix="/viz"
    )
    # Load the knowledge data without creating a new search core
    visualization_module.load_from_pickle(
        Path("output/linked_knowledge.pkl"),
        create_search_core=False  # We'll use the existing one
    )
    # Use the existing search core
    visualization_module.search_core = knowledge_search_core
    # Mount the visualization module
    app.mount_module(visualization_module)

    @app.app.get("/")
    async def root():
        return {
            "status": "healthy",
            "service": "Knowledge Base Q&A MCP Server",
            "mcp_endpoint": "/kb/api/mcp/sse",
            "available_mcp_tools": [
                "send_message",
                "search_knowledge",
                "add_document",
                "summarize_document",
                "list_documents"
            ],
            "endpoints": {
                "chat_ui": "/kb/api/chat",
                "workflow_run": "/kb/api/workflow/run",
                "workflow_info": "/kb/api/workflow",
                "mcp_sse": "/kb/api/mcp/sse",
                "mcp_messages": "/kb/api/mcp/messages",
                "visualization": "/kb/viz",
                "docs": "/kb/docs",
                "openapi": "/kb/openapi.json",
            },
        }
    return app

app = create_app()

# AWS Lambda handler using Mangum (for serverless deployment)
handler = Mangum(app.app, lifespan="off")

def main():
    """Run the knowledge base Q&A MCP server."""
    import uvicorn

    print("\n" + "=" * 70)
    print("📚 Knowledge Base Q&A MCP Server")
    print("=" * 70)
    print("\n📖 Hosting Knowledge Base Tools:")
    print("  • search_knowledge (Search Knowledge)")
    print("  • add_document (Add Document)")
    print("  • summarize_document (Summarize Document)")
    print("  • list_documents (List Documents)")
    print("\n🔌 MCP Server Features:")
    print("  • TypedLLMCall with instructor for structured responses")
    print("  • Streamable HTTP transport")
    print("  • Knowledge base operations")
    print("  • Ready for external client connections (e.g., Claude Desktop)")
    print("\n📡 Server Endpoints:")
    print("  • Health: http://localhost:8002/")
    print("  • Chat UI: http://localhost:8002/kb/api/chat")
    print("  • Visualization: http://localhost:8002/kb/viz")
    print("  • MCP SSE: http://localhost:8002/kb/api/mcp/sse")
    print("  • Workflow: http://localhost:8002/kb/api/workflow")
    print("  • API Docs: http://localhost:8002/kb/docs")
    print("\n🎨 Knowledge Visualization Features:")
    print("  • Interactive term explorer")
    print("  • Document browser")
    print("  • Search interface")
    print("  • Relationship graphs")
    print("\n💡 Connect external MCP clients to SSE endpoint: http://localhost:8002/kb/api/mcp/sse")
    print("\n" + "=" * 70)
    print("🚀 Starting MCP SERVER on http://localhost:8002")
    print("Press Ctrl+C to stop")
    print("=" * 70 + "\n")

    uvicorn.run(app.app, host="0.0.0.0", port=8003, reload=False)


if __name__ == "__main__":
    main()
