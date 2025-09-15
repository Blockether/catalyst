"""
Knowledge Base Q&A MCP Server Example using TypedLLMCall with Instructor.

This example demonstrates how to create a knowledge base Q&A system using:
- TypedLLMCall protocol for structured LLM responses
- Instructor library for Pydantic model validation
- Knowledge search core for document retrieval
- MCP server for tool hosting and external client integration
"""

import logging
from textwrap import dedent
from typing import List, Optional, TypedDict

from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAILike
from agno.memory.manager import MemoryManager
from pydantic import Field
from com_blockether_catalyst.asgi.ASGICoreApplication import ASGICoreApplication

from com_blockether_catalyst.knowledge.KnowledgeSearchCore import KnowledgeSearchCore, KnowledgeSearchResult
from com_blockether_catalyst.integrations.agno.AgnoOsASGIModule import AgnoOsASGIModule, ChatConfig, MCPConfig
from fastmcp.tools import Tool

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DOMAIN = "finanance, banking, limits, risk management, compliance, regulations"
APPLICATION = "Catalyst KnowledgeProvider"

search_module = KnowledgeSearchCore.from_pickle(
    "public/knowledge_extraction/knowledge_search.pkl",
    base_url="http://localhost:8002"
)


KnowledgeProviderGPT4oModel = OpenAILike(
    api_key="dummy",
    base_url="http://localhost:3005/v1",
    id="gpt-4o"
)

KnowledgeFormattingAgent = Agent(
    model=KnowledgeProviderGPT4oModel,
    name="KnowledgeProvider",
    description="Agent for formatting knowledge base responses",
    markdown=True,
    retries=2
)

asgi_app = ASGICoreApplication(
    title="Catalyst ASGI",
    description="Knowledge base Q&A hosting MCP server with knowledge tools",
    version="1.0.0",
    prefix="/",
    debug=False,
)

db = SqliteDb()


class DocumentReference(TypedDict):
    document_name: str
    page: int
    author: Optional[str]
    publication_date: Optional[str]
    href: str
    modified_date: Optional[str]


def knowledge_retriever_internal(
    query: str = Field(..., title="query", description="The search query to find relevant documents."),
    num_documents: int = Field(..., title="num_documents", description="The number of documents to retrieve.")
):

    # Helper function to resolve linked terms
    def resolve_linked_terms(term, limit):
        """Resolve term links to get full term information."""
        resolved_links = []
        for link in term.links[:limit]:
            # Resolve the link_to string to get the actual term object
            linked_term = search_module.resolve_term(link.link_to)
            if linked_term:
                resolved_links.append({
                    "term": linked_term.term,
                    "meaning": linked_term.meaning or "N/A",
                    "term_type": linked_term.type,
                    "link_score": link.score,
                    "total_times_occurred_in_knowledgebase": linked_term.total
                })
        resolved_links.sort(key=lambda x: x["link_score"], reverse=True)
        return resolved_links

    num_documents = num_documents or 5

    return search_module.search(
        query=query,
        k=15,
        threshold=0.5,
        max_depth=2,
        max_cooccurrences=5
    )



knowledge_query = Tool.from_function(
    fn=knowledge_retriever_internal,
    name="search_knowledge",
    description="Search the knowledge base for relevant documents and information.",
    enabled=True,
)


def knowledge_retriever(
    query: str,
    agent: Agent,
    num_documents: int,
    **kwargs
) -> Optional[list[dict]]:
    return [result.__dict__ for result in knowledge_retriever_internal(query=query, num_documents=num_documents)]


QuestionReasoningAgent = Agent(
    model=KnowledgeProviderGPT4oModel,
    name="QuestionReasoningAgent",
    description="Agent for reasoning about questions",
    instructions=dedent("""
        Perform the following steps to ensure accurate and relevant responses:
         1. UNDERSTAND: What is the core question being asked?
         2. ANALYZE: What are the key factors/components involved in the question?
         3. REASON: What logical connections can I make taking into account the retrieved knowledge?
         4. SYNTHESIZE: How do these elements from the knowledge base and the user's question come together to form a coherent answer?
         5. CONCLUDE: What is the most accurate/helpful response?
    """),
    markdown=True,
    retries=2,
    telemetry=False
)

MainKnowledgebaseAgent = Agent(
    id="MainKnowledgebaseAgent",
    model=KnowledgeProviderGPT4oModel,
    name="MainKnowledgebaseAgent",
    description="Agent for extracting and formatting knowledge base responses",
    telemetry=False,
    store_events=True,
    read_chat_history=True,
    db=db,
    debug_mode=True,
    cache_session=True,
    enable_agentic_memory=True,
    add_session_summary_to_context=False,
    enable_session_summaries=False,
    timezone_identifier="AT",
    search_session_history=True,
    search_knowledge=True,
    reasoning=True,
    reasoning_agent=QuestionReasoningAgent,
    knowledge_retriever=knowledge_retriever,  # type: ignore
    instructions=dedent(
        """
        Answer user question from {DOMAIN} domain for {APPLICATION} application.

        Query the knowledgebase with question to retrieve relevant documents where NUM_DOCUMENTS to retrieve: 5 or <SPECIFIED_BY_USER_IN_PROMPT>

        Guidelines:
         1. Carefully analyze the retrieved knowledge to understand the context and details and how they relate to the user's question.
         2. Always structure the response in the following format:
            1. Direct Answer: Provide a clear and concise answer to the question.
            2. <reasoning>
               Reasoning Process: Explain the logical steps taken to arrive at the  answer.
               </reasoning>
            3. <references>
                 Claims and References: List the sources of information used, including document names and page numbers.
                 <reference>Claim 1: "Information from Document A, Page B, confidence: 0-1"</reference>
                 <reference>Claim 2: "Information from Document C, Page D, confidence: 0-1"</reference>
               </references>
            4. Images: If relevant taking into account the context of the question and the captions of the images, include any images that support the answer.
                <images>
                  </image_and_caption>
                  </image_and_caption>
                </images>
            5. Tables: If relevant taking into account the context of the question and the content of the tables, include any tables that support the answer.
                <tables>
                  </markdown_table>
                </tables>

            IMPORTANT: Incorporate the <USER_PREFERENCES> if specified by the user in the prompt or the ones from the memory to further tailor the response style and format.
        """),
    memory_manager=MemoryManager(
        model=KnowledgeProviderGPT4oModel,
        db=db,
        additional_instructions=dedent("""
            - Include relevant context from previous messages e.g. memories should always quote the user claims from previous sessions/messages if user asks about the source of memory.
            - User preferences should always be specified as a list,
        """),
        memory_capture_instructions=dedent("""
            Memories should include details that could personalize ongoing interactions with the user, such as:
              - Personal facts: name, age, occupation, location, interests, etc.
              - Preferences: how user likes to receive information, style of communication, etc.
              - Direct user prompts to remember: anything user explicitly asks to recall later.

            Examples:
              - "Remember that my birthday is July 15th."
              - "I work as a software developer in Vienna."
              - "I prefer concise answers with bullet points."
              - "I enjoy hiking and photography."
              - "My favorite programming languages are Python and JavaScript.
              - "Remember that ..."

            Avoid including transient details that are unlikely to be relevant in future conversations.
            Summarize memories concisely to capture key points without excessive detail.
        """)
    )
)

MainKnowledgebaseAgentEphemeral = MainKnowledgebaseAgent.deep_copy(update={
    "id": "MainKnowledgebaseAgentEphemeral",
    "name": "MainKnowledgebaseAgentEphemeral",
    "description": "Ephemeral agent instance without persistent storage",
    "db": None,
    "memory_manager": None,
    "store_events": False,
    "search_session_history": False,
    "read_chat_history": False,
})


agno_asgi_module = AgnoOsASGIModule(
    title="Catalyst Agent OS Module",
    description="Catalyst integration module for Agno Agent OS",
    agents=[
        MainKnowledgebaseAgent,
        MainKnowledgebaseAgentEphemeral
    ],
    workflows=[],
    teams=[],
    chat=ChatConfig(
        chat_agent=MainKnowledgebaseAgent
    ),
    mcp=MCPConfig(
        agent=MainKnowledgebaseAgentEphemeral,
        name="Catalyst Knowledge MCP",
    )
)

asgi_app.mount_module(agno_asgi_module)

# class KnowledgeTypedLLMCall(ArityOneTypedCall[KnowledgeQuery, KnowledgeResponse]):
#     """TypedLLMCall implementation for knowledge base Q&A using instructor."""

#     def __init__(self, base_url: str = "http://localhost:3005/v1", model_id: str = "gpt-4o"):
#         """Initialize the TypedLLMCall with OpenAI-compatible endpoint."""
#         self._client = instructor.from_openai(
#             OpenAI(
#                 base_url=base_url,
#                 api_key="dummy"  # Dummy key for local server
#             )
#         )
#         self._model_id = model_id

#     async def call(self, x: KnowledgeQuery) -> KnowledgeResponse:
#         """Make a typed call to generate a knowledge response."""

#         system_prompt = """You are an intelligent knowledge base assistant that helps users find and understand information from processed document collections.

# CORE CAPABILITIES:
# - Analyze search results from knowledge base
# - Explain concepts, terms, and acronyms found in documents
# - Provide context and relationships between different pieces of information
# - Present information in clear, structured, and meaningful way

# WORKFLOW for answering questions:
# 1. ANALYZE: Examine provided search results to understand context and relationships
# 2. SYNTHESIZE: Combine information from multiple sources when appropriate
# 3. STRUCTURE: Organize response into logical sections
# 4. CITE: Include source document references and page numbers
# 5. RELATE: Suggest related topics for further exploration

# RESPONSE GUIDELINES:
# - Base all answers on the actual content provided in search results
# - Use clear, structured sections for complex topics
# - Include confidence level based on available information
# - Highlight key definitions and important points
# - If information is limited, acknowledge gaps honestly
# - Suggest related search terms when appropriate

# Always be helpful, thorough, and accurate. Focus on presenting actual content from documents."""

#         user_prompt = f"""Question: {x.question}

# Search Results from Knowledge Base:
# {x.search_context}

# Please analyze the search results and provide a comprehensive, well-structured answer to the question. Include proper citations and organize the response into logical sections."""

#         # Use instructor to get structured response
#         start_time = time.time()
#         logger.info(f"🤖 Starting HTTP request to LLM model: {self._model_id}")

#         response = self._client.chat.completions.create(
#             model=self._model_id,
#             messages=[
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": user_prompt}
#             ],
#             response_model=KnowledgeResponse
#         )

#         end_time = time.time()
#         duration = end_time - start_time
#         logger.info(f"🤖 HTTP request to LLM completed in {duration:.3f}s - Confidence: {response.confidence:.1%}, Sections: {len(response.sections)}, Sources: {len(response.sources)}")

#         return response

# Create the typed LLM call instance
# knowledge_llm_call = KnowledgeTypedLLMCall()


# ============================================================================
# Main Workflow with MCP Server Integration
# ============================================================================

# async def knowledge_workflow(_workflow_instance: , payload: WorkflowRunRequestWithRequestContext) -> RunResponse:
#     """
#     Knowledge base Q&A workflow that can be called via REST API or MCP tools.

#     This workflow:
#     1. First searches the knowledge base for relevant information
#     2. Then uses the TypedLLMCall to provide a comprehensive structured answer
#     3. Includes proper citations and source references
#     """
#     total_start_time = time.time()
#     logger.info(f"🚀 Starting knowledge workflow for query: '{payload.message}'")

#     # First, search the knowledge base for relevant information
#     search_start_time = time.time()
#     logger.info("📊 Beginning knowledge base search...")
#     # search_results = search_module.search(
#     #     query=message,
#     #     k=10,
#     #     threshold=0.1,
#     #     max_depth=2,
#     #     max_cooccurrences=5
#     # )
#     search_time = time.time() - search_start_time
#     logger.info(f"📊 Knowledge search completed in {search_time:.3f}s, found {len([])} results")

#     # Prepare context for the TypedLLMCall with search results
#     context_start_time = time.time()
#     search_results_text = ""
#     # if search_results:
#     #     search_results_text = "\n".join([
#     #         f"Result {i+1} (Score: {result.score:.3f}):\n"
#     #         f"Document: {result.document_name}\n"
#     #         f"Page: {result.page or 'Unknown'}\n"
#     #         f"Text: {result.text}\n"
#     #         f"Primary Terms: {[term.term for term in result.primary_terms]}\n"
#     #         f"Related Terms: {[term.term for term in result.related_terms]}\n"
#     #         for i, result in enumerate(search_results)
#     #     ])
#     # else:
#     #     search_results_text = "No results found in the knowledge base for this query."

#     # Create query object for TypedLLMCall
#     query = KnowledgeQuery(
#         question=payload.message,
#         search_context=search_results_text
#     )
#     context_time = time.time() - context_start_time
#     logger.info(f"🔄 Context preparation completed in {context_time:.3f}s")

#     # Use TypedLLMCall to get structured response
#     llm_start_time = time.time()
#     logger.info("🤖 Starting LLM call to generate structured response...")
#     structured_response = await knowledge_llm_call.call(query)
#     llm_time = time.time() - llm_start_time
#     logger.info(f"🤖 LLM call completed in {llm_time:.3f}s")

#     # Format the structured response as markdown
#     markdown_content = f"# {structured_response.summary}\n\n"

#     for section in structured_response.sections:
#         markdown_content += f"## {section.title}\n\n{section.content}\n\n"

#         if section.key_terms:
#             markdown_content += f"**Key Terms:** {', '.join(section.key_terms)}\n\n"

#     if structured_response.sources:
#         markdown_content += "## Sources\n\n"
#         for i, source in enumerate(structured_response.sources, 1):
#             page_info = f", Page {source.page_number}" if source.page_number else ""
#             markdown_content += f"{i}. {source.document_name}{page_info} (Relevance: {source.relevance_score:.2f})\n"
#         markdown_content += "\n"

#     if structured_response.related_topics:
#         markdown_content += "## Related Topics\n\n"
#         markdown_content += "\n".join([f"- {topic}" for topic in structured_response.related_topics])
#         markdown_content += "\n\n"

#     markdown_content += f"*Confidence Level: {structured_response.confidence:.1%}*"

#     # Calculate and log total workflow time
#     total_time = time.time() - total_start_time
#     logger.info(f"✅ Knowledge workflow completed in {total_time:.3f}s (search: {search_time:.3f}s, context: {context_time:.3f}s, llm: {llm_time:.3f}s)")

#     return RunResponse(
#         content=markdown_content,
#     )


# ============================================================================
# Application Setup
# ============================================================================

# def create_fastapi_application() -> FastAPI:
#     """
#     Create the knowledge base Q&A MCP server application.

#     This creates a single ASGI application that:
#     1. Uses TypedLLMCall with instructor for structured LLM responses
#     2. Hosts knowledge base tools as custom MCP tools
#     3. Provides MCP server endpoint for external clients to connect
#     4. Includes REST API endpoints for workflow execution
#     5. Ready for external MCP clients (like Claude Desktop) to connect and use tools
#     """
# Configure workflow
# workflow_config = WorkflowConfig(
#     run_callback=knowledge_workflow,
#     alias="Knowledge Base Q&A",  # Display name shown in chat UI
#     description="Knowledge base question answering with MCP server hosting knowledge tools",
#     storage=InMemoryStorage(mode="workflow")
# )

# Configure MCP with knowledge base tools (optional)
# mcp_config = MCPConfig(
#     name="knowledge-base-mcp",
#     tools={},  # Add custom tools here if needed
# )

# Create the workflow API configuration
# api_configuration = WorkflowAPIConfiguration(
#     workflow=workflow_config,
#     # mcp=mcp_config  # Uncomment to enable MCP
# )

# # Create the workflow module with proper fields
# workflow_module = WorkflowApiASGIModule(
#     title="Knowledge Base Q&A API",
#     description="Knowledge base Q&A with MCP server",
#     prefix="/workflow",
#     config=api_configuration,
# )

# Create the ASGI application
# app_config = ASGIApplicationConfig(
#     title="Knowledge Base Q&A with MCP",
#     description="Knowledge base Q&A hosting MCP server with knowledge tools",
#     version="1.0.0",
#     prefix="/kb",
#     debug=True,
#     cors_config=ASGICORSConfig(
#         allow_origins=["*"],
#         allow_methods=["*"],
#         allow_headers=["*"],
#     ),
# )

# asgi_app = ASGICoreApplication(
#     modules=[workflow_module],
#     prefix="/kb",
#     debug=True,
# )
# fastapi_app = asgi_app._app

# # Create and mount the visualization module
# visualization_module = KnowledgeVisualizationASGIModule(
#     prefix="/viz",
#     search_module=search_module
# )

# Mount the visualization module
# app.mount_module(visualization_module)

# @fastapi_app.get("/health", tags=["Health"], summary="Health Check")
# async def health_check():
#     return {
#         "status": "healthy",
#         "service": "Knowledge Base Q&A MCP Server",
#         "mcp_endpoint": "/kb/api/mcp/sse",
#         "available_mcp_tools": [
#             "send_message",
#             "search_knowledge",
#             "add_document",
#             "summarize_document",
#             "list_documents"
#         ],
#         "endpoints": {
#             "chat_ui": "/kb/api/chat",
#             "workflow_run": "/kb/api/workflow/run",
#             "workflow_info": "/kb/api/workflow",
#             "mcp_sse": "/kb/api/mcp/sse",
#             "mcp_messages": "/kb/api/mcp/messages",
#             "visualization": "/kb/viz",
#             "docs": "/kb/docs",
#             "openapi": "/kb/openapi.json",
#         },
#     }
# return fastapi_app
#
# fastapi_app = create_fastapi_application()

# AWS Lambda handler using Mangum (for serverless deployment)
# handler = Mangum(fastapi_app, lifespan="off")

def main():
    """Run the knowledge base Q&A MCP server."""
    import uvicorn

    # KnowledgeFormattingAgent.print_response("SIEMMANKO")
    # print("\n" + "=" * 70)
    # print("📚 Knowledge Base Q&A MCP Server")
    # print("=" * 70)
    # print("\n📖 Hosting Knowledge Base Tools:")
    # print("  • search_knowledge (Search Knowledge)")
    # print("  • add_document (Add Document)")
    # print("  • summarize_document (Summarize Document)")
    # print("  • list_documents (List Documents)")
    # print("\n🔌 MCP Server Features:")
    # print("  • TypedLLMCall with instructor for structured responses")
    # print("  • Streamable HTTP transport")
    # print("  • Knowledge base operations")
    # print("  • Ready for external client connections (e.g., Claude Desktop)")
    # print("\n📡 Server Endpoints:")
    # print("  • Health: http://localhost:8002/")
    # print("  • Chat UI: http://localhost:8002/kb/api/chat")
    # print("  • Visualization: http://localhost:8002/kb/viz")
    # print("  • MCP SSE: http://localhost:8002/kb/api/mcp/sse")
    # print("  • Workflow: http://localhost:8002/kb/api/workflow")
    # print("  • API Docs: http://localhost:8002/kb/docs")
    # print("\n🎨 Knowledge Visualization Features:")
    # print("  • Interactive term explorer")
    # print("  • Document browser")
    # print("  • Search interface")
    # print("  • Relationship graphs")
    # print("\n💡 Connect external MCP clients to SSE endpoint: http://localhost:8002/kb/api/mcp/sse")
    # print("\n" + "=" * 70)
    # print("🚀 Starting MCP SERVER on http://localhost:8002")
    # print("Press Ctrl+C to stop")
    # print("=" * 70 + "\n")

    uvicorn.run(asgi_app.app, host="0.0.0.0", port=8002, reload=False)


if __name__ == "__main__":
    main()
