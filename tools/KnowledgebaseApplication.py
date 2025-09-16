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
from urllib.parse import quote

from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAILike
from agno.memory.manager import MemoryManager
from agno.reasoning.step import ReasoningSteps
from pydantic import Field
from blockether_catalyst.asgi.ASGICoreApplication import ASGICoreApplication

from blockether_catalyst.knowledge.KnowledgeSearchCore import KnowledgeSearchCore, KnowledgeSearchResult
from blockether_catalyst.integrations.agno.AgnoOsASGIModule import AgnoOsASGIModule, ChatConfig, MCPConfig
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


def _format_result_as_markdown(result: dict) -> str:
    """Format a single search result as markdown."""
    lines = []

    # Document header with metadata
    doc_name = result.get("document_name", "Unknown Document")
    page = result.get("page")
    score = result.get("score", 0)

    lines.append(f"## THIS IS PART OF {doc_name} on page {page}, RELEVANCE of this excerpt: {score:.2%}")

    # Metadata section
    metadata_parts = []
    if result.get("author"):
        metadata_parts.append(f"Author: {result['author']}")
    if result.get("publication_date"):
        metadata_parts.append(f"Published: {result['publication_date']}")

    if metadata_parts:
        lines.append("")
        lines.append("### Metadata:")
        lines.append(f"*{' | '.join(metadata_parts)}*")
        lines.append("")

    # Document link if available
    if result.get("href"):
        lines.append(f"[View Document]({result['href']})")
        lines.append("")

    # Main content
    lines.append("### Content")
    lines.append(result.get("content", "No content available"))
    lines.append("")

    # Primary terms section
    primary_terms = result.get("primary_terms", [])
    if primary_terms:
        lines.append("### 🔑 Key Terms")
        for term in primary_terms[:4]:
            term_name = term.get("term", "")
            term_meaning = term.get("meaning", "")
            term_type = term.get("term_type", "")

            if term_meaning:
                lines.append(f"- **{term_name}** ({term_type}): {term_meaning[:150]}...")
            else:
                lines.append(f"- **{term_name}** ({term_type})")
        lines.append("")

    # Related terms section (brief)
    related_terms = result.get("related_terms", [])
    if related_terms:
        lines.append("### 🔗 Related Terms")
        related_list = []
        for term in related_terms[:5]:
            term_name = term.get("term", "")

            if term_name:
                related_list.append(
                    f"`- {term_name} (type: {term.get('term_type', '')})`, meaning: {term.get('meaning', '')}")
        if related_list:
            lines.append("\n  ".join(related_list))
            lines.append("")

    # Images section
    images = result.get("images", [])
    if images:
        lines.append("### 🖼️ Images")
        for img in images[:3]:  # Limit to 3 images
            caption = img.get("caption", "Image")
            img_page = img.get("page", "")
            img_href = img.get("href", "")
            document_name = img.get("document_name")

            if img_href:
                lines.append(f"\n![{caption} - {document_name} - (Page {img_page})]({img_href})\n <center>{caption} - {document_name} - (Page {img_page})</center>\n")
            else:
                lines.append(f"{caption} (Page {img_page})")
        lines.append("")

    # Tables section
    tables = result.get("tables", [])
    if tables:
        lines.append("### 📊 Tables")
        for idx, table in enumerate(tables[:2], 1):  # Limit to 2 tables
            table_md = table.get("markdown", "")
            table_page = table.get("page", "")
            if table_md:
                lines.append(f"**Table {idx} (Page {table_page})**")
                lines.append(table_md)
                lines.append("")

    lines.append("---")
    return "\n".join(lines)


def KnowledgeRetriever(
    query: str = Field(..., title="query", description="The search query to find relevant documents."),
    max_documents: int = Field(..., title="max_documents", description="The number of documents to retrieve.")
):
    max_documents = 10

    results = search_module.search(
        query=query,
        k=max_documents,
        threshold=0.5,
        max_depth=2,
        max_cooccurrences=3
    )

    return [
        {
            "content": _format_result_as_markdown(result)
        }
        for result in results
    ]


knowledge_query = Tool.from_function(
    fn=KnowledgeRetriever,
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
    print(KnowledgeRetriever(query=query, max_documents=num_documents))
    return KnowledgeRetriever(query=query, max_documents=num_documents)


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
    cache_session=False,
    enable_agentic_memory=True,
    add_session_summary_to_context=False,
    add_name_to_context=True,
    add_datetime_to_context=True,
    add_history_to_context=True,
    enable_session_summaries=False,
    timezone_identifier="AT",
    search_session_history=True,
    search_knowledge=True,
    reasoning=False,
    introduction="Your name is Catalyst KnowledgeProvider. You are an expert in finance, banking, limits, risk management, compliance, and regulations. You help users find and understand information from a knowledge base of documents related to these topics.",
    knowledge_retriever=knowledge_retriever,  # type: ignore
    dependencies={
        "terms_count": len(search_module.linked_knowledge.terms),
        "documents_count": len(search_module.linked_knowledge.documents),
        "all_keywords_count": search_module.linked_knowledge.total_keywords,
        "all_acronyms_count": search_module.linked_knowledge.total_acronyms,
        "all_images_count": search_module.linked_knowledge.total_images,
        "all_tables_count": search_module.linked_knowledge.total_tables,
        "all_chunks_count": search_module.linked_knowledge.total_chunks,
        "documents": [{
            "document_filename": doc.document_filename,
            "document_author": doc.author,
            "document_title": doc.title,
            "document_pages": doc.total_pages,
            "document_chunks": doc.total_chunks,
            "document_images": doc.total_images,
            "document_tables": doc.total_tables,
            "document_terms": doc.total_terms,
            "document_publication_date": doc.publication_date,
            "document_href": f"{search_module._base_url}/{quote(doc.document_path, safe='/')}"
        } for doc in search_module.linked_knowledge.documents.values() if doc],

    },
    instructions=[
        dedent(f"""Answer user question from {DOMAIN} domain for {APPLICATION} application. MANDATORY: ALWAYS USE KNOWLEDGE SEARCH TO ANSWER USER QUESTIONS.
        ALWAYS RETURN VALID MARKDOWN."""),
        dedent("""
           # CORE CAPABILITIES:
            - Analyze search results from knowledge base
            - Explain concepts, terms, and acronyms found in documents
            - Provide context and relationships between different pieces of information
            - Present information in clear, structured, and meaningful way
            - YOU CAN show images, all you need to do is to embed the image URL in markdown format ![Alt text](image_url)

            WORKFLOW for answering questions:
            1. ANALYZE: Examine provided search results to understand context and relationships
            2. SYNTHESIZE: Combine information from multiple sources when appropriate
            3. STRUCTURE: Organize response into logical sections
            4. CITE: Include source document references and page numbers
            5. RELATE: Suggest related topics for further exploration

            RESPONSE GUIDELINES:
            - Base all answers on the actual content provided in search results
            - DO NOT FABRICATE INFORMATION BASED ON GUESSWORK, YOUR TASK IS TO PROVIDE FACTUAL ANSWERS BASED ON THE SEARCH RESULTS, NOTHING MORE.
            - If information is insufficient, indicate this clearly and suggest next steps.
            - Use clear, structured sections for complex topics
            - Include RELEVANCE level based on available information
            - Highlight key definitions and important points
            - If information is limited, acknowledge gaps honestly
            - Suggest related search terms when appropriate
        """),
        """Align to the following guidelines:
           1. Common questions answering strategies:
             - If the user asks about "showing/finding/listing documents/images/tables" then you should perform the corresponding query respectively with word "document", "image", "table" and return the results in a markdown table format.
             - If the user asks the questions like:
                - How many documents/images/tables you have then you should respond the "you have" to the knowledge base and respond with the message: "I don't have yet such capability to perform this operation.",
             - If the user asks mentions word "all" or "every" in the form which doesn't explicitly mentions document name then you should assume a context switch.
            2. Answer in markdown format:
               - Use headings, subheadings, bullet points, and numbered lists to organize information clearly. Embed the direct answer to the question in the beginning of the response under a heading "Answer".
               - Use blockquotes for quoting definitions or important excerpts from documents.
               - Add header "Citations" and place each citation in bullet point in the form of:
                - [document_name](document_link), document_author, publication_date, [each page number of the document relevant to the answer], (relevance: 0-1) then newline, header under which the citation is (if possible) then blockquote with the excerpt from the document.
               - Add header "Keywords and Acronyms" with related terms, their meanings in bullet points and relevance to the question. Remember to have two subheaders "Primary Terms" first and then "Related Terms".
               - Use horizontal rules (---) to separate different sections of the response.
               - Use bold and italics to highlight key terms and concepts.
               - When presenting data or statistics, use markdown tables if applicable.
               - Add section "Attachments" with images and tables if relevant to the answer. Each attachemnt (image/table) should have a caption with document name and page number.
               - As a last section add "Overall Confidence" header with the number between 0-1 indicating your confidence in the answer being a mean of the relevance of individual citations.
        """

    ],
    markdown=True,
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
    "read_chat_history": False
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
