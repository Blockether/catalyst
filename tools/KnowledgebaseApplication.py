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
from typing import List, Optional

from agno.workflow import Workflow
from agno.workflow.step import Step
from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAILike
from agno.memory.manager import MemoryManager
from mangum import Mangum
from pydantic import Field
from blockether_catalyst.asgi.ASGICoreApplication import ASGICoreApplication

from blockether_catalyst.knowledge.KnowledgeSearchCore import KnowledgeSearchCore
from blockether_catalyst.knowledge.KnowledgeTypes import NormalizedSearchResult
from blockether_catalyst.integrations.agno.AgnoOsASGIModule import (
    AgnoOsASGIModule,
    ChatConfig,
    MCPConfig,
    AssistantConfig,
)
from blockether_catalyst.asgi.ASGICoreModule import StaticMount
from pathlib import Path
from fastmcp.tools import Tool

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DOMAIN = "finanance, banking, limits, risk management, compliance, regulations"
APPLICATION = "Catalyst KnowledgeProvider"

# Separate base URL and prefix for proper path construction
BASE_URL = "http://localhost:8002"
PREFIX = "/os"
RESOURCES_URL = f"{BASE_URL}{PREFIX}"

search_module = KnowledgeSearchCore.from_pickle(
    "public/knowledge_extraction_old/knowledge_search.pkl",
    resources_base_url=RESOURCES_URL
)


KnowledgeProviderGPT4oModel = OpenAILike(
    api_key="dummy",
    base_url="http://localhost:3005/v1",
    id="gpt-4o"
)

asgi_app = ASGICoreApplication(
    title="Catalyst ASGI",
    description="Knowledge base Q&A hosting MCP server with knowledge tools",
    version="1.0.0",
    prefix="/",
    debug=False,
)

db = SqliteDb()



def KnowledgeRetriever(
    query: str = Field(..., title="query", description="The search query to find relevant documents."),
    max_documents: int = Field(..., title="max_documents", description="The number of documents to retrieve.")
) -> List[dict]:
    max_documents = 10

    # search() now returns List[NormalizedSearchResult] Pydantic models
    results: List[NormalizedSearchResult] = search_module.search(
        query=query,
        k=max_documents,
        threshold=0.5,
        max_depth=2,
        max_cooccurrences=3
    )

    # Use the markdown() method from the Pydantic model
    return [
        {
            "content": result.markdown()
        }
        for result in results
    ]


knowledge_query = Tool.from_function(
    fn=KnowledgeRetriever,
    name="search_knowledge",
    description="Search the knowledge base for relevant documents and information.",
    enabled=True,
)


def knowledge_retriever_step(
    query: str,
    num_documents: int = 10,
    **kwargs: dict
) -> Optional[List[dict]]:
    """Workflow step for retrieving knowledge."""
    print(KnowledgeRetriever(query=query, max_documents=num_documents))
    return KnowledgeRetriever(query=query, max_documents=num_documents)


# Create the main agent that will be used as a workflow step
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
    knowledge_retriever=knowledge_retriever_step,  # type: ignore
    dependencies=search_module.get_extraction_details(),
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

# Create a step wrapper for the Agent
knowledge_agent_step = Step(
    name="knowledge_agent",
    description="Process user query with knowledge base agent",
    agent=MainKnowledgebaseAgent,  # Use agent parameter instead of function
)

# Create the main workflow using the Agent step
MainKnowledgebaseWorkflow = Workflow(
    id="MainKnowledgebaseWorkflow",
    name="Knowledge Base Q&A Workflow",
    description="Workflow for extracting and formatting knowledge base responses",
    db=db,
    telemetry=False,
    debug_mode=True,
    steps=[knowledge_agent_step],  # Using the Agent step
    store_events=True,
    store_executor_outputs=True,
    cache_session=True,  # Enable session caching to ensure sessions are created and tracked
)



agno_asgi_module = AgnoOsASGIModule(
    title="Catalyst Agent OS Module",
    description="Catalyst integration module for Agno Agent OS",
    workflows=[MainKnowledgebaseWorkflow],
    teams=[],
    chat=ChatConfig(
        assistant=AssistantConfig(
            name="Catalyst KnowledgeProvider",
            short="C",
            runner=MainKnowledgebaseWorkflow
        ),
        base_url=BASE_URL
    ),
    mcp=MCPConfig(
        workflow=MainKnowledgebaseWorkflow,
        name="Catalyst Knowledge MCP",
        tools=[knowledge_query],
    ),
    statics=[
        StaticMount(
            mount="/public",
            directory=Path("public"),
            name="public",
            html=False
        )
    ]
)

asgi_app.mount_module(agno_asgi_module)

# AWS Lambda handler using Mangum (for serverless deployment)
handler = Mangum(asgi_app.app, lifespan="off")

def main() -> None:
    """Run the knowledge base Q&A MCP server."""
    import uvicorn

    uvicorn.run(asgi_app.app, host="0.0.0.0", port=8002, reload=False)


if __name__ == "__main__":
    main()
