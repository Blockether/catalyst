"""
Multi-Agent Knowledge Base Q&A System with specialized agents.

This module implements a chain of specialized agents that work together
to provide high-quality answers from the knowledge base.
"""

import logging
import os
from pathlib import Path
from typing import Optional

from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAILike
from fastmcp.tools import Tool
from mangum import Mangum
from pydantic import Field

from blockether_catalyst.asgi.ASGICoreApplication import ASGICoreApplication
from blockether_catalyst.asgi.ASGICoreModule import StaticMount
from blockether_catalyst.consensus.ConsensusTypes import ConsensusSettings
from blockether_catalyst.encoder.PotionEightEncoder import PotionEightEncoder
from blockether_catalyst.integrations.agno.AgnoOsASGIModule import (
    AgnoOsASGIModule,
    AssistantConfig,
    ChatConfig,
    MCPConfig,
)
from blockether_catalyst.knowledge.answering.StepsWorkflowCore import (
    create_steps_workflow,
    StepsWorkflowConfig,
    MessageFormatters,
)
from blockether_catalyst.knowledge.KnowledgeTypes import OptimizedSearchResponse
from blockether_catalyst.knowledge.search.SearchCore import KnowledgeSearchCore

os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DOMAIN = "finance, banking, limits, risk management, compliance, regulations, internal policies of Raiffeisen Bank International (RBI), information about LMS (Limit Management System) application, and related financial topics."
APPLICATION = "I'm OLA, an AI-powered assistant designed to provide expert guidance in the finance and banking sectors. OLA operates in the context of the banking and financial industry, in a strictly regulated environment following RBI Group's internal directives and Standard Operating Procedures (SUPs). It is designed to provide information, guidance, and support strictly based on the policies, protocols, and compliance requirements set forth by the RBI Group. All responses are aligned with the latest internal regulations to ensure accuracy and consistency for RBI Group employees."

# Separate base URL and prefix for proper path construction
BASE_URL = "http://localhost:8002"
PREFIX = "/os"
RESOURCES_URL = f"{BASE_URL}{PREFIX}"
STORAGE = SqliteDb(db_file="./agno.db")

# Initialized once to prevent the parallelism transformers issues.
PotionEightEncoder._initialize()

# Load search module
search_module = KnowledgeSearchCore.from_pickle(
    "public/knowledge_extraction/knowledge_search.pkl", resources_base_url=RESOURCES_URL
)

# Model configuration
KnowledgeProviderGPT4oModel = OpenAILike(api_key="dummy", base_url="http://localhost:3005/v1", id="gpt-4o")


def knowledge_retrieval_mcp_tool(
    query: str = Field(..., title="query", description="The search query to find relevant documents."),
    max_documents: Optional[int] = Field(title="max_documents", description="The number of documents to retrieve."),
) -> OptimizedSearchResponse:
    """Execute knowledge base search."""
    optimized_response: OptimizedSearchResponse = search_module.search(
        query=query, k=max_documents or 10, threshold=0.5, max_depth=2, max_cooccurrences=3
    )
    return optimized_response


StepBasedAgnoWorkflow = create_steps_workflow(
    search_module=search_module,
    model=KnowledgeProviderGPT4oModel,
    db=STORAGE,
    domain=DOMAIN,
    application=APPLICATION,
    config=StepsWorkflowConfig(
        min_overall_confidence=0.75,
        max_iterations=3,
        min_information_completeness=0.7,
        min_source_corroboration=0.65,
        min_factual_density=0.6,
        include_images=True,
        include_tables=True,
        max_images_per_answer=5,
        max_tables_per_answer=3,
        citation_style="inline_numeric",
        consensus_settings=ConsensusSettings(
            first_round_threshold=0.65,  # 65% for first round (achievable with 2/3 models)
            threshold=0.65,  # 65% for subsequent rounds (simple majority)
            max_rounds=3,
        ),
        message_formatters=MessageFormatters(
            greeting_formatter_fn=lambda d, a, r: (
                "Witam! Jestem OLA, asystentem AI zaprojektowanym do dostarczania eksperckiej wiedzy w sektorze finansowym i bankowym, ze specjalizacją w zarządzaniu ryzykiem, regulacjach zgodności oraz wewnętrznych ramach instytucji takich jak Raiffeisen Bank International (RBI). Moja ekspertyza obejmuje również specjalistyczne systemy, takie jak System Zarządzania Limitami (LMS) używany w zarządzaniu ryzykiem finansowym i zgodnością. Zapewniam, że moje odpowiedzi są szczegółowe, dokładne i dostosowane do Twoich potrzeb. Zapraszam do zadawania pytań z tych dziedzin."
                if "Polish" in r or "Polski" in r or "polsk" in r.lower()
                else "Hello, I am OLA, an AI-powered assistant designed to provide expert guidance in the finance and banking sectors, with a specialization in risk management, compliance regulations, and the internal frameworks of institutions like Raiffeisen Bank International (RBI). My expertise also extends to specific systems such as the Limit Management System (LMS) used in financial risk and compliance. I ensure my insights are detailed, accurate, and tailored to your needs. Please feel free to explore any topic you'd like to discuss within these domains."
            )
        ),
    )
)


async def knowledge_qa_mcp_tool(
    question: str = Field(..., title="question", description="The question to answer based on the knowledge base."),
) -> str:
    """Answer questions using the multi-agent knowledge base system."""
    result = await StepBasedAgnoWorkflow.arun(question)
    return str(result.content)


# =====================================
# MCP Tool
# =====================================

knowledge_search_tool = Tool.from_function(
    fn=knowledge_retrieval_mcp_tool,
    name="rbi_document_search",
    description="Search and retrieve specific documents from RBI's knowledge base. Returns raw document excerpts, metadata, and relevance scores for manual review. Use this when you need to find specific documents, policies, or procedures without interpretation.",
    enabled=True,
)

ola_expert_consultation_tool = Tool.from_function(
    fn=knowledge_qa_mcp_tool,
    name="ola_banking_advisor",
    description="Ask OLA expert questions and get comprehensive, interpreted answers with analysis and recommendations. Uses multi-agent consensus to provide authoritative guidance on banking operations, compliance, and risk management. Use this when you need expert interpretation, analysis, or actionable advice.",
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
    title="OLA Banking Intelligence Platform",
    description="Advanced multi-agent AI system providing expert consultation on Raiffeisen Bank International's risk management, compliance frameworks, and LMS operations through consensus-driven knowledge synthesis.",
    workflows=[StepBasedAgnoWorkflow],
    teams=[],
    chat=ChatConfig(
        assistant=AssistantConfig(
            name="OLA Iterative Assistant", short="O", runner=StepBasedAgnoWorkflow
        ),
        base_url=BASE_URL,
    ),
    mcp=MCPConfig(
        name="RBI Knowledge & Compliance Assistant",
        tools=[knowledge_search_tool, ola_expert_consultation_tool],
        instructions="""
You have access to two distinct banking intelligence tools for Raiffeisen Bank International:

🔍 DOCUMENT SEARCH (rbi_document_search):
• Purpose: Find and retrieve specific documents, policies, procedures
• Returns: Raw document excerpts with metadata and confidence scores
• Use when: User needs specific documents, wants to review source materials, or requires exact policy text
• Output: Uninterpreted search results for manual review

🧠 EXPERT CONSULTATION (ola_banking_advisor):
• Purpose: Get expert analysis, interpretation, and recommendations
• Returns: Comprehensive answers with reasoning and actionable guidance
• Use when: User needs analysis, interpretation, advice, or complex problem-solving
• Output: Authoritative expert responses with multi-agent consensus

DECISION FRAMEWORK:
- "Find me the policy on..." → Use rbi_document_search
- "What does this policy mean..." → Use ola_banking_advisor
- "Show me documents about..." → Use rbi_document_search
- "How should I handle..." → Use ola_banking_advisor
- "What are the requirements for..." → Use ola_banking_advisor
- "Where can I find information on..." → Use rbi_document_search

EXPERTISE AREAS:
• Risk Management & Compliance
• Limit Management System (LMS)
• Banking Regulations & Internal Policies
• Financial Risk Assessment
• Operational Procedures & SOPs

Always prioritize compliance and regulatory accuracy in all responses.
        """
    ),
    statics=[
        StaticMount(
            mount="/public",
            directory=Path("public"),
            name="public",
        )
    ],
)

asgi_app.mount_module(agno_asgi_module)

handler = Mangum(asgi_app.app)

if __name__ == "__main__":
    asgi_app.run(host="0.0.0.0", port=8002)
