"""
Steps-based Answer Generation Workflow.

A modern Agno workflow implementation using Steps and Loop patterns for iterative
answer generation with consensus-driven quality refinement and rich formatting.
"""

import logging
from textwrap import dedent
from typing import Any, Callable, Dict, List, Literal, Optional

from agno.db.base import BaseDb
from agno.memory.manager import MemoryManager
from agno.models.base import Model
from agno.workflow import Workflow
from agno.workflow.loop import Loop
from agno.workflow.step import Step
from agno.workflow.types import StepInput, StepOutput
from pydantic import BaseModel, ConfigDict, Field

from blockether_catalyst.consensus.Consensus import ConsensusManager
from blockether_catalyst.consensus.ConsensusTypes import ConsensusSettings
from blockether_catalyst.knowledge.answering.AnswerProviderAgent import (
    AnswerOutput,
    create_answer_provider_agent,
    AnswerProviderInput,
    Citation,
    ImageAttachment,
    SuggestedFollowUp,
    TableAttachment,
)
from blockether_catalyst.knowledge.answering.CitationExtractor import (
    CitationExtractor,
)
from blockether_catalyst.knowledge.answering.CitationFormatter import (
    CitationFormatter,
)
from blockether_catalyst.knowledge.answering.InitialAnalysisAgent import (
    InitialAnalysisAgent,
    InitialAnalysisInput,
    InitialAnalysisOutput,
)
from blockether_catalyst.knowledge.KnowledgeTypes import (
    OptimizedSearchResponse,
)
from blockether_catalyst.knowledge.search.SearchCore import KnowledgeSearchCore

logger = logging.getLogger(__name__)

# Citation style type definition
CitationStyleType = Literal[
    "inline_numeric",  # [1], [2], [3] within text
    "inline_author",  # (Smith, 2023) within text
    "footnote",  # Superscript numbers¹²³ with footnotes
    "academic",  # Formal academic style: Author (Year). Title. Source.
    "simplified",  # Simple numbered list at the end
]

# Citation style descriptions
CITATION_STYLE_DESCRIPTIONS = {
    "inline_numeric": """Numeric references within text [1], [2], etc.
    Example: 'The risk limit is 1M EUR [1] for corporate clients [2].'
    References listed at end with full details.""",
    "inline_author": """Author-date references within text (Author, Year).
    Example: 'The risk limit is 1M EUR (RBI Policy, 2023) for corporate clients (Risk Manual, 2024).'
    Full bibliography at the end.""",
    "footnote": """Superscript numbers¹²³ with detailed footnotes.
    Example: 'The risk limit is 1M EUR¹ for corporate clients².'
    Footnotes at the bottom with full source details.""",
    "academic": """Formal academic citation style.
    Example: 'According to the RBI Risk Management Policy (2023), the limit is 1M EUR.'
    Bibliography: RBI Group. (2023). Risk Management Policy. Internal Document, Version 2.1, pp. 45-47.""",
    "simplified": """Simple numbered source list at the end.
    Example: 'The risk limit is 1M EUR for corporate clients.'
    Sources:
    1. RBI Risk Management Policy - Section 4.2
    2. Corporate Banking Guidelines - Chapter 3""",
}


class MessageFormatters(BaseModel):
    """Configurable message formatters for various workflow responses."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    greeting_formatter_fn: Optional[Callable[[str, str, str], str]] = Field(
        default=None,
        description="Formatter for greeting messages. Args: (domain, application, analysis_reasoning)",
    )

    rejection_not_meaningful_formatter_fn: Optional[Callable[[], str]] = Field(
        default=None, description="Formatter for rejection when query is not meaningful"
    )

    rejection_no_info_formatter_fn: Optional[Callable[[str], str]] = Field(
        default=None,
        description="Formatter for rejection when no relevant info found. Args: (reasoning)",
    )

    error_formatter_fn: Optional[Callable[[str], str]] = Field(
        default=None, description="Formatter for error messages. Args: (error_message)"
    )

    max_iterations_reached_formatter_fn: Optional[Callable[[int], str]] = Field(
        default=None,
        description="Formatter for max iterations reached. Args: (iteration_count)",
    )

    answer_body_formatter_fn: Optional[Callable[[str, List[Citation]], str]] = Field(
        default=None,
        description="Formatter for answer body. Args: (content, citations)",
    )

    suggested_follow_ups_formatter_fn: Optional[Callable[[List[str]], str]] = Field(
        default=None,
        description="Formatter for suggested follow-ups. Args: (follow_ups)",
    )

    summary_formatter_fn: Optional[Callable[[str], str]] = Field(
        default=None,
        description="Formatter for answer summary. Args: (summary)",
    )

    references_formatter_fn: Optional[Callable[[List[Citation]], str]] = Field(
        default=None,
        description="Formatter for references section. Args: (citations)",
    )

    images_formatter_fn: Optional[Callable[[List[ImageAttachment]], str]] = Field(
        default=None,
        description="Formatter for images section. Args: (images)",
    )

    tables_formatter_fn: Optional[Callable[[List[TableAttachment]], str]] = Field(
        default=None,
        description="Formatter for tables section. Args: (tables)",
    )


class StepsWorkflowConfig(BaseModel):
    """Configuration for steps-based iterative answer generation with rich formatting."""

    # Quality thresholds
    min_overall_confidence: float = Field(default=0.75, description="Minimum overall confidence score to accept answer")
    max_iterations: int = Field(default=3, description="Maximum refinement iterations")
    min_information_completeness: float = Field(default=0.7, description="Minimum information completeness score")
    min_source_corroboration: float = Field(default=0.65, description="Minimum source corroboration score")
    min_factual_density: float = Field(default=0.6, description="Minimum factual density score")

    # Formatting options
    include_images: bool = Field(default=True, description="Include relevant images")
    include_tables: bool = Field(default=True, description="Include relevant tables")
    max_images_per_answer: int = Field(default=5, description="Maximum images to include")
    max_tables_per_answer: int = Field(default=3, description="Maximum tables to include")
    citation_style: CitationStyleType = Field(
        default="inline_numeric",
        description="""Citation format style. Options:
        - inline_numeric: [1], [2] references within text
        - inline_author: (Author, Year) references within text
        - footnote: Superscript numbers with footnotes
        - academic: Formal academic citation style
        - simplified: Simple numbered list at the end
        """,
    )

    # Consensus settings
    consensus_settings: ConsensusSettings = Field(
        default_factory=lambda: ConsensusSettings(first_round_threshold=0.65, threshold=0.65, max_rounds=3),
        description="Settings for consensus across multiple models",
    )

    # Message formatters
    message_formatters: Optional[MessageFormatters] = Field(
        default=None,
        description="Custom message formatters for various workflow responses",
    )


def create_steps_workflow(
    search_module: KnowledgeSearchCore,
    model: Model,
    db: BaseDb,
    domain: str = "general knowledge",
    application: str = "knowledge base Q&A system",
    config: Optional[StepsWorkflowConfig] = None,
) -> Workflow:
    """
    Create a Steps-based iterative answer workflow with dynamic loops using Agno's Loop pattern.

    This workflow uses proper Agno Steps pattern with:
    - Simple executor functions for each step
    - Dynamic Loop for iterative refinement
    - Quality-based loop termination
    - Shared context across steps via StepInput/StepOutput
    - Rich formatting with images, tables, and multiple citation styles
    - Consensus-driven multi-agent analysis
    """
    config = config or StepsWorkflowConfig()
    # Extract ConsensusSettings from config for use in consensus agents
    consensus_settings = config.consensus_settings  # type: ConsensusSettings

    # =====================================
    # Rich Formatting Helper Functions
    # =====================================

    def format_images(citations: List[Citation]) -> str:
        """Extract and format images from citations."""
        all_images: List[str] = []
        for citation in citations:
            if citation.images:  # No hasattr needed - it's an Optional field in Pydantic
                for img in citation.images[: config.max_images_per_answer]:
                    all_images.append(
                        f"- ![{img.caption}]({img.href})\n" f"  *{img.caption} (from {citation.title}, p.{img.page})*"
                    )

        if not all_images:
            return ""

        return "\n\n".join(all_images[: config.max_images_per_answer])

    def format_tables(citations: List[Citation]) -> str:
        """Extract and format tables from citations."""
        all_tables = []
        for citation in citations:
            if citation.tables:  # No hasattr needed - it's an Optional field in Pydantic
                for table in citation.tables[: config.max_tables_per_answer]:
                    all_tables.append(
                        f"### {table.caption}\n\n" f"{table.content}\n\n" f"*Source: {citation.title}, p.{table.page}*"
                    )

        if not all_tables:
            return ""

        return "\n\n".join(all_tables[: config.max_tables_per_answer])

    def format_citations(citations: List[Citation]) -> str:
        """Format citations according to the configured style."""
        if not citations:
            return ""

        style = config.citation_style

        if style == "inline_numeric":
            formatted = []
            for i, cite in enumerate(citations, 1):
                author = cite.author or "Unknown Author"  # Pydantic Optional field
                date = cite.publication_date or "n.d."  # Pydantic Optional field
                quote = f'\n   > "{cite.quote}"' if cite.quote else ""  # Pydantic Optional field
                formatted.append(
                    f"[{i}] {author} ({date}). *{cite.title}*. " f"Page {cite.page}. [View]({cite.href}){quote}"
                )
            return "\n\n".join(formatted)

        elif style == "inline_author":
            formatted = []
            for cite in citations:
                author = cite.author or "Unknown Author"  # Pydantic Optional field
                date = cite.publication_date or "n.d."  # Pydantic Optional field
                quote = f'\n   > "{cite.quote}"' if cite.quote else ""  # Pydantic Optional field
                formatted.append(f"{author} ({date}). *{cite.title}*. " f"Page {cite.page}. [View]({cite.href}){quote}")
            return "\n\n".join(formatted)

        elif style == "footnote":
            superscripts = "¹²³⁴⁵⁶⁷⁸⁹"
            formatted = []
            for i, cite in enumerate(citations):
                if i < len(superscripts):
                    num = superscripts[i]
                else:
                    num = f"[{i + 1}]"
                author = cite.author or "Unknown Author"  # Pydantic Optional field
                date = cite.publication_date or "n.d."  # Pydantic Optional field
                quote = f'\n   > "{cite.quote}"' if cite.quote else ""  # Pydantic Optional field
                formatted.append(
                    f"{num} {author} ({date}). *{cite.title}*. " f"Page {cite.page}. [View]({cite.href}){quote}"
                )
            return "\n\n".join(formatted)

        elif style == "academic":
            formatted = []
            for cite in citations:
                author = cite.author or "Unknown Author"  # Pydantic Optional field
                date = cite.publication_date or "n.d."  # Pydantic Optional field
                quote = f'\n   > "{cite.quote}"' if cite.quote else ""  # Pydantic Optional field
                formatted.append(
                    f"{author}. ({date}). {cite.title}. "
                    f"Internal Document, pp. {cite.page}. "
                    f"Retrieved from {cite.href}{quote}"
                )
            return "\n\n".join(formatted)

        else:  # simplified
            formatted = ["**Sources:**"]
            for i, cite in enumerate(citations, 1):
                formatted.append(f"{i}. {cite.title} - Page {cite.page}")
                if cite.quote:  # Pydantic Optional field
                    formatted.append(f'   > "{cite.quote}"')
            return "\n".join(formatted)

    def score_emoji(score: float) -> str:
        """Return emoji based on score."""
        if score >= 0.9:
            return "✅ Excellent"
        elif score >= 0.75:
            return "👍 Good"
        elif score >= 0.6:
            return "⚠️ Fair"
        else:
            return "❌ Poor"

    def format_confidence(factors) -> str:
        """Format confidence metrics with scores, assessment, and detailed reasoning in a comprehensive table."""
        return dedent(
            f"""
        | Metric | Score | Assessment | Reasoning |
        |--------|-------|------------|-----------|
        | Information Completeness | {factors.information_completeness.score:.0%} | {score_emoji(factors.information_completeness.score)} | {factors.information_completeness.reasoning} |
        | Source Corroboration | {factors.source_corroboration.score:.0%} | {score_emoji(factors.source_corroboration.score)} | {factors.source_corroboration.reasoning} |
        | Temporal Validity | {factors.temporal_validity.score:.0%} | {score_emoji(factors.temporal_validity.score)} | {factors.temporal_validity.reasoning} |
        | No Contradictions | {factors.contradiction_presence.score:.0%} | {score_emoji(factors.contradiction_presence.score)} | {factors.contradiction_presence.reasoning} |
        | Factual Density | {factors.factual_density.score:.0%} | {score_emoji(factors.factual_density.score)} | {factors.factual_density.reasoning} |
        | Coverage Quality | {factors.coverage_quality.score:.0%} | {score_emoji(factors.coverage_quality.score)} | {factors.coverage_quality.reasoning} |
        | **Overall Confidence** | **{factors.overall_confidence:.0%}** | **{score_emoji(factors.overall_confidence)}** | **Combined assessment based on all factors above** |
        """
        ).strip()

    def format_followups(followups: List[SuggestedFollowUp]) -> str:
        """Format follow-up questions with their rationales hidden under collapsible details."""
        formatted: List[str] = []
        for i, followup in enumerate(followups, 1):
            formatted.append(
                f"**{i}. {followup.question}**\n\n"
                f"<details>\n"
                f"<summary>^</summary>\n\n"
                f"{followup.reasoning}\n\n"
                f"</details>"
            )
        return "\n\n---\n\n".join(formatted)


    def get_citation_style_examples() -> str:
        """Get few-shot examples for the configured citation style."""
        style = config.citation_style
        # These are more detailed examples than CITATION_STYLE_DESCRIPTIONS
        # Used for providing comprehensive few-shot examples to the model
        examples = {
            "inline_numeric": """
                Example answer with citations:
                "The daily transaction limit is 500,000 EUR [1] for standard corporate accounts,
                but can be increased to 1M EUR [2] with proper approval from the risk department [3]."

                References:
                [1] RBI Corporate Banking Manual (2023). Transaction Limits. Page 45.
                [2] Risk Management Policy (2024). Enhanced Limits Procedure. Page 12.
                [3] Internal Memo 2024-03. Risk Approval Guidelines. Page 3.
            """,
            "inline_author": """
                Example answer with citations:
                "The daily transaction limit is 500,000 EUR (RBI Manual, 2023) for standard accounts,
                but can be increased to 1M EUR (Risk Policy, 2024) with proper approval (Internal Memo, 2024)."

                References:
                RBI Corporate Banking Manual (2023). Transaction Limits. Page 45.
                Risk Management Policy (2024). Enhanced Limits Procedure. Page 12.
                Internal Memo (2024). Risk Approval Guidelines. Page 3.
            """,
            "footnote": """
                Example answer with citations:
                "The daily transaction limit is 500,000 EUR¹ for standard corporate accounts,
                but can be increased to 1M EUR² with proper approval from the risk department³."

                Footnotes:
                ¹ RBI Corporate Banking Manual (2023). Transaction Limits. Page 45.
                ² Risk Management Policy (2024). Enhanced Limits Procedure. Page 12.
                ³ Internal Memo 2024-03. Risk Approval Guidelines. Page 3.
            """,
            "academic": """
                Example answer with citations:
                "According to the RBI Corporate Banking Manual (2023), the daily transaction limit
                is set at 500,000 EUR for standard corporate accounts. However, the Risk Management
                Policy (2024) allows for an increase to 1M EUR with proper departmental approval."

                Bibliography:
                RBI Group. (2023). Corporate Banking Manual. Internal Document, Version 4.2, pp. 45-47.
                RBI Group. (2024). Risk Management Policy. Internal Document, Version 2.1, pp. 12-15.
                RBI Group. (2024). Internal Memorandum 2024-03: Risk Approval Guidelines. pp. 3-4.
            """,
            "simplified": """
                Example answer with citations:
                "The daily transaction limit is 500,000 EUR for standard corporate accounts,
                but can be increased to 1M EUR with proper approval from the risk department."

                Sources:
                1. RBI Corporate Banking Manual - Page 45
                2. Risk Management Policy - Page 12
                3. Internal Memo 2024-03 - Page 3
            """,
        }
        return dedent(examples.get(style, examples["inline_numeric"])).strip()

    def enhance_query_from_feedback(original_query: str, feedback: List[str]) -> str:
        """Enhance search query based on feedback from previous iterations."""
        if not feedback:
            return original_query

        # Combine original query with feedback insights
        feedback_str = " ".join(feedback[-2:])  # Use last 2 feedback items
        return f"{original_query} {feedback_str}"

    def format_knowledge_base_rich(search_results: OptimizedSearchResponse) -> str:
        """Format search results into structured knowledge base context with images/tables."""
        if not search_results.results:
            return "No relevant information found in knowledge base."

        sections = []

        # Group results by document
        by_document = {}
        for result in search_results.results:
            doc_name = result.document_name
            if doc_name not in by_document:
                by_document[doc_name] = []
            by_document[doc_name].append(result)

        # Format each document's content
        for doc_name, results in by_document.items():
            section = f"**Document: {doc_name}**\n\n"
            for result in results:
                section += f"- Page {result.page} (score: {result.score:.2f}): {result.content}\n"

                # Add images if available
                if result.images:
                    section += "  Images: " + ", ".join([img.caption for img in result.images]) + "\n"

                # Add tables if available
                if result.tables:
                    section += "  Tables: " + ", ".join([tbl.caption for tbl in result.tables]) + "\n"

            sections.append(section)

        return "\n\n".join(sections)

    def get_context(step_input: StepInput) -> Dict[str, Any]:
        """Extract a mutable workflow context from step input."""
        context_data: Dict[str, Any] = {}
        if isinstance(step_input.previous_step_content, dict):
            context_data.update(step_input.previous_step_content)
        if step_input.additional_data:
            context_data.update(step_input.additional_data)
        return context_data

    def get_search_results_from_context(context: Dict[str, Any]) -> OptimizedSearchResponse:
        """Get search results from context, reconstructing from dict if needed."""
        search_results_data = context.get("search_results")
        if isinstance(search_results_data, dict):
            return OptimizedSearchResponse.model_validate(search_results_data)
        elif isinstance(search_results_data, OptimizedSearchResponse):
            return search_results_data
        else:
            # Fallback empty response
            return OptimizedSearchResponse(results=[], terms={}, total_results=0)

    def get_initial_analysis_from_context(context: Dict[str, Any]) -> Optional[InitialAnalysisOutput]:
        """Get initial analysis from context, reconstructing from dict if needed."""
        initial_analysis_data = context.get("initial_analysis")
        if isinstance(initial_analysis_data, dict):
            return InitialAnalysisOutput.model_validate(initial_analysis_data)
        elif isinstance(initial_analysis_data, InitialAnalysisOutput):
            return initial_analysis_data
        else:
            return None

    def get_answer_output_from_context(context: Dict[str, Any]) -> Optional[AnswerOutput]:
        """Get answer output from context, reconstructing from dict if needed."""
        answer_data = context.get("current_answer")
        if isinstance(answer_data, dict):
            return AnswerOutput.model_validate(answer_data)
        elif isinstance(answer_data, AnswerOutput):
            return answer_data
        else:
            return None

    def build_knowledge_context(search_results: OptimizedSearchResponse, kb_overview: str, max_items: int = 5) -> str:
        """Render a human-readable knowledge context string."""
        sections: List[str] = [kb_overview]
        if search_results.results:
            sections.append("\nTop supporting evidence:")
            for idx, result in enumerate(search_results.results[:max_items], 1):
                title = result.document_name
                snippet = result.content.replace("\n", " ").strip()
                if len(snippet) > 280:
                    snippet = snippet[:500] + "..."
                sections.append(f"{idx}. {title} — score {result.score:.2f}\n    {snippet}")
        return "\n".join(sections)

    # Create consensus managers for each response type
    initial_analysis_manager = ConsensusManager(InitialAnalysisOutput)
    answer_manager = ConsensusManager(AnswerOutput)

    # =====================================
    # Step 1: Initial Analysis
    # =====================================
    async def initial_analysis_executor(step_input: StepInput) -> StepOutput:
        """Analyze user question and determine processing route."""
        context = get_context(step_input)
        user_prompt_input = step_input.input if isinstance(step_input.input, str) else ""
        user_prompt = context.get("user_prompt") or user_prompt_input
        context["user_prompt"] = user_prompt

        logger.info("━━━ STEP 1: Initial Analysis ━━━")
        logger.info(f"Analyzing question: {user_prompt[:100]}...")
        logger.info("Temperature setting: 0.3 (deterministic mode)")

        # Perform search
        search_results = search_module.search(
            query=user_prompt,
            k=10,
            threshold=0.5,
            max_depth=2,
            max_cooccurrences=3,
        )

        context["search_results"] = search_results.model_dump(mode="json")
        linked_knowledge = search_module.linked_knowledge
        total_documents = len(linked_knowledge.documents)
        total_chunks = linked_knowledge.total_chunks
        total_terms = len(linked_knowledge.terms)
        kb_overview = (
            f"{total_documents} documents | " f"{total_chunks} chunks | " f"{total_terms} terms | " f"Domain: {domain}"
        )

        context["kb_overview"] = kb_overview
        context["knowledge_base"] = build_knowledge_context(search_results, kb_overview)

        # Analyze with consensus
        analysis_input = InitialAnalysisInput(
            user_prompt=user_prompt,
            search_results=search_results,
            knowledge_base_overview=kb_overview,
        )

        # Create consensus agent for initial analysis
        initial_consensus_agent = initial_analysis_manager.agno_consensus(
            runner=InitialAnalysisAgent,
            ids=["initial_1", "initial_2", "initial_3"],
            perspectives=[
                f"Critical validator for {domain} ensuring query validity. Remember you are {application}!",
                f"Intent clarifier understanding user needs in {domain}. Remember you are {application}!",
                f"Domain expert providing {domain} routing decisions. Remember you are {application}!",
            ],
            weights=[1.0, 1.0, 1.0],
            runner_settings=[
                {
                    "model": model,
                    "db": db,
                    "temperature": 0.3,
                    "enable_agentic_memory": True,
                    "add_history_to_context": True,
                    "enable_user_memories": True,
                    "add_memories_to_context": True,
                    "memory_manager": MemoryManager(
                        db=db,
                        model=model,
                        additional_instructions=dedent(
                        """
                            Always remember the memories of the user and their preferences when answering.
                            Use these memories to tailor your responses to their style and needs.
                            If no memories are available, proceed without them.

                            Do not make up memories - only use what is stored in the memory database.
                            Keep the memories concise and relevant to the user's preferences.

                            Example of phrases which should trigger memory saving:
                            - "Remember that I prefer concise answers."
                            - "Save this context for future reference."
                            - "I like responses that include examples."
                            - "Keep in mind that I am a beginner in this topic."
                            - "Store this information for next time."
                            - "I like..."
                            - "My preference is..."

                            Example of phrases which should trigger memory deletion:
                            - "Forget my previous preferences."
                            - "I no longer want you to remember that."
                            - "Remove my saved preferences."
                            - "Forget that I like..."
                            - "I changed my mind about..."
                            - "Delete my preference for..."

                            Example of phrases which should trigger memory recall:
                            - "What do you remember about my preferences?"
                            - "Do you remember anything about me?"
                            - "What have I told you about my interests?"
                            - "What should you keep in mind when answering?"
                            - "What do you know about me?"
                            - "Recall my preferences."
                            - "According to my preferences.
                        """
                        ),
                    ),
                }
            ],
            consensus_settings={"settings": consensus_settings},
        )

        # Run consensus analysis
        result = await initial_consensus_agent.call(analysis_input)
        analysis_output = result.final_response

        # Determine if we should continue
        should_continue = analysis_output.decision == "specific_answer"
        final_response = None

        if analysis_output.decision in ["greeting", "answer_general", "reject"]:
            logger.info(f"📌 Decision: {analysis_output.decision} - Will SKIP answer generation steps")
            logger.info(f"   Reason: {analysis_output.reasoning[:200]}...")
            formatters = config.message_formatters or MessageFormatters()

            # Use default formatters if not provided
            greeting_fn = formatters.greeting_formatter_fn or (
                lambda d, a, r: f"Hello! I'm the {d} assistant for {a}. {r}"
            )
            reject_not_meaningful_fn = formatters.rejection_not_meaningful_formatter_fn or (
                lambda: "I'm unable to understand your question. Could you please rephrase it?"
            )
            reject_no_info_fn = formatters.rejection_no_info_formatter_fn or (
                lambda r: f"I couldn't find relevant information in the knowledge base. {r}"
            )

            if analysis_output.decision == "greeting":
                final_response = greeting_fn(domain, application, analysis_output.reasoning or "")
            elif analysis_output.decision == "answer_general":
                final_response = (
                    "📚 **Knowledge Base Overview**\n\n"
                    f"{kb_overview}\n\n"
                    "You can ask specific questions about any of these topics, and I'll provide\n"
                    "detailed answers with citations and supporting materials."
                )
            elif analysis_output.decision == "reject":
                if not analysis_output.is_meaningful:
                    final_response = reject_not_meaningful_fn()
                else:
                    final_response = reject_no_info_fn(analysis_output.reasoning or "No relevant information found")
        else:
            logger.info(f"✅ Decision: {analysis_output.decision} - Will CONTINUE to answer generation")
            logger.info(f"   Intent: {analysis_output.intent}")

        context.update(
            {
                "initial_analysis": analysis_output.model_dump(mode="json"),
                "should_continue": should_continue,
                "final_response": final_response,
                "iteration_count": 0,
                "refinement_history": [],
            }
        )

        return StepOutput(content=context)

    # =====================================
    # Step 2: Generate Initial Answer
    # =====================================
    async def answer_generation_executor(step_input: StepInput) -> StepOutput:
        """Generate initial answer with quality assessment."""
        context = get_context(step_input)

        if not context.get("should_continue") or not context.get("initial_analysis"):
            logger.info(f"⏭️  STEP 2: Answer Generation - SKIPPED (should_continue={context.get('should_continue', False)})")
            return StepOutput(content=context)

        logger.info("━━━ STEP 2: Answer Generation ━━━")
        logger.info(f"Generating initial answer (iteration {context.get('iteration_count', 0) + 1})")

        initial_analysis = get_initial_analysis_from_context(context)
        if not initial_analysis:
            return StepOutput(content=context)

        search_results = get_search_results_from_context(context)

        # Prepare input for answer generation with proper citation examples
        citation_examples = get_citation_style_examples()

        # Use rich formatting for knowledge base
        knowledge_base = context.get("knowledge_base") or format_knowledge_base_rich(search_results)

        # Pre-extract citations for agents to SELECT from (not create)
        pre_extracted_citations = CitationExtractor.extract_all_citations(search_results, max_citations=10)
        citation_context = CitationExtractor.create_citation_context(search_results, max_citations=10)
        enhanced_knowledge_base = f"{knowledge_base}\n\n{citation_context}"
        context["knowledge_base"] = enhanced_knowledge_base
        context["pre_extracted_citations"] = [c.model_dump() for c in pre_extracted_citations]

        answer_input = AnswerProviderInput(
            user_prompt=context["user_prompt"],
            knowledge_base=enhanced_knowledge_base,
            intent=initial_analysis.intent,
            missing_context=initial_analysis.missing_context,
            missing_terms=initial_analysis.missing_terms,
            available_citations=pre_extracted_citations,  # Pass pre-extracted citations
            citation_style=config.citation_style,  # Pass style config
            citation_style_examples=citation_examples,
            reasoning="Generating initial answer based on retrieved knowledge base information. This first pass will synthesize the available documents to provide a comprehensive response addressing the user's query. CRITICAL: Only use citations from the 'Available citations' section provided.",
        )

        # Create answer provider agent with domain and application context
        answer_provider_agent = create_answer_provider_agent(domain=domain, application=application)

        # Create consensus agent for answer generation
        answer_consensus_agent = answer_manager.agno_consensus(
            runner=answer_provider_agent,
            ids=["answer_1", "answer_2", "answer_3"],
            perspectives=[
                f"Comprehensive synthesizer focusing on completeness for {domain}. CRITICAL: Each evaluation factor reasoning must be detailed (100+ chars) explaining the specific rationale behind scores. IMPORTANT: Always respond in the same language as the user's question.",
                f"Accuracy validator ensuring factual correctness in {domain}. CRITICAL: Each evaluation factor reasoning must be detailed (100+ chars) explaining the specific rationale behind scores. IMPORTANT: Always respond in the same language as the user's question.",
                f"Clarity optimizer making complex {domain} concepts accessible. CRITICAL: Each evaluation factor reasoning must be detailed (100+ chars) explaining the specific rationale behind scores. IMPORTANT: Always respond in the same language as the user's question.",
            ],
            weights=[1.0, 1.0, 1.0],
            runner_settings=[{"model": model, "db": db, "temperature": 0.3}],
            consensus_settings={"settings": consensus_settings},
        )

        # Run consensus answer generation
        result = await answer_consensus_agent.call(answer_input)
        answer_output = result.final_response

        # Map citation indices back to Citation objects
        if hasattr(answer_output, 'citation_indices') and answer_output.citation_indices:
            mapped_citations = []
            for idx in answer_output.citation_indices:
                if 0 <= idx < len(pre_extracted_citations):
                    mapped_citations.append(pre_extracted_citations[idx])
                else:
                    logger.warning(f"Citation index {idx} out of range (available: {len(pre_extracted_citations)})")

            # Add the mapped citations to the output (for backward compatibility)
            answer_output.citations = mapped_citations
            logger.info(f"Mapped {len(mapped_citations)} citation indices to Citation objects")

        # Update context
        context["current_answer"] = answer_output.model_dump(mode="json")
        context["iteration_count"] = context.get("iteration_count", 0) + 1

        # Add to refinement history
        refinement_history = context.get("refinement_history", [])
        refinement_history.append(
            {
                "iteration": context["iteration_count"],
                "confidence": answer_output.evaluation_factors.overall_confidence,
                "completeness": answer_output.evaluation_factors.information_completeness.score,
                "missing_context": initial_analysis.missing_context if initial_analysis else None,
                "missing_terms": initial_analysis.missing_terms if initial_analysis else None,
            }
        )
        context["refinement_history"] = refinement_history

        return StepOutput(content=context)

    # =====================================
    # Loop Steps for Refinement
    # =====================================
    def additional_search_executor(step_input: StepInput) -> StepOutput:
        """Perform additional search based on missing information."""
        context = get_context(step_input)
        current_answer = get_answer_output_from_context(context)
        initial_analysis = get_initial_analysis_from_context(context)

        logger.info(f"━━━ STEP 3A: Additional Search (Loop iteration {context.get('iteration_count', 0)}) ━━━")

        if not current_answer or not initial_analysis:
            logger.info("⏭️  Additional Search - SKIPPED (no answer or analysis to refine)")
            return StepOutput(content=context)

        # Check if we need additional search based on low completeness or missing context/terms
        needs_more_search = (
            current_answer.evaluation_factors.information_completeness.score < 0.8
            or initial_analysis.missing_context
            or initial_analysis.missing_terms
        )

        if not needs_more_search:
            logger.info("⏭️  Additional Search - SKIPPED")
            logger.info(f"   Completeness: {current_answer.evaluation_factors.information_completeness.score:.2%} (threshold: 80%)")
            logger.info(f"   Missing context: {bool(initial_analysis.missing_context)}")
            logger.info(f"   Missing terms: {bool(initial_analysis.missing_terms)}")
            return StepOutput(content=context)

        # Combine missing context and terms for search query
        missing_info_parts = []
        if initial_analysis.missing_context:
            missing_info_parts.append(initial_analysis.missing_context)
        if initial_analysis.missing_terms:
            missing_info_parts.append(initial_analysis.missing_terms)

        missing_info = " ".join(missing_info_parts) if missing_info_parts else context.get("user_prompt", "")

        logger.info("✅ Performing additional search")
        logger.info(f"   Completeness: {current_answer.evaluation_factors.information_completeness.score:.2%}")
        logger.info(f"   Missing context: {initial_analysis.missing_context or 'None'}")
        logger.info(f"   Missing terms: {initial_analysis.missing_terms or 'None'}")

        # Use enhanced query with feedback
        feedback_history = context.get("feedback_history", [])
        enhanced_query = enhance_query_from_feedback(missing_info, feedback_history)

        additional_results = search_module.search(
            query=enhanced_query,
            k=5,
            threshold=0.5,
            max_depth=1,
            max_cooccurrences=2,
        )

        # Merge additional results with existing
        search_results = get_search_results_from_context(context)
        if search_results and additional_results.results:
            existing_contents = {r.content for r in search_results.results}
            for result in additional_results.results:
                if result.content not in existing_contents:
                    search_results.results.append(result)
                    search_results.total_results += 1
            context["search_results"] = search_results.model_dump(mode="json")
            context["knowledge_base"] = format_knowledge_base_rich(search_results)

        return StepOutput(content=context)

    async def refined_answer_executor(step_input: StepInput) -> StepOutput:
        """Generate refined answer based on additional search results."""
        context = get_context(step_input)
        initial_analysis = get_initial_analysis_from_context(context)

        logger.info(f"━━━ STEP 3B: Refined Answer (Loop iteration {context.get('iteration_count', 0)}) ━━━")

        if not initial_analysis:
            logger.info("⏭️  Refined Answer - SKIPPED (no analysis available)")
            return StepOutput(content=context)

        logger.info(f"✅ Generating refined answer (iteration {context.get('iteration_count', 0) + 1})")

        # Prepare input for answer generation with proper citation examples
        citation_examples = get_citation_style_examples()

        # Get search results for citation extraction
        search_results = get_search_results_from_context(context)

        knowledge_base = context.get("knowledge_base") or build_knowledge_context(
            search_results, context.get("kb_overview", "")
        )

        # Pre-extract citations for agents to SELECT from (not create)
        pre_extracted_citations = CitationExtractor.extract_all_citations(search_results, max_citations=10)
        citation_context = CitationExtractor.create_citation_context(search_results, max_citations=10)
        enhanced_knowledge_base = f"{knowledge_base}\n\n{citation_context}"
        context["knowledge_base"] = enhanced_knowledge_base
        context["pre_extracted_citations"] = [c.model_dump() for c in pre_extracted_citations]

        answer_input = AnswerProviderInput(
            user_prompt=context["user_prompt"],
            knowledge_base=enhanced_knowledge_base,
            intent=initial_analysis.intent,
            missing_context=initial_analysis.missing_context,
            missing_terms=initial_analysis.missing_terms,
            available_citations=pre_extracted_citations,  # Pass pre-extracted citations
            citation_style=config.citation_style,  # Pass style config
            citation_style_examples=citation_examples,
            reasoning="Generating refined answer after additional search and context gathering. This iteration incorporates new information to improve accuracy, completeness, and clarity of the response. CRITICAL: Only use citations from the 'Available citations' section provided.",
        )

        # Create answer provider agent with domain and application context
        answer_provider_agent = create_answer_provider_agent(domain=domain, application=application)

        # Create consensus agent for answer generation
        answer_consensus_agent = answer_manager.agno_consensus(
            runner=answer_provider_agent,
            ids=["answer_1", "answer_2", "answer_3"],
            perspectives=[
                f"Comprehensive synthesizer focusing on completeness for {domain}. CRITICAL: Each evaluation factor reasoning must be detailed (100+ chars) explaining the specific rationale behind scores. IMPORTANT: Always respond in the same language as the user's question.",
                f"Accuracy validator ensuring factual correctness in {domain}. CRITICAL: Each evaluation factor reasoning must be detailed (100+ chars) explaining the specific rationale behind scores. IMPORTANT: Always respond in the same language as the user's question.",
                f"Clarity optimizer making complex {domain} concepts accessible. CRITICAL: Each evaluation factor reasoning must be detailed (100+ chars) explaining the specific rationale behind scores. IMPORTANT: Always respond in the same language as the user's question.",
            ],
            weights=[1.0, 1.0, 1.0],
            runner_settings=[{"model": model, "db": db, "temperature": 0.3}],
            consensus_settings={"settings": consensus_settings},
        )

        # Run consensus answer generation
        result = await answer_consensus_agent.call(answer_input)
        answer_output = result.final_response

        # Map citation indices back to Citation objects
        if hasattr(answer_output, 'citation_indices') and answer_output.citation_indices:
            mapped_citations = []
            for idx in answer_output.citation_indices:
                if 0 <= idx < len(pre_extracted_citations):
                    mapped_citations.append(pre_extracted_citations[idx])
                else:
                    logger.warning(f"Citation index {idx} out of range (available: {len(pre_extracted_citations)})")

            # Add the mapped citations to the output (for backward compatibility)
            answer_output.citations = mapped_citations
            logger.info(f"Mapped {len(mapped_citations)} citation indices to Citation objects")

        # Update context
        context["current_answer"] = answer_output.model_dump(mode="json")
        context["iteration_count"] = context.get("iteration_count", 0) + 1

        # Add to refinement history
        refinement_history = context.get("refinement_history", [])
        refinement_history.append(
            {
                "iteration": context["iteration_count"],
                "confidence": answer_output.evaluation_factors.overall_confidence,
                "completeness": answer_output.evaluation_factors.information_completeness.score,
                "missing_context": initial_analysis.missing_context if initial_analysis else None,
                "missing_terms": initial_analysis.missing_terms if initial_analysis else None,
            }
        )
        context["refinement_history"] = refinement_history

        return StepOutput(content=context)

    # =====================================
    # Quality Evaluator for Loop
    # =====================================
    def quality_evaluator(loop_outputs: List[StepOutput]) -> bool:
        """
        Evaluate if the refinement loop should end.
        Returns True to end the loop if quality thresholds are met.
        """
        if not loop_outputs:
            return False

        # Get the latest context from the last step output
        latest_output = loop_outputs[-1]
        context = latest_output.content if isinstance(latest_output.content, dict) else {}

        current_answer = get_answer_output_from_context(context)
        iteration_count = context.get("iteration_count", 0)

        logger.info(f"━━━ Loop Quality Evaluator (iteration {iteration_count}) ━━━")

        if not current_answer:
            logger.info("🛑 No answer generated, ending loop")
            return True

        # Check iteration limit
        if iteration_count >= config.max_iterations:
            logger.info(f"🛑 Reached max iterations ({config.max_iterations}), ending loop")
            return True

        # Check quality thresholds
        meets_confidence = current_answer.evaluation_factors.overall_confidence >= config.min_overall_confidence
        meets_completeness = (
            current_answer.evaluation_factors.information_completeness.score >= config.min_information_completeness
        )
        meets_corroboration = (
            current_answer.evaluation_factors.source_corroboration.score >= config.min_source_corroboration
        )
        meets_density = current_answer.evaluation_factors.factual_density.score >= config.min_factual_density

        all_met = meets_confidence and meets_completeness and meets_corroboration and meets_density

        if all_met:
            logger.info("✅ Quality thresholds MET - Ending refinement loop")
            logger.info(f"   Overall Confidence: {current_answer.evaluation_factors.overall_confidence:.2%} ✓")
            logger.info(f"   Completeness: {current_answer.evaluation_factors.information_completeness.score:.2%} ✓")
            logger.info(f"   Corroboration: {current_answer.evaluation_factors.source_corroboration.score:.2%} ✓")
            logger.info(f"   Factual Density: {current_answer.evaluation_factors.factual_density.score:.2%} ✓")
        else:
            # Generate feedback for next iteration
            feedback_items = []
            if not meets_confidence:
                feedback_items.append(f"confidence below {config.min_overall_confidence}")
            if not meets_completeness:
                feedback_items.append("incomplete information")
            if not meets_corroboration:
                feedback_items.append("weak source support")
            if not meets_density:
                feedback_items.append("low factual density")

            feedback = "Quality issues: " + "; ".join(feedback_items)

            # Store feedback in context for next iteration
            feedback_history = context.get("feedback_history", [])
            feedback_history.append(feedback)
            context["feedback_history"] = feedback_history

            logger.info("⚠️  Quality thresholds NOT MET - Continuing refinement")
            logger.info(
                f"   Confidence: {current_answer.evaluation_factors.overall_confidence:.2%} "
                f"{'✓' if meets_confidence else f'✗ (need {config.min_overall_confidence:.0%})'}"
            )
            logger.info(
                f"   Completeness: {current_answer.evaluation_factors.information_completeness.score:.2%} "
                f"{'✓' if meets_completeness else f'✗ (need {config.min_information_completeness:.0%})'}"
            )
            logger.info(
                f"   Corroboration: {current_answer.evaluation_factors.source_corroboration.score:.2%} "
                f"{'✓' if meets_corroboration else f'✗ (need {config.min_source_corroboration:.0%})'}"
            )
            logger.info(
                f"   Density: {current_answer.evaluation_factors.factual_density.score:.2%} "
                f"{'✓' if meets_density else f'✗ (need {config.min_factual_density:.0%})'}"
            )

        return all_met

    # =====================================
    # Step 4: Final Formatting
    # =====================================
    def final_formatting_executor(step_input: StepInput) -> StepOutput:
        """Format the final answer with proper citations and structure."""
        logger.info("━━━ STEP 4: Final Formatting ━━━")

        context = get_context(step_input)
        current_answer = get_answer_output_from_context(context)

        if not current_answer:
            logger.info("📝 Using early exit response (no answer to format)")
            context["final_response"] = context.get("final_response", "Unable to generate an answer.")
            return StepOutput(content=context["final_response"])

        logger.info(f"✅ Formatting answer with {len(current_answer.citations or [])} citations")

        # Format the complete response with all rich features
        sections = []

        # Main answer - apply programmatic citation style transformation
        formatted_answer = current_answer.answer
        if current_answer.citations and config.citation_style != "inline_numeric":
            # Transform citations from default inline_numeric to target style
            formatted_answer, _ = CitationFormatter.transform_citations_in_text(
                current_answer.answer,
                current_answer.citations,
                from_style="inline_numeric",
                to_style=config.citation_style
            )
        sections.append(formatted_answer)

        # Add relevant images
        if config.include_images and current_answer.citations:
            image_section = format_images(current_answer.citations)
            if image_section:
                sections.append("\n## 📸 Relevant Images\n" + image_section)

        # Add relevant tables
        if config.include_tables and current_answer.citations:
            table_section = format_tables(current_answer.citations)
            if table_section:
                sections.append("\n## 📊 Relevant Tables\n" + table_section)

        # Add citations with rich formatting
        if current_answer.citations:
            citation_section = format_citations(current_answer.citations)
            if citation_section:
                sections.append("\n## 📚 References\n" + citation_section)

        # Add quality metrics - ALWAYS collapsed for better UX
        if current_answer.evaluation_factors:
            confidence_section = format_confidence(current_answer.evaluation_factors)
            overall_score = current_answer.evaluation_factors.overall_confidence

            # Use color-coded emoji based on confidence level
            if overall_score >= 0.9:
                emoji = "🟢"
            elif overall_score >= 0.75:
                emoji = "🟡"
            elif overall_score >= 0.6:
                emoji = "🟠"
            else:
                emoji = "🔴"

            # Always show as collapsed/expandable details
            quality_indicator = (
                f"\n\n<div id=\"quality_details\">\n"
                f"<details>\n"
                f"<summary>{emoji} <b>Quality: {overall_score:.0%}</b></summary>\n\n"
                f"{confidence_section}\n\n"
                f"</details>\n"
                f"</div>"
            )
            sections.append(quality_indicator)

        # Add follow-up suggestions
        if current_answer.suggested_follow_ups:
            followup_section = format_followups(current_answer.suggested_follow_ups)
            sections.append("\n## 💡 Suggested Follow-up Questions\n" + followup_section)

        # Add quality metadata if in debug mode
        if logger.isEnabledFor(logging.DEBUG):
            sections.append(
                f"\n\n---\n*Debug: {context.get('iteration_count', 0)} iterations, {context.get('feedback_history', [])}*"
            )

        context["final_response"] = "\n\n".join(sections)

        # Log completion summary
        logger.info("━━━ Workflow Complete ━━━")
        logger.info(f"   Total iterations: {context.get('iteration_count', 0)}")
        logger.info(f"   Decision type: {context.get('initial_analysis', {}).get('decision', 'unknown')}")

        return StepOutput(content=context["final_response"])

    # =====================================
    # Create the workflow with Steps and Loop
    # =====================================

    # Create Steps with executor functions
    initial_analysis_step = Step(
        name="initial_analysis",
        description="Analyze user question for validity and intent",
        executor=initial_analysis_executor,
    )

    answer_generation_step = Step(
        name="answer_generation",
        description="Generate initial answer from search results",
        executor=answer_generation_executor,
    )

    additional_search_step = Step(
        name="additional_search",
        description="Search for missing information",
        executor=additional_search_executor,
    )

    refined_answer_step = Step(
        name="refined_answer",
        description="Generate refined answer with additional context",
        executor=refined_answer_executor,
    )

    final_formatting_step = Step(
        name="final_formatting",
        description="Format final answer with citations and structure",
        executor=final_formatting_executor,
    )

    # Create the refinement loop
    refinement_loop = Loop(
        name="refinement_loop",
        description="Iteratively refine answer until quality thresholds are met",
        steps=[
            additional_search_step,
            refined_answer_step,
        ],
        end_condition=quality_evaluator,
        max_iterations=config.max_iterations,
    )

    # Create workflow with steps and loop
    workflow = Workflow(
        id="StepsWorkflow",
        name="Steps-based Knowledge Answer Workflow",
        description="Modern workflow with Steps and Loop patterns for quality-driven answer generation",
        db=db,
        telemetry=False,
        debug_mode=False,
        steps=[
            initial_analysis_step,  # Step 1: Initial analysis
            answer_generation_step,  # Step 2: Generate first answer
            refinement_loop,  # Step 3: Dynamic refinement loop
            final_formatting_step,  # Step 4: Format final answer
        ],
        store_events=True,
        store_executor_outputs=True,
        cache_session=True,
    )

    return workflow
