"""
Multi-step knowledge answering pipeline with consensus-based agents.

This module implements a 6-step pipeline for generating high-quality answers
from the knowledge base, with each step using consensus across multiple perspectives.
"""

from textwrap import dedent
from typing import Literal, Optional, Tuple

from agno.agent import Agent
from agno.memory.manager import MemoryManager
from pydantic import BaseModel, Field

from blockether_catalyst.consensus.VotingComparison import (
    BaseModelWithReasoning,
    ComparisonStrategy,
    VotingField,
)
from blockether_catalyst.knowledge.KnowledgeTypes import (
    CompactSearchResult,
    OptimizedSearchResponse,
)


class InitialAnalysisOutput(BaseModelWithReasoning):
    """Output from unified analysis and intent clarification."""

    # Core analysis
    is_meaningful: bool = VotingField(
        description="True if the prompt is meaningful in the context of the knowledge base and the intent is resolvable (not gibberish, spam, random words or offensive)",
        threshold=0.8,
    )

    is_greeting: bool = VotingField(
        description="True if the user is greeting (e.g., 'hello', 'hi', 'good morning', 'hey') or asking who we are",
        threshold=0.8,
    )

    is_answer_general_question: bool = VotingField(
        description="True if asking about KB contents/topics rather than specific information",
        threshold=0.8,
    )

    is_answerable: bool = VotingField(
        description="True if we have relevant information to answer the question",
        threshold=0.8,
    )

    intent: str = VotingField(
        description="Clear, concise statement of what the user wants to know",
        threshold=0.7,
        comparison=ComparisonStrategy.SEMANTIC,
    )

    user_preferences: Optional[str] = VotingField(
        default=None,
        description="Any known user preferences or answer style memories that should be considered when answering. Should be retrieved from the memory via the MemoryManager. Otherwise should be empty!",
        comparison=ComparisonStrategy.SEMANTIC,
        threshold=0.6,
    )

    missing_context: Optional[str] = VotingField(
        default=None,
        description="Additional contextual information that would help provide a more complete answer (e.g., application domain, use case, constraints)",
        threshold=0.7,
        comparison=ComparisonStrategy.SEMANTIC,
    )

    missing_terms: Optional[str] = VotingField(
        default=None,
        description="Specific terms, concepts, or aspects that need clarification to better answer the query",
        threshold=0.7,
        comparison=ComparisonStrategy.SEMANTIC,
    )

    decision: Literal["greeting", "answer_general", "specific_answer", "reject"] = VotingField(
        description="What to do now: 'greeting' (for greetings), 'answer_general', 'specific_answer', or 'reject'",
        comparison=ComparisonStrategy.EXACT,
    )



class InitialAnalysisInput(BaseModel):
    """Input for unified analysis agent."""

    user_prompt: str = Field(description="Original user question or prompt")
    search_results: OptimizedSearchResponse = Field(description="Raw search results from the knowledge base")
    knowledge_base_overview: str = Field(description="Overview of KB contents for general questions")


def _initial_analysis_example(
    input: InitialAnalysisInput, output: InitialAnalysisOutput
) -> Tuple[str, InitialAnalysisOutput]:
    """Helper function for creating few-shot examples."""
    return (
        f"User query:\n\n{input.model_dump()}",
        output,
    )


INITIAL_ANALYSIS_FEW_SHOT_EXAMPLES = [
    *_initial_analysis_example(
        InitialAnalysisInput(
            user_prompt="Hello",
            search_results=OptimizedSearchResponse(results=[], total_results=0),
            knowledge_base_overview="Knowledge base for finance, banking, risk management, and compliance",
        ),
        InitialAnalysisOutput(
            is_meaningful=True,
            is_greeting=True,
            is_answer_general_question=False,
            is_answerable=False,
            intent="User is greeting the system",
            missing_context=None,
            missing_terms=None,
            decision="greeting",
            reasoning="The user has provided a simple greeting. This is a social interaction rather than a knowledge request. The appropriate response is to greet back and introduce the system's capabilities based on the domain and application context. This helps establish rapport and sets expectations for the types of questions the system can answer.",
        ),
    ),
    *_initial_analysis_example(
        InitialAnalysisInput(
            user_prompt="Cześć",
            search_results=OptimizedSearchResponse(results=[], total_results=0),
            knowledge_base_overview="Knowledge base for finance, banking, risk management, and compliance",
        ),
        InitialAnalysisOutput(
            is_meaningful=True,
            is_greeting=True,
            is_answer_general_question=False,
            is_answerable=False,
            intent="User is greeting the system in Polish",
            missing_context=None,
            missing_terms=None,
            decision="greeting",
            reasoning="The user has provided a simple greeting in Polish ('Cześć' means 'Hello'). This is a social interaction rather than a knowledge request. The appropriate response is to greet back in the same language and introduce the system's capabilities. Responding to greetings in the user's language shows cultural awareness and helps establish better rapport.",
        ),
    ),
    *_initial_analysis_example(
        InitialAnalysisInput(
            user_prompt="What topics are covered in the knowledge base?",
            search_results=OptimizedSearchResponse(results=[], total_results=0),
            knowledge_base_overview="15 documents | Machine Learning (5), NLP (3), Computer Vision (4), Data Science (3) | 1,234 pages | 567 terms",
        ),
        InitialAnalysisOutput(
            is_meaningful=True,
            is_greeting=False,
            is_answer_general_question=True,
            is_answerable=True,
            intent="User wants an overview of topics and content available in the knowledge base",
            missing_context=None,
            missing_terms=None,
            decision="answer_general",
            reasoning="This is a clear meta-question about the knowledge base contents itself rather than seeking specific information from within it. The user wants to understand what topics and types of information are available. The agent should use the knowledge_base_overview input to generate a comprehensive and helpful response describing the knowledge base contents.",
        ),
    ),
    *_initial_analysis_example(
        InitialAnalysisInput(
            user_prompt="What is gradient descent?",
            search_results=OptimizedSearchResponse(
                results=[
                    CompactSearchResult(
                        score=0.92,
                        document_name="ml_basics.pdf",
                        content="Gradient descent is an optimization algorithm used to minimize the cost function...",
                    )
                ],
                total_results=1,
            ),
            knowledge_base_overview="No knowledge base overview provided",
        ),
        InitialAnalysisOutput(
            is_meaningful=True,
            is_greeting=False,
            is_answer_general_question=False,
            is_answerable=True,
            intent="User wants to understand the gradient descent optimization algorithm",
            missing_context="Application domain or use case (e.g., training neural networks, linear regression)",
            missing_terms="Specific aspects like learning rate, convergence criteria, or batch vs stochastic gradient descent",
            decision="specific_answer",
            reasoning="This is a specific technical question with relevant search results showing high-quality matches from machine learning documentation. The knowledge base contains detailed information about gradient descent that can be used to provide a comprehensive answer. We should process through the full pipeline to synthesize the information effectively.",
        ),
    ),
    *_initial_analysis_example(
        InitialAnalysisInput(
            user_prompt="blah blah xyz 123 !!!",
            search_results=OptimizedSearchResponse(results=[], total_results=0),
            knowledge_base_overview="No knowledge base overview provided",
        ),
        InitialAnalysisOutput(
            is_meaningful=False,
            is_greeting=False,
            is_answer_general_question=False,
            is_answerable=False,
            intent="No clear intent - appears to be random text",
            missing_context=None,
            missing_terms=None,
            decision="reject",
            reasoning="The prompt contains only random characters and symbols without forming any coherent question or meaningful request. This appears to be gibberish or potentially spam input that cannot be processed into a knowledge base query. The input lacks semantic meaning and structure needed for information retrieval.",
        ),
    ),
]


InitialAnalysisAgent = Agent(
    id="InitialAnalysisAgent",
    name="Initial Question Analyzer",
    description="Analyzes user question to determine validity, relevance, and clarifies intent",
    telemetry=False,
    output_schema=InitialAnalysisOutput,
    debug_mode=False,
    cache_session=True,
    add_datetime_to_context=True,
    timezone_identifier="Europe/Vienna",
    additional_input=INITIAL_ANALYSIS_FEW_SHOT_EXAMPLES,  # type: ignore
    instructions=[
        dedent(
            """
        Given the user question and search results from the knowledge base:

        1. VALIDATION - Determine if the question is valid:
           - Is it meaningful? (not gibberish, spam, trolling, offensive)
           - Is it a greeting? (e.g., "hello", "hi", "good morning", "who are you", "cześć", "hej", "dzień dobry", "witam", "siema")
             * IMPORTANT: Detect greetings in ANY language (English, Polish, German, etc.)
             * Simple greetings should ALWAYS be classified as 'greeting', not 'specific_answer'
           - Is it a general KB overview question? (e.g., "What topics are covered?, What knowledge can you answer/provide?")
           - Can we answer it with available knowledge?

        2. INTENT CLARIFICATION:
           - Clarify what the user actually wants to know.
           - Identify missing context and terms that would improve the answer
           - Determine the processing route

        missing_context: Additional contextual information that would help provide a more complete answer.
        missing_terms: Specific terms, concepts, or aspects that need clarification.
        Keep both fields concise and practical - focus on information that would significantly enhance the response.

        3. Retrieve the USER ANSWER PREFERENCES/MEMORIES if available...

        4. ROUTING DECISIONS:
           - 'greeting': For greetings and introductions
             * User is greeting or asking who we are
             * CRITICAL: Any simple greeting (hello, hi, cześć, hej, etc.) MUST be classified as 'greeting'
             * DO NOT process greetings as 'specific_answer' even if search results exist
             * Respond with friendly introduction based on domain/application
           - 'answer_general': For meta-questions about KB contents
             * Use knowledge_base_overview input to generate answer_general_answer
             * Create a helpful response describing what's in the KB
           - 'specific_answer': For specific content questions
             * Process through full pipeline
             * Only use this for actual questions that need KB information
           - 'reject': For invalid/off-topic questions
             * Not meaningful or no relevant information

        REMEMBER THAT EACH REASONING FIELD SHOULD BE AT LEAST 100 characters long! Try to always converge on reasoning!
        Focus on understanding user needs. Be practical about information gaps.
        """
        )
    ],
    reasoning=False,
)
