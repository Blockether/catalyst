"""
Multi-step knowledge answering pipeline with consensus-based agents.

This module implements a 6-step pipeline for generating high-quality answers
from the knowledge base, with each step using consensus across multiple perspectives.
"""

from textwrap import dedent
from typing import List, Literal, Optional, Tuple

from agno.agent import Agent
from pydantic import BaseModel, Field

from blockether_catalyst.consensus.VotingComparison import (
    BaseModelWithReasoning,
    ComparisonStrategy,
    VotingField,
)


class SuggestedFollowUp(BaseModelWithReasoning):
    """A suggested follow-up question with context."""

    question: str = VotingField(
        description="The follow-up question the user could ask",
        comparison=ComparisonStrategy.SEMANTIC,
        threshold=0.85,
    )

class ImageAttachment(BaseModel):
    """An image attachment from search results."""

    caption: str = Field(description="Image caption or description")
    href: str = Field(description="URL to the image")
    page: int = Field(description="Page number where image appears")
    document_name: str = Field(description="Document containing the image")
    score: float = Field(default=0.0, description="Relevance score to the query (0-1)")


class TableAttachment(BaseModel):
    """A table attachment from search results."""

    content: str = Field(description="Table content in markdown format")
    page: int = Field(description="Page number where table appears")
    document_name: str = Field(description="Document containing the table")
    score: float = Field(default=0.0, description="Relevance score to the query (0-1)")
    caption: str = Field(description="Brief summary of table contents")


class AnswerProviderInput(BaseModelWithReasoning):
    """Output from unified analysis and intent clarification."""

    user_prompt: str = VotingField(
        description="Original user question or prompt",
        comparison=ComparisonStrategy.EXACT,
        threshold=1.0,
    )
    knowledge_base: str = VotingField(
        description="Knowledge base for the given user prompt",
        comparison=ComparisonStrategy.SEMANTIC,
        threshold=0.9,
    )
    intent: str = VotingField(
        description="Clear, concise statement of what the user wants to know",
        comparison=ComparisonStrategy.SEMANTIC,
        threshold=0.85,
    )
    missing_context: Optional[str] = VotingField(
        default=None,
        description="Additional contextual information that would help provide a more complete answer (e.g., application domain, use case, constraints)",
        comparison=ComparisonStrategy.SEMANTIC,
        threshold=0.75,
    )
    missing_terms: Optional[str] = VotingField(
        default=None,
        description="Specific terms, concepts, or knowledge base related aspects that need clarification to better answer the query",
        comparison=ComparisonStrategy.SEMANTIC,
        threshold=0.75,
    )
    suggested_follow_ups: Optional[List[SuggestedFollowUp]] = VotingField(
        default=None,
        description="Suggested follow-up questions the user might find valuable to explore.",
        comparison=ComparisonStrategy.DERIVED_UNORDERED,
        threshold=0.7,
    )
    available_citations: Optional[List["Citation"]] = VotingField(
        default=None,
        description="Pre-extracted citations from search results. Agents must ONLY reference these by index, never create new ones.",
        comparison=ComparisonStrategy.EXACT,
        threshold=1.0,
    )
    citation_style: Optional[str] = VotingField(
        default="inline_numeric",
        description="Citation style to use: 'inline_numeric', 'footnote', 'author_date', 'superscript'",
        comparison=ComparisonStrategy.EXACT,
        threshold=1.0,
    )
    citation_style_examples: Optional[str] = VotingField(
        default=None,
        description="Examples of how to format citations in the answer",
        comparison=ComparisonStrategy.EXACT,
        threshold=1.0,
    )


class Citation(BaseModel):
    """A single citation."""

    title: str = Field(description="Title of the document")
    author: Optional[str] = VotingField(default=None, description="Document author")

    publication_date: Optional[str] = VotingField(
        default=None,
        description="Publication date",
        comparison=ComparisonStrategy.EXACT,
        threshold=1.0,
    )
    page: int = VotingField(
        description="Page number where content appears",
        comparison=ComparisonStrategy.RANGE,
        threshold=0.95,
    )
    href: Optional[str] = VotingField(
        default=None,
        description="URL to the source document",
        comparison=ComparisonStrategy.EXACT,
        threshold=1.0,
    )
    quote: Optional[str] = VotingField(
        default=None,
        description="Relevant quote from source",
        comparison=ComparisonStrategy.SEMANTIC,
        threshold=0.8,
    )
    images: Optional[List[ImageAttachment]] = Field(default=None)
    tables: Optional[List[TableAttachment]] = Field(default=None)


class EvaluationFactor(BaseModelWithReasoning):
    """Individual evaluation factor with score and reasoning."""

    score: float = VotingField(
        ge=0.0,
        le=1.0,
        description="Score between 0 and 1",
        comparison=ComparisonStrategy.RANGE,
        threshold=0.9,
    )


class AnswerEvaluationFactors(BaseModelWithReasoning):
    """Comprehensive evaluation of answer quality across multiple dimensions."""

    # Evaluate information completeness (0-1): Do we have all parts of the answer?
    information_completeness: EvaluationFactor = VotingField(
        comparison=ComparisonStrategy.DERIVED,
        threshold=0.8,
    )

    # Evaluate source corroboration (0-1): Do multiple sources agree?
    source_corroboration: EvaluationFactor = VotingField(
        comparison=ComparisonStrategy.DERIVED,
        threshold=0.8,
    )

    # Evaluate temporal validity (0-1): Is the information current?
    temporal_validity: EvaluationFactor = VotingField(
        comparison=ComparisonStrategy.DERIVED,
        threshold=0.8,
    )

    # Evaluate contradiction presence (0-1): Any conflicts? (1 = no conflicts)
    # Note: Score 1.0 means NO contradictions, 0.0 means severe contradictions
    contradiction_presence: EvaluationFactor = VotingField(
        comparison=ComparisonStrategy.DERIVED,
        threshold=0.8,
    )

    # Evaluate factual density (0-1): Concrete facts vs vague statements
    factual_density: EvaluationFactor = VotingField(
        comparison=ComparisonStrategy.DERIVED,
        threshold=0.8,
    )

    # Evaluate coverage quality (0-1): How well does the answer address the question?
    coverage_quality: EvaluationFactor = VotingField(
        comparison=ComparisonStrategy.DERIVED,
        threshold=0.8,
    )

    @property
    def overall_confidence(self) -> float:
        """Calculate overall confidence score from all factors."""
        scores = [
            self.information_completeness.score,
            self.source_corroboration.score,
            self.temporal_validity.score,
            self.contradiction_presence.score,
            self.factual_density.score,
            self.coverage_quality.score,
        ]
        return sum(scores) / len(scores)


class AnswerOutput(BaseModelWithReasoning):
    """Final answer output with quality evaluation and follow-up suggestions."""

    answer: str = Field(description="Comprehensive, accurate, and well-structured answer to the user's question. Ensure the reasoning field is at least 50 characters long! Try to always converge on reasoning!")
    citation_indices: List[int] = VotingField(
        description="Indices (0-based) of citations from available_citations that support the answer",
        comparison=ComparisonStrategy.DERIVED_UNORDERED,
        threshold=0.75,
    )
    # This field will be populated by the workflow after mapping indices to objects
    citations: Optional[List[Citation]] = None
    evaluation_factors: AnswerEvaluationFactors = VotingField(
        comparison=ComparisonStrategy.DERIVED,
        threshold=0.8,
    )
    suggested_follow_ups: List[SuggestedFollowUp] = VotingField(
        comparison=ComparisonStrategy.DERIVED_UNORDERED,
        threshold=0.7,
    )



def _answer_provider_example(input: AnswerProviderInput, output: AnswerOutput) -> Tuple[str, AnswerOutput]:
    """Helper function for creating few-shot examples."""
    return (
        f"Input context:\n\n{input.model_dump()}",
        output,
    )


ANSWER_PROVIDER_FEW_SHOT_EXAMPLES = [
    *_answer_provider_example(
        AnswerProviderInput(
            user_prompt="What is machine learning and how does it work?",
            knowledge_base="Technical documentation covering ML fundamentals, algorithms, neural networks, deep learning, and practical applications",
            intent="User wants a comprehensive understanding of machine learning concepts and mechanisms",
            missing_context="Specific application domain or level of technical depth desired",
            missing_terms="Type of ML algorithms of interest (supervised, unsupervised, reinforcement)",
            suggested_follow_ups=None,
            reasoning="Clear technical question requiring comprehensive explanation from knowledge base. The user is asking for foundational understanding of machine learning concepts and mechanisms. This requires synthesizing information about algorithms, training processes, and practical applications to provide a complete answer.",
        ),
        AnswerOutput(
            answer=dedent(
                """
                Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It works through pattern recognition and statistical inference from data.

                **Core Mechanism:**
                Machine learning algorithms build mathematical models based on training data to make predictions or decisions. The process involves:
                1. **Data Collection**: Gathering relevant examples for the system to learn from
                2. **Feature Extraction**: Identifying important characteristics in the data
                3. **Model Training**: Algorithm adjusts parameters to minimize prediction errors
                4. **Validation**: Testing the model on unseen data to ensure generalization
                5. **Deployment**: Using the trained model for real-world predictions

                **Types of Learning:**
                - **Supervised Learning**: Learns from labeled examples (input-output pairs)
                - **Unsupervised Learning**: Discovers patterns in unlabeled data
                - **Reinforcement Learning**: Learns through interaction and feedback

                The fundamental principle is that algorithms iteratively improve their performance by adjusting internal parameters based on the error between predicted and actual outcomes, ultimately finding patterns that generalize to new, unseen data.
                """
            ),
            citation_indices=[0, 1],  # References citations at indices 0 and 1 from available_citations
            evaluation_factors=AnswerEvaluationFactors(
                information_completeness=EvaluationFactor(
                    score=0.85,
                    reasoning="The answer covers all major aspects of machine learning including definition, mechanism, types, and process. Some advanced topics like specific algorithms could be added for completeness.",
                ),
                source_corroboration=EvaluationFactor(
                    score=0.9,
                    reasoning="Multiple authoritative sources support the information provided. Both fundamental and advanced texts agree on the core concepts and mechanisms described.",
                ),
                temporal_validity=EvaluationFactor(
                    score=0.95,
                    reasoning="Information is current and reflects modern understanding of machine learning. Sources are from 2023 and include latest developments in the field.",
                ),
                contradiction_presence=EvaluationFactor(
                    score=1.0,
                    reasoning="No contradictions found. All sources present consistent information about machine learning concepts and mechanisms. The fundamental principles, algorithms, and processes described across different references align perfectly, providing a coherent and unified understanding of the subject matter without any conflicting information.",
                ),
                factual_density=EvaluationFactor(
                    score=0.88,
                    reasoning="Answer contains numerous specific facts, technical terms, and concrete examples. Each section provides verifiable information with clear definitions.",
                ),
                coverage_quality=EvaluationFactor(
                    score=0.92,
                    reasoning="Directly addresses the user's question about what ML is and how it works. Provides both conceptual understanding and technical details appropriately.",
                ),
                reasoning="Comprehensive evaluation shows high-quality answer with strong factual basis. The answer successfully addresses all aspects of machine learning with clear explanations, appropriate technical depth, and reliable sources. All evaluation metrics indicate excellent quality with no significant gaps or issues identified.",
            ),
            suggested_follow_ups=[
                SuggestedFollowUp(
                    question="What are the most common machine learning algorithms and when should each be used?",
                    reasoning="Natural progression from understanding concepts to practical application. This follow-up question would help the user move from theoretical knowledge to hands-on implementation. Understanding practical applications is crucial for fully grasping machine learning concepts and seeing how the theoretical foundations translate into real-world solutions.",
                ),
                SuggestedFollowUp(
                    question="How do neural networks differ from traditional machine learning approaches?",
                    reasoning="Important distinction for understanding modern AI capabilities and their relationship to machine learning. This question would clarify the hierarchical relationship between ML and DL, helping users understand when to use each approach and why deep learning has become dominant in certain application areas like computer vision and natural language processing.",
                ),
                SuggestedFollowUp(
                    question="What are the key challenges and limitations of machine learning systems?",
                    reasoning="Critical for realistic expectations and proper application of machine learning technologies. Understanding limitations helps practitioners make informed decisions about when ML is appropriate, what challenges to expect, and how to design robust systems that account for inherent constraints like data quality requirements, computational costs, and interpretability issues.",
                ),
            ],
            reasoning="Comprehensive answer with strong citations and high confidence across all evaluation factors. The response provides a thorough explanation of machine learning concepts, mechanisms, and applications. Multiple authoritative sources support the content, and the answer directly addresses the user's question with appropriate technical depth and clarity.",
        ),
    ),
    *_answer_provider_example(
        AnswerProviderInput(
            user_prompt="How do I implement error handling?",
            knowledge_base="Software engineering best practices, programming languages documentation, error handling patterns",
            intent="User wants practical guidance on implementing error handling in code",
            missing_context="Programming language, application type, error handling requirements",
            missing_terms="Specific error types, exception handling vs error codes, recovery strategies",
            suggested_follow_ups=None,
            reasoning="Practical implementation question requiring code examples and best practices. The user needs guidance on error handling implementation but hasn't specified the programming language or context. Must provide general principles while noting that specific implementation details would vary based on the technology stack and use case.",
        ),
        AnswerOutput(
            answer=dedent(
                """
                Error handling implementation varies by language and context, but follows common principles:

                **General Best Practices:**
                1. **Fail Fast**: Detect errors early and handle them immediately
                2. **Be Specific**: Catch specific exceptions rather than generic ones
                3. **Clean Resources**: Always release resources in finally blocks
                4. **Log Appropriately**: Record errors with context for debugging

                **Implementation Patterns:**
                - **Try-Catch-Finally**: Standard exception handling structure
                - **Error Codes**: Return values indicating success/failure
                - **Result Types**: Wrapping values with success/error states
                - **Circuit Breakers**: Preventing cascading failures

                Without knowing your specific language or use case, focus on:
                - Identifying potential failure points
                - Choosing appropriate handling strategies
                - Providing meaningful error messages
                - Ensuring system stability after errors
                """
            ),
            citation_indices=[0],  # References citation at index 0 from available_citations
            evaluation_factors=AnswerEvaluationFactors(
                information_completeness=EvaluationFactor(
                    score=0.65,
                    reasoning="Answer provides general guidance but lacks specific implementation details due to missing context about programming language and use case. More concrete examples would improve completeness.",
                ),
                source_corroboration=EvaluationFactor(
                    score=0.7,
                    reasoning="Limited source material available without specific language context. General principles are well-supported but specific implementations vary.",
                ),
                temporal_validity=EvaluationFactor(
                    score=0.9,
                    reasoning="Error handling principles are timeless and remain valid. Sources are recent and reflect current best practices across the software engineering discipline. While implementation details evolve with new programming languages and frameworks, the fundamental concepts of error detection, handling, and recovery remain consistent and applicable across all modern development contexts.",
                ),
                contradiction_presence=EvaluationFactor(
                    score=1.0,
                    reasoning="No contradictions present. All guidance aligns with established software engineering principles that have been validated through decades of software development practice. The error handling strategies presented are universally accepted across the industry, forming a coherent framework that applies regardless of specific technology choices or implementation contexts.",
                ),
                factual_density=EvaluationFactor(
                    score=0.6,
                    reasoning="Contains general principles and patterns but lacks specific code examples and concrete implementation details due to missing context.",
                ),
                coverage_quality=EvaluationFactor(
                    score=0.7,
                    reasoning="Addresses the question at a high level but cannot provide specific implementation without knowing the programming language and context.",
                ),
                reasoning="Medium confidence due to lack of specific context; general guidance provided. Without knowing the specific programming language or application requirements, the answer focuses on universal error handling principles. While these principles are valuable, more targeted advice would be possible with additional context about the implementation environment.",
            ),
            suggested_follow_ups=[
                SuggestedFollowUp(
                    question="What are the best error handling practices for Python specifically?",
                    reasoning="Narrows scope to specific language for actionable guidance. By focusing on a particular programming language, this follow-up would enable providing concrete code examples, language-specific best practices, and detailed implementation patterns. This specificity transforms general principles into immediately applicable knowledge with practical code samples.",
                ),
                SuggestedFollowUp(
                    question="How should I handle errors in distributed systems and microservices?",
                    reasoning="Advanced topic building on basic error handling concepts to explore sophisticated error management strategies. Understanding async error handling is crucial for modern applications that rely on concurrent operations, network requests, and event-driven architectures. This knowledge gap represents a significant learning opportunity for intermediate developers.",
                ),
                SuggestedFollowUp(
                    question="What's the difference between checked and unchecked exceptions?",
                    reasoning="Important conceptual distinction for proper error handling design that affects architectural decisions and code organization. Understanding when to use exceptions versus error codes influences API design, performance characteristics, and error recovery strategies. This knowledge helps developers choose appropriate error handling mechanisms for different scenarios and requirements.",
                ),
            ],
            reasoning="General answer provided due to missing context; suggested follow-ups help narrow scope for more specific guidance. The answer covers fundamental error handling concepts applicable across programming paradigms, but specific implementation details would vary significantly based on the chosen technology stack and application requirements.",
        ),
    ),
    *_answer_provider_example(
        AnswerProviderInput(
            user_prompt="Explain quantum computing advantages over classical computing",
            knowledge_base="Quantum computing principles, quantum algorithms, computational complexity, quantum supremacy demonstrations",
            intent="User wants to understand the specific advantages and use cases where quantum computing excels",
            missing_context=None,
            missing_terms=None,
            suggested_follow_ups=None,
            reasoning="Well-formed technical question with clear intent and sufficient context. The user wants to understand quantum computing's advantages over classical computing. This requires explaining quantum phenomena like superposition and entanglement, and how they enable computational advantages for specific problem classes like cryptography and optimization.",
        ),
        AnswerOutput(
            answer=dedent(
                """
                Quantum computing offers several fundamental advantages over classical computing for specific problem types:

                **Quantum Parallelism:**
                Quantum computers leverage superposition to explore multiple solution paths simultaneously. A quantum system with n qubits can represent 2^n states at once, enabling massive parallel processing that classical computers cannot achieve.

                **Key Advantages:**

                1. **Exponential Speedup for Specific Problems:**
                   - **Factorization**: Shor's algorithm factors large numbers exponentially faster
                   - **Database Search**: Grover's algorithm provides quadratic speedup
                   - **Simulation**: Natural quantum system simulation is exponentially more efficient

                2. **Quantum Entanglement:**
                   Enables correlations between qubits that have no classical equivalent, allowing for:
                   - Enhanced communication protocols
                   - Distributed computing advantages
                   - Novel algorithmic approaches

                3. **Optimization Problems:**
                   Quantum annealing and variational algorithms excel at:
                   - Portfolio optimization
                   - Route planning
                   - Machine learning feature mapping

                **Current Limitations:**
                However, quantum computers are not universally superior:
                - Limited to specific algorithm types
                - High error rates require correction
                - Extremely sensitive to environmental noise
                - Currently limited qubit counts

                **Practical Applications:**
                Quantum advantage is demonstrated in:
                - Drug discovery and molecular simulation
                - Cryptography and security
                - Financial modeling
                - Materials science

                The advantage is problem-specific rather than universal, with quantum computers complementing rather than replacing classical systems.
                """
            ),
            citation_indices=[0, 1, 2],  # References citations at indices 0, 1, and 2 from available_citations
            evaluation_factors=AnswerEvaluationFactors(
                information_completeness=EvaluationFactor(
                    score=0.93,
                    reasoning="Answer comprehensively covers quantum advantages including parallelism, specific algorithms, applications, and limitations. All major aspects addressed with appropriate detail.",
                ),
                source_corroboration=EvaluationFactor(
                    score=0.95,
                    reasoning="Multiple authoritative sources from research institutions and industry leaders. Strong agreement across sources on quantum computing advantages and applications.",
                ),
                temporal_validity=EvaluationFactor(
                    score=0.98,
                    reasoning="All sources from 2023, reflecting current state of quantum computing. Information includes latest developments and experimental results.",
                ),
                contradiction_presence=EvaluationFactor(
                    score=1.0,
                    reasoning="No contradictions found. All sources align on fundamental quantum advantages and current limitations of the technology. The scientific consensus is remarkably consistent regarding quantum computing's theoretical advantages, practical challenges, and current state of development. This alignment across academic and industry sources strengthens confidence in the presented information.",
                ),
                factual_density=EvaluationFactor(
                    score=0.91,
                    reasoning="Rich in specific facts, algorithms, complexity comparisons, and concrete examples. Includes quantitative data and verifiable claims.",
                ),
                coverage_quality=EvaluationFactor(
                    score=0.96,
                    reasoning="Excellently addresses the question with balanced coverage of advantages, limitations, and practical applications. Directly answers what was asked.",
                ),
                reasoning="High-quality comprehensive answer with excellent source support and factual accuracy. The response effectively explains quantum computing advantages through specific algorithms, applications, and theoretical foundations. All evaluation metrics indicate exceptional quality with strong evidence from authoritative sources and clear explanations of complex concepts.",
            ),
            suggested_follow_ups=[
                SuggestedFollowUp(
                    question="How do quantum error correction techniques work and why are they necessary?",
                    reasoning="Natural progression to understanding practical challenges in quantum computing implementation. This follow-up question would help users appreciate the engineering difficulties in building quantum computers, including coherence times, error rates, and the need for extreme cooling. Understanding these challenges provides context for why quantum computing remains largely experimental.",
                ),
                SuggestedFollowUp(
                    question="What specific quantum algorithms exist beyond Shor's and Grover's algorithms?",
                    reasoning="Deepens understanding of quantum computational capabilities by exploring specific algorithmic advantages. This question would lead to detailed exploration of how quantum algorithms achieve exponential speedups for certain problems, including the mathematical foundations and complexity theory that underlies quantum computational advantages over classical approaches.",
                ),
                SuggestedFollowUp(
                    question="What are the current hardware approaches to building quantum computers?",
                    reasoning="Connects theoretical advantages to practical implementation challenges that currently limit quantum computing applications. Understanding the gap between theoretical potential and practical reality helps set realistic expectations about quantum computing's near-term impact and the ongoing research efforts to overcome technical barriers like decoherence and error correction.",
                ),
                SuggestedFollowUp(
                    question="How does quantum computing impact cryptography and cybersecurity?",
                    reasoning="Important practical implication of quantum computing advances for cybersecurity and data protection strategies. This question addresses critical concerns about how quantum computers could break current encryption standards, necessitating the development of quantum-resistant cryptography. Understanding these implications is essential for long-term security planning in organizations.",
                ),
            ],
            reasoning="Excellent answer quality with comprehensive coverage and strong supporting evidence from multiple authoritative sources. The response successfully balances technical accuracy with accessibility, providing clear explanations of quantum advantages while acknowledging current limitations. Follow-up questions offered to explore specific aspects in greater depth.",
        ),
    ),
    *_answer_provider_example(
        AnswerProviderInput(
            user_prompt="Who am I and what do I like?",
            knowledge_base="No relevant information found in knowledge base.",
            intent="User asking about their personal identity and preferences",
            missing_context="This is a personal question about the user themselves",
            missing_terms="User's identity, personal preferences",
            suggested_follow_ups=None,
            reasoning="This is a personal question about the user's identity and preferences, which cannot be answered from the knowledge base. The knowledge base contains technical documentation and does not have information about individual users' personal identities or preferences. Must clearly state this limitation.",
        ),
        AnswerOutput(
            answer="I don't have information about your personal identity or preferences in my knowledge base. The knowledge base contains technical documentation and reference materials, but does not include personal information about individual users. I can only provide information based on the documents and resources available in the knowledge base.",
            citation_indices=[],  # No citations for personal questions
            evaluation_factors=AnswerEvaluationFactors(
                information_completeness=EvaluationFactor(
                    score=0.0,
                    reasoning="The knowledge base contains no information about user identities or personal preferences. Searched through available documents but found no relevant content addressing personal user information. Score of 0.0 reflects complete absence of relevant information for this query.",
                ),
                source_corroboration=EvaluationFactor(
                    score=0.0,
                    reasoning="No sources available for personal user information as this type of data is not stored in the knowledge base. The knowledge base focuses on technical documentation and reference materials rather than personal user profiles or preference data.",
                ),
                temporal_validity=EvaluationFactor(
                    score=0.0,
                    reasoning="Not applicable for this query as no temporal information exists about user identity in the knowledge base. Personal user information would require real-time user profile access which is outside the scope of the document-based knowledge base system.",
                ),
                contradiction_presence=EvaluationFactor(
                    score=1.0,
                    reasoning="No contradictions possible as no information exists about this topic in the knowledge base. The absence of information is consistently reflected across all available documents. This consistency in lack of personal user data confirms the knowledge base's technical documentation focus.",
                ),
                factual_density=EvaluationFactor(
                    score=0.0,
                    reasoning="No facts available about user identity in the knowledge base. The response accurately reflects this absence of information rather than attempting to generate speculative content. Factual density is zero due to complete lack of relevant source material.",
                ),
                coverage_quality=EvaluationFactor(
                    score=0.1,
                    reasoning="While the knowledge base cannot answer the personal identity question, the response appropriately acknowledges this limitation and explains what types of information are available. This transparent communication about system capabilities represents minimal but appropriate coverage given the constraints.",
                ),
                reasoning="Very low confidence scores accurately reflect the complete absence of relevant information in the knowledge base for this personal identity query. The response correctly identifies the limitation and avoids generating speculative or philosophical content, maintaining integrity by only stating what the knowledge base can and cannot provide.",
            ),
            suggested_follow_ups=[
                SuggestedFollowUp(
                    question="What types of technical documentation are available in the knowledge base?",
                    reasoning="This follow-up helps redirect the conversation toward the actual capabilities of the knowledge base system. By understanding what documentation is available, users can ask more targeted questions that the system can effectively answer with citations and detailed information from the available resources.",
                ),
                SuggestedFollowUp(
                    question="Can you explain any specific technical concept from your knowledge base?",
                    reasoning="Shifting focus to technical concepts allows the system to provide value through its actual strengths - synthesizing and explaining technical information from documentation. This helps users understand the appropriate use cases for the knowledge base system and how to formulate answerable queries.",
                ),
            ],
            reasoning="Answer correctly acknowledges the absence of personal user information in the knowledge base. The response maintains integrity by not generating speculative content and clearly communicates system limitations. All evaluation scores accurately reflect the lack of relevant information while the response appropriately explains what the knowledge base can provide.",
        ),
    ),
]


def create_answer_provider_agent(domain: Optional[str] = None, application: Optional[str] = None):
    """
    Create an AnswerProviderAgent with configurable domain and application context.

    Args:
        domain: The domain/field of expertise (e.g., "finance, banking, risk management")
        application: The application description including assistant name and context

    Returns:
        Configured Agent instance
    """

    # Build dynamic instructions based on domain and application
    identity_instruction = ""
    if application:
        identity_instruction = f"""
            {application}

            When asked about your identity (name, who you are, what you do), always clearly state your name and role as described above.
            """
    else:
        identity_instruction = """
            You are an expert knowledge synthesis agent. Generate comprehensive, accurate answers from the knowledge base.
            """

    # Add domain context if provided
    domain_context = ""
    if domain:
        domain_context = f"""
            Your expertise covers: {domain}
            """

    return Agent(
        id="AnswerProviderAgent",
        name="Knowledge Base Answer Provider",
        description="Generates comprehensive answers from knowledge base with quality evaluation and citations",
        telemetry=False,
        output_schema=AnswerOutput,
        debug_mode=False,
        cache_session=True,
        add_datetime_to_context=True,
        additional_input=ANSWER_PROVIDER_FEW_SHOT_EXAMPLES,  # type: ignore
        instructions=[
            dedent(
                identity_instruction + domain_context + """

            **Your Process:**

            1. **ANSWER SYNTHESIS**:
               - STRICTLY use only information available in the knowledge base
               - If no relevant information is found, clearly state: "I don't have information about [topic] in my knowledge base."
               - Integrate information from multiple sources when available
               - Structure the answer logically with clear sections
               - Use markdown formatting for readability
               - Balance depth with clarity
               - Address the user's intent directly using knowledge base content

            2. **CITATION MANAGEMENT**:
               - CRITICAL: Citations are PRE-EXTRACTED in the 'available_citations' field
               - OUTPUT: Use 'citation_indices' field with 0-based indices into available_citations
               - Example: citation_indices=[0, 2] selects available_citations[0] and available_citations[2]
               - In your answer text, use 1-based notation: [1] for index 0, [2] for index 1, etc.
               - Example: "The limit is 1M EUR [1]" with citation_indices=[0]
               - NEVER create Citation objects - only output indices
               - NEVER reference indices beyond available_citations length
               - If no suitable citation exists, use empty list: citation_indices=[]
               - Follow the citation_style for text formatting (inline_numeric, footnote, etc.)
               - The system will map your indices to actual Citation objects

            3. **QUALITY EVALUATION**:
               For each evaluation factor, provide both a score AND detailed reasoning (minimum 100 characters):

               - information_completeness: Do we have all needed information?
                 * Score: 1.0 for greetings/tool questions (complete for their purpose), 0.0-1.0 for KB-based answers
                 * Reasoning: For greetings/meta: "Complete - this greeting/tool question requires no additional information"
                 * For KB questions: Reference specific sources/documents by name, explain what information is present, what might be missing, and why you scored it this way.

               - source_corroboration: Do multiple sources agree? (N/A for non-KB questions)
                 * Score: 1.0 for greetings/tool questions (not applicable), 0.0-1.0 for KB-based answers
                 * Reasoning: For greetings/meta: "Not applicable - this type of response doesn't require source corroboration"
                 * For KB questions: Name specific sources (document titles, pages) that agree/disagree, cite any conflicting information with source references.

               - temporal_validity: Is the information current?
                 * Score: 1.0 for greetings/tool questions (always valid), 0.0-1.0 for KB-based answers
                 * Reasoning: For greetings/meta: "Always valid - greetings and tool information are not time-sensitive"
                 * For KB questions: Reference publication dates and document versions, assess time-sensitivity based on source dates.

               - contradiction_presence: Any conflicts? (1.0 = no contradictions)
                 * Score: 1.0 for greetings/tool questions (no contradictions possible), 0.0-1.0 for KB answers
                 * Reasoning: For greetings/meta: "No contradictions - single-source response without conflicting information"
                 * For KB questions: Cite specific sources and pages where contradictions exist (or confirm no contradictions).

               - factual_density: Ratio of facts to general statements
                 * Score: 0.8-1.0 for greetings/tool questions (appropriate for context), 0.0-1.0 for KB answers
                 * Reasoning: For greetings/meta: "Appropriate density - greeting/tool responses balance information with accessibility"
                 * For KB questions: Count and reference specific facts from named sources, cite concrete examples from documents.

               - coverage_quality: How well does answer address the question?
                 * Score: 0.0-1.0 based on how directly and completely the question is answered
                 * Reasoning: Reference which sources and documents address each part of the user's question, identify any gaps by noting missing information from available sources, cite specific document sections that support the answer.

               **CRITICAL REQUIREMENTS**:
               - Each reasoning field must be at least 100 characters - NO EXCEPTIONS
               - NEVER leave reasoning fields empty or with generic text
               - If knowledge base content is insufficient, explain WHY in detail (100+ chars)
               - Be realistic in scoring - perfect scores should be rare and well-justified
               - Even when information is limited, provide detailed reasoning explaining the limitations

               **EXAMPLE of proper reasoning when content is limited:**
               "The knowledge base contains no specific information about [topic]. I searched through available documents including [doc1], [doc2] but found no relevant content addressing this question. Score reflects this complete absence of relevant information."

            4. **FOLLOW-UP SUGGESTIONS**:
               Generate 3-5 meaningful follow-up questions, each with detailed reasoning (minimum 100 characters):
               - Explore different aspects of the topic
               - Progress from current understanding
               - Offer practical next steps
               - Address potential gaps or clarifications

               **For each follow-up question, provide detailed reasoning explaining:**
               - Why this question would be valuable to explore next
               - How it builds on the current answer
               - What specific insights or benefits it would provide
               - How it addresses gaps or extends understanding

            **Important Considerations:**
            - ONLY answer using information available in the knowledge base
            - If the knowledge base lacks relevant information, clearly state this limitation
            - DO NOT generate speculative, general, or philosophical responses when knowledge base content is insufficient
            - When knowledge base content is limited, acknowledge gaps and focus on what IS available
            - When confidence is low, reflect this in evaluation scores
            - Prioritize accuracy over completeness
            - Use tables and images to enhance understanding when available
            - Maintain objectivity in evaluation - be honest about limitations

            **For Iterative Refinement:**
            If this is a refinement iteration:
            - Build upon previous answer strengths
            - Address identified weaknesses
            - Incorporate additional context if provided
            - Improve clarity and structure
            - Update evaluation factors based on improvements
            """
            )
        ],
        reasoning=False,
    )


# Create default instance for backward compatibility
AnswerProviderAgent = create_answer_provider_agent()
