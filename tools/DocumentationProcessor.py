#!/usr/bin/env python3
"""
Unified Documentation Processor using Catalyst Consensus with Agno Agents and Serena.

This tool leverages Serena MCP server for semantic code analysis:
1. Generate documentation - Serena provides symbol-level understanding
2. Assess documentation - Verifies claims against actual code structure
3. Refine documentation - Improves iteratively with Serena's memories

Serena handles its own initialization and onboarding - we just use its tools.
"""

import ast
import asyncio
import json
import logging
import re
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, List, Literal, Optional, cast

from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.base import Model
from agno.workflow import Workflow
from agno.workflow.loop import Loop
from agno.workflow.step import Step
from agno.workflow.types import StepInput, StepOutput
from pydantic import BaseModel, Field

from blockether_catalyst.consensus.Consensus import ConsensusManager
from blockether_catalyst.consensus.ConsensusTypes import ConsensusSettings
from blockether_catalyst.consensus.VotingComparison import BaseModelWithReasoning, ComparisonStrategy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mode of operation
OperationMode = Literal["generate", "assess", "refine"]


# Note: Serena tools are accessed directly via MCP by agents
# No wrapper tools needed - agents call mcp__serena__* functions directly


# ============================================================================
# SHARED MODELS WITH VOTING COMPARISON
# ============================================================================

class DocumentationOutput(BaseModelWithReasoning):
    """Output model for documentation generation/assessment with consensus voting."""

    # Content sections
    content: str = Field(
        default="",
        description="Main documentation content",
        json_schema_extra={"voting_comparison": {"strategy": ComparisonStrategy.SEMANTIC, "threshold": 0.6}},
    )

    # Quality scores with RANGE comparison
    clarity_score: float = Field(
        default=5.0,
        ge=0,
        le=10,
        description="Clarity and readability score",
        json_schema_extra={"voting_comparison": {"strategy": ComparisonStrategy.RANGE, "tolerance": 1.0}},
    )

    completeness_score: float = Field(
        default=5.0,
        ge=0,
        le=10,
        description="Documentation completeness",
        json_schema_extra={"voting_comparison": {"strategy": ComparisonStrategy.RANGE, "tolerance": 1.0}},
    )

    accuracy_score: float = Field(
        default=5.0,
        ge=0,
        le=10,
        description="Technical accuracy",
        json_schema_extra={"voting_comparison": {"strategy": ComparisonStrategy.RANGE, "tolerance": 1.0}},
    )

    usability_score: float = Field(
        default=5.0,
        ge=0,
        le=10,
        description="Developer usability",
        json_schema_extra={"voting_comparison": {"strategy": ComparisonStrategy.RANGE, "tolerance": 1.0}},
    )

    # Boolean checks with EXACT comparison
    has_examples: bool = Field(
        default=False,
        description="Contains code examples",
        json_schema_extra={"voting_comparison": {"strategy": ComparisonStrategy.EXACT}},
    )

    has_quickstart: bool = Field(
        default=False,
        description="Has quickstart guide",
        json_schema_extra={"voting_comparison": {"strategy": ComparisonStrategy.EXACT}},
    )

    has_api_docs: bool = Field(
        default=False,
        description="Has API documentation",
        json_schema_extra={"voting_comparison": {"strategy": ComparisonStrategy.EXACT}},
    )

    examples_valid: bool = Field(
        default=False,
        description="Examples are syntactically valid",
        json_schema_extra={"voting_comparison": {"strategy": ComparisonStrategy.EXACT}},
    )

    # Lists with SEMANTIC comparison
    sections_generated: List[str] = Field(
        default_factory=list,
        description="Documentation sections created/found",
        json_schema_extra={"voting_comparison": {"strategy": ComparisonStrategy.SEMANTIC, "threshold": 0.7}},
    )

    missing_sections: List[str] = Field(
        default_factory=list,
        description="Missing documentation sections",
        json_schema_extra={"voting_comparison": {"strategy": ComparisonStrategy.SEMANTIC, "threshold": 0.7}},
    )

    improvements_needed: List[str] = Field(
        default_factory=list,
        description="Suggested improvements",
        json_schema_extra={"voting_comparison": {"strategy": ComparisonStrategy.SEMANTIC, "threshold": 0.6}},
    )

    verified_claims: List[str] = Field(
        default_factory=list,
        description="Claims verified against code",
        json_schema_extra={"voting_comparison": {"strategy": ComparisonStrategy.SEMANTIC, "threshold": 0.7}},
    )

    invalid_claims: List[str] = Field(
        default_factory=list,
        description="Claims that are incorrect",
        json_schema_extra={"voting_comparison": {"strategy": ComparisonStrategy.SEMANTIC, "threshold": 0.7}},
    )

    # Metadata
    files_analyzed: int = Field(
        default=0,
        description="Number of files analyzed",
        json_schema_extra={"voting_comparison": {"strategy": ComparisonStrategy.EXACT}},
    )

    patterns_found: int = Field(
        default=0,
        description="Number of patterns discovered",
        json_schema_extra={"voting_comparison": {"strategy": ComparisonStrategy.EXACT}},
    )

    confidence: float = Field(
        default=0.5,
        ge=0,
        le=1,
        description="Confidence in output",
        json_schema_extra={"voting_comparison": {"strategy": ComparisonStrategy.RANGE, "tolerance": 0.1}},
    )

    summary: str = Field(
        default="",
        description="Overall summary",
        json_schema_extra={"voting_comparison": {"strategy": ComparisonStrategy.SEMANTIC, "threshold": 0.7}},
    )


class ProcessorInput(BaseModel):
    """Input for documentation processing agents."""

    mode: OperationMode = Field(description="Operation mode: generate/assess/refine")
    content: Optional[str] = Field(default=None, description="Existing documentation content (for assess/refine)")
    codebase_context: Optional[Dict[str, Any]] = Field(default=None, description="Analyzed codebase context")
    target_module: Optional[str] = Field(default=None, description="Specific module to focus on")
    requirements: List[str] = Field(default_factory=list, description="Specific requirements or focus areas")


class DocumentationAgent(Agent):
    """Agent that processes documentation with Serena MCP tool support."""

    def __init__(
        self,
        id: str,
        model: Model,
        db: SqliteDb,
        perspective: str,
    ):
        """Initialize documentation agent with Serena integration."""
        super().__init__(
            id=id,
            name=f"Documentation {perspective} Agent",
            model=model,
            description=f"Processes documentation from {perspective} perspective using Serena",
            instructions=[
                f"You are a documentation expert focusing on {perspective}.",
                "",
                "BEFORE STARTING: Check if Serena is ready:",
                "1. Call mcp__serena__check_onboarding_performed()",
                "2. If false, call mcp__serena__onboarding()",
                "",
                "Available Serena MCP tools for semantic analysis:",
                "- mcp__serena__list_dir: List directory structure",
                "- mcp__serena__find_file: Find files by pattern",
                "- mcp__serena__get_symbols_overview: Get symbol overview of files",
                "- mcp__serena__find_symbol: Find specific symbols with depth",
                "- mcp__serena__find_referencing_symbols: Find symbol references",
                "- mcp__serena__search_for_pattern: Search patterns with regex",
                "- mcp__serena__read_memory: Read project memories",
                "- mcp__serena__write_memory: Store documentation insights",
                "- mcp__serena__list_memories: List available memories",
                "",
                "Use these Serena tools to gather deep semantic context.",
                "Return a structured DocumentationOutput response.",
            ],
            tools=[],  # Agents use MCP tools directly
            db=db,
            telemetry=False,
        )
        self.perspective = perspective

    def run_sync(self, input_data: ProcessorInput) -> DocumentationOutput:
        """Run synchronous processing."""
        prompt = self._build_prompt(input_data)
        result = self.run(prompt)
        return self._parse_result(result)

    def _build_prompt(self, input_data: ProcessorInput) -> str:
        """Build prompt based on mode and perspective."""
        # Constants for content limits
        CONTENT_PREVIEW_LIMIT = 3000
        CONTENT_REFINE_LIMIT = 2000

        base_prompt = ""

        if input_data.mode == "generate":

            base_prompt = f"""
            Generate comprehensive documentation for the codebase using Serena's semantic analysis.

            Target module: {input_data.target_module or "Full codebase"}

            First, ensure Serena is ready:
            1. Call mcp__serena__check_onboarding_performed()
            2. If false, call mcp__serena__onboarding()

            Then use Serena MCP tools for semantic analysis:
            1. mcp__serena__list_dir to explore structure
            2. mcp__serena__find_file to locate key files (*Core.py, *Types.py, etc)
            3. mcp__serena__get_symbols_overview for each major file
            4. mcp__serena__find_symbol with depth=2 for important classes
            5. mcp__serena__search_for_pattern to find patterns and conventions
            6. mcp__serena__read_memory to get existing project insights
            7. mcp__serena__write_memory to store new documentation insights

            Create documentation with:
            - Overview based on symbol analysis
            - Architecture from Core/Types/Internal patterns
            - API reference from symbol documentation
            - Code examples validated with validate_example
            - Installation and quickstart guides
            - Relationships from find_referencing_symbols
            """

        elif input_data.mode == "assess":
            base_prompt = f"""
            Assess the following documentation for quality and accuracy using Serena.

            Documentation:
            {input_data.content[:CONTENT_PREVIEW_LIMIT] if input_data.content else "No content provided"}

            Use Serena MCP tools to verify claims:
            1. For each mentioned class/function, use mcp__serena__find_symbol to verify existence
            2. For architectural claims, use mcp__serena__search_for_pattern
            3. For API documentation, use mcp__serena__get_symbols_overview
            4. For relationships, use mcp__serena__find_referencing_symbols
            5. Check project memories with mcp__serena__read_memory
            6. Validate code examples with validate_example tool

            Verify:
            - Component existence at symbol level
            - Architectural patterns accuracy
            - API documentation completeness
            - Code example validity
            """

        else:  # refine
            base_prompt = f"""
            Refine and improve the following documentation.

            Current documentation:
            {input_data.content[:CONTENT_REFINE_LIMIT] if input_data.content else "No content provided"}

            Requirements:
            {chr(10).join('- ' + req for req in input_data.requirements)}

            Use tools to:
            - Fill missing information
            - Correct inaccuracies
            - Add missing examples
            - Improve clarity
            """

        # Add perspective-specific instructions
        if self.perspective == "architect":
            base_prompt += """
            Focus on:
            - System architecture and design patterns
            - Component relationships and dependencies
            - Technical accuracy and completeness
            - Architectural decisions and rationale
            """
        elif self.perspective == "developer":
            base_prompt += """
            Focus on:
            - Practical usage and examples
            - API documentation and interfaces
            - Quickstart guides and tutorials
            - Common use cases and patterns
            """
        elif self.perspective == "reviewer":
            base_prompt += """
            Focus on:
            - Documentation quality and clarity
            - Missing sections and information
            - Consistency and organization
            - Best practices and standards
            """

        base_prompt += """
        Return a JSON object matching DocumentationOutput schema with all fields.
        """

        return base_prompt

    def _parse_result(self, result: Any) -> DocumentationOutput:
        """Parse agent result into structured output."""
        OUTPUT_PREVIEW_LIMIT = 1000
        try:
            # Use getattr instead of hasattr per CLAUDE.md
            result_content = getattr(result, 'content', None)
            if result_content is not None:
                content = str(result_content)
            else:
                content = str(result)

            # Extract JSON
            json_match = re.search(r"\{[\s\S]*\}", content)
            if json_match:
                data = json.loads(json_match.group())
                return DocumentationOutput(**data)
            else:
                # Fallback - try to extract what we can
                return DocumentationOutput(
                    content=content[:OUTPUT_PREVIEW_LIMIT],
                    summary="Failed to parse structured output",
                    confidence=0.3,
                )
        except (json.JSONDecodeError, ValueError) as e:
            logger.exception("Parse error")
            return DocumentationOutput(
                summary=f"Parse error: {str(e)}",
                confidence=0.1,
            )


# ============================================================================
# UNIFIED PROCESSOR CONFIGURATION
# ============================================================================

class DocumentationProcessorConfig(BaseModel):
    """Configuration for unified documentation processor."""

    mode: OperationMode = Field(default="assess", description="Operation mode")
    min_quality_threshold: float = Field(default=7.0, description="Minimum quality score")
    max_iterations: int = Field(default=3, description="Maximum refinement iterations")

    include_examples: bool = Field(default=True, description="Generate/validate code examples")
    include_api_docs: bool = Field(default=True, description="Generate/check API documentation")

    consensus_settings: ConsensusSettings = Field(
        default_factory=lambda: ConsensusSettings(
            first_round_threshold=0.7,
            threshold=0.6,
            max_rounds=3,
        ),
        description="Consensus settings for multi-agent processing",
    )


# ============================================================================
# UNIFIED DOCUMENTATION PROCESSOR WORKFLOW
# ============================================================================

def create_documentation_processor_workflow(
    model: Model,
    db_path: str = "doc_processor.db",
    codebase_path: str = ".",
    config: Optional[DocumentationProcessorConfig] = None,
) -> Workflow:
    """
    Create a unified documentation processor workflow powered by Serena.

    This workflow leverages Serena MCP server for:
    - Semantic code analysis at the symbol level
    - Pattern detection across the codebase
    - Memory persistence for documentation insights
    - Advanced symbol referencing and relationships

    All using Catalyst consensus pattern with Agno agents calling Serena tools.
    """
    config = config or DocumentationProcessorConfig()
    db = SqliteDb(path=db_path)

    # Agents will use Serena MCP tools directly - no wrapper tools needed

    # Create ConsensusManager
    output_manager = ConsensusManager(DocumentationOutput)

    def get_context(step_input: StepInput) -> Dict[str, Any]:
        """Extract workflow context."""
        context: Dict[str, Any] = {}
        if isinstance(step_input.previous_step_content, dict):
            context.update(step_input.previous_step_content)
        if step_input.additional_data:
            context.update(step_input.additional_data)
        return context

    # =====================================
    # Step 1: Initialize (Serena handles its own setup)
    # =====================================
    def initialize_executor(step_input: StepInput) -> StepOutput:
        """Initialize workflow - Serena handles its own configuration."""
        context = get_context(step_input)

        logger.info("Initializing documentation processor...")

        # Serena handles its own setup through MCP
        # Agents will call:
        # 1. mcp__serena__check_onboarding_performed()
        # 2. mcp__serena__onboarding() if needed
        # This is Serena's responsibility, not ours

        # Determine mode and input
        if isinstance(step_input.input, dict):
            context["mode"] = step_input.input.get("mode", config.mode)
            context["documentation_path"] = step_input.input.get("path")
            context["target_module"] = step_input.input.get("module")
        else:
            # Try to infer mode
            input_str = str(step_input.input)
            if Path(input_str).exists() and Path(input_str).suffix == ".md":
                context["mode"] = "assess"
                context["documentation_path"] = input_str
            else:
                context["mode"] = config.mode

        mode = context["mode"]
        logger.info(f"Step 1: Initializing in {mode} mode")

        # Load existing documentation if assessing/refining
        if mode in ["assess", "refine"]:
            doc_path = context.get("documentation_path", "README.md")
            doc_file = Path(doc_path)

            if doc_file.exists():
                context["existing_content"] = doc_file.read_text()
                logger.info(f"Loaded documentation from {doc_path}")
            else:
                context["error"] = f"Documentation not found: {doc_path}"
                return StepOutput(content=context)

        # Agents will use Serena directly for analysis
        context["target_module"] = context.get("target_module", "src")
        logger.info(f"Target module: {context['target_module']}")

        context["iteration_count"] = 0
        context["processing_history"] = []

        return StepOutput(content=context)

    # =====================================
    # Step 2: Process with Consensus
    # =====================================
    def process_executor(step_input: StepInput) -> StepOutput:
        """Process documentation using consensus across agents."""
        context = get_context(step_input)

        mode = context.get("mode", "assess")
        logger.info(f"Step 2: Processing documentation ({mode} mode)")

        # Prepare input
        processor_input = ProcessorInput(
            mode=mode,
            content=context.get("existing_content"),
            codebase_context=context.get("codebase_context"),
            target_module=context.get("target_module"),
            requirements=context.get("requirements", []),
        )

        # Create consensus agent
        consensus_agent = cast(
            Any,
            output_manager.agno_consensus(
                runner=DocumentationAgent,
                ids=["architect", "developer", "reviewer"],
                perspectives=[
                    f"architect focusing on system design for {mode}",
                    f"developer focusing on practical usage for {mode}",
                    f"reviewer focusing on quality standards for {mode}",
                ],
                weights=[1.0, 1.0, 1.0],
                runner_settings=[
                    {"model": model, "db": db, "perspective": "architect"},
                    {"model": model, "db": db, "perspective": "developer"},
                    {"model": model, "db": db, "perspective": "reviewer"},
                ],
                consensus_settings={"settings": config.consensus_settings},
            ),
        )

        # Run consensus processing
        result = consensus_agent.run_sync(processor_input)
        output = result.final_response

        # Calculate average quality score
        avg_score = (
            output.clarity_score +
            output.completeness_score +
            output.accuracy_score +
            output.usability_score
        ) / 4

        context.update({
            "current_output": output,
            "average_score": avg_score,
            "iteration_count": context.get("iteration_count", 0) + 1,
            "consensus_metadata": result.metrics if hasattr(result, "metrics") else None,
        })

        # Add to history
        history = context.get("processing_history", [])
        history.append({
            "iteration": context["iteration_count"],
            "mode": mode,
            "score": avg_score,
            "improvements": len(output.improvements_needed),
        })
        context["processing_history"] = history

        logger.info(f"Processing complete. Score: {avg_score:.1f}/10")

        return StepOutput(content=context)

    # =====================================
    # Refinement Loop Steps
    # =====================================
    def identify_improvements_executor(step_input: StepInput) -> StepOutput:
        """Identify specific improvements needed."""
        context = get_context(step_input)

        output = context.get("current_output")
        if not output:
            return StepOutput(content=context)

        logger.info("Identifying improvements...")

        improvements = []

        # Add specific improvements based on scores
        if output.clarity_score < 7:
            improvements.append("Improve clarity and organization")

        if not output.has_quickstart:
            improvements.append("Add quickstart guide with hello world example")

        if not output.has_api_docs and config.include_api_docs:
            improvements.append("Add comprehensive API documentation")

        if not output.has_examples and config.include_examples:
            improvements.append("Add practical code examples")

        if output.invalid_claims:
            for claim in output.invalid_claims[:3]:
                improvements.append(f"Correct invalid claim: {claim}")

        if output.missing_sections:
            for section in output.missing_sections[:3]:
                improvements.append(f"Add missing section: {section}")

        # Add consensus suggestions
        improvements.extend(output.improvements_needed)

        context["identified_improvements"] = improvements
        context["requirements"] = improvements  # For next iteration

        return StepOutput(content=context)

    def refine_documentation_executor(step_input: StepInput) -> StepOutput:
        """Refine documentation based on improvements."""
        context = get_context(step_input)

        improvements = context.get("identified_improvements", [])
        if not improvements:
            return StepOutput(content=context)

        logger.info(f"Refining with {len(improvements)} improvements")

        # Switch to refine mode
        context["mode"] = "refine"

        # Use current output as new input
        current_output = context.get("current_output")
        if current_output and current_output.content:
            context["existing_content"] = current_output.content

        # Re-run processing with improvements as requirements
        return process_executor(step_input)

    # =====================================
    # Quality Check for Loop
    # =====================================
    def quality_check(loop_outputs: List[StepOutput]) -> bool:
        """Check if quality threshold is met or max iterations reached."""
        if not loop_outputs:
            return False

        latest = loop_outputs[-1]
        context = latest.content if isinstance(latest.content, dict) else {}

        avg_score = context.get("average_score", 0)
        iteration = context.get("iteration_count", 0)
        improvements = context.get("identified_improvements", [])

        # End if:
        # 1. Quality threshold met
        # 2. Max iterations reached
        # 3. No improvements needed

        if avg_score >= config.min_quality_threshold:
            logger.info(f"Quality threshold met: {avg_score:.1f}")
            return True

        if iteration >= config.max_iterations:
            logger.info(f"Max iterations reached: {iteration}")
            return True

        if not improvements:
            logger.info("No improvements identified")
            return True

        logger.info(f"Iteration {iteration}: Score {avg_score:.1f}, {len(improvements)} improvements needed")
        return False

    # =====================================
    # Step 4: Generate Final Output
    # =====================================
    def finalize_executor(step_input: StepInput) -> StepOutput:
        """Generate final documentation or report and save to docs directory."""
        context = get_context(step_input)

        mode = context.get("mode")
        output = context.get("current_output")

        if not output:
            return StepOutput(content="Processing failed - no output generated")

        logger.info("Step 4: Generating final output")

        avg_score = context.get("average_score", 0)

        if mode == "generate":
            # Return generated documentation
            final_content = output.content

            # Add metadata footer
            final_content += f"\n\n---\n*Generated with confidence: {output.confidence:.0%}*"
            final_content += f"\n*Quality score: {avg_score:.1f}/10*"

        else:  # assess or refine
            # Generate assessment report
            sections = []

            # Header
            sections.append("# 📊 Documentation Assessment Report")
            sections.append(f"*Mode: {mode.title()}*\n")

            # Overall score
            emoji = "🟢" if avg_score >= 8 else ("🟡" if avg_score >= 6 else "🔴")
            sections.append(f"## Overall Score: {emoji} **{avg_score:.1f}/10**\n")

            # Consensus info
            if context.get("consensus_metadata"):
                meta = context["consensus_metadata"]
                sections.append("### Consensus Metrics")
                sections.append(f"- Dissent Rate: {getattr(meta, 'dissent_rate', 0):.2%}")
                sections.append(f"- Confidence: {getattr(meta, 'consensus_confidence', 0):.2%}\n")

            # Detailed scores
            sections.append("### Detailed Scores")
            sections.append(f"- **Clarity**: {output.clarity_score:.1f}/10")
            sections.append(f"- **Completeness**: {output.completeness_score:.1f}/10")
            sections.append(f"- **Accuracy**: {output.accuracy_score:.1f}/10")
            sections.append(f"- **Usability**: {output.usability_score:.1f}/10\n")

            # Checklist
            sections.append("### Feature Checklist")
            sections.append(f"- Examples: {'✅' if output.has_examples else '❌'}")
            sections.append(f"- Quickstart: {'✅' if output.has_quickstart else '❌'}")
            sections.append(f"- API Docs: {'✅' if output.has_api_docs else '❌'}")
            sections.append(f"- Valid Examples: {'✅' if output.examples_valid else '❌'}\n")

            # Analysis depth
            sections.append("### Analysis Depth")
            sections.append(f"- Files Analyzed: {output.files_analyzed}")
            sections.append(f"- Patterns Found: {output.patterns_found}\n")

            # Issues
            if output.invalid_claims:
                sections.append("### ❌ Invalid Claims")
                for claim in output.invalid_claims:
                    sections.append(f"- {claim}")
                sections.append("")

            if output.missing_sections:
                sections.append("### 📝 Missing Sections")
                for section in output.missing_sections:
                    sections.append(f"- {section}")
                sections.append("")

            # Improvements
            if output.improvements_needed:
                sections.append("### 🔧 Improvements Needed")
                for i, improvement in enumerate(output.improvements_needed, 1):
                    sections.append(f"{i}. {improvement}")
                sections.append("")

            # Summary
            sections.append("### 📋 Summary")
            sections.append(output.summary)

            # Processing history
            if context.get("processing_history"):
                sections.append("\n### 🔄 Processing History")
                for entry in context["processing_history"]:
                    sections.append(
                        f"- Iteration {entry['iteration']}: "
                        f"{entry['mode']} mode, "
                        f"score {entry['score']:.1f}"
                    )

            final_content = "\n".join(sections)

        # Save to docs directory
        docs_dir = Path("docs")
        docs_dir.mkdir(exist_ok=True)

        # Generate filename based on mode and target
        timestamp = _get_timestamp()
        target = context.get("target_module", "codebase").replace("/", "_")

        if mode == "generate":
            filename = f"generated_{target}_{timestamp}.md"
        elif mode == "assess":
            filename = f"assessment_{target}_{timestamp}.md"
        else:  # refine
            filename = f"refined_{target}_{timestamp}.md"

        output_path = docs_dir / filename
        output_path.write_text(final_content)
        logger.info(f"Documentation saved to {output_path}")

        context["final_output"] = final_content
        context["output_path"] = str(output_path)
        return StepOutput(content=final_content)

    # =====================================
    # Create Workflow
    # =====================================

    # Create steps
    initialize = Step(
        name="initialize",
        description="Initialize and analyze codebase",
        executor=initialize_executor,
    )

    process = Step(
        name="process",
        description="Process documentation with consensus",
        executor=process_executor,
    )

    identify_improvements = Step(
        name="identify_improvements",
        description="Identify needed improvements",
        executor=identify_improvements_executor,
    )

    refine = Step(
        name="refine",
        description="Refine documentation",
        executor=refine_documentation_executor,
    )

    finalize = Step(
        name="finalize",
        description="Generate final output",
        executor=finalize_executor,
    )

    # Create refinement loop
    refinement_loop = Loop(
        name="refinement_loop",
        description="Iteratively improve documentation",
        steps=[identify_improvements, refine],
        end_condition=quality_check,
        max_iterations=config.max_iterations,
    )

    # Create workflow
    workflow = Workflow(
        id="UnifiedDocumentationProcessor",
        name="Unified Documentation Processor Workflow",
        description="Single workflow for generating, assessing, and refining documentation using Catalyst consensus",
        db=db,
        telemetry=False,
        debug_mode=False,
        steps=[
            initialize,
            process,
            refinement_loop,
            finalize,
        ],
    )

    return workflow


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def _get_timestamp() -> str:
    """Get timestamp for file naming."""
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def _detect_project_language(codebase_path: Path) -> str:
    """Detect primary language of the project."""
    # Check for language-specific files
    if (codebase_path / "pyproject.toml").exists() or (codebase_path / "setup.py").exists():
        return "python"
    elif (codebase_path / "package.json").exists():
        return "javascript"
    elif (codebase_path / "Cargo.toml").exists():
        return "rust"
    elif (codebase_path / "go.mod").exists():
        return "go"
    else:
        return "unknown"

def _get_language_ignores(language: str) -> List[str]:
    """Get ignore patterns for language."""
    common = [".git", "__pycache__", "node_modules", ".env", ".venv", "venv"]

    language_specific = {
        "python": ["*.pyc", "*.pyo", "*.pyd", ".Python", "*.egg-info", "dist", "build"],
        "javascript": ["node_modules", "dist", "build", "*.min.js", "coverage"],
        "rust": ["target", "Cargo.lock"],
        "go": ["vendor", "*.exe", "*.test"],
    }

    return common + language_specific.get(language, [])

# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def generate_documentation(
    module_path: str,
    model: Model,
    config: Optional[DocumentationProcessorConfig] = None,
) -> str:
    """Generate new documentation for a module."""
    config = config or DocumentationProcessorConfig(mode="generate")
    workflow = create_documentation_processor_workflow(model=model, config=config)

    result = workflow.run({
        "mode": "generate",
        "module": module_path,
    })

    return result


def assess_documentation(
    doc_path: str,
    model: Model,
    config: Optional[DocumentationProcessorConfig] = None,
) -> str:
    """Assess existing documentation quality."""
    config = config or DocumentationProcessorConfig(mode="assess")
    workflow = create_documentation_processor_workflow(model=model, config=config)

    result = workflow.run({
        "mode": "assess",
        "path": doc_path,
    })

    return result


def refine_documentation(
    doc_path: str,
    model: Model,
    requirements: List[str] = None,
    config: Optional[DocumentationProcessorConfig] = None,
) -> str:
    """Refine existing documentation."""
    config = config or DocumentationProcessorConfig(mode="refine")
    workflow = create_documentation_processor_workflow(model=model, config=config)

    result = workflow.run({
        "mode": "refine",
        "path": doc_path,
        "requirements": requirements or [],
    })

    return result


# ============================================================================
# MAIN USAGE EXAMPLE
# ============================================================================

def main():
    """Example usage of the unified documentation processor."""
    import os
    from agno.models.openai import OpenAILike

    # Initialize model
    model = OpenAILike(
        api_key=os.getenv("LLM_API_KEY", "dummy"),
        base_url=os.getenv("LLM_BASE_URL", "http://localhost:3005/v1"),
        id=os.getenv("LLM_MODEL", "gpt-4o"),
        temperature=0.3,
    )

    # Example 1: Assess existing documentation
    print("=" * 50)
    print("ASSESSING DOCUMENTATION")
    print("=" * 50)

    assessment = assess_documentation("README.md", model)
    print(assessment)
    print("\nDocumentation saved to docs/")

    # Example 2: Generate new documentation
    print("\n" + "=" * 50)
    print("GENERATING DOCUMENTATION")
    print("=" * 50)

    OUTPUT_PREVIEW_LIMIT = 1000
    generated = generate_documentation("src/blockether_catalyst/consensus", model)
    print(generated[:OUTPUT_PREVIEW_LIMIT] + "...")
    print("\nDocumentation saved to docs/")

    # Example 3: Refine with specific requirements
    print("\n" + "=" * 50)
    print("REFINING DOCUMENTATION")
    print("=" * 50)

    refined = refine_documentation(
        "README.md",
        model,
        requirements=[
            "Add installation instructions",
            "Include API examples",
            "Improve clarity",
        ]
    )
    print(refined[:OUTPUT_PREVIEW_LIMIT] + "...")
    print("\nDocumentation saved to docs/")


if __name__ == "__main__":
    main()