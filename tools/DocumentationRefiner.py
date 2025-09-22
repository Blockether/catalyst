#!/usr/bin/env python3
"""
Documentation Quality Assessment using Catalyst's Consensus Mechanism.

Uses the real Hashgraph-inspired consensus from blockether_catalyst.consensus
for multi-perspective evaluation with proper voting and refinement.
"""

import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from blockether_catalyst.consensus.Consensus import Consensus
from blockether_catalyst.consensus.ConsensusTypes import (
    ModelConfiguration,
    ConsensusSettings,
    ConsensusResult,
    VerbosityLevel
)
from blockether_catalyst.consensus.VotingComparison import (
    BaseModelWithReasoning,
    ComparisonStrategy,
    VotingMetadata
)
from blockether_catalyst.utils.TypedCalls import ArityOneTypedCall
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentationEvaluation(BaseModelWithReasoning):
    """Structured evaluation of documentation quality with voting strategies."""
    
    # Scores with RANGE comparison (tolerance of 1.0 point)
    clarity_score: float = Field(
        default=5.0, 
        ge=0, 
        le=10,
        description="How clear and understandable is the documentation?",
        json_schema_extra={"voting_comparison": {"strategy": ComparisonStrategy.RANGE, "tolerance": 1.0}}
    )
    
    completeness_score: float = Field(
        default=5.0,
        ge=0,
        le=10, 
        description="How complete is the documentation?",
        json_schema_extra={"voting_comparison": {"strategy": ComparisonStrategy.RANGE, "tolerance": 1.0}}
    )
    
    technical_accuracy_score: float = Field(
        default=5.0,
        ge=0,
        le=10,
        description="How technically accurate is the documentation?",
        json_schema_extra={"voting_comparison": {"strategy": ComparisonStrategy.RANGE, "tolerance": 1.0}}
    )
    
    value_proposition_score: float = Field(
        default=5.0,
        ge=0,
        le=10,
        description="How clear is the value proposition?",
        json_schema_extra={"voting_comparison": {"strategy": ComparisonStrategy.RANGE, "tolerance": 1.0}}
    )
    
    # Boolean checks with EXACT comparison
    has_hello_world: bool = Field(
        default=False,
        description="Is there a hello world example after installation?",
        json_schema_extra={"voting_comparison": {"strategy": ComparisonStrategy.EXACT}}
    )
    
    has_api_docs_link: bool = Field(
        default=False,
        description="Are API docs properly linked?",
        json_schema_extra={"voting_comparison": {"strategy": ComparisonStrategy.EXACT}}
    )
    
    has_examples_section: bool = Field(
        default=False,
        description="Does the Examples section exist (not just linked)?",
        json_schema_extra={"voting_comparison": {"strategy": ComparisonStrategy.EXACT}}
    )
    
    navigation_matches_content: bool = Field(
        default=True,
        description="Do navigation links match actual sections?",
        json_schema_extra={"voting_comparison": {"strategy": ComparisonStrategy.EXACT}}
    )
    
    # Lists with SEMANTIC comparison (they contain similar text)
    strengths: List[str] = Field(
        default_factory=list,
        description="Key strengths identified",
        json_schema_extra={"voting_comparison": {"strategy": ComparisonStrategy.SEMANTIC, "threshold": 0.5}}
    )
    
    critical_issues: List[str] = Field(
        default_factory=list,
        description="Critical issues that must be fixed",
        json_schema_extra={"voting_comparison": {"strategy": ComparisonStrategy.SEMANTIC, "threshold": 0.5}}
    )
    
    missing_sections: List[str] = Field(
        default_factory=list,
        description="Important missing sections",
        json_schema_extra={"voting_comparison": {"strategy": ComparisonStrategy.SEMANTIC, "threshold": 0.5}}
    )
    
    immediate_fixes: List[str] = Field(
        default_factory=list,
        description="Quick fixes that can be done in < 30 minutes",
        json_schema_extra={"voting_comparison": {"strategy": ComparisonStrategy.SEMANTIC, "threshold": 0.5}}
    )
    
    # Overall assessment with SEMANTIC comparison
    overall_assessment: str = Field(
        default="",
        description="One paragraph overall assessment",
        json_schema_extra={"voting_comparison": {"strategy": ComparisonStrategy.SEMANTIC, "threshold": 0.7}}
    )
    
    # Reasoning is IGNORED in voting (from BaseModelWithReasoning)
    # reasoning field inherited from parent class


class PerspectiveEvaluator(ArityOneTypedCall[str, DocumentationEvaluation]):
    """Evaluator from a specific perspective (implements ArityOneTypedCall)."""
    
    def __init__(self, perspective: str, llm_model: Any):
        """Initialize with perspective and LLM model."""
        self.perspective = perspective
        self.llm = llm_model
        self.prompt_template = self._get_perspective_prompt()
    
    def _get_perspective_prompt(self) -> str:
        """Get the evaluation prompt for this perspective."""
        if self.perspective == "new_user":
            return """You are a BRAND NEW USER evaluating documentation.

Documentation to evaluate:
{content}

Evaluate and provide a JSON response with these exact fields:
- clarity_score: 0-10 (how clear for new users)
- completeness_score: 0-10 (has everything needed to start)
- technical_accuracy_score: 0-10 (appears correct)
- value_proposition_score: 0-10 (clear what this does)
- has_hello_world: true/false (hello world example after install)
- has_api_docs_link: true/false (API documentation linked)
- has_examples_section: true/false (Examples section exists, not just linked)
- navigation_matches_content: true/false (nav links work)
- strengths: list of 3-5 key strengths
- critical_issues: list of critical problems
- missing_sections: list of missing content
- immediate_fixes: list of quick fixes
- overall_assessment: one paragraph summary
- reasoning: your detailed reasoning

BE SPECIFIC. Check if Examples link has actual Examples section."""

        elif self.perspective == "experienced_dev":
            return """You are an EXPERIENCED DEVELOPER evaluating documentation.

Documentation to evaluate:
{content}

Evaluate for production use and provide a JSON response with these exact fields:
- clarity_score: 0-10 (technical clarity)
- completeness_score: 0-10 (API coverage, advanced features)
- technical_accuracy_score: 0-10 (correct, no errors)
- value_proposition_score: 0-10 (clear differentiation)
- has_hello_world: true/false (basic example exists)
- has_api_docs_link: true/false (API reference available)
- has_examples_section: true/false (comprehensive examples)
- navigation_matches_content: true/false (broken promises?)
- strengths: list of technical strengths
- critical_issues: list of blockers for production
- missing_sections: list of missing technical docs
- immediate_fixes: list of quick improvements
- overall_assessment: one paragraph assessment
- reasoning: your detailed technical reasoning

CHECK for broken navigation promises (e.g., Examples link but no Examples section)."""

        else:  # auditor
            return """You are a TECHNICAL AUDITOR evaluating documentation.

Documentation to evaluate:
{content}

Audit for accuracy and provide a JSON response with these exact fields:
- clarity_score: 0-10 (unambiguous claims)
- completeness_score: 0-10 (all claims substantiated)
- technical_accuracy_score: 0-10 (factually correct)
- value_proposition_score: 0-10 (honest positioning)
- has_hello_world: true/false 
- has_api_docs_link: true/false
- has_examples_section: true/false
- navigation_matches_content: true/false
- strengths: list of well-supported claims
- critical_issues: list of unsubstantiated claims
- missing_sections: list of missing evidence
- immediate_fixes: list of quick accuracy fixes
- overall_assessment: audit summary
- reasoning: your audit reasoning

Verify all claims are backed by evidence."""
    
    async def call(self, x: str) -> DocumentationEvaluation:
        """Execute evaluation from this perspective."""
        # Use Agno to get response and parse it
        from agno.agent import Agent
        from agno.db.sqlite import SqliteDb
        import json
        
        prompt = self.prompt_template.format(content=x)
        
        # Create agent for this evaluation
        db = SqliteDb()
        agent = Agent(
            id=f"{self.perspective}_agent",
            model=self.llm,
            name=f"{self.perspective.replace('_', ' ').title()} Evaluator",
            description=f"Evaluates documentation from {self.perspective} perspective",
            instructions=[prompt + "\n\nReturn ONLY valid JSON matching the DocumentationEvaluation schema."],
            db=db,
            telemetry=False
        )
        
        result = agent.run("Evaluate the documentation and return structured JSON")
        
        # Parse the response to get structured data
        if hasattr(result, 'content'):
            response_text = str(result.content)
        else:
            response_text = str(result)
        
        # Try to extract JSON from the response
        try:
            # Find JSON in response (might be wrapped in markdown)
            import re
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                json_str = json_match.group()
                data = json.loads(json_str)
            else:
                data = json.loads(response_text)
            
            # Create DocumentationEvaluation from the parsed data
            return DocumentationEvaluation(**data)
        except Exception as e:
            logger.error(f"Failed to parse response from {self.perspective}: {e}")
            # Return a default evaluation on parse error
            return DocumentationEvaluation(
                reasoning=f"Failed to parse: {str(e)[:200]}",
                overall_assessment="Evaluation failed due to parsing error"
            )


class DocumentationConsensusRefiner:
    """Documentation refiner using Catalyst's consensus mechanism."""
    
    def __init__(self, llm_models: List[Any], judge_model: Any, settings: Optional[ConsensusSettings] = None):
        """Initialize with LLM models for consensus.
        
        Args:
            llm_models: List of LLM models (need odd number, min 3)
            judge_model: Judge model for tie-breaking
            settings: Optional consensus settings
        """
        self.llm_models = llm_models
        self.judge_model = judge_model
        self.settings = settings or ConsensusSettings(
            max_rounds=3,
            threshold=0.6,  # 60% agreement needed
            first_round_threshold=0.8,  # 80% for first round consensus
            verbosity=VerbosityLevel.VERBOSE
        )
    
    async def evaluate_documentation(self, content: str) -> ConsensusResult[DocumentationEvaluation]:
        """Evaluate documentation using consensus across perspectives.
        
        Creates multiple evaluators with different perspectives and
        uses consensus to get the final evaluation.
        """
        
        # Create evaluators with different perspectives
        perspectives = ["new_user", "experienced_dev", "auditor"]
        
        if len(self.llm_models) < 3:
            raise ValueError("Need at least 3 models for consensus")
        
        # Create model configurations for consensus
        model_configs = []
        for i, (llm, perspective) in enumerate(zip(self.llm_models[:3], perspectives)):
            evaluator = PerspectiveEvaluator(perspective, llm)
            
            config = ModelConfiguration[DocumentationEvaluation](
                id=f"{perspective}_evaluator",
                executor=evaluator,
                perspective=f"Evaluate documentation from {perspective} perspective",
                weight_multiplier=1.0
            )
            model_configs.append(config)
        
        # If we have more models, add them with repeated perspectives
        if len(self.llm_models) > 3:
            for i, llm in enumerate(self.llm_models[3:]):
                perspective = perspectives[i % 3]
                evaluator = PerspectiveEvaluator(perspective, llm)
                
                config = ModelConfiguration[DocumentationEvaluation](
                    id=f"{perspective}_evaluator_{i+2}",
                    executor=evaluator,
                    perspective=f"Additional {perspective} perspective for consensus",
                    weight_multiplier=1.0
                )
                model_configs.append(config)
        
        # Create judge evaluator
        judge_evaluator = PerspectiveEvaluator("auditor", self.judge_model)
        
        # Initialize consensus mechanism
        consensus = Consensus[DocumentationEvaluation](
            models=model_configs,
            judge=judge_evaluator,
            settings=self.settings
        )
        
        # Run consensus evaluation
        logger.info("Starting consensus-based documentation evaluation...")
        result = await consensus.call(content)
        
        return result
    
    def generate_report(self, result: ConsensusResult[DocumentationEvaluation]) -> str:
        """Generate human-readable report from consensus result."""
        
        eval = result.final_response
        metrics = result.metrics
        
        # Calculate overall score
        overall_score = (
            eval.clarity_score * 0.3 +
            eval.completeness_score * 0.3 +
            eval.technical_accuracy_score * 0.2 +
            eval.value_proposition_score * 0.2
        )
        
        report = f"""# 📊 Documentation Quality Report (Consensus-Based)

## Overall Score: **{overall_score:.1f}/10**

### 🎯 Consensus Metrics
- **Consensus Achieved**: {"✅ Yes" if result.consensus_achieved else "⚠️ No (used " + (metrics.fallback_method or "majority vote") + ")"}
- **Rounds to Convergence**: {result.total_rounds}
- **Convergence Score**: {result.convergence_score:.3f}
- **Model Agreement**: {(1 - metrics.dissent_rate):.1%}
- **Consensus Confidence**: {metrics.consensus_confidence:.1%}

### 📈 Individual Scores (Consensus Values)
- **Clarity**: {eval.clarity_score:.1f}/10
- **Completeness**: {eval.completeness_score:.1f}/10  
- **Technical Accuracy**: {eval.technical_accuracy_score:.1f}/10
- **Value Proposition**: {eval.value_proposition_score:.1f}/10

### ✅ Verification Checklist
- Has Hello World: {"✅" if eval.has_hello_world else "❌"}
- Has API Docs Link: {"✅" if eval.has_api_docs_link else "❌"}
- Has Examples Section: {"✅" if eval.has_examples_section else "❌"}
- Navigation Matches Content: {"✅" if eval.navigation_matches_content else "❌"}

### 💪 Strengths (Consensus)
"""
        for strength in eval.strengths[:5]:
            report += f"- {strength}\n"
        
        report += "\n### 🚨 Critical Issues (Consensus)\n"
        for issue in eval.critical_issues[:5]:
            report += f"- ❌ {issue}\n"
        
        if eval.missing_sections:
            report += "\n### 📝 Missing Sections\n"
            for section in eval.missing_sections[:5]:
                report += f"- {section}\n"
        
        if eval.immediate_fixes:
            report += "\n### 🏃 Immediate Fixes (< 30 minutes)\n"
            for i, fix in enumerate(eval.immediate_fixes[:5], 1):
                report += f"{i}. {fix}\n"
        
        # Model contributions
        if result.model_contributions:
            report += "\n### 🤝 Evaluator Contributions\n"
            for model_id, score in sorted(result.model_contributions.items(), key=lambda x: x[1], reverse=True):
                report += f"- {model_id}: {score:.3f}\n"
        
        # Overall assessment
        report += f"\n### 📋 Overall Assessment\n{eval.overall_assessment}\n"
        
        # Consensus reasoning
        report += f"\n### 💭 Consensus Process\n{result.reasoning}\n"
        
        return report


async def main():
    """Example usage with real consensus."""
    import os
    from agno.models.openai import OpenAILike
    
    # Create multiple LLM instances for consensus (need odd number, min 3)
    base_llm = OpenAILike(
        api_key=os.getenv("LLM_API_KEY", "dummy"),
        base_url=os.getenv("LLM_BASE_URL", "http://localhost:3005/v1"),
        id=os.getenv("LLM_MODEL", "gpt-4o"),
        temperature=0.3
    )
    
    # For testing, use same model multiple times (in production, use different models)
    llm_models = [base_llm, base_llm, base_llm]  # Minimum 3 for consensus
    judge_model = base_llm  # Judge for tie-breaking
    
    # Initialize refiner with consensus
    refiner = DocumentationConsensusRefiner(
        llm_models=llm_models,
        judge_model=judge_model,
        settings=ConsensusSettings(
            max_rounds=3,
            threshold=0.6,
            first_round_threshold=0.8,
            verbosity=VerbosityLevel.VERBOSE
        )
    )
    
    # Load README
    doc_path = Path("README.md")
    if not doc_path.exists():
        logger.error("README.md not found")
        return
    
    content = doc_path.read_text()
    
    logger.info("Evaluating documentation with Catalyst Consensus mechanism...")
    logger.info("This uses REAL Hashgraph-inspired consensus with voting!")
    
    # Run consensus evaluation
    result = await refiner.evaluate_documentation(content)
    
    # Generate report
    report = refiner.generate_report(result)
    
    print("\n" + report)
    
    # Save report
    report_path = Path("documentation_consensus_report.md")
    report_path.write_text(report)
    logger.info(f"Report saved to {report_path}")


if __name__ == "__main__":
    asyncio.run(main())