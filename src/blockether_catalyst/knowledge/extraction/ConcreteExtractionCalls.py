"""
Concrete extraction call implementations with full consensus support.

These are the actual implementations that will be used in production.
They take ExtractionModelSettings and create proper consensus-based calls.
"""

import os
from pathlib import Path
from typing import Dict, List

from jinja2 import Environment, FileSystemLoader, Template

from blockether_catalyst.consensus.Consensus import Consensus
from blockether_catalyst.consensus.ConsensusTypes import ConsensusSettings as CoreConsensusSettings
from blockether_catalyst.consensus.ConsensusTypes import (
    ModelConfiguration,
)
from blockether_catalyst.knowledge.extraction.internal.KnowledgeExtractionCallBase import (
    BaseChunkContentClassificationCall,
    BaseDocumentChunkingCall,
    BaseTableCaptionExtractionCall,
    BaseTermExtractionCall,
    ExtractionCallsSettings,
)
from blockether_catalyst.knowledge.extraction.ModelSettings import (
    ConsensusSettings,
    ExtractionModelSettings,
)
from blockether_catalyst.knowledge.KnowledgeTypes import (
    ChunkingDecisionResponse,
    DocumentMetadata,
    KnowledgeChunkClassificationResponse,
    KnowledgePageData,
    TableCaptionExtractionResponse,
    TermMeaningExtractionResponse,
)
from blockether_catalyst.utils.instructor.InstructorLLMCall import InstructorLLMCall


def get_template_environment() -> Environment:
    """
    Get Jinja2 environment with templates from configurable path.

    The path can be configured via KNOWLEDGE_TEMPLATES_PATH environment variable.
    If not set, defaults to src/blockether_catalyst/assets/knowledge/prompts relative to project root.

    Individual templates can be overridden via:
    - KNOWLEDGE_TEMPLATE_TERM_REFINEMENT: Path to custom term_refinement.j2
    - KNOWLEDGE_TEMPLATE_DOCUMENT_CHUNKING: Path to custom document_chunking.j2
    - KNOWLEDGE_TEMPLATE_CHUNK_CLASSIFICATION: Path to custom chunk_classification.j2

    Returns:
        Configured Jinja2 Environment
    """
    # Check for environment variable
    templates_path = os.getenv("KNOWLEDGE_TEMPLATES_PATH")

    if templates_path:
        # Use user-specified path
        templates_dir = Path(templates_path)
        if not templates_dir.exists():
            raise ValueError(f"Template path does not exist: {templates_dir}")
        if not templates_dir.is_dir():
            raise ValueError(f"Template path is not a directory: {templates_dir}")
    else:
        # Default to src/blockether_catalyst/assets/knowledge/prompts
        # Go up from src/blockether_catalyst/knowledge/extraction/ to src/blockether_catalyst/
        templates_dir = Path(__file__).parent.parent.parent / "assets" / "knowledge" / "prompts"

        if not templates_dir.exists():
            # Fallback to checking if we're in a different directory structure
            templates_dir = Path("src/blockether_catalyst/assets") / "knowledge" / "prompts"
            if not templates_dir.exists():
                raise ValueError(
                    "Default template directory not found. "
                    "Please set KNOWLEDGE_TEMPLATES_PATH environment variable to the template directory path."
                )

    return Environment(
        loader=FileSystemLoader(templates_dir),
        trim_blocks=True,
        lstrip_blocks=True,
    )


# Create template environment once
template_env = get_template_environment()


def load_template(
    template_name: str,
    env_var_name: str = "",
    template_env: Environment = template_env,
) -> Template:
    """
    Load a template with optional override from environment variable.

    This higher-order function handles the common pattern of:
    1. Checking for a custom template path via environment variable
    2. Loading the custom template if it exists
    3. Falling back to the default template from the template environment
    4. Providing proper error messages for missing templates

    Args:
        template_name: Default template filename (e.g., "document_chunking.j2")
        env_var_name: Environment variable name for custom override
                      (e.g., "KNOWLEDGE_TEMPLATE_DOCUMENT_CHUNKING")
        template_env: Jinja2 environment to load default templates from

    Returns:
        Loaded Jinja2 Template object

    Raises:
        FileNotFoundError: If custom template path is specified but file doesn't exist
        jinja2.TemplateNotFound: If default template is not found in template directory
        ValueError: If custom template file cannot be read
    """
    # Check for custom template override
    custom_template_path = os.getenv(env_var_name) if env_var_name else None

    if custom_template_path:
        # Use custom template file
        template_path = Path(custom_template_path)

        if not template_path.exists():
            raise FileNotFoundError(
                f"Custom template specified via {env_var_name} not found: {template_path}\n"
                f"Please check that the file exists or unset the environment variable to use the default template."
            )

        if not template_path.is_file():
            raise ValueError(
                f"Custom template path is not a file: {template_path}\n"
                f"The {env_var_name} environment variable must point to a template file, not a directory."
            )

        try:
            # Load template directly from file
            with open(template_path, "r", encoding="utf-8") as f:
                template_content = f.read()

            # Create a one-off template (not using environment)
            return Template(template_content)

        except IOError as e:
            raise ValueError(
                f"Failed to read custom template from {template_path}: {e}\n"
                f"Please check file permissions and that the file is a valid text file."
            )
        except Exception as e:
            raise ValueError(
                f"Failed to parse custom template from {template_path}: {e}\n"
                f"Please check that the file contains valid Jinja2 template syntax."
            )

    else:
        # Use default template from configured directory
        try:
            return template_env.get_template(template_name)
        except Exception as e:
            # Provide helpful error message with available templates
            template_dir = "unknown"
            available_templates = []

            try:
                # Try to get template directory from loader
                if isinstance(template_env.loader, FileSystemLoader) and hasattr(template_env.loader, "searchpath"):
                    searchpath = getattr(template_env.loader, "searchpath", [])
                    template_dir = str(searchpath[0]) if searchpath else "unknown"
                else:
                    template_dir = "unknown"

                # Try to list available templates for helpful error message
                if template_dir != "unknown":
                    template_path = Path(template_dir)
                    if template_path.exists():
                        available_templates = [f.name for f in template_path.glob("*.j2")]
            except Exception:
                # Ignore errors when trying to list templates
                pass

            error_msg = f"Failed to load template '{template_name}': {e}\n"
            error_msg += f"Template directory: {template_dir}\n"

            if available_templates:
                error_msg += f"Available templates: {', '.join(sorted(available_templates))}\n"
            else:
                error_msg += "No templates found in template directory.\n"

            error_msg += (
                f"You can either:\n"
                f"1. Add '{template_name}' to the template directory\n"
                f"2. Set {env_var_name or 'an environment variable'} to point to a custom template file\n"
                f"3. Set KNOWLEDGE_TEMPLATES_PATH to a directory containing the templates"
            )

            raise FileNotFoundError(error_msg)


class ConcreteTermExtractionCall(BaseTermExtractionCall):
    """
    Concrete term extraction call with full consensus configuration.
    """

    def __init__(self, settings: ConsensusSettings):
        """
        Initialize with consensus settings containing all model configurations.

        Args:
            settings: Complete consensus settings with models, thresholds, etc.
        """
        # Build consensus with all the settings
        models = []
        for model_config in settings.models:
            instructor_call = InstructorLLMCall(
                response_model=TermMeaningExtractionResponse,
                temperature=model_config.temperature,
                model=model_config.model,
                base_url=model_config.api_url,
                api_key=model_config.api_key,
            )
            models.append(
                {
                    "instructor_call": instructor_call,
                    "weight": model_config.weight,
                    "perspective": model_config.perspective,
                }
            )

        if not models:
            raise ValueError("At least one term extraction model must be configured.")

        # Create judge call for tie-breaking
        judge_call = models[0]["instructor_call"]

        # Create model configurations
        model_configs = [
            ModelConfiguration[TermMeaningExtractionResponse](
                id=f"model_{i}",
                executor=m["instructor_call"],
                perspective=m.get("perspective", ""),
                weight=m.get("weight", 1.0),
            )
            for i, m in enumerate(models)
        ]

        # Create consensus settings
        consensus_settings = CoreConsensusSettings(
            threshold=settings.consensus_threshold,
            first_round_threshold=settings.consensus_threshold,
            max_rounds=settings.max_rounds,
        )

        consensus = Consensus[TermMeaningExtractionResponse](
            models=model_configs,
            judge=judge_call,
            settings=consensus_settings,
        )

        super().__init__(consensus=consensus)
        self.settings = settings

    def fill_template(
        self,
        term: str,
        type: str,
        occurrences_contexts: List[str],
        cooccurring_terms: Dict[str, List[str]],
        domain: str = "",
        application: str = "",
    ) -> str:
        """
        Fill the prompt for term extraction using templates from assets.

        Can be overridden via KNOWLEDGE_TEMPLATE_TERM_REFINEMENT environment variable
        pointing to a custom template file path.
        """
        template = load_template("term_refinement.j2", "KNOWLEDGE_TEMPLATE_TERM_REFINEMENT")

        return template.render(
            term=term,
            type=type,
            occurrences_contexts=occurrences_contexts,
            cooccurring_terms=cooccurring_terms,
            domain=domain,
            application=application,
        )


class ConcreteChunkingCall(BaseDocumentChunkingCall):
    """
    Concrete document chunking call with full consensus configuration.
    """

    def __init__(self, settings: ConsensusSettings):
        """
        Initialize with consensus settings containing all model configurations.

        Args:
            settings: Complete consensus settings with models, thresholds, etc.
        """
        models = []
        for model_config in settings.models:
            instructor_call = InstructorLLMCall(
                response_model=ChunkingDecisionResponse,
                temperature=model_config.temperature,
                model=model_config.model,
                base_url=model_config.api_url,
                api_key=model_config.api_key,
            )
            models.append(
                {
                    "instructor_call": instructor_call,
                    "weight": model_config.weight,
                    "perspective": model_config.perspective,
                }
            )

        if not models:
            raise ValueError("At least one document chunking model must be configured.")

        # Create judge call for tie-breaking
        judge_call = models[0]["instructor_call"]

        # Create model configurations
        model_configs = [
            ModelConfiguration[ChunkingDecisionResponse](
                id=f"model_{i}",
                executor=m["instructor_call"],
                perspective=m.get("perspective", ""),
                weight=m.get("weight", 1.0),
            )
            for i, m in enumerate(models)
        ]

        # Create consensus settings
        consensus_settings = CoreConsensusSettings(
            threshold=settings.consensus_threshold,
            first_round_threshold=settings.consensus_threshold,
            max_rounds=settings.max_rounds,
        )

        consensus = Consensus[ChunkingDecisionResponse](
            models=model_configs,
            judge=judge_call,
            settings=consensus_settings,
        )

        super().__init__(consensus=consensus)
        self.settings = settings

    def fill_template(
        self,
        page: KnowledgePageData,
        document_name: str,
        metadata: DocumentMetadata,
    ) -> str:
        """
        Fill the prompt for document chunking using templates from assets.

        Can be overridden via KNOWLEDGE_TEMPLATE_DOCUMENT_CHUNKING environment variable
        pointing to a custom template file path.
        """
        template = load_template("document_chunking.j2", "KNOWLEDGE_TEMPLATE_DOCUMENT_CHUNKING")

        return template.render(
            page=page,
            document_name=document_name,
            metadata=metadata,
        )


class ConcreteChunkClassificationCall(BaseChunkContentClassificationCall):
    """
    Concrete chunk classification call with full consensus configuration.
    """

    def __init__(self, settings: ConsensusSettings):
        """
        Initialize with consensus settings containing all model configurations.

        Args:
            settings: Complete consensus settings with models, thresholds, etc.
        """
        models = []
        for model_config in settings.models:
            instructor_call = InstructorLLMCall(
                response_model=KnowledgeChunkClassificationResponse,
                temperature=model_config.temperature,
                model=model_config.model,
                base_url=model_config.api_url,
                api_key=model_config.api_key,
            )
            models.append(
                {
                    "instructor_call": instructor_call,
                    "weight": model_config.weight,
                    "perspective": model_config.perspective,
                }
            )

        if not models:
            raise ValueError("At least one chunk classification model must be configured.")

        # Create judge call for tie-breaking
        judge_call = models[0]["instructor_call"]

        # Create model configurations
        model_configs = [
            ModelConfiguration[KnowledgeChunkClassificationResponse](
                id=f"model_{i}",
                executor=m["instructor_call"],
                perspective=m.get("perspective", ""),
                weight=m.get("weight", 1.0),
            )
            for i, m in enumerate(models)
        ]

        # Create consensus settings
        consensus_settings = CoreConsensusSettings(
            threshold=settings.consensus_threshold,
            first_round_threshold=settings.consensus_threshold,
            max_rounds=settings.max_rounds,
        )

        consensus = Consensus[KnowledgeChunkClassificationResponse](
            models=model_configs,
            judge=judge_call,
            settings=consensus_settings,
        )

        super().__init__(consensus=consensus)
        self.settings = settings

    def fill_template(
        self,
        chunk_text: str,
        document_name: str,
        page_number: int,
        content_types: List[str],
    ) -> str:
        """
        Fill the prompt for chunk classification using templates from assets.

        Can be overridden via KNOWLEDGE_TEMPLATE_CHUNK_CLASSIFICATION environment variable
        pointing to a custom template file path.
        """
        template = load_template("chunk_classification.j2", "KNOWLEDGE_TEMPLATE_CHUNK_CLASSIFICATION")

        return template.render(
            chunk_text=chunk_text,
            document_name=document_name,
            page_number=page_number,
            content_types=content_types,
        )


class ConcreteTableCaptionExtractionCall(BaseTableCaptionExtractionCall):
    """
    Concrete table caption extraction call with full consensus configuration.
    """

    def __init__(self, settings: ConsensusSettings):
        """
        Initialize with consensus settings containing all model configurations.

        Args:
            settings: Complete consensus settings with models, thresholds, etc.
        """
        models = []
        for model_config in settings.models:
            instructor_call = InstructorLLMCall(
                response_model=TableCaptionExtractionResponse,
                temperature=model_config.temperature,
                model=model_config.model,
                base_url=model_config.api_url,
                api_key=model_config.api_key,
            )
            models.append(
                {
                    "instructor_call": instructor_call,
                    "weight": model_config.weight,
                    "perspective": model_config.perspective,
                }
            )

        if not models:
            raise ValueError("At least one table caption model must be configured.")

        # Create judge call for tie-breaking
        judge_call = models[0]["instructor_call"]

        # Create model configurations
        model_configs = [
            ModelConfiguration[TableCaptionExtractionResponse](
                id=f"model_{i}",
                executor=m["instructor_call"],
                perspective=m.get("perspective", ""),
                weight=m.get("weight", 1.0),
            )
            for i, m in enumerate(models)
        ]

        # Create consensus settings
        consensus_settings = CoreConsensusSettings(
            threshold=settings.consensus_threshold,
            first_round_threshold=settings.consensus_threshold,
            max_rounds=settings.max_rounds,
        )

        consensus = Consensus[TableCaptionExtractionResponse](
            models=model_configs,
            judge=judge_call,
            settings=consensus_settings,
        )

        super().__init__(consensus=consensus)
        self.settings = settings

    def fill_template(
        self,
        table_content: str,
        document_name: str,
        page_number: int,
    ) -> str:
        """
        Fill the prompt for table caption extraction.

        Uses a simple inline template for now. Can be overridden via
        KNOWLEDGE_TEMPLATE_TABLE_CAPTION environment variable if needed.
        """
        # For now, use inline template since it's simple
        prompt = f"""Analyze this table and generate a concise caption describing its content and purpose.

Table preview (first 500 characters):
{table_content}

Document: {document_name}
Page: {page_number}

Create a descriptive caption that clearly explains what data this table contains.
Focus on the subject matter and purpose. Keep it under 120 characters.
Do NOT include document name or page number in your caption."""

        return prompt


def create_extraction_calls(
    model_settings: ExtractionModelSettings,
) -> ExtractionCallsSettings:
    """
    Create extraction calls from model settings.

    This is the main factory function that creates all the extraction calls
    with proper consensus configuration based on the provided settings.

    Args:
        model_settings: Complete model settings for all extraction types

    Returns:
        ExtractionCallsSettings with all configured calls
    """
    # Create table caption call if settings are provided
    table_caption_call = None
    if model_settings.table_caption:
        table_caption_call = ConcreteTableCaptionExtractionCall(settings=model_settings.table_caption)

    return ExtractionCallsSettings(
        term_extraction_call=ConcreteTermExtractionCall(settings=model_settings.term_extraction),
        document_chunking_call=ConcreteChunkingCall(settings=model_settings.document_chunking),
        chunk_content_classification_call=ConcreteChunkClassificationCall(settings=model_settings.chunk_classification),
        table_caption_extraction_call=table_caption_call,
    )
