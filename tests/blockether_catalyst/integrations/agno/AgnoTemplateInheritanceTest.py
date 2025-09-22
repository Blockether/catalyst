"""Test template inheritance for AgnoOsASGIModule."""

from pathlib import Path
from typing import Any

import pytest
from agno.workflow import Workflow
from jinja2 import ChoiceLoader, FileSystemLoader

from blockether_catalyst.integrations.agno.AgnoOsASGIModule import (
    AgnoOsASGIModule,
    AssistantConfig,
    ChatConfig,
)


class TestAgnoTemplateInheritance:
    """Test class for Agno template inheritance functionality."""

    # Constants to avoid magic numbers
    EXPECTED_LOADER_COUNT = 2
    AGNO_TEMPLATE_LOADER_INDEX = 0
    ASGI_TEMPLATE_LOADER_INDEX = 1

    @pytest.fixture
    def test_workflow(self) -> Workflow:
        """Create a test workflow for testing."""
        return Workflow(
            id="test-workflow",
            name="Test Workflow",
            description="Test workflow for template verification",
        )

    @pytest.fixture
    def chat_config(self, test_workflow: Workflow) -> ChatConfig:
        """Create a test chat configuration."""
        return ChatConfig(
            assistant=AssistantConfig(name="Test Assistant", short="T", runner=test_workflow),
            base_url="http://localhost:8000",
        )

    @pytest.fixture
    def agno_module(self, chat_config: ChatConfig, test_workflow: Workflow) -> AgnoOsASGIModule:
        """Create an AgnoOsASGIModule instance for testing."""
        return AgnoOsASGIModule(
            title="Test Module",
            description="Test module for template verification",
            chat=chat_config,
            workflows=[test_workflow],
        )

    def test_module_instantiation(self, agno_module: AgnoOsASGIModule) -> None:
        """Test that AgnoOsASGIModule instantiates successfully."""
        assert agno_module is not None
        assert agno_module.title == "Test Module"
        assert agno_module.description == "Test module for template verification"

    def test_templates_are_configured(self, agno_module: AgnoOsASGIModule) -> None:
        """Test that templates are properly configured."""
        assert agno_module.templates is not None
        assert hasattr(agno_module.templates, "env")

    def test_template_loader_is_choice_loader(self, agno_module: AgnoOsASGIModule) -> None:
        """Test that the template loader is a ChoiceLoader."""
        loader = agno_module.templates.env.loader
        assert isinstance(loader, ChoiceLoader)

    def test_choice_loader_has_correct_number_of_loaders(self, agno_module: AgnoOsASGIModule) -> None:
        """Test that ChoiceLoader has exactly 2 loaders configured."""
        loader = agno_module.templates.env.loader
        assert len(loader.loaders) == self.EXPECTED_LOADER_COUNT

    def test_first_loader_searches_agno_templates(self, agno_module: AgnoOsASGIModule) -> None:
        """Test that the first loader searches in Agno templates directory."""
        loader = agno_module.templates.env.loader
        first_loader = loader.loaders[self.AGNO_TEMPLATE_LOADER_INDEX]
        assert isinstance(first_loader, FileSystemLoader)

        # Check the search path
        search_paths = first_loader.searchpath
        assert len(search_paths) == 1

        # Verify it points to Agno templates
        expected_path = (
            Path(__file__).parent.parent.parent.parent.parent
            / "src"
            / "blockether_catalyst"
            / "integrations"
            / "agno"
            / "templates"
        )
        actual_path = Path(search_paths[0])
        assert actual_path.resolve() == expected_path.resolve()

    def test_second_loader_searches_asgi_templates(self, agno_module: AgnoOsASGIModule) -> None:
        """Test that the second loader searches in ASGI templates directory."""
        loader = agno_module.templates.env.loader
        second_loader = loader.loaders[self.ASGI_TEMPLATE_LOADER_INDEX]
        assert isinstance(second_loader, FileSystemLoader)

        # Check the search path
        search_paths = second_loader.searchpath
        assert len(search_paths) == 1

        # Verify it points to ASGI templates
        expected_path = (
            Path(__file__).parent.parent.parent.parent.parent / "src" / "blockether_catalyst" / "asgi" / "templates"
        )
        actual_path = Path(search_paths[0])
        assert actual_path.resolve() == expected_path.resolve()

    def test_chat_template_loads_from_agno_directory(self, agno_module: AgnoOsASGIModule) -> None:
        """Test that chat.j2 loads successfully from Agno templates."""
        template = agno_module.templates.env.get_template("chat.j2")
        assert template is not None
        assert template.name == "chat.j2"

    def test_base_template_loads_from_asgi_directory(self, agno_module: AgnoOsASGIModule) -> None:
        """Test that base.j2 loads successfully from ASGI templates."""
        template = agno_module.templates.env.get_template("base.j2")
        assert template is not None
        assert template.name == "base.j2"

    def test_partial_templates_load_correctly(self, agno_module: AgnoOsASGIModule) -> None:
        """Test that partial templates load from Agno templates."""
        # Test workflow_message partial
        workflow_template = agno_module.templates.env.get_template("partials/workflow_message.j2")
        assert workflow_template is not None
        assert workflow_template.name == "partials/workflow_message.j2"

        # Test user_message partial
        user_template = agno_module.templates.env.get_template("partials/user_message.j2")
        assert user_template is not None
        assert user_template.name == "partials/user_message.j2"

    def test_chat_template_extends_base_template(self, agno_module: AgnoOsASGIModule) -> None:
        """Test that chat.j2 correctly extends base.j2."""
        # Read the chat.j2 file content
        chat_file_path = (
            Path(__file__).parent.parent.parent.parent.parent
            / "src"
            / "blockether_catalyst"
            / "integrations"
            / "agno"
            / "templates"
            / "chat.j2"
        )
        assert chat_file_path.exists()

        with open(chat_file_path, "r") as f:
            chat_content = f.read()

        # Verify it extends base.j2
        assert '{% extends "base.j2" %}' in chat_content

    def test_duplicate_base_template_does_not_exist(self) -> None:
        """Test that duplicate base.j2 has been removed from Agno templates."""
        agno_base_path = (
            Path(__file__).parent.parent.parent.parent.parent
            / "src"
            / "blockether_catalyst"
            / "integrations"
            / "agno"
            / "templates"
            / "base.j2"
        )
        assert not agno_base_path.exists()

    def test_template_rendering_with_context(self, agno_module: AgnoOsASGIModule) -> None:
        """Test that templates can be rendered with context data."""
        # Test rendering base template with minimal context
        base_template = agno_module.templates.env.get_template("base.j2")
        rendered = base_template.render(
            module_title="Test Title",
            module_prefix="/test",
        )

        assert "Test Title" in rendered
        assert "<!DOCTYPE html>" in rendered
        assert "</html>" in rendered

    def test_template_autoescape_is_enabled(self, agno_module: AgnoOsASGIModule) -> None:
        """Test that autoescape is enabled for security."""
        assert agno_module.templates.env.autoescape is True
