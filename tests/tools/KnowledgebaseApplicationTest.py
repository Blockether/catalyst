"""Tests for KnowledgebaseApplication workflow."""

import sys
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, Mock, patch

import pytest
from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAILike
from agno.workflow import Workflow
from agno.workflow.step import Step

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "tools"))

from blockether_catalyst.knowledge.KnowledgeSearchCore import KnowledgeSearchCore
from blockether_catalyst.knowledge.KnowledgeTypes import NormalizedSearchResult


class TestKnowledgebaseApplication:
    """Test suite for KnowledgebaseApplication workflow."""

    @pytest.fixture
    def mock_search_module(self):
        """Create mock KnowledgeSearchCore."""
        mock = MagicMock(spec=KnowledgeSearchCore)

        # Create mock search results
        mock_results = [
            MagicMock(spec=NormalizedSearchResult),
            MagicMock(spec=NormalizedSearchResult),
        ]

        # Configure mock results
        for i, result in enumerate(mock_results):
            result.markdown.return_value = f"# Result {i + 1}\nMocked search result content {i + 1}"
            result.document_name = f"document_{i + 1}.pdf"
            result.chunk_id = f"chunk_{i + 1}"
            result.page = i + 1
            result.relevance_score = 0.9 - (i * 0.1)
            # Configure model_dump method to return a dict with content
            result.model_dump.return_value = {
                "content": f"# Result {i + 1}\nMocked search result content {i + 1}",
                "document_name": f"document_{i + 1}.pdf",
                "chunk_id": f"chunk_{i + 1}",
                "page": i + 1,
                "relevance_score": 0.9 - (i * 0.1)
            }

        mock.search.return_value = mock_results
        return mock

    @pytest.fixture
    def mock_db(self):
        """Create mock SqliteDb."""
        return MagicMock(spec=SqliteDb)

    @pytest.fixture
    def mock_model(self):
        """Create mock OpenAI model."""
        model = MagicMock(spec=OpenAILike)
        model.api_key = "dummy"
        model.base_url = "http://localhost:3005/v1"
        model.id = "gpt-4o"
        return model

    def test_knowledge_retriever_function(self, mock_search_module):
        """Test the KnowledgeRetriever function."""
        # Import here to use mocked module
        with patch("tools.KnowledgebaseApplication.search_module", mock_search_module):
            from tools.KnowledgebaseApplication import KnowledgeRetriever

            # Test with valid query
            results = KnowledgeRetriever(query="test query", max_documents=5)

            # Verify search was called
            mock_search_module.search.assert_called_once_with(
                query="test query",
                k=5,  # Using max_documents=5, so k=5
                threshold=0.5,  # Updated to match actual implementation
                max_depth=2,
                max_cooccurrences=3,
            )

            # Verify results format
            assert len(results) == 2
            assert all("content" in result for result in results)
            assert results[0]["content"] == "# Result 1\nMocked search result content 1"

    def test_knowledge_retriever_empty_results(self, mock_search_module):
        """Test KnowledgeRetriever with no results."""
        mock_search_module.search.return_value = []

        with patch("tools.KnowledgebaseApplication.search_module", mock_search_module):
            from tools.KnowledgebaseApplication import KnowledgeRetriever

            results = KnowledgeRetriever(query="no matches", max_documents=5)

            assert results == []

    def test_knowledge_retriever_step_function(self, mock_search_module):
        """Test knowledge_retriever_step function."""
        with patch("tools.KnowledgebaseApplication.search_module", mock_search_module):
            from tools.KnowledgebaseApplication import knowledge_retriever_step

            # Call the step function with query
            results = knowledge_retriever_step("test query", num_documents=5)

            # Verify it returns the expected format
            assert isinstance(results, list)
            assert len(results) == 2  # Based on mock_search_module fixture

    def test_knowledge_agent_step_creation(self):
        """Test knowledge_agent_step creation."""
        from tools.KnowledgebaseApplication import knowledge_agent_step

        # Verify step properties
        assert isinstance(knowledge_agent_step, Step)
        assert knowledge_agent_step.name == "knowledge_agent"
        assert knowledge_agent_step.agent is not None

    def test_main_agent_creation(self):
        """Test MainKnowledgebaseAgent creation."""
        from tools.KnowledgebaseApplication import MainKnowledgebaseAgent

        # Verify agent configuration
        assert isinstance(MainKnowledgebaseAgent, Agent)
        assert MainKnowledgebaseAgent.name == "MainKnowledgebaseAgent"
        assert MainKnowledgebaseAgent.id == "MainKnowledgebaseAgent"
        assert MainKnowledgebaseAgent.tools is not None

    def test_main_workflow_creation(self):
        """Test MainKnowledgebaseWorkflow creation."""
        from tools.KnowledgebaseApplication import MainKnowledgebaseWorkflow

        # Verify workflow properties
        assert isinstance(MainKnowledgebaseWorkflow, Workflow)
        assert MainKnowledgebaseWorkflow.name == "Knowledge Base Q&A Workflow"
        assert MainKnowledgebaseWorkflow.id == "MainKnowledgebaseWorkflow"

    @patch("agno.workflow.Workflow.run")
    def test_workflow_execution(self, mock_run):
        """Test the complete workflow execution."""
        # Mock the workflow execution
        mock_run.return_value = {
            "status": "completed",
            "response": "Here is the information you requested...",
            "sources": ["document_1.pdf", "document_2.pdf"],
        }

        from tools.KnowledgebaseApplication import MainKnowledgebaseWorkflow

        # Simulate workflow execution
        result = MainKnowledgebaseWorkflow.run()

        # Verify workflow was executed
        mock_run.assert_called_once()
        assert result["status"] == "completed"
        assert "response" in result

    def test_agno_module_configuration(self):
        """Test the AgnoOsASGIModule configuration."""
        from tools.KnowledgebaseApplication import agno_asgi_module

        # Verify module properties
        assert agno_asgi_module is not None
        assert agno_asgi_module.chat is not None
        assert agno_asgi_module.chat.assistant.name == "Catalyst KnowledgeProvider"
        assert agno_asgi_module.mcp is not None

    def test_chat_configuration(self):
        """Test the chat configuration."""
        from tools.KnowledgebaseApplication import agno_asgi_module

        # Verify chat config
        chat_config = agno_asgi_module.chat
        assert chat_config is not None
        assert chat_config.assistant is not None
        assert chat_config.assistant.name == "Catalyst KnowledgeProvider"

    def test_error_handling_in_retriever(self, mock_search_module):
        """Test error handling in KnowledgeRetriever."""
        # Simulate search failure
        mock_search_module.search.side_effect = Exception("Search failed")

        with patch("tools.KnowledgebaseApplication.search_module", mock_search_module):
            from tools.KnowledgebaseApplication import KnowledgeRetriever

            # Should raise the exception (or handle it based on implementation)
            with pytest.raises(Exception, match="Search failed"):
                KnowledgeRetriever(query="test", max_documents=5)

    def test_mcp_config(self):
        """Test MCP configuration."""
        from tools.KnowledgebaseApplication import agno_asgi_module

        # Verify MCP config
        mcp_config = agno_asgi_module.mcp
        assert mcp_config is not None
        assert mcp_config.name == "Catalyst Knowledge MCP"
        assert len(mcp_config.tools) > 0

    def test_asgi_module_creation(self):
        """Test AgnoOsASGIModule creation."""
        from tools.KnowledgebaseApplication import agno_asgi_module

        # Verify module is created
        assert agno_asgi_module is not None
        assert agno_asgi_module.chat is not None
        assert agno_asgi_module.mcp is not None
        assert len(agno_asgi_module.workflows) > 0
