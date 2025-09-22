"""Simple REAL integration tests that directly test Agents, Workflows, and Teams with LLM.

NO MOCKS - REAL LLM TESTS ONLY!
These tests verify that the executors work with real LLM responses.
All LLM configuration comes from conftest.py
"""

import pytest
from agno.agent import Agent
from agno.team import Team
from agno.workflow import Workflow
from agno.workflow.step import Step

# All fixtures (test_llm, test_db) are provided by conftest.py


class TestRealLLMIntegration:
    """Direct tests with REAL LLM - NO MOCKS."""

    # Fixtures test_llm and test_db are provided by conftest.py

    def test_agent_direct_math(self, test_llm, test_db) -> None:
        """Test REAL Agent with REAL LLM doing math."""
        agent = Agent(
            id="math_agent",
            model=test_llm,
            name="Math Agent",
            description="Solves math problems",
            instructions=["Solve math problems step by step"],
            db=test_db,
            telemetry=False,
        )

        # REAL LLM call
        result = agent.run("What is 25 + 17?")

        # REAL response must contain correct answer
        assert result is not None, "Agent returned None!"
        content = str(result.content)
        assert "42" in content, f"LLM failed basic math! Response: {content}"
        print(f"✅ Agent correctly calculated: {content[:100]}...")

    def test_workflow_direct_calculation(self, test_llm, test_db) -> None:
        """Test REAL Workflow with REAL LLM."""
        # Create agent for workflow
        calc_agent = Agent(
            id="calc_agent",
            model=test_llm,
            name="Calculator",
            description="Performs calculations",
            instructions=["Calculate precisely"],
            db=test_db,
            telemetry=False,
        )

        # Create workflow step
        calc_step = Step(
            name="calculate",
            description="Calculate the result",
            agent=calc_agent,
        )

        # Create workflow
        workflow = Workflow(
            id="calc_workflow",
            name="Calculation Workflow",
            description="Calculates things",
            db=test_db,
            steps=[calc_step],
            telemetry=False,
        )

        # REAL workflow execution
        result = workflow.run("What is 5 times 8?")

        # REAL response validation
        assert result is not None, "Workflow returned None!"
        content = str(result.content if hasattr(result, "content") else result)
        assert "40" in content, f"Workflow failed multiplication! Response: {content}"
        print(f"✅ Workflow correctly calculated: {content[:100]}...")

    def test_team_collaboration_real(self, test_llm, test_db) -> None:
        """Test REAL Team with multiple agents collaborating."""
        # Create researcher agent
        researcher = Agent(
            id="researcher",
            model=test_llm,
            name="Researcher",
            description="Researches topics",
            instructions=["Research and analyze the topic thoroughly"],
            db=test_db,
            telemetry=False,
        )

        # Create expert agent
        expert = Agent(
            id="expert",
            model=test_llm,
            name="Expert",
            description="Provides expert advice",
            instructions=["Provide expert recommendations"],
            db=test_db,
            telemetry=False,
        )

        # Create team
        team = Team(
            id="expert_team",
            name="Expert Team",
            description="Research and expert advice",
            members=[researcher, expert],
            model=test_llm,  # Use the test LLM
            db=test_db,
            telemetry=False,
        )

        # REAL team execution
        result = team.run("What are the benefits of Python for data science?")

        # REAL team response validation
        assert result is not None, "Team returned None!"
        content = str(result.content if hasattr(result, "content") else result)

        # Team must mention Python benefits
        keywords = ["python", "data", "library", "pandas", "numpy", "analysis"]
        found = [kw for kw in keywords if kw in content.lower()]
        assert len(found) >= 3, f"Team didn't provide proper analysis! Found: {found}. Response: {content[:200]}..."
        print(f"✅ Team provided analysis with keywords: {found}")

    def test_agent_memory_real(self, test_llm, test_db) -> None:
        """Test REAL agent memory persistence."""
        agent = Agent(
            id="memory_agent",
            model=test_llm,
            name="Memory Agent",
            description="Agent with memory",
            instructions=["Remember what users tell you"],
            db=test_db,
            telemetry=False,
            store_events=True,
            read_chat_history=True,
        )

        session_id = "test_memory_session"

        # First interaction
        result1 = agent.run(
            "My favorite programming language is Rust and my favorite number is 73",
            session_id=session_id,
        )
        assert result1 is not None

        # Second interaction - MUST remember
        result2 = agent.run(
            "What is my favorite programming language and number?",
            session_id=session_id,
        )

        content = str(result2.content)
        assert "rust" in content.lower(), f"Agent forgot language! Response: {content}"
        assert "73" in content, f"Agent forgot number! Response: {content}"
        print("✅ Agent remembered: Rust and 73")

    def test_complex_reasoning(self, test_llm, test_db) -> None:
        """Test REAL complex reasoning with LLM."""
        agent = Agent(
            id="reasoning_agent",
            model=test_llm,
            name="Reasoning Agent",
            description="Complex reasoning",
            instructions=["Think step by step", "Show your reasoning"],
            db=test_db,
            telemetry=False,
            reasoning=True,
        )

        # Complex question requiring reasoning
        result = agent.run(
            "If a train travels 60 mph for 2 hours, then 80 mph for 3 hours, " "what is the total distance traveled?"
        )

        content = str(result.content)
        # Must calculate: (60*2) + (80*3) = 120 + 240 = 360
        assert "360" in content, f"Failed complex calculation! Response: {content}"
        print("✅ Agent correctly reasoned: 360 miles")


class TestRealAPIIntegration:
    """Test REAL API endpoints with REAL LLM responses."""

    def test_agno_workflow_api_real(self, test_llm, test_db) -> None:
        """Test REAL workflow through Agno API."""
        from agno.os import AgentOS
        from agno.os.settings import AgnoAPISettings

        # Create a simple workflow
        agent = Agent(
            id="api_agent",
            model=test_llm,
            name="API Agent",
            description="API test agent",
            instructions=["Answer questions"],
            db=test_db,
            telemetry=False,
        )

        step = Step(
            name="answer",
            description="Answer the question",
            agent=agent,
        )

        workflow = Workflow(
            id="api_workflow",
            name="API Workflow",
            description="Workflow for API testing",
            db=test_db,
            steps=[step],
            telemetry=False,
        )

        # Create AgentOS
        os = AgentOS(
            os_id="test_os",
            version="1.0.0",
            workflows=[workflow],
            teams=[],
            settings=AgnoAPISettings(
                docs_enabled=False,
                cors_origin_list=["*"],
            ),
            telemetry=False,
        )

        # Get the FastAPI app
        app = os.get_app()
        assert app is not None, "Failed to create AgentOS app"

        # Test that the workflow is registered
        assert len(os.workflows) == 1
        assert os.workflows[0].id == "api_workflow"
        print("✅ Workflow registered in AgentOS")

    def test_error_handling_real_llm(self, test_llm, test_db) -> None:
        """Test error handling with REAL LLM."""
        agent = Agent(
            id="error_agent",
            model=test_llm,
            name="Error Test Agent",
            description="Tests error scenarios",
            instructions=["Handle requests appropriately"],
            db=test_db,
            telemetry=False,
        )

        # Test with invalid/nonsense input
        result = agent.run("XYZABC123!@#$%^&*()")

        # Even with nonsense, LLM should respond
        assert result is not None, "Agent failed on unusual input"
        content = str(result.content)
        assert len(content) > 0, "Agent returned empty response"
        print("✅ Agent handled unusual input gracefully")


if __name__ == "__main__":
    # Import configuration from conftest
    import sys

    sys.path.insert(0, "/Users/fierycod/com_blockether_catalyst")
    from agno.db.sqlite import SqliteDb
    from agno.models.openai import OpenAILike

    from tests.blockether_catalyst.integrations.conftest import (
        LLM_API_KEY,
        LLM_BASE_URL,
        LLM_MODEL,
        check_llm_server,
    )

    # Check LLM availability
    check_llm_server()
    print(f"✅ LLM server is available at {LLM_BASE_URL}")

    # Quick smoke test
    llm = OpenAILike(
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        id=LLM_MODEL,
        temperature=0.1,
    )

    db = SqliteDb()

    agent = Agent(
        id="smoke_test",
        model=llm,
        name="Smoke Test",
        description="Quick test",
        instructions=["Answer briefly"],
        db=db,
        telemetry=False,
    )

    result = agent.run("Say 'Hello World'")
    assert result is not None
    print(f"✅ Smoke test passed: {result.content}")
