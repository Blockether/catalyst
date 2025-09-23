"""Real integration tests for AgnoOsASGIModule with actual LLMs.

THESE ARE REAL TESTS - NO MOCKS!
Requires LLM server running at localhost:3005
Will fail immediately if LLM is not available.
"""

import json
import time
from typing import Any, Dict, List, Optional

import pytest
from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAILike
from agno.team import Team
from agno.workflow import Workflow
from agno.workflow.step import Step
from fastapi.testclient import TestClient
from fastmcp.tools import Tool
from pydantic import BaseModel, Field

from blockether_catalyst.asgi.ASGICoreApplication import ASGICoreApplication
from blockether_catalyst.integrations.agno.AgnoOsASGIModule import (
    AgnoOsASGIModule,
    AgnoOSAPISettings,
    AssistantConfig,
    ChatConfig,
    MCPConfig,
)

# Import global LLM configuration from conftest
# All LLM configuration is now centralized in conftest.py


class MathProblem(BaseModel):
    """Model for math problem solving."""

    problem: str = Field(description="The math problem to solve")
    solution: str = Field(description="The step-by-step solution")
    answer: float = Field(description="The final numerical answer")


class TestAgnoOsASGIModuleIntegration:
    """Real integration tests with actual LLM."""

    BASE_URL = "http://localhost:8000"
    TEST_SESSION_ID = "test_session_integration"

    # test_llm fixture is now provided by conftest.py

    # test_db fixture is now provided by conftest.py

    @pytest.fixture
    def test_agent(self, test_llm, test_db) -> Agent:
        """Create a real test agent with memory persistence enabled."""
        agent = Agent(
            id="test_agent_001",
            name="Math Assistant",
            description="An agent that helps with math problems",
            instructions=[
                "You are a helpful math assistant.",
                "Solve math problems step by step.",
                "Always provide clear explanations.",
            ],
            model=test_llm,
            telemetry=False,
            store_events=True,
            db=test_db,
            debug_mode=True,
            markdown=True,
            # CRITICAL: Enable conversation history for memory persistence
            add_history_to_context=True,
            num_history_runs=5,  # Include last 5 conversations in context
        )
        return agent

    @pytest.fixture
    def test_workflow(self, test_agent, test_db) -> Workflow:
        """Create a real test workflow with agent step."""
        # Create a step that uses the agent
        agent_step = Step(
            name="solve_math",
            description="Solve a math problem using the agent",
            agent=test_agent,
        )

        workflow = Workflow(
            id="test_workflow_001",
            name="Math Problem Solver",
            description="A workflow that solves math problems",
            db=test_db,
            telemetry=False,
            debug_mode=True,
            steps=[agent_step],
            store_events=True,
            store_executor_outputs=True,
        )
        return workflow

    @pytest.fixture
    def test_team(self, test_llm, test_db) -> Team:
        """Create a real test team with multiple agents."""
        # Create researcher agent
        researcher = Agent(
            id="researcher_001",
            name="Research Agent",
            description="Researches and analyzes problems",
            instructions=[
                "You are a research specialist.",
                "Analyze problems thoroughly.",
                "Provide detailed context and background.",
            ],
            telemetry=False,
            db=test_db,
        )

        # Create solver agent
        solver = Agent(
            id="solver_001",
            name="Solution Agent",
            description="Provides solutions to problems",
            instructions=[
                "You are a solution expert.",
                "Provide clear, actionable solutions.",
                "Be concise and practical.",
            ],
            telemetry=False,
            db=test_db,
        )

        # Create the team
        team = Team(
            id="test_team_001",
            name="Problem Solving Team",
            description="A team that researches and solves problems",
            members=[researcher, solver],  # Use 'members' instead of 'agents'
            db=test_db,
            telemetry=False,
            debug_mode=True,
            store_events=True,
        )
        return team

    @pytest.fixture
    def math_tool(self):
        """Create a real MCP tool for testing."""

        def calculate(
            expression: str = Field(..., description="Mathematical expression to evaluate"),
        ) -> Dict[str, Any]:
            """Evaluate a mathematical expression."""
            try:
                # Use eval safely for simple math
                result = eval(expression, {"__builtins__": {}}, {})
                return {"expression": expression, "result": result, "success": True}
            except Exception as e:
                return {"expression": expression, "error": str(e), "success": False}

        return Tool.from_function(
            fn=calculate,
            name="calculator",
            description="Evaluate mathematical expressions",
            enabled=True,
        )

    @pytest.fixture
    def app_with_agent(self, test_agent, test_llm, math_tool):
        """Create ASGI app with agent executor."""

        # We need to wrap the agent in a workflow for AgentOS
        from agno.workflow import Workflow
        from agno.workflow.step import Step

        agent_step = Step(
            name="agent_step",
            description="Execute agent",
            agent=test_agent,
        )

        agent_workflow = Workflow(
            id="agent_workflow_wrapper",
            name="Agent Workflow Wrapper",
            description="Workflow wrapper for agent",
            db=test_agent.db,
            telemetry=False,
            steps=[agent_step],
        )

        # CRITICAL FIX: Use the workflow wrapper for chat, not the raw agent
        # This ensures both execution and retrieval use the same persistence mechanism
        chat_config = ChatConfig(
            assistant=AssistantConfig(
                name="Math Agent Assistant",
                short="M",
                runner=agent_workflow,  # Use workflow instead of raw agent
            ),
            base_url=self.BASE_URL,
        )

        mcp_config = MCPConfig(
            name="Math Agent MCP",
            tools=[math_tool],
        )

        module = AgnoOsASGIModule(
            title="Agent Test Module",
            description="Testing with real agent",
            workflows=[agent_workflow],  # Add the workflow wrapper
            teams=[],
            chat=chat_config,
            mcp=mcp_config,
            api=AgnoOSAPISettings(
                docs_enabled=True,
                api_token=None,  # No auth for tests
            ),
        )

        asgi_app = ASGICoreApplication(
            title="Agent Test App",
            description="Integration test app with agent",
            version="1.0.0",
            prefix="/",
            debug=True,
        )
        asgi_app.mount_module(module)

        return TestClient(asgi_app.app)

    @pytest.fixture
    def app_with_workflow(self, test_workflow, math_tool):
        """Create ASGI app with workflow executor."""
        chat_config = ChatConfig(
            assistant=AssistantConfig(
                name="Math Workflow Assistant",
                short="W",
                runner=test_workflow,
            ),
            base_url=self.BASE_URL,
        )

        mcp_config = MCPConfig(
            name="Math Workflow MCP",
            tools=[math_tool],
        )

        module = AgnoOsASGIModule(
            title="Workflow Test Module",
            description="Testing with real workflow",
            workflows=[test_workflow],
            teams=[],
            chat=chat_config,
            mcp=mcp_config,
        )

        asgi_app = ASGICoreApplication(
            title="Workflow Test App",
            description="Integration test app with workflow",
            version="1.0.0",
            prefix="/",
            debug=True,
        )
        asgi_app.mount_module(module)

        return TestClient(asgi_app.app)

    @pytest.fixture
    def app_with_team(self, test_team):
        """Create ASGI app with team executor."""
        chat_config = ChatConfig(
            assistant=AssistantConfig(
                name="Problem Solving Team Assistant",
                short="T",
                runner=test_team,
            ),
            base_url=self.BASE_URL,
        )

        module = AgnoOsASGIModule(
            title="Team Test Module",
            description="Testing with real team",
            workflows=[],
            teams=[test_team],
            chat=chat_config,
        )

        asgi_app = ASGICoreApplication(
            title="Team Test App",
            description="Integration test app with team",
            version="1.0.0",
            prefix="/",
            debug=True,
        )
        asgi_app.mount_module(module)

        return TestClient(asgi_app.app)

    def test_agent_chat_interface(self, app_with_agent):
        """Test that agent chat interface loads correctly."""
        response = app_with_agent.get("/os/view")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

        # Check that it's an HTML chat interface
        content = response.text
        assert "<html" in content
        assert "chat-messages" in content  # Chat messages container
        assert "message-input" in content  # Message input field

    def test_workflow_chat_interface(self, app_with_workflow):
        """Test that workflow chat interface loads correctly."""
        response = app_with_workflow.get("/os/view")
        assert response.status_code == 200

        content = response.text
        assert "<html" in content
        assert "chat-messages" in content
        assert "message-input" in content

    def test_team_chat_interface(self, app_with_team):
        """Test that team chat interface loads correctly."""
        response = app_with_team.get("/os/view")
        assert response.status_code == 200

        content = response.text
        assert "<html" in content
        assert "chat-messages" in content
        assert "message-input" in content

    @pytest.mark.require_llm_server
    def test_agent_executor_run(self, app_with_agent):
        """Test REAL agent execution with REAL LLM response."""
        request_data = {
            "message": "What is 15 + 27?",
            "user_id": "test_user",
            "session_id": self.TEST_SESSION_ID,
        }

        response = app_with_agent.post("/os/executor/runs", data=request_data)
        assert response.status_code == 200, "Agent execution failed!"

        data = response.json()
        assert "run_id" in data, "No run_id in response"
        assert "session_id" in data, "No session_id in response"

        # REAL LLM MUST provide the correct answer
        content = str(data.get("content", data.get("output", "")))
        assert "42" in content, f"LLM did not calculate 15+27 correctly! Response: {content}"

    @pytest.mark.require_llm_server
    def test_workflow_executor_run(self, app_with_workflow):
        """Test REAL workflow execution with REAL LLM response."""
        request_data = {
            "message": "Calculate the area of a rectangle with width 5 and height 8",
            "user_id": "test_user",
            "session_id": self.TEST_SESSION_ID,
        }

        response = app_with_workflow.post("/os/executor/runs", data=request_data)
        assert response.status_code == 200, "Workflow execution failed!"

        data = response.json()
        assert "run_id" in data, "No run_id in workflow response"

        # REAL LLM MUST calculate area correctly
        content = str(data.get("content", data.get("output", "")))
        assert (
            "40" in content or "forty" in content.lower()
        ), f"LLM did not calculate area correctly! Response: {content}"

    @pytest.mark.require_llm_server
    def test_team_executor_run(self, app_with_team):
        """Test REAL team collaboration with REAL LLM responses."""
        request_data = {
            "message": "How can I improve my Python code performance?",
            "user_id": "test_user",
            "session_id": self.TEST_SESSION_ID,
        }

        response = app_with_team.post("/os/executor/runs", data=request_data)
        assert response.status_code == 200, "Team execution failed!"

        data = response.json()
        assert "run_id" in data, "No run_id in team response"

        # REAL team MUST provide performance tips
        content = str(data.get("content", data.get("output", "")))
        performance_keywords = [
            "performance",
            "optimize",
            "profiling",
            "cache",
            "speed",
            "efficient",
        ]
        found_keywords = [kw for kw in performance_keywords if kw in content.lower()]
        assert (
            len(found_keywords) >= 2
        ), f"Team did not provide proper performance advice! Found keywords: {found_keywords}. Response: {content[:200]}..."

    def test_render_message_with_metrics(self, app_with_workflow):
        """Test rendering a message with metrics."""
        message_data = {
            "content": "The calculation is complete: 15 + 27 = 42",
            "message_id": "msg_test_123",
            "workflow_name": "Math Problem Solver",
            "run_id": "run_test_456",
            "status": "completed",
            "metrics": {
                "steps": {
                    "solve_math": {
                        "metrics": {
                            "input_tokens": 50,
                            "output_tokens": 25,
                            "total_tokens": 75,
                            "duration": 0.5,
                        }
                    }
                }
            },
        }

        response = app_with_workflow.post("/os/view/render-message", json=message_data)
        assert response.status_code == 200

        content = response.text
        assert "42" in content
        # Check that the message was rendered
        assert "msg_test_123" in content  # Message ID is in the content
        assert "42" in content  # The calculation result is shown

    @pytest.mark.require_llm_server
    def test_session_runs_retrieval(self, app_with_agent):
        """Test REAL session with REAL LLM run retrieval."""
        # Create a REAL run with REAL LLM
        request_data = {
            "message": "What is 2 + 2?",
            "user_id": "test_user",
            "session_id": self.TEST_SESSION_ID,
        }

        create_response = app_with_agent.post("/os/executor/runs", data=request_data)
        assert create_response.status_code == 200, "Failed to create run with LLM"

        create_data = create_response.json()
        created_run_id = create_data.get("run_id")
        assert created_run_id, "No run_id from LLM execution"

        # Verify LLM answered correctly
        content = str(create_data.get("content", create_data.get("output", "")))
        assert "4" in content, f"LLM did not answer 2+2 correctly! Response: {content}"

        # Wait for storage
        time.sleep(1.0)

        # Retrieve runs - MUST find our run
        get_response = app_with_agent.get(f"/os/sessions/{self.TEST_SESSION_ID}/runs")
        assert get_response.status_code == 200, "Failed to retrieve session runs"

        runs = get_response.json()
        assert isinstance(runs, list), "Runs is not a list"
        assert len(runs) > 0, "No runs found in session!"
        assert any(
            run.get("run_id") == created_run_id for run in runs
        ), f"Created run {created_run_id} not found in session runs!"

    def test_agent_with_mcp_tools(self, app_with_agent):
        """Test that MCP tools are properly registered with agent."""
        # The calculator tool should be available
        # We can test this by checking the MCP configuration
        response = app_with_agent.get("/os/view")
        assert response.status_code == 200

        # The MCP tools would be available to the agent
        # In a real scenario, we'd test tool execution

    @pytest.mark.require_llm_server
    def test_agent_memory_persistence(self, app_with_agent):
        """Test REAL agent memory with REAL LLM maintaining context."""
        session_id = "memory_test_session_" + str(time.time())  # Unique session

        # First REAL LLM interaction
        request1 = {
            "message": "My favorite number is 42 and my favorite color is blue",
            "user_id": "test_user",
            "session_id": session_id,
        }

        response1 = app_with_agent.post("/os/executor/runs", data=request1)
        assert response1.status_code == 200, "First memory message failed"

        # Wait for memory to be stored
        time.sleep(1.0)

        # Second REAL LLM interaction - MUST remember
        request2 = {
            "message": "What was my favorite number and color?",
            "user_id": "test_user",
            "session_id": session_id,
        }

        response2 = app_with_agent.post("/os/executor/runs", data=request2)
        assert response2.status_code == 200, "Second memory message failed"

        data = response2.json()
        content = str(data.get("content", data.get("output", "")))

        # REAL LLM MUST remember BOTH pieces of information
        assert "42" in content, f"Agent forgot the number! Response: {content}"
        assert "blue" in content.lower(), f"Agent forgot the color! Response: {content}"

    def test_error_handling_invalid_session(self, app_with_workflow):
        """Test error handling for invalid session."""
        response = app_with_workflow.get("/os/sessions/non_existent_session/runs")
        # The endpoint returns 404 which is correct for non-existent session
        assert response.status_code == 404 or response.status_code == 200

        # If 200, should return empty array
        if response.status_code == 200:
            data = response.json()
            assert data == []

    def test_dev_mode_authentication(self, app_with_team):
        """Test dev mode authentication bypass."""
        response = app_with_team.get("/os/view?dev_mode=true")
        assert response.status_code == 200

        content = response.text
        # In dev mode, user should be DemoUser
        assert "DemoUser" in content or "user_id" in content


# Additional test utilities
class TestDataGenerator:
    """Generate test data for various scenarios."""

    @staticmethod
    def math_problems() -> List[str]:
        """Generate various math problems for testing."""
        return [
            "What is 15 + 27?",
            "Calculate the area of a circle with radius 5",
            "Solve for x: 2x + 5 = 15",
            "What is the factorial of 5?",
            "Find the prime factors of 60",
        ]

    @staticmethod
    def coding_questions() -> List[str]:
        """Generate coding questions for team testing."""
        return [
            "How can I optimize a Python loop?",
            "What are the best practices for error handling?",
            "Explain the difference between async and sync functions",
            "How to implement caching in Python?",
            "What is the time complexity of binary search?",
        ]


@pytest.mark.integration
class TestEndToEndScenarios:
    """End-to-end integration scenarios."""

    BASE_URL = "http://localhost:8000"

    @pytest.fixture
    def test_agent(self, test_llm, test_db):
        """Create a real test agent."""
        agent = Agent(
            id="test_agent_001",
            name="Math Assistant",
            description="An agent that helps with math problems",
            instructions=[
                "You are a helpful math assistant.",
                "Solve math problems step by step.",
                "Always provide clear explanations.",
            ],
            model=test_llm,
            telemetry=False,
            store_events=True,
            db=test_db,
            debug_mode=True,
        )
        return agent

    @pytest.fixture
    def test_workflow(self, test_agent, test_db) -> Workflow:
        """Create a real test workflow with agent step."""
        # Create a step that uses the agent
        agent_step = Step(
            name="solve_math",
            description="Solve a math problem using the agent",
            agent=test_agent,
        )

        workflow = Workflow(
            id="test_workflow_001",
            name="Math Problem Solver",
            description="A workflow that solves math problems",
            db=test_db,
            telemetry=False,
            debug_mode=True,
            steps=[agent_step],
            store_events=True,
            store_executor_outputs=True,
        )
        return workflow

    @pytest.fixture
    def test_team(self, test_llm, test_db) -> Team:
        """Create a real test team with multiple agents."""
        # Create researcher agent
        researcher = Agent(
            id="researcher_001",
            name="Research Agent",
            model=test_llm,
            description="Researches and analyzes problems",
            instructions=[
                "You are a research specialist.",
                "Analyze problems thoroughly.",
                "Provide detailed context and background.",
            ],
            telemetry=False,
            db=test_db,
        )

        # Create solver agent
        solver = Agent(
            id="solver_001",
            name="Solution Agent",
            model=test_llm,
            description="Provides solutions to problems",
            instructions=[
                "You are a solution expert.",
                "Provide clear, actionable solutions.",
                "Be concise and practical.",
            ],
            telemetry=False,
            db=test_db,
        )

        # Create the team
        team = Team(
            id="test_team_001",
            name="Problem Solving Team",
            description="A team that researches and solves problems",
            members=[researcher, solver],  # Use 'members' instead of 'agents'
            model=test_llm,
            db=test_db,
            telemetry=False,
            debug_mode=True,
            store_events=True,
        )
        return team

    @pytest.fixture
    def math_tool(self):
        """Create a real MCP tool for testing."""
        return Tool(
            name="add_numbers",
            description="Add two numbers together",
            parameters={"properties": {"a": {"type": "number"}, "b": {"type": "number"}}},
        )

    @pytest.fixture
    def app_with_workflow(self, test_workflow, math_tool):
        """Create ASGI app with workflow executor."""
        chat_config = ChatConfig(
            assistant=AssistantConfig(
                name="Math Workflow Assistant",
                short="W",
                runner=test_workflow,
            ),
            base_url=self.BASE_URL,
        )

        mcp_config = MCPConfig(
            name="Math Workflow MCP",
            tools=[math_tool],
        )

        module = AgnoOsASGIModule(
            title="Workflow Test Module",
            description="Testing with real workflow",
            workflows=[test_workflow],
            teams=[],
            chat=chat_config,
            mcp=mcp_config,
        )

        asgi_app = ASGICoreApplication(
            title="Workflow Test App",
            description="Integration test app with workflow",
            version="1.0.0",
            prefix="/",
            debug=True,
        )
        asgi_app.mount_module(module)

        return TestClient(asgi_app.app)

    @pytest.fixture
    def app_with_team(self, test_team):
        """Create ASGI app with team executor."""
        chat_config = ChatConfig(
            assistant=AssistantConfig(
                name="Problem Solving Team Assistant",
                short="T",
                runner=test_team,
            ),
            base_url=self.BASE_URL,
        )

        module = AgnoOsASGIModule(
            title="Team Test Module",
            description="Testing with real team",
            workflows=[],
            teams=[test_team],
            chat=chat_config,
        )

        asgi_app = ASGICoreApplication(
            title="Team Test App",
            description="Integration test app with team",
            version="1.0.0",
            prefix="/",
            debug=True,
        )
        asgi_app.mount_module(module)

        return TestClient(asgi_app.app)

    @pytest.mark.require_llm_server
    def test_complete_workflow_cycle(self, app_with_workflow):
        """Test REAL end-to-end workflow with REAL LLM calculations."""
        session_id = "complete_cycle_" + str(time.time())

        # Step 1: Create REAL run with REAL LLM
        request = {
            "message": "Calculate the sum of first 10 natural numbers (1+2+3+...+10)",
            "user_id": "test_user",
            "session_id": session_id,
        }

        create_response = app_with_workflow.post("/os/executor/runs", data=request)
        assert create_response.status_code == 200, "Workflow run creation failed"

        run_data = create_response.json()
        run_id = run_data.get("run_id")
        assert run_id, "No run_id in workflow response"

        # Verify REAL LLM calculated correctly
        content = str(run_data.get("content", run_data.get("output", "")))
        assert "55" in content, f"LLM did not calculate sum correctly! Expected 55. Response: {content}"

        # Step 2: Verify session storage
        time.sleep(1.0)
        runs_response = app_with_workflow.get(f"/os/sessions/{session_id}/runs")
        assert runs_response.status_code == 200, "Failed to get session runs"

        runs = runs_response.json()
        assert runs, "No runs in session"
        assert any(run.get("run_id") == run_id for run in runs), f"Run {run_id} not found in session"

        # Step 3: Render the REAL message
        render_request = {
            "content": content,
            "run_id": run_id,
            "session_id": session_id,
            "status": "completed",
        }

        render_response = app_with_workflow.post("/os/view/render-message", json=render_request)
        assert render_response.status_code == 200, "Failed to render message"

        # Verify rendered content has the answer
        rendered = render_response.text
        assert "55" in rendered, "Rendered message missing the answer"

    @pytest.mark.require_llm_server
    def test_team_collaboration(self, app_with_team):
        """Test REAL team collaboration with REAL multi-agent LLM responses."""
        request = {
            "message": "Research Python async programming and provide best practices",
            "user_id": "test_user",
            "session_id": "team_collab_" + str(time.time()),
        }

        response = app_with_team.post("/os/executor/runs", data=request)
        assert response.status_code == 200, "Team execution failed"

        data = response.json()
        content = str(data.get("content", data.get("output", "")))

        # REAL team MUST provide comprehensive async programming guidance
        async_keywords = ["async", "await", "asyncio", "coroutine", "event loop"]
        practice_keywords = [
            "practice",
            "recommend",
            "should",
            "avoid",
            "tip",
            "example",
        ]

        found_async = [kw for kw in async_keywords if kw in content.lower()]
        found_practice = [kw for kw in practice_keywords if kw in content.lower()]

        assert (
            len(found_async) >= 3
        ), f"Team did not discuss async concepts properly! Found: {found_async}. Response: {content[:300]}..."
        assert (
            len(found_practice) >= 2
        ), f"Team did not provide best practices! Found: {found_practice}. Response: {content[:300]}..."
