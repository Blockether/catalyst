"""
Arity-Specified Typed Calls Protocol - Generic interface for typed calls with specific arity.

This defines the protocol that any implementation must follow for typed calls,
allowing your code to be implementation-agnostic. Currently supports arity-one calls
with plans for future expansion to other arities.
"""

import inspect
from abc import abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Protocol,
    TypeVar,
    Union,
    cast,
    runtime_checkable,
)

if TYPE_CHECKING:
    from agno.agent import Agent
    from agno.models.base import Model
    from agno.team import Team
    from agno.workflow import Workflow

from dataclasses import replace as copy_with_merge

from pydantic import BaseModel, RootModel

T = TypeVar("T", bound=BaseModel, covariant=True)
X = TypeVar("X", bound=Union[str, BaseModel, RootModel], contravariant=True)


@runtime_checkable
class ArityOneTypedCall(Protocol, Generic[X, T]):
    """
    Protocol for arity-one typed calls that return structured Pydantic models.

    This is the interface that any implementation must follow.
    Users can implement this protocol for LLMs (BAML, OpenAI, Anthropic, etc.)
    or any other service that takes input and returns typed output.

    The input type X can be:
    - A string (for simple prompts)
    - A BaseModel (for structured requests)
    - A RootModel (for wrapped primitive types)
    """

    @abstractmethod
    async def call(
        self,
        x: X,
    ) -> T:
        """
        Make a typed call and return a structured response.

        Args:
            x: The input which can be a string, BaseModel, or RootModel.

        Returns:
            Structured response as type T (Pydantic model)
        """
        ...


class AgnoRunnerToArityTypedCallAdapter(Generic[X, T]):
    """
    Adapter class that converts Agno Agents/Teams into ArityOneTypedCall implementations.

    This adapter wraps an Agno Agent or Team and provides the ArityOneTypedCall interface,
    allowing runners to be used wherever typed calls are expected.
    """

    @staticmethod
    def create_typed_call(runner: "Agent | Team | Workflow", **runner_kwargs) -> ArityOneTypedCall[str, T]:
        """
        Create an ArityOneTypedCall from an Agno Agent or Team.

        Args:
            runner: The Agno Agent, Team, Workflow instance to wrap
            **runner_kwargs: Additional keyword arguments to pass to runner when cloning

        Returns:
            An ArityOneTypedCall instance that wraps the runner
        """
        from agno.agent import Agent
        from agno.team import Team
        from agno.workflow import Workflow

        if not isinstance(runner, (Agent, Team, Workflow)):
            raise ValueError(
                "runner must be an instance of agno.agent.Agent, agno.team.Team, or agno.workflow.Workflow"
            )

        is_agent = isinstance(runner, Agent)
        is_team = isinstance(runner, Team)
        is_workflow = isinstance(runner, Workflow)

        from blockether_catalyst.consensus.VotingComparison import (
            BaseModelWithReasoning,
        )

        clazz = Agent if is_agent else Team if is_team else Workflow if is_workflow else None

        if clazz is None:
            raise ValueError("Unsupported runner type. Shouldn't happen due to earlier check.")

        sig = inspect.signature(clazz.__init__)
        valid_params = set(sig.parameters.keys()) - {"self"}

        # Start with the runner's existing configuration
        # We allow here for the hasattr/getattr as a fallback in case
        # Agno changes their Agent/Team class
        clone_kwargs = {
            param: getattr(runner, param)
            for param in valid_params
            if hasattr(runner, param) and getattr(runner, param) is not None
        }

        # Override with any provided runner_kwargs
        # Filter to only valid parameters for the class
        for key, value in runner_kwargs.items():
            if key in valid_params:
                clone_kwargs[key] = value

        # Valid check via
        if (is_agent or is_team) and getattr(runner, "output_schema", None) is None:
            raise ValueError("Runner must have an output_schema defined")

        runner = clazz(**clone_kwargs)

        class RunnerTypedCall(ArityOneTypedCall):
            """Inner class that implements the ArityOneTypedCall protocol for the runner."""

            async def call(self, x: X) -> Any:
                run_output = await runner.arun(str(x))

                # Extract the content from the RunOutput
                content = run_output.content

                if not content:
                    raise ValueError("Runner returned empty content")

                # Check if the content is BaseModelWithReasoning
                if not isinstance(content, BaseModelWithReasoning):
                    raise ValueError(
                        f"Runner returned content type {type(content).__name__}, expected BaseModelWithReasoning"
                    )

                # Verify it has the reasoning attribute
                if not isinstance(content, BaseModel) or not hasattr(content, "reasoning"):
                    raise ValueError(
                        f"Runner returned content type {type(content).__name__}, "
                        f"expected BaseModel with 'reasoning' attribute"
                    )

                return content

        return cast(ArityOneTypedCall[str, T], RunnerTypedCall())
