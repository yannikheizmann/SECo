from abc import ABC
from langchain_core.messages import AIMessage
from typing import override, Union

from ...states import SystemState, GenerationState
from ..tools import ToolBase
from .. import NodeBase
from .models import ProxyChatModel
from ....utils import log_node


class AgentBase(NodeBase, ABC):
    """
    Abstract base class for agent nodes within the application.

    Combines a LangChain-compatible model with optional tool bindings.
    Acts as a node in the graph, capable of being extended for specialized agent behavior.

    Attributes:
        _model (ProxyChatModel): The wrapped language model and its configuration.
    """
    def __init__(self, name: str, model: str, prompt: str, tool_choice: Union[dict, str, bool, None]):
        """
        Initializes a base agent node with a language model and tool configuration.

        Args:
            name (str): Name of the agent node.
            model (str): Model identifier (e.g., OpenAI or custom backend).
            prompt (str): Prompt template for guiding the model.
            tool_choice (Union[dict, str, bool, None]): Tool usage policy for the model.
        """
        NodeBase.__init__(self, name)
        self._model = ProxyChatModel(model, prompt, tool_choice)

    def add_tool(self, tool: ToolBase) -> None:
        """
        Binds a single tool to the agent's model for use during execution.

        Args:
            tool (ToolBase): A tool implementing the ToolBase interface.
        """
        self._model.bind_tools([tool.as_tool()])


class SpecialistAgentBase(AgentBase, ToolBase, ABC):
    """
    Abstract base class for specialist agents that also act as tools.

    Extends both AgentBase and ToolBase to allow dual-role usage in the graph.
    Suitable for specialist nodes that produce tool-augmented output and respond to invocations.
    """
    def __init__(self, name: str, model: str, prompt: str, description: str):
        """
        Initializes a specialist agent that can act as both a tool and an agent.

        Args:
            name (str): Unique name of the agent/tool.
            model (str): Model identifier.
            prompt (str): Prompt string to guide agent behavior.
            description (str): Tool description for discovery and invocation.
        """
        AgentBase.__init__(self, name, model, prompt, tool_choice=None)
        ToolBase.__init__(self, name, description)

    @override
    def node(self, state: SystemState) -> GenerationState:
        """
        Executes the specialist agent node using the current system state.

        Uses conversation history and context to invoke the model.
        May disable tools if repeated invocations exceed a threshold.

        Args:
            state (SystemState): The input application state.

        Returns:
            GenerationState: The updated generation state with response, counter, and agent tracking.
        """
        log_node(self, state)
        no_tools = state.counter.count >= 5
        response: AIMessage = self._model.invoke(
            input=state.with_context(state.with_history(state.specialist_query)),
            no_tools=no_tools)
        return {
            "counter": self._name,
            "specialist_responses": (self._name, response),
            "agents": self._name
        }

    @override
    def invoke(self, query: str) -> None:
        """
        Placeholder for direct query invocation. Not implemented for specialist agents.

        Args:
            query (str): The input query string.
        """
        pass