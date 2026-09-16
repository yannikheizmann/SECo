from typing import override

from langgraph.prebuilt import ToolNode
from langgraph.types import Command

from ..states import InteractionState
from ._base import NodeBase
from .tools import ToolRegistry
from ...utils import log_node, Responses
from ...config.graph import TOOLS


class Tools(NodeBase):
    """
    Node responsible for managing and invoking the collection of tools registered in the system.

    This node wraps all registered tools into a LangGraph ToolNode, executes the node logic
    by invoking the tools based on the current interaction state, and updates the state with
    the results of the tool invocations.

    Inherits from:
        NodeBase: Base class for graph nodes.
    """

    def __init__(self):
        """
        Initializes the Tools node with the configured name from the graph constants.
        """
        NodeBase.__init__(self, TOOLS)

    def _get_node(self) -> ToolNode:
        """
        Retrieves the ToolNode composed of all registered tools.

        Returns:
            ToolNode: A LangGraph node encapsulating all available tools.
        """
        tools = ToolRegistry.get_all_as_tools()
        return ToolNode(tools=tools)

    @override
    def node(self, state: InteractionState) -> Command:
        """
        Executes the tools node by invoking the registered tools with the current state.

        Args:
            state (InteractionState): The current interaction state, including specialist responses.

        Returns:
            Command: Updates the state with new context derived from tool calls and
                     redirects to the last node that was invoked.
        """
        log_node(self, state)
        node = self._get_node()
        agent = state.counter.last_node
        specialist_response = state.specialist_responses.get(agent)
        result = node.invoke({"messages": [specialist_response]})
        responses = Responses.from_calls(specialist_response.tool_calls, result)
        target = state.counter.last_node
        return Command(
            update={
                "counter": self._name,
                "context": responses
            },
            goto=target)