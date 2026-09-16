from typing import Union, override

from langgraph.types import Send

from ...config.graph import (
    END,
    INTENT_ROUTER
)
from ._base import EdgeBase
from ..states import SystemState
from ...utils import log_edge

# https://langchain-ai.github.io/langgraph/how-tos/branching/


class IntentRouter(EdgeBase):
    """
    Edge that routes execution based on the detected user intent.

    Evaluates the current system state to determine if there are any detected intent tool calls.
    If so, creates a list of `Send` commands, each directing the graph to invoke a specialist
    agent with an updated state containing the specialist query.

    If no intent is found, routes directly to the end node.

    Inherits from:
        EdgeBase: Base class for graph edges.
    """

    def __init__(self):
        """
        Initializes the IntentRouter edge with a predefined name constant.
        """
        super().__init__(
            name=INTENT_ROUTER,
        )

    @override
    def edge(self, state: SystemState) -> Union[str, list[Send]]:
        """
        Determines the next nodes to route to based on the intent present in the state.

        Args:
            state (SystemState): The current system state with intent information.

        Returns:
            Union[str, list[Send]]: List of Send commands targeting specialist agents if intent exists,
                                   otherwise the end node name string.
        """
        log_edge(self)
        intent = state.intent
        if len(intent) > 0:
            specialists = [
                Send(node=agent["name"], arg=state.model_copy(update={"specialist_query": agent["args"]["query"]}, deep=True))
                for agent in intent
            ]
            return specialists
        else:
            return END