from typing import override, Union

from langgraph.types import Send

from ...config.graph import (
    AGGREGATOR_CHECKPOINT, 
    TOOLS,
    TOOL_ROUTER
)
from ._base import EdgeBase
from ..states import InteractionState
from ...utils import log_edge


class ToolRouter(EdgeBase):
    """
    Edge that routes execution based on whether the last specialist agent made any tool calls.

    Checks the most recent specialist response in the interaction state. If there are one or more tool calls,
    it routes the graph execution to the tools node to process these calls. Otherwise, it proceeds to
    the aggregator checkpoint node.

    Inherits from:
        EdgeBase: Base class for graph edges.
    """

    def __init__(self):
        """
        Initializes the ToolRouter edge with its configured name.
        """
        super().__init__(
            name=TOOL_ROUTER,
        )

    @override
    def edge(self, state: InteractionState) -> Union[str, Send]:
        """
        Determines routing based on tool calls in the last specialist response.

        Args:
            state (InteractionState): The current interaction state.

        Returns:
            Union[str, Send]: Sends to the tools node if tool calls exist; otherwise, routes to aggregator checkpoint.
        """
        log_edge(self)
        agent = state.counter.last_node
        specialist_response = state.specialist_responses.get(agent)
        if len(specialist_response.tool_calls) >= 1:
            return Send(node=TOOLS, arg=state.model_copy(deep=True))
        else:
            return AGGREGATOR_CHECKPOINT