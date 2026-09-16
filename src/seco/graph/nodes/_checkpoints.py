from typing import override

from langchain_core.messages import AIMessage
from langgraph.types import Command

from ...config.graph import (
    AGGREGATOR_CHECKPOINT,
    AGGREGATOR_AGENT,
    END
)
from ..states import SystemState
from ._base import NodeBase
from ...utils import log_node


class AggregatorCheckpoint(NodeBase):
    """
    A checkpoint node that determines if all specialist agents have completed their responses.

    This node inspects the system state to verify if each scheduled specialist agent
    has provided a non-empty response. If all have finished, it directs the graph to
    proceed to the aggregator agent; otherwise, it directs to the end node.

    Inherits from:
        NodeBase: Base graph node class.
    """

    def __init__(self):
        """
        Initializes the checkpoint node with a predefined name constant.
        """
        NodeBase.__init__(self, AGGREGATOR_CHECKPOINT)

    def _all_specialist_agents_finished(self, state: SystemState) -> bool:
        """
        Checks if all specialist agents scheduled in the intent have provided responses.

        Args:
            state (SystemState): The current system state containing agent intents and responses.

        Returns:
            bool: True if all specialist agents have non-empty responses; False otherwise.
        """
        scheduled_agents = [call["name"] for call in state.intent]
        for agent in scheduled_agents:
            specialist_response = state.specialist_responses.get(agent, AIMessage(content=""))
            if not specialist_response.content:
                return False
        return True

    @override
    def node(self, state) -> Command:
        """
        The main node logic to determine the next graph node.

        Args:
            state (SystemState): The current system state.

        Returns:
            Command: Directs the graph to either the aggregator agent node if all specialists
                     finished, or the end node otherwise.
        """
        log_node(self, state)
        if not self._all_specialist_agents_finished(state):
            return Command(goto=END)
        else:
            return Command(goto=AGGREGATOR_AGENT)