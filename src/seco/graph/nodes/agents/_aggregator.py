from __future__ import annotations
from typing import override

from langchain_core.messages import AIMessage, HumanMessage

from ....config.graph import AGGREGATOR_AGENT
from ....config.agents import (
    AGGREGATOR_AGENT_MODEL, 
    AGGREGATOR_AGENT_PROMPT
)
from ...states import (
    SystemState, 
    InteractionState
)
from ._base import AgentBase
from ....utils import log_node, time_execution


class AggregatorAgent(AgentBase):
    """
    Aggregator agent responsible for combining specialist responses into a final output.

    Uses a language model to process responses from specialists and produce a final message
    summarizing or deciding based on the inputs. Executed as the final step in the graph.

    Inherits from:
        AgentBase: Provides core model and tool-handling functionality.
    """
    def __init__(self):
        """
        Initializes the AggregatorAgent with configuration from settings.

        Sets name, model, and prompt using AGGREGATOR_AGENT constants.
        """
        super().__init__(
            name=AGGREGATOR_AGENT,
            model=AGGREGATOR_AGENT_MODEL,
            prompt=AGGREGATOR_AGENT_PROMPT,
            tool_choice=None)

    @override
    @time_execution("et_aggregator", end_key="et_overall_end")
    def node(self, state: SystemState) -> InteractionState:
        """
        Executes the aggregator agent to produce a final response.

        Gathers and processes specialist responses using the model, logs execution,
        and returns an InteractionState with the updated response and history.

        Args:
            state (SystemState): The system state containing query and specialist responses.

        Returns:
            InteractionState: Contains the model's response and conversation history.
        """
        log_node(self, state)
        response: AIMessage = self._model.invoke([state.get_responses()])
        return {
            "response": response,
            "history": [HumanMessage(content=state.query), response]
        }