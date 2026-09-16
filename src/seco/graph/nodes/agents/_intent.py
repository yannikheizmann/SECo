from __future__ import annotations
from typing import override

from langchain_core.messages import AIMessage

from ....config.graph import INTENT_AGENT
from ....config.agents import INTENT_AGENT_MODEL, INTENT_AGENT_PROMPT
from ...states import SystemState, ResolutionState
from ._base import AgentBase
from ....utils import log_node, time_execution


class IntentAgent(AgentBase):
    """
    Agent responsible for detecting user intent from the input query.

    Utilizes a language model with the ability to call any tool to interpret the query
    and extract relevant tool calls, which guide further routing in the graph.

    Inherits from:
        AgentBase: Provides base LLM interaction and optional tool handling.
    """
    def __init__(self):
        """
        Initializes the IntentAgent with configuration for name, model, and prompt.

        The agent is configured to allow calling any available tool to assist with intent classification.
        """
        super().__init__(
            name=INTENT_AGENT,
            model=INTENT_AGENT_MODEL,
            prompt=INTENT_AGENT_PROMPT,
            tool_choice="any")

    @override
    @time_execution("et_intent", start_key="et_overall_start")
    def node(self, state: SystemState) -> ResolutionState:
        """
        Executes the intent detection logic using the language model.

        Args:
            state (SystemState): The input system state containing the user query.

        Returns:
            ResolutionState: A dictionary containing extracted tool calls under the "intent" key.
        """
        log_node(self, state)
        response: AIMessage = self._model.invoke(state.with_history(state.query))
        return {
            "intent": response.tool_calls,
        }