from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Annotated, Union

from langchain_core.messages import AIMessage, ToolCall, SystemMessage, AnyMessage, ToolMessage

from ...utils import Counter, Responses

# https://langchain-ai.github.io/langgraph/how-tos/state-model/#working-with-message-models
# https://langchain-ai.github.io/langgraph/how-tos/input_output_schema/

# state persistentcy etc
#https://langchain-ai.github.io/langgraph/concepts/persistence/?utm_source=chatgpt.com#get-state

class GenerationState(BaseModel):
    """State tracking for message generation and tool interactions.

    Attributes:
        specialist_query (str): The current specialist query text.
        messages (list[ToolMessage]): Tool-generated messages in the conversation.
        context (Responses): Accumulated previous tool responses.
        counter (Counter): Stateful count of node interactions.

    Internal:
        _append_context: Merges new context into the existing one.
        _increment_counter: Updates interaction counter based on last node.
    """
    specialist_query: str = Field(default="")
    messages: list[ToolMessage] = Field(default=[])
    context: Annotated[Responses, GenerationState._append_context] = Field(default=Responses())
    counter: Annotated[Counter, GenerationState._increment_counter] = Field(default=Counter())

    @classmethod
    def _append_context(cls, context: Responses, new_context: Responses) -> Responses:
        """Append new tool responses to context, resetting if empty.

        Args:
            context (Responses): Existing stored responses.
            new_context (Responses): Newly incoming responses.

        Returns:
            Responses: Combined or reset context.
        """
        if new_context == Responses():
            return Responses()
        return Responses(
            tool_responses=context.tool_responses + new_context.tool_responses
        )

    @classmethod
    def _increment_counter(cls, counter: Counter, new_counter_or_node: Union[str, Counter]) -> Counter:
        """Update or reset counter based on the node interaction.

        Args:
            counter (Counter): Current interaction counter.
            new_counter_or_node (str | Counter): New node name or counter object.

        Returns:
            Counter: Updated interaction counter.
        """
        if new_counter_or_node == Counter():
            return Counter()
        else:
            node = new_counter_or_node if isinstance(new_counter_or_node, str) else new_counter_or_node.last_node
            count = counter.count + 1 if counter.prev_node == node else 1
            return Counter(
                prev_node=counter.last_node,
                last_node=node,
                count=count)

    def with_context(self, messages: list[AnyMessage]) -> list[AnyMessage]:
        """Attach tool-context to a message sequence as a SystemMessage.

        Args:
            messages (list[AnyMessage]): Original message sequence.

        Returns:
            list[AnyMessage]: Extended sequence including context summary.
        """
        content = "\n".join([f"Context {i + 1}: {tool_response.as_string()}" for i, tool_response in
                             enumerate(self.context.tool_responses)])
        if len(content) == 0:
            return messages
        return messages + [SystemMessage(content=content)]

    def as_string(self) -> str:
        """Generate a formatted multiline string of the current generation state.

        Returns:
            str: Human-readable representation of current state.
        """
        context_str = "\n".join([
            f"""                                                {i+1}. {rsp.as_string(length=100)}"""
            for i, rsp in enumerate(self.context.tool_responses)
        ])
        return f"""
        Generation State:
            Specialist Query:                   {self.specialist_query}
            Context:                            {f"\n{context_str}" if context_str != "" else ""}
            Counter:                            {self.counter}"""


class ResolutionState(BaseModel):
    """State tracking for query resolution and AI intent planning.

    Attributes:
        query (str): The input query being resolved.
        intent (list[ToolCall]): Detected intent tool calls.
        agents (list[str]): Names of specialist agents involved.
        specialist_responses (dict[str, AIMessage]): Agent-specific AI responses.
        response (AIMessage): Final AIMessage result.
    """
    query: str = Field(default="")
    intent: list[ToolCall] = Field(default=[])
    agents: Annotated[list[str], ResolutionState._add_agent] = Field(default=[])
    specialist_responses: Annotated[dict[str, AIMessage], ResolutionState._add_response] = Field(default={})
    response: AIMessage = Field(default=AIMessage(content=""))

    @classmethod
    def _add_agent(cls, agents: list[str], new_agents: Union[str, list[str]]) -> list[str]:
        """Add agent names without duplicates.

        Args:
            agents (list[str]): Existing list of agents.
            new_agents (str | list[str]): Agent(s) to add.

        Returns:
            list[str]: Updated list of agents.
        """
        if new_agents == []:
            return []
        else:
            if new_agents not in agents:
                agents.append(new_agents)
        return agents

    @classmethod
    def _add_response(cls, responses: dict[str, AIMessage], new_responses: Union[tuple[str, AIMessage], dict[str, AIMessage]]) -> dict[str, AIMessage]:
        """Add or reset specialist agent responses.

        Args:
            responses (dict[str, AIMessage]): Existing responses.
            new_responses (tuple[str, AIMessage] | dict[str, AIMessage]): New single or multiple responses.

        Returns:
            dict[str, AIMessage]: Updated specialist_responses dict.
        """
        if new_responses == {}:
            return {}
        else:
            responses[new_responses[0]] = new_responses[1]
        return responses

    def get_responses(self) -> AIMessage:
        """Combine all specialist agent responses into a single AIMessage.

        Returns:
            AIMessage: Aggregated responses or empty message if none exist.
        """
        if not self.specialist_responses:
            return AIMessage(content="")
        return AIMessage(
            content="\n".join([
                f"{agent}: {rsp.content}" for agent, rsp in self.specialist_responses.items()
            ])
        )

    def as_string(self) -> str:
        """Generate a formatted multiline string of the current resolution state.

        Returns:
            str: Human-readable representation including intent, agents, and responses.
        """
        specialist_responses_str = "\n".join([
            f"                                                {i + 1}. {agent}: {rsp.content}"
            for i, (agent, rsp) in enumerate(self.specialist_responses.items())
        ])
        return f"""
        ResolutionState:
            Query:                              {self.query}
            Intent:                             {", ".join(call["name"] for call in self.intent) if self.intent else ""}
            Agents:                             {", ".join(self.agents) if self.agents else ""}
            Specialist Responses:               {f"\n{specialist_responses_str}" if specialist_responses_str != "" else ""}
            Response:                           {self.response}"""


class InteractionState(GenerationState, ResolutionState):
    """Combined state model for both interaction generation and resolution phases."""

    @classmethod
    def reset(cls) -> dict:
        """Reset all model fields to their default values.

        Returns:
            dict: Default field values for the entire interaction state.
        """
        defaults = {}
        for field_name, field_info in cls.model_fields.items():
            default_value = field_info.get_default(call_default_factory=True)
            defaults[field_name] = default_value
        return defaults

    def as_string(self) -> str:
        """Generate a combined formatted string of resolution and generation states.

        Returns:
            str: Human-readable printout of full interaction state.
        """
        return f"""
    InteractionState: {ResolutionState.as_string(self)}{GenerationState.as_string(self)}"""