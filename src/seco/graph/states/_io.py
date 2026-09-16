from pydantic import BaseModel, Field

from langchain_core.messages import AIMessage
from typing import Annotated

from ._interaction import ResolutionState


class InputState(BaseModel):
    """
    Represents the input payload passed into the graph during invocation.

    Attributes:
        query (str): The user query to process.
    """
    query: str = Field(default="")


class OutputState(BaseModel):
    """
    Represents the output payload returned by the graph after execution.

    Attributes:
        agents (list[str]): Names of specialist agents involved in the response.
        response (AIMessage): The final AI message returned.
        history (list[AIMessage]): The conversation history.
        specialist_responses (dict[str, AIMessage]): Responses from individual specialist agents, using ResolutionState logic.
    """
    agents: list[str] = Field(default=[])
    response: AIMessage = Field(default=AIMessage(content=""))
    history: list[AIMessage] = Field(default=[])
    specialist_responses: Annotated[dict[str, AIMessage], ResolutionState._add_response] = Field(default={})