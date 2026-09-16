from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional

from langchain_core.messages import ToolMessage, ToolCall


class Counter(BaseModel):
    """
    A class to keep track of the previous and last nodes visited, along with a count.

    Attributes:
        prev_node (str): The identifier of the previous node. Defaults to an empty string.
        last_node (str): The identifier of the last node. Defaults to an empty string.
        count (int): A counter to track occurrences or visits. Defaults to 0.

    Usage:
        Used to maintain state information about nodes during processing or traversal.
    """
    prev_node: str = Field(default="")
    last_node: str = Field(default="")
    count: int = Field(default=0)


class Responses(BaseModel):
    """
    A class representing a collection of tool responses.

    Attributes:
        tool_responses (list[ToolResponse]): A list of ToolResponse instances. Defaults to an empty list.

    Usage:
        Used to aggregate multiple tool responses, typically constructed from tool calls and their results.
    """
    tool_responses: list[ToolResponse] = Field(default=[])

    @classmethod
    def from_calls(cls, tool_calls: list[ToolCall], result: dict) -> Responses:
        """
        Creates a Responses instance from a list of tool calls and a result dictionary.

        Args:
            tool_calls (list[ToolCall]): A list of ToolCall objects representing the tool calls made.
            result (dict): A dictionary containing the result data, expected to have a "messages" key with a list of ToolMessage objects.

        Returns:
            Responses: An instance of Responses containing the processed tool responses.

        Functionality:
            Iterates over the messages in the result, matches each message to its corresponding tool call by ID,
            and constructs ToolResponse objects which are collected into a Responses instance.
        """
        tool_messages: list[ToolMessage] = result.get("messages", [])
        tool_responses: list[ToolResponse] = []
        for message in tool_messages:
            for tool_call in tool_calls:
                if tool_call["id"] == message.tool_call_id:
                    call = tool_call
                    break
            response = ToolResponse(
                  name=call["name"],
                  args=call["args"],
                response=message.content
            )
            tool_responses.append(response)
        return Responses(
            tool_responses=tool_responses
        )

class ToolResponse(BaseModel):
    """
    A class representing a single response from a tool.

    Attributes:
        name (str): The name of the tool.
        args (dict): The arguments passed to the tool.
        response (str): The response content from the tool.

    Usage:
        Used to encapsulate the details of a tool's response including the input arguments and output.
    """
    name: str = Field(default="")
    args: dict = Field(default={})
    response: str = Field(default="")

    def as_string(self, length: Optional[int] = None) -> str:
        """
        Returns a string representation of the ToolResponse, optionally truncated to a specified length.

        Args:
            length (Optional[int]): The maximum length of the response string to include. If None, includes the full response.

        Returns:
            str: A formatted string representing the ToolResponse with the possibly truncated response content.
        """
        length = length if length is not None else len(self.response)
        return f"ToolResponse(name={self.name}, args={self.args}, response={self.response[:length]})"