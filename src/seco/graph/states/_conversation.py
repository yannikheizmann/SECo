from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Annotated, Union

from langchain_core.messages import AnyMessage, HumanMessage

from ...config.application import MAX_HISTORY_LENGTH


class ConversationState(BaseModel):
    """
    Manages the conversational history state using LangChain message models.

    Attributes:
        history (list[AnyMessage]): List of messages exchanged in the conversation. Appended via the `append` reducer.
    """
    history: Annotated[list[AnyMessage], ConversationState.append] = Field(default=[])

    @classmethod
    def append(cls, history: list[AnyMessage], new_messages: Union[AnyMessage, list[AnyMessage]]) -> list[AnyMessage]:
        """
        Appends new non-null string messages to the history. If history exceeds MAX_HISTORY_LENGTH, oldest messages are truncated.

        Args:
            history (list[AnyMessage]): Existing list of conversation messages.
            new_messages (Union[AnyMessage, list[AnyMessage]]): A message or list of messages to be appended.

        Returns:
            list[AnyMessage]: Updated conversation history.
        """
        if not isinstance(new_messages, list):
            new_messages = [new_messages]
        non_null_messages = [message for message in new_messages if isinstance(message.content, str) and message.content is not None]
        if len(history) + len(non_null_messages) > MAX_HISTORY_LENGTH:
            history = history[len(non_null_messages):]
        history.extend(non_null_messages) 
        return history
    
    def with_history(self, query: str) -> list[AnyMessage]:
        """
        Combines current history with a new user query as a HumanMessage.

        Args:
            query (str): The user's input query.

        Returns:
            list[AnyMessage]: Full list of messages including the new query.
        """
        return self.history + [HumanMessage(content=query)]
    
    def as_string(self) -> str:
        """
        Returns a formatted string representation of the conversation history.

        Returns:
            str: Human-readable conversation history.
        """
        history_str = "\n".join([
            f"                                                {i + 1}. {str.upper(msg.type)}: {msg.content}"
            for i, msg in enumerate(self.history)
        ])
        return f"""
    ConversationState: 
        History:                              {f"\n{history_str}"}"""