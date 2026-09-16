from typing import Any

from ._conversation import ConversationState
from ._interaction import InteractionState


# https://langchain-ai.github.io/langgraph/how-tos/state-model/#working-with-message-models
# https://langchain-ai.github.io/langgraph/how-tos/input_output_schema/

# state persistentcy etc
#https://langchain-ai.github.io/langgraph/concepts/persistence/?utm_source=chatgpt.com#get-state


class SystemState(InteractionState, ConversationState):
    """System-level state combining interaction and conversation contexts.

    Inherits from both InteractionState and ConversationState, providing
    a unified state model for system operations that involve both
    interaction tracking and conversational history.
    """

    def reset(self) -> dict[str, Any]:
        """Reset internal interaction state to its initial model dump.

        Overrides InteractionState.reset to provide a fresh interaction state.

        Returns:
            dict[str, Any]: A snapshot of the new default interaction state.
        """
        return InteractionState().model_dump()

    def as_string(self) -> str:
        """Generate a string representation combining interaction and conversation state.

        Returns:
            str: Formatted human-readable string representing the combined state.
        """
        return f"""
SystemState: {InteractionState.as_string(self)}{ConversationState.as_string(self)}"""