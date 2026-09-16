from typing import Union

from ._conversation import ConversationState
from ._interaction import InteractionState, ResolutionState, GenerationState
from ._io import InputState, OutputState
from ._system import SystemState


StateBase = Union[
    InputState,
    SystemState,
        InteractionState,
            ResolutionState,
            GenerationState,
        ConversationState,
    OutputState, 
]