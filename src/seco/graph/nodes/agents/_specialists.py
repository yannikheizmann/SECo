from typing import override

from ....config.graph import (
    GENERAL_AGENT,
    SWEATY_AGENT,
    HSO_AGENT,
)
from ....config.agents import (
    GENERAL_AGENT_MODEL,
    SWEATY_AGENT_MODEL,
    HSO_AGENT_MODEL,
    GENERAL_AGENT_PROMPT,
    SWEATY_AGENT_PROMPT,
    HSO_AGENT_PROMPT,
)
from ._base import SpecialistAgentBase
from ....utils import time_execution


class GeneralAgent(SpecialistAgentBase):
    """
    Specialist agent responsible for answering general-purpose questions.

    Inherits from:
        SpecialistAgentBase: Provides dual-role tool and agent behavior.
    """
    def __init__(self):
        """
        Initializes the GeneralAgent with its name, model, prompt, and description.
        """
        super().__init__(
            name=GENERAL_AGENT,
            model=GENERAL_AGENT_MODEL,
            prompt=GENERAL_AGENT_PROMPT,
            description="Beantwortet generelle Fragen.")
        
    @override
    @time_execution("et_general_avg", start_key="et_general_start", end_key="et_general_end")
    def node(self, state):
        """
        Executes the general agent node and tracks execution timing.

        Args:
            state (SystemState): The current graph state.

        Returns:
            GenerationState: The updated generation state from the base class.
        """
        return super().node(state)
    

class SweatyAgent(SpecialistAgentBase):
    """
    Specialist agent focused on answering questions related to the humanoid robot Sweaty
    and related topics at Hochschule Offenburg.
    """
    def __init__(self):
        """
        Initializes the SweatyAgent with its configuration and descriptive prompt.
        """
        super().__init__(
            name=SWEATY_AGENT,
            model=SWEATY_AGENT_MODEL,
            prompt=SWEATY_AGENT_PROMPT,
            description="Beantwortet Fragen zu dem humanoiden Roboter Sweaty der Hochschule Offenburg und verwandten Themen.")
        
    @override
    @time_execution("et_sweaty_avg", start_key="et_sweaty_start", end_key="et_sweaty_end")
    def node(self, state):
        """
        Executes the Sweaty agent node and tracks execution timing.

        Args:
            state (SystemState): The current graph state.

        Returns:
            GenerationState: The updated generation state from the base class.
        """
        return super().node(state)


class HSOAgent(SpecialistAgentBase):
    """
    Specialist agent dedicated to answering questions about Hochschule Offenburg (HSO)
    and associated subjects.
    """
    def __init__(self):
        """
        Initializes the HSOAgent with its model settings and purpose description.
        """
        super().__init__(
            name=HSO_AGENT,
            model=HSO_AGENT_MODEL,
            prompt=HSO_AGENT_PROMPT,
            description="Beantwortet Fragen zu HSO (Hochschule Offenburg) und verwandten Themen.")
        
    @override
    @time_execution("et_hso_avg", start_key="et_hso_start", end_key="et_hso_end")
    def node(self, state):
        """
        Executes the HSO agent node and tracks execution timing.

        Args:
            state (SystemState): The current graph state.

        Returns:
            GenerationState: The updated generation state from the base class.
        """
        return super().node(state)