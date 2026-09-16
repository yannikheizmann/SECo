from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Union, Optional

from ..states import StateBase
from ...utils import RegistryMeta


class EdgeRegistry:
    """
    Registry for managing edges within the graph system.

    Provides methods to register, retrieve, and list all defined edges.
    """
    _registry: dict[str, EdgeBase] = {}

    @classmethod
    def register(cls, name: str, edge: EdgeBase) -> None:
        """
        Registers an edge by name in the global edge registry.

        Args:
            name (str): Unique name to associate with the edge.
            edge (EdgeBase): The edge instance to register.
        """
        cls._registry[name] = edge

    @classmethod
    def get(cls, name: str) -> Optional[EdgeBase]:
        """
        Retrieves an edge from the registry by name.

        Args:
            name (str): The name of the edge to look up.

        Returns:
            Optional[EdgeBase]: The found edge, or None if not registered.
        """
        return cls._registry.get(name)
    
    @classmethod
    def get_all(cls) -> list[EdgeBase]:
        """
        Returns a list of all registered edge instances.

        Returns:
            list[EdgeBase]: All currently registered edges.
        """
        return list(cls._registry.values())


class EdgeBase(ABC, metaclass=RegistryMeta):
    """
    Abstract base class for defining edges within the graph.

    Each edge must implement the `edge` method that processes an input state
    and returns one or more next node names to traverse.

    Attributes:
        _name (str): The identifier name for the edge, used in registry lookups.
    """
    __registry__ = EdgeRegistry

    def __init__(self, name: str) -> None:
        """
        Initializes an edge with a given name.

        Args:
            name (str): The name to assign to this edge.
        """
        self._name = name

    @abstractmethod
    def edge(self, state: StateBase) -> Union[str, list[str]]:
        """
        Abstract method to be implemented by all edges.

        Defines the logic to determine the next node(s) given the current graph state.

        Args:
            state (StateBase): The current graph state passed into the edge.

        Returns:
            Union[str, list[str]]: The next node name(s) to transition to.
        """
        pass