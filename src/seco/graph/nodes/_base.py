from __future__ import annotations
from typing import Optional
from abc import ABC, abstractmethod

from ..states import StateBase
from ...utils import RegistryMeta


class NodeRegistry:
    """
    Registry for managing node instances (e.g., agents) in the graph system.

    Provides methods to register, retrieve, and list all defined nodes.
    """
    _registry: dict[str, NodeBase] = {}

    @classmethod
    def register(cls, name: str, node: NodeBase) -> None:
        """
        Registers a node by name in the global node registry.

        Args:
            name (str): Unique name to associate with the node.
            node (NodeBase): The node instance to register.
        """
        cls._registry[name] = node

    @classmethod
    def get(cls, name: str) -> Optional[NodeBase]:
        """
        Retrieves a node from the registry by name.

        Args:
            name (str): The name of the node to look up.

        Returns:
            Optional[NodeBase]: The found node, or None if not registered.
        """
        return cls._registry.get(name)
    
    @classmethod
    def get_all(cls) -> list[NodeBase]:
        """
        Returns a list of all registered node instances.

        Returns:
            list[NodeBase]: All currently registered nodes.
        """
        return list(cls._registry.values())


class NodeBase(ABC, metaclass=RegistryMeta):
    """
    Abstract base class for defining nodes within the graph, such as agents.

    Each node must implement the `node` method that processes an input state
    and returns an updated state.

    Attributes:
        _name (str): The identifier name for the node, used in registry lookups.
    """
    __registry__ = NodeRegistry

    def __init__(self, name: str) -> None:
        """
        Initializes a node with a given name.

        Args:
            name (str): The name to assign to this node.
        """
        self._name = name
        
    @abstractmethod
    def node(self, state: StateBase) -> StateBase:
        """
        Abstract method to be implemented by all nodes.

        Defines the logic to be applied when this node is invoked in the graph.

        Args:
            state (StateBase): The current graph state passed into the node.

        Returns:
            StateBase: The updated graph state returned by the node.
        """
        pass
    def get_name(self) -> str:
        """
        Retrieve the tool's name.

        Returns:
            str: The unique identifier for this tool.
        """
        return self._name