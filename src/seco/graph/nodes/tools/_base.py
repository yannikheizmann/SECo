from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable, Optional
import inspect
from functools import wraps

from langchain_core.tools import tool, BaseTool
from langgraph.types import Command

from ....utils import RegistryMeta

# https://langchain-ai.github.io/langgraph/reference/prebuilt/#langgraph.prebuilt.tool_node.ToolNode
# https://langchain-ai.github.io/langgraph/how-tos/tool-calling/#define-tools
# https://langchain-ai.github.io/langgraph/how-tos/tool-calling/


class ToolRegistry:
    """Registry for managing available tools in the graph system."""

    _registry: dict[str, ToolBase] = {}

    @classmethod
    def register(cls, name: str, tool: ToolBase) -> None:
        """
        Add a tool to the registry.

        Args:
            name (str): Unique identifier for the tool.
            tool (ToolBase): The tool instance to register.
        """
        cls._registry[name] = tool

    @classmethod
    def get(cls, name: str) -> Optional[ToolBase]:
        """
        Retrieve a registered tool by name.

        Args:
            name (str): The tool's identifier.

        Returns:
            Optional[ToolBase]: The tool instance, or None if not found.
        """
        return cls._registry.get(name)

    @classmethod
    def get_all(cls) -> list[ToolBase]:
        """
        List all registered tool instances.

        Returns:
            list[ToolBase]: All tools in the registry.
        """
        return list(cls._registry.values())

    @classmethod
    def get_all_as_tools(cls) -> list[BaseTool]:
        """
        Return all registered tools converted to LangChain BaseTool wrappers.

        Returns:
            list[BaseTool]: Tools wrapped for LangChain invocation.
        """
        return [tool.as_tool() for tool in cls._registry.values()]


class ToolBase(ABC, metaclass=RegistryMeta):
    """Abstract base class for defining custom tools within the system.

    Acts as a registry-backed interface layer, enabling conversion of custom logic
    into a form usable by LangChain agents.

    Attributes:
        _name (str): Identifier used for both registry and tooling APIs.
        _description (str): Human-readable description of the tool.
    """

    __registry__ = ToolRegistry

    def __init__(self, name: str, description: str) -> None:
        """
        Initialize a new ToolBase instance.

        Args:
            name (str): Unique name of the tool.
            description (str): Short description of what the tool does.
        """
        self._name = name
        self._description = description

    @staticmethod
    def _update_signature(target_func: Callable, source_func: Callable) -> Callable:
        """
        Copy the signature from source_func to target_func for accurate schema inference.

        Args:
            target_func (Callable): The function to have its signature updated.
            source_func (Callable): The source providing desired signature.

        Returns:
            Callable: Wrapped function retaining target behavior and updated signature.
        """
        source_signature = inspect.signature(source_func)

        @wraps(target_func)
        def wrapper(*args, **kwargs):
            return target_func(*args, **kwargs)

        wrapper.__signature__ = source_signature
        return wrapper

    def _tool_method(self) -> Callable:
        """
        Wrap the invoke method to be compatible with the LangChain tool decorator.

        Returns:
            Callable: A function ready for wrapping with @tool decorator.
        """
        @wraps(self.invoke)
        def convert_to_func(*args, **kwargs):
            return self.invoke(*args, **kwargs)

        return self._update_signature(convert_to_func, self.invoke)

    def as_tool(self) -> BaseTool:
        """
        Convert this custom tool into a LangChain BaseTool via decorator metadata.

        Returns:
            BaseTool: LangChain-compatible tool instance.
        """
        return tool(name_or_callable=self._name, description=self._description)(self._tool_method())

    @abstractmethod
    def invoke(self, *args, **kwargs) -> Command:
        """
        Execute the tool's core logic.

        Args:
            *args: Positional arguments for tool execution.
            **kwargs: Keyword arguments for tool execution.

        Returns:
            Command: The output command structure expected by LangGraph.
        """
        pass