from typing import Any, Type, Union

from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from ..config.graph import (
    # nodes
    INTENT_AGENT,
    HSO_AGENT,
    SWEATY_AGENT,
    GENERAL_AGENT,
    AGGREGATOR_AGENT,
    AGGREGATOR_CHECKPOINT,
    TOOLS,
    # tools
    RAG_TOOL,
    TAVILY_WS_TOOL,
    GOOGLE_SERPER_WS_TOOL,
    # edges
    INTENT_ROUTER,
    TOOL_ROUTER
)
from .nodes import NodeRegistry, NodeBase
from .nodes.agents import AgentBase
from .nodes.tools import ToolRegistry
from .edges import EdgeRegistry
from .states import (
    StateBase,
    SystemState,
    InputState,
    OutputState,
    InteractionState
)
from ..utils import log_registry, ExecutionTime


class Graph:
    """
    A wrapper around LangChain's StateGraph to construct and manage a configurable graph structure 
    of nodes and edges, suitable for agent-based workflows.

    Attributes:
        _state_graph (StateGraph): The underlying state graph instance from LangGraph.
        _compiled_graph (Any): The compiled version of the state graph after calling compile().
        _config (dict[str, Any]): Configuration dictionary for the compiled graph.

    See Also:
        https://langchain-ai.github.io/langgraph/concepts/low_level/#stategraph
    """
    def __init__(
            self, 
            state_schema: Type[StateBase],
            input: Type[StateBase],
            output: Type[StateBase]
        ) -> None:
        """
        Initializes the Graph with a given state schema, input type, and output type.

        Args:
            state_schema (Type[StateBase]): The schema used to define state transitions.
            input (Type[StateBase]): The input state type.
            output (Type[StateBase]): The output state type.
        """
        self._state_graph = StateGraph(
            state_schema=state_schema,
            input=input,
            output=output
        )

    def _reset_state(self) -> None:
        """
        Resets the interaction state of the compiled graph to its default state.
        """
        self._compiled_graph.update_state(self._config, InteractionState.reset())
        ExecutionTime.reset()

    def _check_for_nodes(self, names: Union[str, list[str]]) -> Union[NodeBase, list[NodeBase]]:
        """
        Validates and retrieves node classes from the registry.

        Args:
            names (Union[str, list[str]]): Name or list of names of nodes to validate.

        Returns:
            Union[NodeBase, list[NodeBase]]: The corresponding node class(es) from the registry.

        Raises:
            ValueError: If a node name is not found in the registry.
        """
        if isinstance(names, str):
            names = [names]
        for name in names:
            if NodeRegistry.get(name) is None and name not in [START, END]:
                raise ValueError(f"Node with name '{name}' not found in registry.")
        nodes = [NodeRegistry.get(name) for name in names]
        return nodes[0] if len(nodes) == 1 else nodes

    def add_node(self, name: str) -> None:
        """
        Adds a node to the graph.

        Args:
            name (str): The name of the node to add.
        """
        node_class: NodeBase = self._check_for_nodes(name)
        self._state_graph.add_node(name, node_class.node)

    def add_node_with_tools(self, name: str, tools: list[str]) -> None:
        """
        Adds a node to the graph and attaches tools to it.

        Args:
            name (str): The name of the node.
            tools (list[str]): A list of tool names to attach to the node.

        Raises:
            TypeError: If the node is not an instance of AgentBase.
            ValueError: If any tool name is not found in the registry.
        """
        node_class: NodeBase = self._check_for_nodes(name)
        if not isinstance(node_class, AgentBase):
            raise TypeError(f"Node '{name}' must be an instance of AgentBase to add tools.")
        for tool in tools:
            tool_class = ToolRegistry.get(tool)
            if tool_class is None:
                raise ValueError(f"Tool with name '{tool}' not found in registry.")
            node_class.add_tool(tool_class)
        self._state_graph.add_node(name, node_class.node)

    def add_edge(self, from_node: str, to_node: str) -> None:
        """
        Adds a direct edge between two nodes in the graph.

        Args:
            from_node (str): The name of the starting node.
            to_node (str): The name of the destination node.
        """
        self._check_for_nodes([from_node, to_node])
        self._state_graph.add_edge(from_node, to_node)

    def add_conditional_edges(self, from_node: Union[str, list[str]], edge: str) -> None:
        """
        Adds conditional edges from one or multiple nodes based on edge logic.

        Args:
            from_node (Union[str, list[str]]): Name or list of names of source nodes.
            edge (str): The name of the edge logic to apply.

        Raises:
            ValueError: If the edge name is not found in the registry.
        """
        if isinstance(from_node, list):
            for node in from_node:
                self.add_conditional_edges(node, edge)
        else:
            edge_class = EdgeRegistry.get(edge)
            if edge_class is None:
                raise ValueError(f"Edge with name '{edge}' not found in registry.")
            self._check_for_nodes(from_node)
            self._state_graph.add_conditional_edges(from_node, edge_class.edge)
        
    def compile(self) -> None:
        """
        Compiles the graph and sets up the checkpointer and configuration.
        """
        self._config = {"configurable": {"thread_id": "unique_thread_id"}}
        checkpointer = MemorySaver()
        self._compiled_graph = self._state_graph.compile(checkpointer=checkpointer)
            
    def invoke(self, query: str) -> dict[str, Any] | Any:
        """
        Executes the compiled graph with a given query.

        Args:
            query (str): The input query to pass into the graph.

        Returns:
            dict[str, Any] | Any: The response generated by the graph execution.

        Raises:
            RuntimeError: If the graph has not been compiled before invoking.
        """
        if not hasattr(self, '_compiled_graph'):
            raise RuntimeError("Graph must be compiled before invoking.")
        try:
            response = self._compiled_graph.invoke({"query": query}, config=self._config)
            return response
        finally:
            self._reset_state()



class GraphBuilder:
    """
    Factory class for building and initializing a Graph instance with predefined nodes and edges.
    """
    @classmethod
    def _create_nodes(cls, graph: Graph) -> None:
        """
        Adds and configures all predefined nodes in the graph.

        Args:
            graph (Graph): The graph instance to modify.
        """
        # -- agents -- 
        graph.add_node_with_tools(
            name=INTENT_AGENT,
            tools=[HSO_AGENT, GENERAL_AGENT, SWEATY_AGENT]
        )
        graph.add_node_with_tools(
            name=HSO_AGENT,
            tools=[GOOGLE_SERPER_WS_TOOL]
        )
        graph.add_node_with_tools(
            name=SWEATY_AGENT,
            tools=[RAG_TOOL]
        )
        graph.add_node(
            name=GENERAL_AGENT,
        )
        graph.add_node(
            name=AGGREGATOR_AGENT
        )
        # -- others --
        graph.add_node(
            name=AGGREGATOR_CHECKPOINT
        )
        graph.add_node(
            name=TOOLS
        )
    
    @classmethod
    def _create_edges(cls, graph: Graph) -> None:
        """
        Adds and configures all predefined edges between nodes in the graph.

        Args:
            graph (Graph): The graph instance to modify.
        """
        graph.add_edge(
            from_node=START,
            to_node=INTENT_AGENT
        )
        graph.add_conditional_edges(
            from_node=INTENT_AGENT,
            edge=INTENT_ROUTER
        )
        graph.add_conditional_edges(
            from_node=[GENERAL_AGENT, HSO_AGENT, SWEATY_AGENT],
            edge=TOOL_ROUTER
        )
        graph.add_edge(
            from_node=AGGREGATOR_AGENT,
            to_node=END
        )

    @classmethod
    def create(cls) -> Graph:
        """
        Constructs, configures, and compiles a complete Graph instance.

        Returns:
            Graph: A fully built and compiled graph object.
        """
        graph = Graph(
            state_schema=SystemState,
            input=InputState,
            output=OutputState
        )
        log_registry(ToolRegistry)
        log_registry(NodeRegistry)
        log_registry(EdgeRegistry)
        cls._create_nodes(graph)
        cls._create_edges(graph)
        graph.compile()
        
        return graph       