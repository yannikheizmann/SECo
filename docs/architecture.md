# Architecture and extension

The graph is assembled by
[`GraphBuilder`](../src/seco/graph/_graph.py). It wraps LangGraph's `StateGraph`
and connects three stages of agents:

1. **Intent agent:** splits the user's query into requests for relevant specialists.
2. **Specialist agents:** answer their assigned requests, using tools when needed.
   The HSO agent uses Google Serper for university information; the Sweaty agent
   retrieves robot facts from Chroma; the general agent handles other conversation.
3. **Aggregator agent:** combines specialist responses into the final answer.

![Agent graph](graph.png)

The intent router dispatches specialist requests with LangGraph `Send`. The tool
router checks for tool calls and sends work to the tools node. An aggregation
checkpoint checks whether all scheduled specialists have produced answers before
forwarding execution to the aggregator.

## Package responsibilities

| Package | Responsibility |
| --- | --- |
| [`graph/nodes`](../src/seco/graph/nodes) | Agent execution, tool execution, aggregation checkpoint |
| [`graph/edges`](../src/seco/graph/edges) | Conditional routing through `EdgeBase` implementations |
| [`graph/states`](../src/seco/graph/states) | Conversation, generation, resolution, input and output models |
| [`config`](../src/seco/config) | Shared identifiers, model and prompt defaults, application settings |
| [`utils`](../src/seco/utils) | Registration, timing, logging, and shared response containers |

`SystemState` combines conversation and interaction state. Reducers merge updates
such as specialist responses and tool context. After invocation, the graph resets
interaction fields while preserving conversation history. An in-memory checkpointer
uses a fixed thread identifier; history is not persisted across application restarts.

## Registration

[`RegistryMeta`](../src/seco/utils/_meta.py) creates an instance of each concrete
registered class as its module is imported. Node, tool, and edge registries then
let the builder resolve components by their configured names. Classes intended
for registration must be imported by the corresponding package `__init__.py`.

Constructors can initialize external clients and the RAG index. Environment
configuration must therefore be ready before graph import. `@skip_registry`
excludes a class from automatic registration.

## Add a specialist agent

1. Add a unique identifier to `config/graph/_agents.py` and export it through
   `config/graph/__init__.py`.
2. Add the model and prompt constants to `config/agents` and export them.
3. Implement a subclass of
   [`SpecialistAgentBase`](../src/seco/graph/nodes/agents/_base.py), following
   [`_specialists.py`](../src/seco/graph/nodes/agents/_specialists.py). Pass its name,
   model, prompt, and description to the base constructor.
4. Import the class in the agents package so the registry sees it.
5. Add it to the intent agent's tools in `GraphBuilder._create_nodes`, add its node
   with any retrieval tools, and connect its conditional tool-routing edge in
   `GraphBuilder._create_edges`.
6. Update the intent prompt to describe the new domain and verify routing with
   representative single-domain and mixed-domain requests.

## Add a tool

1. Define and export its identifier in `config/graph/_tools.py`.
2. Subclass [`ToolBase`](../src/seco/graph/nodes/tools/_base.py), provide a name and
   description, and implement `invoke(query)`.
3. Import the implementation in `graph/nodes/tools/__init__.py` for registration.
4. Attach its identifier to an agent using `graph.add_node_with_tools` in the builder.
5. Document any additional credentials and declare its runtime dependencies.

The base class adapts tool calls into LangChain tools. The shared tools node
executes them and updates graph state with the retrieved context.
