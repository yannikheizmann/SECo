# Configuration

Configuration is defined as module-level constants under
[`src/seco/config`](../src/seco/config). Changes take effect when the application
restarts. There is no configuration-file or CLI override framework; the only
application flag is `--frontend`.

## Application defaults

[`application/_static.py`](../src/seco/config/application/_static.py) contains:

| Constant | Default / purpose |
| --- | --- |
| `MAX_HISTORY_LENGTH` | 10; conversation history retention setting |
| `MAX_RETRIEVAL_CONTEXT` | 5; retrieved documents and search result limit |
| `LLM_PROXY_URL` | University endpoint used for chat and embeddings |
| `EMBEDDINGS_MODEL` | `text-embedding-3-large` |
| `DATA_PATH` | Repository `data/` directory |
| `RAG_DATA_PATH` | `data/sweaty/`, containing the robot knowledge base |
| `RAG_VS_PATH` | `data/rag/`, containing generated Chroma storage |
| `LOGGING_PATH` | Repository `seco.log` |

Paths are resolved from the source checkout, independently of the current working
directory. Running from a source checkout with `uv sync` is the supported setup;
the facts file is not bundled into the Python wheel.

The RAG implementation reads the knowledge files, splits them, embeds them, and
builds its vector store during initialization. It currently uses 35-token chunks
with 10-token overlap in [`_rag.py`](../src/seco/graph/nodes/tools/_rag.py).

## Agents

[`agents/_models.py`](../src/seco/config/agents/_models.py) defines a shared
`DEFAULT_AGENT_MODEL` (`gpt-4o-mini`) and individual constants for intent, HSO,
Sweaty, general, and aggregator agents. Change an individual constant to select a
different model for that agent. The selected model must be available through the
configured proxy.

[`agents/_prompts.py`](../src/seco/config/agents/_prompts.py) holds the German system
prompts. They define each agent's domain, response style, and use of supporting
context. Prompts are static and contain no import-time timestamp.

## Graph identifiers

[`config/graph`](../src/seco/config/graph) defines the names used by graph builders,
registries, and routing. Import these constants wherever a component is registered
or referenced. LangGraph's `START` and `END` are imported directly from LangGraph.

## Environment and startup

[`main.py`](../src/seco/main.py) loads `.env` and configures logging before importing
the graph. This ordering matters: the registry instantiates concrete tools and
agents during import, including the RAG tool and its embedding client.
Importing `seco.config` alone does not load `.env` or initialize clients.

The default graph uses Google Serper for the HSO specialist and RAG for the Sweaty
specialist. Tavily is also instantiated by the registry, so its key is required
even though the default graph does not attach it to an agent. DuckDuckGo has
`@skip_registry` and is not instantiated automatically.
