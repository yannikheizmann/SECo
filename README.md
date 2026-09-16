# SECo

**Sweaty Conversation Agent System** is a LangGraph-based conversational agent for
Hochschule Offenburg's humanoid robot Sweaty. It answers questions about the
university and the robot, and handles general conversation. Specialist agents use
web search and a local retrieval-augmented generation (RAG) knowledge base.

## Setup

Use Python 3.13 and uv. Run these commands from the repository root:

```sh
uv sync --locked
cp .env.example .env
```

Set the following credentials in `.env`:

| Variable | Purpose |
| --- | --- |
| `OPENAI_API_KEY` | Access to the university's LLM proxy for chat and embeddings |
| `SERPER_API_KEY` | University web search through Google Serper |
| `TAVILY_API_KEY` | Required by the Tavily tool registered during startup |

The default proxy is `https://llm-proxy.imla.hs-offenburg.de`. The original
university deployment requires access through the HSO network/VPN. Credentials
are not included. Existing environment variables take precedence over `.env`.

## Run

```sh
uv run seco              # Terminal conversation; enter exit to quit
uv run seco --frontend   # Gradio interface
uv run seco --help       # Show options without initializing the graph
```

You can also run `uv run python -m seco.main` with the same options.
Startup builds the RAG index from `data/sweaty/`, which requires the embedding
service. The generated Chroma files are stored in `data/rag/` and logs in
`seco.log`; both are ignored by Git.

The current application uses a single conversation thread. The Gradio interface
is intended for a local demonstration, not isolated multi-user sessions.

## Configuration and architecture

- [Configuration](docs/configuration.md): models, prompts, paths, and graph identifiers.
- [Architecture and extension](docs/architecture.md): routing, states, registries,
  and adding agents or tools.

```text
src/seco/
├── config/   # Application defaults, agent prompts/models, graph identifiers
├── graph/    # Graph builder, nodes, tools, routing, and state models
├── utils/    # Registries, logging, timing, and shared containers
└── main.py   # Terminal and Gradio application
```

## Background

SECo grew out of a university project for Sweaty at Hochschule Offenburg.
This repository contains the conversational agent implementation by Yannik
Heizmann. The original university project was supervised by Prof. Dr. Daniela Oelke.
