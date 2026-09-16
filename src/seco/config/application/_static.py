from pathlib import Path

MAX_HISTORY_LENGTH = 10
MAX_RETRIEVAL_CONTEXT = 5

LLM_PROXY_URL = "https://llm-proxy.imla.hs-offenburg.de"
EMBEDDINGS_MODEL = "text-embedding-3-large"

# Resolve relative to the source checkout, independently of the working directory.
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATA_PATH = _PROJECT_ROOT / "data"
LOGGING_PATH = _PROJECT_ROOT / "seco.log"
RAG_DATA_PATH = DATA_PATH / "sweaty"
RAG_VS_PATH = DATA_PATH / "rag"
