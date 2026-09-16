import os
from typing import override

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from ._base import ToolBase
from ....config.graph import RAG_TOOL
from ....config.application import (
    RAG_DATA_PATH,
    EMBEDDINGS_MODEL,
    LLM_PROXY_URL,
    RAG_VS_PATH,
    MAX_RETRIEVAL_CONTEXT
)


class RAG(ToolBase):
    """
    A Retrieval-Augmented Generation (RAG) tool for answering queries using external Sweaty-related documents.

    Loads documents from a configured path, splits them, embeds and indexes them in a vector store,
    and performs similarity-based retrieval to provide relevant context for agent queries.

    Inherits from:
        ToolBase: Provides interface compatibility with LangChain and LangGraph tools.
    """
    def __init__(self):
        """
        Initializes the RAG tool with configuration, metadata, and prepares the retriever.
        """
        super().__init__(
            name=RAG_TOOL, 
            description="This tool searches the external dataset with data on Sweaty for context to answer a specified query.")
        self._initialize()
    
    def _get_data(self) -> list:
        """
        Loads raw text documents from the RAG data path.

        Returns:
            list: List of document objects loaded from the file system.
        """
        data_file_paths = [os.path.join(RAG_DATA_PATH, filename) for filename in os.listdir(RAG_DATA_PATH)]
        data = []
        for path in data_file_paths:
            loader = TextLoader(path)
            data.extend(loader.load())
        return data
    
    def _split_data(self, data: list) -> list:
        """
        Splits loaded documents into manageable chunks using a recursive text splitter.

        Args:
            data (list): List of loaded documents.

        Returns:
            list: List of chunked document segments.
        """
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=35, chunk_overlap=10) #TODO WHAT IS GOOD VALUE?!?
        return text_splitter.split_documents(data)
    
    def _initialize_vector_store(self, data_splits: list) -> None:
        """
        Creates a Chroma vector store from embedded document chunks.

        Args:
            data_splits (list): The split documents to index in the vector store.
        """
        embeddings = OpenAIEmbeddings(
            model=EMBEDDINGS_MODEL,
            openai_api_base=LLM_PROXY_URL
        )
        self._vector_store: Chroma = Chroma.from_documents(
            documents=data_splits,
            embedding=embeddings,
            collection_name="sweaty_collection",
            persist_directory=self.vs_path()
        )

    def _initialize_retriever(self) -> None:
        """
        Initializes a similarity-based retriever using the Chroma vector store.
        """
        self._retriever: VectorStoreRetriever = self._vector_store.as_retriever(
            search_type="similarity", 
            search_kwargs={
                "k": MAX_RETRIEVAL_CONTEXT,  
            }
        )
        
    def _initialize(self) -> None:
        """
        Executes full initialization: loads data, splits it, embeds and stores in vector DB, and sets up retriever.
        """
        data: list = self._get_data()
        data_splits: list = self._split_data(data)
        self._initialize_vector_store(data_splits)
        self._initialize_retriever()

    def vs_path(self) -> str:
        """
        Returns the local file system path where the vector store should persist.

        Returns:
            str: Path to Chroma database directory.
        """
        return f"{RAG_VS_PATH}/chroma_sweaty_db"

    @override
    def invoke(self, query: str) -> str:
        """
        Retrieves relevant documents for a given query using similarity search.

        Args:
            query (str): The user's input query.

        Returns:
            str: Concatenated page contents of retrieved documents as a single string context.
        """
        docs = self._retriever.invoke(query)
        context = "\n".join(doc.page_content for doc in docs)
        return context
    
  