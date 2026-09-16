from typing import override

from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_tavily import TavilySearch

from ._base import ToolBase
from ....config.graph import (
    DDG_WS_TOOL,
    GOOGLE_SERPER_WS_TOOL,
    TAVILY_WS_TOOL
)
from ....config.application import MAX_RETRIEVAL_CONTEXT
from ....utils import log_tool, skip_registry


# tools could have been implemented as nodes by using injected state, command allows to implement state change directly in


@skip_registry
class DDGWebSearch(ToolBase):
    """
    A web search tool that uses DuckDuckGo to fetch search results as context for a given query.

    Attributes:
        _client (DuckDuckGoSearchResults): Configured LangChain-compatible search utility.
    """
    def __init__(self):
        """
        Initializes the DuckDuckGo-based web search tool with output formatting and result limits.
        """
        ToolBase.__init__(
            self, 
            name=DDG_WS_TOOL, 
            description="This tool leverages DuckDuckGo to search the web for context to answer a query.")
        self._client = DuckDuckGoSearchResults(
            output_format="string", 
            num_results=MAX_RETRIEVAL_CONTEXT)

    @override
    def invoke(self, query: str) -> str:
        """
        Executes the DuckDuckGo web search and logs the query.

        Args:
            query (str): The query to search online.

        Returns:
            str: Search results as a formatted string.
        """
        log_tool(self, query=query)
        result = self._client.invoke(query)
        return result


class GoogleSerperWebSearch(ToolBase):
    """
    A web search tool using the Google Serper API to retrieve contextual search data.

    Attributes:
        _client (GoogleSerperAPIWrapper): Configured Serper API client.
    """
    def __init__(self):
        """
        Initializes the Google Serper-based tool with search type and result cap.
        """
        ToolBase.__init__(
            self, 
            name=GOOGLE_SERPER_WS_TOOL, 
            description="This tool leverages Google Serper to search the web for context to answer a query.")
        self._client = GoogleSerperAPIWrapper(
            k=MAX_RETRIEVAL_CONTEXT, 
            type="search")

    @override
    def invoke(self, query: str) -> str:
        """
        Executes a query via Google Serper and logs it.

        Args:
            query (str): User's web search input.

        Returns:
            str: Combined text results from the API.
        """
        log_tool(self, query=query)
        result = self._client.run(query=query)
        return result
    

class TavilyWebSearch(ToolBase):
    """
    A web search tool powered by Tavily for retrieving general-topic context via online search.

    Attributes:
        _client (TavilySearch): Tavily client instance for querying and retrieving results.
    """
    def __init__(self):
        """
        Initializes the Tavily tool with search topic and result configuration.
        """
        ToolBase.__init__(
            self, 
            name=TAVILY_WS_TOOL, 
            description="This tool leverages Tavily to search the web for context to answer a query.")
        self._client = TavilySearch(
            max_results=MAX_RETRIEVAL_CONTEXT,
            topic="general")

    @override
    def invoke(self, query: str) -> str:
        """
        Executes a Tavily search and concatenates result contents.

        Args:
            query (str): The user-provided query string.

        Returns:
            str: Aggregated content from all search results.
        """
        log_tool(self, query=query)
        result = self._client.invoke({"query": query})
        storage = ""
        for i in range(len(result.get("results"))):
            storage += result.get("results")[i].get("content")
        return storage