from typing import Any, Union, override

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

from .....config.application import LLM_PROXY_URL
from .....utils import log_inference


class ProxyChatModel(BaseChatModel):
    """
    A proxy-based chat model used by agents to interface with an external LLM server.

    Wraps LangChain's `ChatOpenAI` model, routing calls through a proxy server (e.g., Hochschule Offenburg's hosted endpoint),
    with optional tool binding support and prompt injection.

    Attributes:
        _prompt (SystemMessage): A system-level message prepended to all inputs.
        _tools (list[BaseTool]): Tools available for invocation by the model.
        _tool_choice (Union[dict, str, bool, None]): Strategy for tool selection.
        _client (ChatOpenAI): The base LLM client.
        _client_with_tools (ChatOpenAI): Client bound with tools, if applicable.
    """
    def __init__(self, model: str, prompt: str, tool_choice: Union[dict, str, bool, None], **kwargs: dict[str, Any]):
        """
        Initializes the proxy chat model with model type, prompt, tool choice, and optional kwargs.

        Args:
            model (str): Identifier of the model to invoke.
            prompt (str): The prompt text to use as a system message.
            tool_choice (Union[dict, str, bool, None]): Tool selection strategy.
            **kwargs (dict[str, Any]): Additional keyword arguments for BaseChatModel.
        """
        super().__init__(**kwargs)
        self._initialize_client(model, LLM_PROXY_URL)
        self._prompt = SystemMessage(content=prompt)
        self._tools: dict[str, BaseTool] = {}
        self._tool_choice = tool_choice

    def _initialize_client(self, model: str, proxy_url: str):
        """
        Initializes the ChatOpenAI client using the proxy URL.

        Args:
            model (str): The model name to use.
            proxy_url (str): The proxy server endpoint.
        """
        self._client = ChatOpenAI(
            model=model,
            openai_api_base=proxy_url
        )
        self._client_with_tools = self._client

    @override
    @property
    def _llm_type(self) -> str:
        return "proxy-chat-model"

    @override
    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "client": self._client,
            "prompt": self._prompt,
            "tools": self._tools
            }

    @override
    def _generate(self, messages: list[BaseMessage], **kwargs: Any) -> ChatResult:
        """
        Generates a chat response from the model, optionally using tools.

        Prepends the system prompt, logs inference, and invokes the LLM client with or without tool bindings.

        Args:
            messages (list[BaseMessage]): List of input messages.
            **kwargs (Any): Additional options (e.g., `no_tools` flag).

        Returns:
            ChatResult: The result containing model generations.
        """
        prompt_messages = [self._prompt] + messages[:]
        log_inference(self, prompt_messages)
        if kwargs.get("no_tools", False):
            response: AIMessage = self._client.invoke(prompt_messages)
        else:
            response: AIMessage = self._client_with_tools.invoke(prompt_messages)
        generation = ChatGeneration(message=response)
        result = ChatResult(generations=[generation])
        return result
    
    def bind_tools(self, tools: list[BaseTool]) -> None:
        """
        Binds tools to the chat model, enabling tool-assisted LLM responses.

        Args:
            tools (list[BaseTool]): Tools to bind to the client for use during inference.
        """
        self._tools.update({tools[0].get_name(): tools[0]})
        #print(f'länge {len(self._tools)}')
        self._client_with_tools = self._client.bind_tools(list(self._tools.values()), tool_choice=self._tool_choice)
        