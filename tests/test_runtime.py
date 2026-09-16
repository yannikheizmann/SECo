import os
import socket
import subprocess
import sys
import unittest
from unittest.mock import MagicMock, patch


class RuntimeTests(unittest.TestCase):
    def test_help_without_credentials(self):
        environment = {
            key: value for key, value in os.environ.items()
            if key not in {"OPENAI_API_KEY", "SERPER_API_KEY", "TAVILY_API_KEY"}
        }
        result = subprocess.run(
            [sys.executable, "-m", "seco.main", "--help"],
            env=environment, capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--frontend", result.stdout)

    def test_offline_conversation(self):
        from langchain_chroma import Chroma
        from langchain_core.messages import AIMessage
        from langchain_core.outputs import ChatGeneration, ChatResult
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        environment = {
            "OPENAI_API_KEY": "offline-test",
            "SERPER_API_KEY": "offline-test",
            "TAVILY_API_KEY": "offline-test",
            "LANGSMITH_TRACING": "false",
            "LANGCHAIN_TRACING_V2": "false",
        }
        store = MagicMock()
        # Substitute external retrieval and model responses, retaining real graph execution.
        with (
            patch.dict(os.environ, environment),
            patch.object(socket.socket, "connect", side_effect=AssertionError("Network forbidden")),
            patch.object(
                RecursiveCharacterTextSplitter, "from_tiktoken_encoder",
                return_value=RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20),
            ),
            patch.object(Chroma, "from_documents", return_value=store) as index,
        ):
            from seco.config.agents import GENERAL_AGENT_PROMPT, INTENT_AGENT_PROMPT
            from seco.graph import GraphBuilder
            from seco.graph.nodes.agents.models import ProxyChatModel

            calls = []

            def generate(model, messages, **kwargs):
                calls.append((model._prompt.content, messages))
                if model._prompt.content == INTENT_AGENT_PROMPT:
                    answer = AIMessage(content="", tool_calls=[{
                        "name": "general_agent", "args": {"query": "Hallo"}, "id": "call-1",
                    }])
                elif model._prompt.content == GENERAL_AGENT_PROMPT:
                    answer = AIMessage(content="Hallo!")
                else:
                    answer = AIMessage(content="Hallo zusammen!")
                return ChatResult(generations=[ChatGeneration(message=answer)])

            with patch.object(ProxyChatModel, "_generate", generate):
                graph = GraphBuilder.create()
                for query in ("Hallo", "Und weiter?"):
                    result = graph.invoke(query)
                    self.assertEqual(result["response"].content, "Hallo zusammen!")
                    self.assertEqual(result["agents"], ["general_agent"])
                intent_calls = [messages for prompt, messages in calls if prompt == INTENT_AGENT_PROMPT]
                self.assertEqual(len(intent_calls), 2)
                self.assertGreater(len(intent_calls[1]), len(intent_calls[0]))
                self.assertTrue(index.call_args.kwargs["documents"])

    def test_frontend_construction(self):
        with patch.dict(os.environ, {"GRADIO_ANALYTICS_ENABLED": "False"}):
            import gradio as gr
            from seco.main import Application

            application = Application.__new__(Application)
            application._graph = MagicMock()
            with patch.object(gr.Blocks, "launch") as launch:
                application.run_in_gradio()
        launch.assert_called_once()


if __name__ == "__main__":
    unittest.main()
