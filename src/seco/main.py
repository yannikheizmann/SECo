import argparse
from colorama import init, Fore
import gradio as gr
import os
import logging
from typing import Any

from dotenv import load_dotenv

from .config.graph import (
    GENERAL_AGENT,
    HSO_AGENT,
    SWEATY_AGENT,
)
from .config.application import LOGGING_PATH




class Application:
    """
    Main application class for querying the graph and interacting via terminal or Gradio UI.

    Attributes:
        _graph (GraphBuilder): The graph-based query engine initialized via GraphBuilder.
    """

    def __init__(self):
        """
        Initializes the application by setting up the colorama console and building the graph.
        """
        load_dotenv()
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(LOGGING_PATH)],
        )
        from .graph import GraphBuilder

        init()
        logging.info("Initializing the graph.")
        self._graph = GraphBuilder.create()

    def _get_prefix(self, response: dict[str, Any]) -> str:
        """
        Generates a prefix emoji string based on the agents in the response.

        Args:
            response (dict[str, Any]): A dictionary representing the system's response.
                Must contain a key `"agents"` which maps to a list of agent identifiers.

        Returns:
            str: Emoji prefix based on present agents:
                - "💬" for GENERAL_AGENT
                - "📚" for HSO_AGENT
                - "🤖" for SWEATY_AGENT
                - "🧠" if none matched
        """
        agents = response["agents"]
        prefix = ""
        for agent in agents:
            if agent == GENERAL_AGENT:
                prefix += "💬"
            elif agent == HSO_AGENT:
                prefix += "📚"
            elif agent == SWEATY_AGENT:
                prefix += "🤖"
        if prefix == "":
            return "🧠"
        return prefix

    def _get_color(self, response: dict[str, Any]) -> str:
        """
        Determines the terminal color for a response based on the involved agent.

        Args:
            response (dict[str, Any]): A dictionary containing a list of agents under the "agents" key.

        Returns:
            str: A color code from `colorama.Fore`:
                - `Fore.BLUE` for GENERAL_AGENT
                - `Fore.RED` for HSO_AGENT
                - `Fore.GREEN` for SWEATY_AGENT
                - `Fore.WHITE` if multiple or no agents matched
        """
        agents = response["agents"]
        color = None
        for agent in agents:
            if agent == GENERAL_AGENT:
                color = Fore.BLUE
            elif agent == HSO_AGENT:
                color = Fore.RED
            elif agent == SWEATY_AGENT:
                color = Fore.GREEN
        if not color or len(agents) >= 2:
            color = Fore.WHITE
        return color

    def _handle_query(
        self,
        query: str,
        chat_history: list[tuple[str, str]]
    ) -> tuple[str, list[tuple[str, str]], list[tuple[str, str]]]:
        """
        Handles a query, appends the result to chat history, and formats response for Gradio.

        Args:
            query (str): The user's input query.
            chat_history (list[tuple[str, str]]): List of (query, response) tuples representing chat history.

        Returns:
            tuple[str, list[tuple[str, str]], list[tuple[str, str]]]:
                - Empty string to reset input box.
                - Updated chat history.
                - Same updated chat history (for chatbot rendering).
        """
        if query.strip().lower() == "exit":
            chat_history.append((query, "🛑 Anwendung wird beendet..."))
            os._exit(0)
            logging.info("Stopping the application.")
            return "", chat_history, chat_history

        response = self._graph.invoke(query)
        prefix = self._get_prefix(response)
        chat_history.append((query, f"{prefix}: {response['response'].content}"))
        return "", chat_history, chat_history

    def run_in_terminal(self) -> None:
        """
        Runs the application in terminal (CLI) mode.

        Continuously prompts the user for input and displays color-coded responses.
        """
        print("Type 'exit' to quit.")
        while True:
            query = input("QUERY: ")

            if query.lower() == "exit":
                logging.info("Stopping the application.")
                break

            response = self._graph.invoke(query)
            prefix = self._get_prefix(response)
            color = self._get_color(response)

            print(f"""
{color}{prefix} RESPONSE: {response["response"].content}{Fore.RESET}
        """)

    def run_in_gradio(self) -> None:
        """
        Launches the application with a Gradio-based user interface.
        Allows users to ask queries and receive responses in a web interface.
        """
        with gr.Blocks() as demo:
            chatbot = gr.Chatbot(label="Graph Query Interface")

            with gr.Row():
                msg = gr.Textbox(placeholder="Frage stellen...", scale=4)
                submit = gr.Button("Senden", scale=1)
            state = gr.State([])

            submit.click(self._handle_query, [msg, state], [msg, state, chatbot])
            msg.submit(self._handle_query, [msg, state], [msg, state, chatbot])

        demo.launch()


def main():
    """
    Entry point of the application.

    Parses CLI arguments and launches either the terminal-based or Gradio-based interface
    depending on the provided flags.

    Command-line Args:
        --frontend (bool): Launch the Gradio interface if specified.
    """
    parser = argparse.ArgumentParser(description="SECo CLI/Frontend")
    parser.add_argument("--frontend", action="store_true", help="Start with Gradio frontend")
    args = parser.parse_args()

    app = Application()
    if args.frontend:
        logging.info("Starting the application with gradio frontend")
        app.run_in_gradio()
    else:
        logging.info("Starting the application (without gradio frontend)")
        app.run_in_terminal()


if __name__ == "__main__":
    main()
