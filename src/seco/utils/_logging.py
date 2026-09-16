import logging


def log_node(node, state) -> None:
    """
    Logs the invocation of a node along with its current state.

    Args:
        node (object): The node instance being invoked.
        state (object): The current state passed to the node, expected to have an `as_string` method.
    """
    logging.info(f"{node.__class__.__name__} node invoked with state: {state.as_string()}")

def log_edge(edge) -> None:
    """
    Logs the invocation of an edge.

    Args:
        edge (object): The edge instance being invoked.
    """
    logging.info(f"{edge.__class__.__name__} edge invoked")

def log_tool(tool, **kwargs) -> None:
    """
    Logs the invocation of a tool along with its keyword arguments.

    Args:
        tool (object): The tool instance being invoked.
        **kwargs: Arbitrary keyword arguments representing the tool's input parameters.
    """
    logging.info(f"{tool.__class__.__name__} tool invoked with arguments: {', '.join([f'{keyword}={argument}' for keyword, argument in kwargs.items()])}")

def log_inference(model, prompt_messages) -> None:
    """
    Logs when a model inference is invoked, including the prompt messages.

    Args:
        model (object): The model instance performing inference.
        prompt_messages (list): The list of messages used as the prompt.
    """
    logging.info(f"{model.__class__.__name__} inference invoked with prompt messages: {prompt_messages}")

def log_registry(registry) -> None:
    """
    Logs the contents of a registry, listing all registered classes.

    Args:
        registry (object): The registry class containing registered classes.
    """
    logging.info(f"{registry.__name__}: {', '.join([cl.__class__.__name__ for cl in registry.get_all()])}")