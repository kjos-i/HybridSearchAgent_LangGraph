"""
Utility functions for the Hybrid Search Agent.

This module provides helper tools for:
    - Debugging and console output management.
    - Visualizing LangGraph agent workflows as PNG files.
    - Configuring dual-handler logging (File + Console).
"""

# Standard library
import logging
import sys

# Local imports
from config import DEBUG_PRINT


# --- Debugging & Output ---
def debug_print(*args, **kwargs):
    """
    Prints messages to the console only if the global DEBUG_PRINT flag is enabled.
    
    Args:
        *args: Variable length argument list to print.
        **kwargs: Arbitrary keyword arguments.
    """
    if DEBUG_PRINT:
        print("[DEBUG]", *args, **kwargs)


# --- Visualization ---
def print_agent_graph(agent, filename="agent_graph.png"):
    """
    Generates and saves a visual representation of the LangGraph workflow.
    
    Uses Mermaid.js rendering to create a PNG schema of nodes and edges.
    
    Args:
        agent: The compiled LangGraph agent instance.
        filename (str): The target path/name for the exported image.
    """
    try:
        # Retrieve the internal graph structure from the compiled agent
        agent_graph = agent.get_graph()

        # Generate binary PNG data using the Mermaid API
        png_bytes = agent_graph.draw_mermaid_png()
        
        # Save the byte stream to a physical file
        with open(filename, "wb") as f:
            f.write(png_bytes)
        
        print(f"Agent graph saved to {filename}. Open it to verify the logic flow.")

    except Exception as e:
        print(f"Could not generate graph: {e}.")


# --- Logging Infrastructure ---
def setup_logger(name=__name__, log_file="agent.log"):
    """Configures a dual-handler logger for robust activity tracking.
    
    Handlers:
        1. File (DEBUG): A detailed 'black box' recording all events with timestamps.
        2. Console (INFO): A clean 'dashboard' for the user showing only high-level info.

    Args:
        name (str): The name of the logger (usually the module __name__).
        log_file (str): The filename for the permanent log record.

    Returns:
        logging.Logger: A configured logger instance.
"""

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Singleton pattern: Prevent duplicate handlers if the setup is called twice
    if not logger.handlers:
        # 1. File Handler: Captures deep technical details for troubleshooting
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)

        # 2. Console Handler: Provides immediate, readable feedback to the operator
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('[%(levelname)s] %(message)s')
        console_handler.setFormatter(console_formatter)

        # Attach handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger
