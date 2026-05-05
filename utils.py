"""
Shared utility helpers for the Hybrid Search Agent.

Bundles a small set of cross-cutting concerns used by the agent, the
retriever, and the dashboard:
    - fmt_score — uniform score formatting for tables and exports.
    - debug_print — gated console output controlled by DEBUG_PRINT.
    - print_agent_graph — Mermaid-rendered PNG of the LangGraph workflow.
    - setup_logger / setup_logger_dual — file (and optional console)
      handlers for runtime activity tracking, written to agent.log.
"""

# Standard library
import logging
import sys
from pathlib import Path

# Local imports
from config import DEBUG_PRINT


# Default log file lives next to utils.py so logs land in the project folder
# regardless of the caller's current working directory.
_DEFAULT_LOG_PATH = str(Path(__file__).parent / "agent.log")


# --- Formatting ---
def fmt_score(score: float | None, decimals: int = 4) -> str:
    """
    Format a retrieval score for display, returning 'N/A' when missing.

    Args:
        score (float | None): The score to format, or None for missing values.
        decimals (int): Number of decimal places to render. Defaults to 4.

    Returns:
        str: The formatted score string, or 'N/A' if score is None.
    """
    return f"{score:.{decimals}f}" if score is not None else "N/A"


# --- Debugging & Output ---
def debug_print(*args, **kwargs):
    """
    Print a [DEBUG]-prefixed message to stdout, gated by the
    DEBUG_PRINT flag in config.py.

    Provides a single switch for ad-hoc tracing inside tools and helpers
    without requiring a logger.  When DEBUG_PRINT is False the call
    is a no-op.

    Args:
        *args: Positional values forwarded to print.
        **kwargs: Keyword arguments forwarded to print (e.g., end,
            flush).
    """
    if DEBUG_PRINT:
        print("[DEBUG]", *args, **kwargs)


# --- Visualization ---
def print_agent_graph(agent, filename="agent_graph.png"):
    """
    Render the compiled LangGraph workflow to a PNG via the Mermaid API.

    Useful for documentation and quick visual inspection of node/edge
    structure.  Failures (no network, Mermaid API errors, file I/O) are
    swallowed and reported on stdout so a dev-time visualisation never
    crashes the agent.

    Args:
        agent: The compiled LangGraph agent (must expose get_graph()).
        filename (str): Destination path for the PNG.  Defaults to
            agent_graph.png in the current working directory.
    """
    try:
        agent_graph = agent.get_graph()
        png_bytes = agent_graph.draw_mermaid_png()
        with open(filename, "wb") as f:
            f.write(png_bytes)
        print(f"Agent graph saved to {filename}. Open it to verify the logic flow.")

    except Exception as e:
        print(f"Could not generate graph: {e}.")


# --- Logging Infrastructure ---
def setup_logger(name=__name__, log_file=_DEFAULT_LOG_PATH):
    """
    Configure a file-only logger for runtime activity tracking.

    All log levels (DEBUG and up) are written to log_file with full
    timestamps; nothing is printed to the console.  Repeated calls with
    the same name reuse the existing logger (no duplicate handlers).

    Args:
        name (str): Logger name.  Conventionally the calling module's
            __name__ so log lines are distinguishable when multiple
            modules share the same file.
        log_file (str): Path to the log file.  Defaults to agent.log
            next to utils.py.

    Returns:
        logging.Logger: A logger configured with a single FileHandler.
    """

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Singleton pattern: Prevent duplicate handlers if setup is called twice.
    if not logger.handlers:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


# --- Logging Infrastructure --- Dual Handler Version ---
def setup_logger_dual(name=__name__, log_file=_DEFAULT_LOG_PATH):
    """
    Configure a dual-handler logger writing to file and console.

    The file handler captures the full DEBUG stream (a permanent 'black
    box' record with timestamps), while the console handler shows only
    INFO and above with a compact [LEVEL] message format suitable for
    operator feedback.  Repeated calls with the same name reuse the
    existing logger (no duplicate handlers).

    Args:
        name (str): Logger name.  Conventionally the calling module's
            __name__.
        log_file (str): Path to the log file.  Defaults to agent.log
            next to utils.py.

    Returns:
        logging.Logger: A logger configured with a FileHandler (DEBUG)
        and a StreamHandler (INFO) on stdout.
    """

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Singleton pattern: Prevent duplicate handlers if setup is called twice.
    if not logger.handlers:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('[%(levelname)s] %(message)s')
        console_handler.setFormatter(console_formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger
