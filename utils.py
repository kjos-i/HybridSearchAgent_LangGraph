"""
Utils.
"""

# Load environment variables, configurations
from config import DEBUG


def debug_print(*args, **kwargs):
    """
    Enable debug prints for troubleshooting.
    """
    if DEBUG:
        print("[DEBUG]", *args, **kwargs)


def print_agent_graph(agent, filename="agent_graph.png"):
    """
    Save the LangGraph agent graph as a PNG file.
    
    Args:
        agent_compiled: Compiled LangGraph agent object.
        filename (str): Path to save PNG.
    """

    agent_graph = agent.get_graph()

    # draw_mermaid_png returns PNG bytes
    png_bytes = agent_graph.draw_mermaid_png()
    
    # Write bytes to file
    with open(filename, "wb") as f:
        f.write(png_bytes)
    
    print(f"Agent graph saved to {filename}. Open it with any image viewer.")
