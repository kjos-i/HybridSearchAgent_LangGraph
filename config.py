"""
Global Configuration Settings.

This module acts as a central 'control panel' for the agent's behavior, 
toggling debugging logs, UI feedback, and graph visualization.
"""

# Enable verbose LangGraph system logs
DEBUG = False

# Show real-time node updates during agent execution
UPDATES = False

# Generate and save a PNG of the agent's logic graph on startup
DRAW = False

# Enable the custom 'debug_print' utility for internal tool logic
DEBUG_PRINT = False

# Control overall console output/logging visibility
PRINT = False
