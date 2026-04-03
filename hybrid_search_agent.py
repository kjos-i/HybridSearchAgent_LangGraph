"""
Hybrid Search Agent using FTS + Vector embeddings.

This module defines a LangGraph-enabled research assistant agent that:
    - Uses a HybridRetriever combining FTS and semantic vector search.
    - Provides a Pydantic-based schema for input arguments, including metadata filters.
    - Exposes a `hybrid_search_tool` for LangGraph agent.
    - Runs an interactive async agent loop with streaming responses and tool tracking.
"""

# Standard library
import asyncio
import sys
import time
from pathlib import Path

# Third-party
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_chroma import Chroma
from chromadb.errors import ChromaError
import sqlite3
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, Field

# Local imports
from fts_search import FTSStore
from hybrid_search import HybridRetriever
from pydantic_models import HybridSearchArgs
from utils import debug_print, print_agent_graph, setup_logger
from config import DEBUG, UPDATES, DRAW, DEBUG_PRINT, PRINT

# Load env from specific path
load_dotenv()


# Initialize logger
logger = setup_logger("hybrid_search_agent")

# --- Database and Store Initialization ---
# Initialize Vector Store (Semantic)
vector_store = Chroma(
    collection_name="document_collection_1", 
    embedding_function=OpenAIEmbeddings(),
    persist_directory="./chroma_db"
)

# Initialize Full-Text Search (Keyword)
fts_store = FTSStore()

# --- Search Strategy Hyperparameters ---
# Preferably, keep FTS_WEIGHT + VECTOR WEIGHT <= 1.0 for better score separation, but not strictly required.
# Ensure that FTS_WEIGHT + VECTOR WEIGHT always equals the same total
FTS_WEIGHT = 0.5
VECTOR_SIMILARITY_WEIGHT = 0.5
VECTOR_MMR_WEIGHT = 0.5
VECTOR_MAX_SCORE = 1.5 # For Default L2 distance Chroma scores
FTS_MAX_SCORE = 20.0 # For SQLite BM25 scores
FTS_MULTI_WEIGHTS = {"phrase": 1.0, "keyword": 1.0, "prefix": 1.0}

# Global retriever instance used by the tool
retriever = HybridRetriever(
    fts_store=fts_store,
    vector_store=vector_store,
    fts_weight=FTS_WEIGHT,
    vector_similarity_weight=VECTOR_SIMILARITY_WEIGHT,
    vector_mmr_weight=VECTOR_MMR_WEIGHT,
    fts_max_score=FTS_MAX_SCORE,
    vector_max_score=VECTOR_MAX_SCORE
)


# --- Hybrid search tool definition ---
@tool(args_schema=HybridSearchArgs)
def hybrid_search_tool(**kwargs):
    """
    Search tool that fuses FTS keyword results with Chroma vector results.
    
    Logic:
        1. Validates flags (disables prefix if phrase is active).
        2. Filters out None values from metadata.
        3. Calls HybridRetriever to perform score-fused retrieval.
    """
    try:
        logger.info(f"hybrid_search_tool called with kwargs: {kwargs}")

        # Extract required query and optional arguments
        query = kwargs.pop("query")
        k = kwargs.pop("k", 3)
        vector_search_method = kwargs.pop("vector_search_method", "similarity")
        use_phrase = kwargs.pop("use_phrase", False)
        use_prefix = kwargs.pop("use_prefix", False)
        multi_fts = kwargs.pop("multi_fts", False)

        # Mutually exclusive search flags guardrail
        if use_phrase and use_prefix:
            logger.debug("Both use_phrase and use_prefix are True. Disabling use_prefix.")
            use_prefix = False

        # Build metadata filter dict from remaining kwargs
        metadata_filters = {key: value for key, value in kwargs.items() if value is not None}
        logger.info(f"Metadata filters applied: {metadata_filters}")
        
        # --- Perform search using HybridRetriever ---
        results = retriever.search(
            query=query, 
            k=k, 
            vector_search_method=vector_search_method, 
            use_phrase=use_phrase, 
            use_prefix=use_prefix, 
            multi_fts=multi_fts,
            fts_multi_weights=FTS_MULTI_WEIGHTS,
            **metadata_filters
        ) 

        
        logger.info(f"Hybrid search returned {len(results)} results")
        for i, result in enumerate(results):
            logger.info(f"{i+1}: backend={result.backend}, score={result.score}, chunk_id={result.chunk_id}")

        # Return structured results as dictionaries
        return {"results": [result.model_dump() for result in results]}
    
    except sqlite3.OperationalError as e:
        # Handle "database is locked" or disk issues
        logger.error(f"SQLite operational error: {e}")
        return "Error: The database is currently busy. Please wait a moment and try one more time."

    except ChromaError as e:
        # Handle collection missing or schema issues
        logger.error(f"ChromaDB error: {e}")
        return f"Error: Retrieval system failure ({type(e).__name__}). Try a simpler keyword search."

    except Exception as e:
            logger.error(f"Unexpected tool error: {str(e)}")
            return f"Error: An unexpected issue occurred during search: {str(e)}."


# --- Agent Configuration ---
tools = [hybrid_search_tool]
model = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True, stream_usage=True)
checkpointer = InMemorySaver()
system_prompt_path = Path(__file__).with_name("system_prompt_hybrid_search.txt")
system_prompt = system_prompt_path.read_text(encoding="utf-8").strip()
logger.info(f"Loaded system prompt from: {system_prompt_path}")

agent = create_agent(model, tools, checkpointer=checkpointer, system_prompt=system_prompt, debug=DEBUG)
if DRAW: print_agent_graph(agent)


# --- Async interactive agent loop ---
async def run_agent():
    """
    Orchestrates the async conversation loop.
    Handles event streaming for real-time AI responses and tool performance tracking.
    """

    config = {"configurable": {"thread_id": "1"}}
    tool_timers = {}
    print("\n\n✧ Agent ready. Type 'exit' or 'quit' to end the chat.")
    
    turn = 0
    while True:
        turn += 1
        user_input = input("\n\nYou: ")
        if user_input.lower() in ['exit', 'quit']: break

        logger.info(f"→ Starting Turn {turn} with input: {user_input}")

        events = agent.astream_events(
            {"messages": [HumanMessage(content=user_input)]}, 
            config=config, version="v2", stream_mode="updates"
        ) 

        async for event in events: 
            kind = event["event"]
            # Unique ID for this specific tool execution
            run_id = event["run_id"] 

            # --- State Updates ---
            if UPDATES and kind == "on_chain_stream" and "updates" in event["data"]:
                for node, delta in event["data"]["updates"].items():
                    # Log the full delta to the FILE, print a summary to TERMINAL
                    logger.debug(f"Node {node} output: {delta}")
                    if node != "__metadata__": 
                        print(f"◈ Node '{node}' updated state.")

            # --- AI Thinking ---
            if kind == "on_chat_model_start":
                print("\n✧ Agent is thinking...\n", end=" ", flush=True)
                sys.stdout.flush()

            # --- Tool Start ---
            if kind == "on_tool_start":
                tool_timers[run_id] = time.time()
                print(f"▶ Tool: {event['name']}")
                logger.debug(f"Tool {event['name']} input: {event['data'].get('input')}")
                sys.stdout.flush()

            # --- Tool End and Latency ---
            if kind == "on_tool_end":
                start_time = tool_timers.pop(run_id, None)
                latency = time.time() - start_time if start_time else 0
                print(f"⏱ {latency:.2f}s {event['name']} finished.")
                logger.info(f"Tool {event['name']} completed in {latency:.2f}s")
                sys.stdout.flush()

             # --- Tool Error ---
            if kind == "on_tool_error":
                start_time = tool_timers.pop(run_id, None)
                error_msg = event.get("data", {}).get("error", "Unknown error")
                print(f"!! ERROR in {event['name']}: {error_msg}")
                logger.error(f"Tool {event['name']} failed: {error_msg}")

            # --- LLM Streaming Content ---
            if kind == "on_chat_model_stream":
                content = event["data"]["chunk"].content
                if content:
                    print(content, end="", flush=True)

            # --- Token Usage ---
            if kind == "on_chat_model_end":
                ai_message = event["data"].get("output")
                
                # 1. Check for modern usage_metadata (Best for v0.3+)
                if ai_message and hasattr(ai_message, "usage_metadata") and ai_message.usage_metadata:
                    usage = ai_message.usage_metadata
                    logger.info(f"Tokens: {usage.get('total_tokens')} "
                                f"[In: {usage.get('input_tokens')}, Out: {usage.get('output_tokens')}]")
    
                # 2. Check for legacy response_metadata (Fallback)
                elif ai_message and hasattr(ai_message, "response_metadata"):
                    usage = ai_message.response_metadata.get("token_usage", {})
                    if usage:
                        logger.info(f"Tokens: {usage.get('total_tokens')}")

            # --- Turn Separator ---
            if kind == "on_chain_end" and event["name"] == "agent":
                print("\n" + "─" * 40)
                logger.info(f"Turn {turn} complete.")


# --- Entry point ---
if __name__ == "__main__":
    asyncio.run(run_agent())
