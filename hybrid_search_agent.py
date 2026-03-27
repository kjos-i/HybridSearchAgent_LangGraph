"""
Hybrid Search Agent using FTS + Vector embeddings.

This module defines a LangGraph-enabled research assistant agent that:
    - Uses a HybridRetriever combining keyword (FTS) and semantic vector search.
    - Provides a Pydantic-based schema for input arguments, including optional metadata filters.
    - Exposes a `hybrid_search_tool` for LangGraph pipelines.
    - Runs an interactive async agent loop with streaming responses.
"""

# Standard library
import asyncio
from typing import Literal

# Third-party
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import Field

# Local application / project imports
from fts_search import FTSStore
from hybrid_search import HybridRetriever
from pydantic_models import ChunkMetadata
from utils import debug_print, print_agent_graph

# Load environment variables, configurations
from config import DEBUG, EVENTS, DRAW
load_dotenv("C:/Users/kjosi/dotenv/.env")


# --- Initialize Vector and FTS stores ---
vector_store = Chroma(
    collection_name="document_collection_1", 
    embedding_function=OpenAIEmbeddings(),
    persist_directory="./chroma_db"
)

fts_store = FTSStore()

# --- Hybrid search configuration weights ---
FTS_WEIGHT = 0.3
VECTOR_SIMILARITY_WEIGHT = 0.7
VECTOR_MMR_WEIGHT = 0.7
VECTOR_MAX_SCORE = 1.0
FTS_MULTI_WEIGHTS = {"phrase": 1.0, "keyword": 1.0, "prefix": 1.0}

# Instantiate the hybrid retriever
retriever = HybridRetriever(
    fts_store=fts_store,
    vector_store=vector_store,
    fts_weight=FTS_WEIGHT,
    vector_similarity_weight=VECTOR_SIMILARITY_WEIGHT,
    vector_mmr_weight=VECTOR_MMR_WEIGHT,
    vector_max_score=VECTOR_MAX_SCORE
)


# --- Input schema for the hybrid search tool ---
class HybridSearchArgs(ChunkMetadata):
    """
    Pydantic schema for hybrid search tool inputs.

    Inherits from ChunkMetadata to allow optional metadata filters
    such as source, category, chunk_id, start_char, and end_char.
    """

    query: str = Field(..., description="Search query.")
    k: int = Field(5, description="Number of results to return.") 
    
    vector_search_method: Literal["similarity", "mmr"] = Field(
        "similarity",
        description="Use 'similarity' for relevance or 'mmr' for diversity."
    )

    use_phrase: bool = Field(False, description="Enable exact phrase matching.")
    use_prefix: bool = Field(False, description="Enable prefix keyword matching (e.g., 'bio*').")

    multi_fts: bool = Field(False, description="Use multi-mode FTS search (keyword + phrase + prefix).")


# --- Hybrid search tool definition ---
@tool(args_schema=HybridSearchArgs)
def hybrid_search_tool(**kwargs):
    """
    Perform a hybrid search combining FTS + Chroma vector database.

    Features:
        - Multi-mode FTS (keyword, phrase, prefix) with weighted score fusion.
        - Vector search: similarity or MMR-based retrieval.
        - Optional metadata filters inherited from ChunkMetadata.
        - Returns structured search results as a list of dictionaries.
    """

    if DEBUG:
        debug_print("hybrid_search_tool called with kwargs:", kwargs)

    # Extract required query and optional arguments
    query = kwargs.pop("query")
    k = kwargs.pop("k", 5)
    vector_search_method = kwargs.pop("vector_search_method", "similarity")
    use_phrase = kwargs.pop("use_phrase", False)
    use_prefix = kwargs.pop("use_prefix", False)
    multi_fts = kwargs.pop("multi_fts", False)

    # Guardrail: cannot use phrase and prefix together
    if use_phrase and use_prefix:
        debug_print("Both use_phrase and use_prefix are True. Disabling use_prefix.")
        use_prefix = False

    # Keep only non-None metadata fields
    metadata_filters = {key: value for key, value in kwargs.items() if value is not None}
    
    if DEBUG:
        debug_print("Metadata filters applied:", metadata_filters)
    
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

    if DEBUG:
        debug_print(f"Hybrid search returned {len(results)} results")
        for i, r in enumerate(results):
            debug_print(f"{i+1}: backend={r.backend}, score={r.score}, chunk_id={r.chunk_id}")

    # Return structured results as dictionaries
    return {"results": [result.model_dump() for result in results]}


# List of tools exposed to the agent
tools = [hybrid_search_tool]

# --- Define LLM model ---
model = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)

# --- LangGraph agent setup --- 
checkpointer = InMemorySaver()
system_prompt = """
You are a research assistant. When answering, first find relevant document chunks using the hybrid search tool. 
Prioritize chunks that exactly match key terms, then consider semantic similarity. 
Return only the top 3 most relevant chunks. 
"""

agent = create_agent(model, tools, checkpointer=checkpointer, system_prompt=system_prompt)
if DEBUG and DRAW:
    print_agent_graph(agent)

# --- Async interactive agent loop ---
async def run_agent():
    """
    Run the interactive hybrid search agent in an async loop.

    Users can type queries and receive streaming responses.
    Type 'exit' or 'quit' to stop the agent.
    """

    config = {"configurable": {"thread_id": "1"}}
    print("Agent ready. Type 'exit' or 'quit' to end the chat.")
    
    turn = 0
    while True:
        turn += 1
        if DEBUG:
            print(f"\n\n🟢 TURN {turn}")

        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            break

        events = agent.astream_events(
            {"messages": [HumanMessage(content=user_input)]}, 
            config=config
        ) 

        async for event in events: 
            if DEBUG and EVENTS:
                 debug_print("Event received:", event["event"])
            if event["event"] == "on_chat_model_stream":
                print(event["data"]["chunk"].content, end="", flush=True)


# --- Entry point ---
if __name__ == "__main__":
    asyncio.run(run_agent())
