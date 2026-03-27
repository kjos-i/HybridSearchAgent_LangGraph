"""
Module defining Pydantic models for the Hybrid Search Agent.

This module provides data structures for representing search results
and associated metadata. These models are used in both full-text search (FTS)
and vector-based retrievals in the hybrid agent implementation.
"""

# Standard library
from typing import Optional, Dict, Any

# Third-party
from pydantic import BaseModel


class ChunkMetadata(BaseModel):
    """
    Metadata for a document chunk, used in search results.

    All fields are optional to provide flexibility in what metadata
    is included. This metadata is used for both FTS and vector search results.

    Note:
        If you update, add, or delete fields in this schema for filtering, update the following
        FTSStore components accordingly:
            1. Virtual table schema
            2. `add_documents` insert statement into `docs`
            3. Inserted values in `add_documents`
            4. `search` SELECT statement
            5. Allowed filter values in `search`
    """

    source: Optional[str] = None
    category: Optional[str] = None
    chunk_id: Optional[int] = None
    start_char: Optional[int] = None
    end_char: Optional[int] = None


class SearchResult(BaseModel):
    """
    Represents a single search result from the Hybrid Search Agent.

    Can be returned from:
        - FTSStore.search()
        - HybridRetriever.search() (vector results)

    Attributes:
        page_content (str): The text content of the result chunk.
        source (Optional[str]): Source document or identifier.
        category (Optional[str]): Category or type of the content.
        chunk_id (Optional[int]): Unique ID for the chunk.
        start_char (Optional[int]): Start character offset in the original document.
        end_char (Optional[int]): End character offset in the original document.
        metadata (Dict[str, Any]): Additional metadata associated with the chunk.
        score (Optional[float]): Relevance score (e.g., similarity or ranking).
        backend (str): Retrieval backend used, e.g., "fts", "vector_similarity", "vector_mmr".
    """
    
    page_content: str
    source: Optional[str] = None
    category: Optional[str] = None
    chunk_id: Optional[int] = None
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    metadata: Dict[str, Any] = {}
    score: Optional[float] = None
    backend: str
