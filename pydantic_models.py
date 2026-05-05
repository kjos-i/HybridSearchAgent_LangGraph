"""
Module defining Pydantic models for the Hybrid Search Agent.

This module provides data structures for representing search results
and associated metadata. These models are used in both full-text search (FTS)
and vector-based retrievals in the hybrid agent implementation.
"""

# Standard library
import datetime
from typing import Optional, Dict, Any, Literal

# Third-party
from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    """
    Schema for document chunk metadata used in Hybrid Search (FTS + Vector).

    This model defines the structured data associated with each text chunk. 
    It uses deterministic MD5 hashing for document and chunk identification 
    to support deduplication and traceable retrieval.

    Attributes:
        ingested_at: Unix timestamp for numeric filtering.
        doc_hash_id: MD5 hash of the parent document's content.
        chunk_hash_id: MD5 hash of the specific chunk (doc_id + chunk_id).

    Note:
        When adding or renaming a field, update _FTS_COLUMNS in
        fts_search.py so the field name matches.  The FTS5 CREATE
        schema, the add_documents INSERT, the search SELECT and
        row-to-SearchResult mapping, and the allowed filter keys
        are all derived from that single tuple, so no further manual
        synchronization with FTSStore is required.
    """

    source: str = Field(..., description="Source identifier")
    doc_hash_id: str = Field(..., description="MD5 hash for document based on document content")
    filename: Optional[str] = None 
    file_type: Optional[str] = None
    folder: Optional[str] = None
    doc_word_count: Optional[int] = None
    doc_char_count: Optional[int] = None
    ingested_at: int = Field(..., description="Unix timestamp for numeric filtering")
    category: Optional[str] = None               
    language: Optional[str] = None     
    chunk_id: int = Field(..., description="Unique ID for the chunk within the document")
    chunk_hash_id: str = Field(..., description="MD5 hash for chunk based on doc_id + chunk_id")
    chunk_start_char: Optional[int] = None
    chunk_end_char: Optional[int] = None


# --- Input schema for the hybrid search tool ---
class HybridSearchArgs(BaseModel):
    """
    Validation schema for hybrid search tool inputs.
    
    Attributes:
        query: The user's search string.
        k: Maximum number of documents to retrieve.
        vector_search_method: Logic for vector ranking ('similarity' or 'mmr').
        use_phrase: Whether to force exact phrase matching in FTS.
        use_prefix: Whether to allow wildcard/prefix matching in FTS.
        multi_fts: Whether to run all FTS modes and fuse scores.
    """

    query: str = Field(..., description="Search query.")
    k: int = Field(3, description="Number of results to retrieve.") 
    vector_search_method: Literal["similarity", "mmr"] = Field(
        "similarity",
        description="Use 'similarity' for relevance or 'mmr' for diversity."
    )

    # Optional filters the agent can 'tweak' if mentioned in user prompt
    category: Optional[str] = Field(None, description="Filter by topic (e.g., 'Norway')")
    language: Optional[str] = Field(None, description="Filter by language code (e.g., 'en')")
    filename: Optional[str] = Field(None, description="Filter by filename")
    file_type: Optional[str] = Field(None, description="Filter by file type")
    folder: Optional[str] = Field(None, description="Filter by folder")

    # Flags for search style
    use_phrase: bool = Field(False, description="Set to True for exact phrase matches")
    use_prefix: bool = Field(False, description="Set to True for partial word/wildcard matches")
    multi_fts: bool = Field(False, description="Use multi-mode FTS search (keyword + phrase + prefix).")


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
    doc_hash_id: Optional[str] = None
    filename: Optional[str] = None 
    file_type: Optional[str] = None
    folder: Optional[str] = None
    doc_word_count: Optional[int] = None
    doc_char_count: Optional[int] = None
    ingested_at: Optional[int] = None
    category: Optional[str] = None               
    language: Optional[str] = None
    chunk_id: Optional[int] = None
    chunk_hash_id: Optional[str] = None
    chunk_start_char: Optional[int] = None
    chunk_end_char: Optional[int] = None
    metadata: Dict[str, Any] = {}
    score: Optional[float] = None
    backend: str
