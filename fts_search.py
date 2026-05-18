"""
Full-text search engine wrapper for Python (a 'mini search engine').

Features:
    - Creates an FTS (Full-Text Search) database using SQLite FTS5.
    - Stores and indexes document chunks with metadata.
    - Supports single-mode (keyword, phrase, prefix) searches.
    - Supports multi-mode search with combined scores (score fusion).
    - Returns results as SearchResult objects compatible with LangGraph tools.
"""

# Standard library
import json
import re
import sqlite3

# Local imports
from pydantic_models import SearchResult


# ---------------------------------------------------------------------------
# FTS5 schema
# ---------------------------------------------------------------------------

# Columns of the FTS5 virtual table, in declaration order. This single tuple
# drives the CREATE/INSERT/SELECT SQL and the row-to-SearchResult mapping, so
# adding or removing a column is a one-place edit here (plus matching changes
# to ChunkMetadata / SearchResult in pydantic_models.py).
_FTS_COLUMNS = (
    "page_content",
    "source",
    "doc_hash_id",
    "filename",
    "file_type",
    "folder",
    "doc_word_count",
    "doc_char_count",
    "ingested_at",
    "category",
    "language",
    "chunk_id",
    "chunk_hash_id",
    "chunk_start_char",
    "chunk_end_char",
    "metadata",
)

# Columns accepted as metadata-filter keys in search queries. Excludes
# page_content (the FTS search target) and metadata (a JSON blob).
_FILTERABLE_COLUMNS = frozenset(_FTS_COLUMNS) - {"page_content", "metadata"}

_FTS_COLUMNS_CSV = ", ".join(_FTS_COLUMNS)
_FTS_PLACEHOLDERS = ", ".join(["?"] * len(_FTS_COLUMNS))

_SEARCHABLE_COLUMNS = " ".join(col for col in _FTS_COLUMNS if col != "metadata")


class FTSStore:
    """
    Wrapper around a SQLite FTS5 database for full-text search.

    This class provides methods to:
        - Initialize the FTS database.
        - Add document chunks with metadata.
        - Perform keyword, phrase, or prefix searches.
        - Perform multi-mode FTS searches with weighted score fusion.
    """

    def __init__(self, db_path="fts.db", fts_max_score=20.0):
        """
        Initialize the FTSStore instance.

        Args:
            db_path (str): Path to the SQLite database file. Defaults to "fts.db".
        """

        self.conn = sqlite3.connect(db_path, check_same_thread=True)
        self.cur = self.conn.cursor()
        self.fts_max_score = fts_max_score
        self._init()


    def _init(self):
        """
        Initialize the FTS5 virtual table if it does not exist.

        The table stores document chunks and associated metadata to allow
        full-text search queries.
        """

        self.cur.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS docs USING fts5({_FTS_COLUMNS_CSV})"
        )
        self.conn.commit()

  
    def add_documents(self, chunks): 
        """
        Add a list of document chunks to the FTS database.

        Args:
            chunks (list of SearchResult): Document chunks to index.

        Notes:
            - Metadata dictionary is serialized as JSON for storage.
            - If chunk_id is missing, the index in the list is used as fallback.
        """

        batch = []
        for i, chunk in enumerate(chunks):
            meta = chunk.metadata
            # Build a value-by-column dict, then materialize the tuple in
            # _FTS_COLUMNS order. page_content and metadata are sourced
            # explicitly (they are not flat metadata fields), and chunk_id
            # falls back to the enumerate index when the chunk lacks one.
            values = dict(meta)
            values["page_content"] = chunk.page_content
            values["metadata"] = json.dumps(meta)
            values.setdefault("chunk_id", i)
            batch.append(tuple(values.get(col) for col in _FTS_COLUMNS))

        self.cur.executemany(
            f"INSERT INTO docs ({_FTS_COLUMNS_CSV}) VALUES ({_FTS_PLACEHOLDERS})",
            batch,
        )
        self.conn.commit()


    def search_single(self, query, k=3, metadata_filters=None, use_phrase=False, use_prefix=False): 
        """
        Perform a single-mode full-text search over the indexed documents.

        Args:
            query (str): Search query string.
            k (int): Maximum number of results to return. Defaults to 3.
            metadata_filters (dict, optional): Key-value filters on metadata.
                Allowed keys are every column in _FTS_COLUMNS except
                page_content (the search target) and metadata (a JSON blob).
                Unknown keys are silently dropped.
            use_phrase (bool): If True, search for the exact phrase.
            use_prefix (bool): If True, search using prefix matching (query*).

        Returns:
            list[SearchResult]: List of results with BM25 relevance scores.
        """

        # Punctuation characters (apostrophes, commas, question marks, periods, etc.) are
        # special characters in FTS5 query syntax and cause parse errors. The unicode61
        # tokenizer already treats all punctuation as word separators, so stripping them
        # before the query parser sees them produces identical tokens with no change in
        # retrieval meaning.
        sanitized = re.sub(r"[^\w\s]", " ", query)

        # Build FTS query based on search mode
        fts_query = sanitized
        if use_phrase:
            fts_query = f'"{sanitized}"'
        elif use_prefix:
            fts_query = f'{sanitized}*'
        
        # This explicitly shields the keyword search from scanning your 'metadata' JSON column.
        fts_query = f"{{{_SEARCHABLE_COLUMNS}}} : {fts_query}"

        # Base SQL query: every column in _FTS_COLUMNS plus the BM25 score.
        sql = f"""
            SELECT {_FTS_COLUMNS_CSV}, bm25(docs) as score
            FROM docs
            WHERE docs MATCH ?
        """

        params = [fts_query]

        # Apply optional metadata filters (silently dropping unknown keys).
        if metadata_filters:
            for key, value in metadata_filters.items():
                if key in _FILTERABLE_COLUMNS:
                    sql += f" AND {key} = ?"
                    params.append(value)

        # Order by score (ascending because lower BM25 is better using SQLite's bm25 function)
        sql += " ORDER BY score ASC"

        # Limit number of results
        sql += " LIMIT ?"
        params.append(k)

        # Execute query and fetch results - WHERE docs MATCH ? AND source = ? AND category = ? LIMIT ?
        self.cur.execute(sql, params)
        rows = self.cur.fetchall()

        # Convert rows to SearchResult objects.  The row layout is
        # (*_FTS_COLUMNS, bm25_score) — zip the column names back onto the
        # values, decode the metadata JSON blob, and splat into SearchResult.
        results = []
        for row in rows:
            row_dict = dict(zip(_FTS_COLUMNS, row))
            row_dict["metadata"] = json.loads(row_dict["metadata"]) if row_dict["metadata"] else {}
            results.append(
                SearchResult(
                    **row_dict,
                    score = abs(min(row[-1] or 0, 0)) / self.fts_max_score,
                    backend = "fts",
                )
            )
        return results


    def search_multi(
        self,
        query_phrases=None,
        query_keywords=None,
        query_prefixes=None,
        k=3,
        metadata_filters=None,
        fts_multi_weights=None
    ):
        """
        Multi-mode full-text search with weighted score fusion.

        This method allows you to combine multiple query types:
            - Keyword queries
            - Exact phrase queries
            - Prefix queries (query*)

        Scores from each mode are weighted and combined for final ranking.

        Args:
            query_phrases (list[str], optional): Exact phrases to search for.
            query_keywords (list[str], optional): Keywords to search for.
            query_prefixes (list[str], optional): Prefixes for prefix search.
            k (int): Maximum number of results to return.
            metadata_filters (dict, optional): Metadata key-value filters.
            fts_multi_weights (dict, optional): Weight per query type, e.g.,
                {"phrase": 1.0, "keyword": 1.0, "prefix": 1.0}.
                Defaults to equal weighting if None.

        Returns:
            list[SearchResult]: Ranked and deduplicated results based on combined scores.
        """

        query_phrases = query_phrases or []
        query_keywords = query_keywords or []
        query_prefixes = query_prefixes or []
        fts_multi_weights = fts_multi_weights or {"phrase": 1.0, "keyword": 1.0, "prefix": 1.0}

        unique_dict_fts = {}

        def add_results(results, weight):
            """
            Merge results into unique_dict_fts under a weighted score.

            Each result is keyed by (page_content, chunk_id) so the same
            chunk hit by multiple modes (keyword + phrase + prefix) is
            deduplicated and its weighted contributions are summed.
            """
            for result in results:
                result_key = (result.page_content, result.chunk_hash_id)
                match_score = result.score * weight

                if result_key not in unique_dict_fts:
                    unique_dict_fts[result_key] = [result, match_score]
                else:
                    unique_dict_fts[result_key][1] += match_score
        

        # --- Keyword search ---
        for keyword in query_keywords:
            results = self.search_single(
                keyword,
                k=k,
                metadata_filters=metadata_filters
            )
            add_results(results, fts_multi_weights["keyword"])
        
        # --- Phrase search ---
        for phrase in query_phrases:
            results = self.search_single(
                phrase,
                k=k,
                use_phrase=True,
                metadata_filters=metadata_filters
            )
            add_results(results, fts_multi_weights["phrase"])

        # --- Prefix search ---
        for prefix in query_prefixes:
            results = self.search_single(
                prefix,
                k=k,
                use_prefix=True,
                metadata_filters=metadata_filters
            )
            add_results(results, fts_multi_weights["prefix"])

        # --- Final ranking ---        
        # Sort based on the combined score (second item in tuple)
        ranked_pairs = sorted(unique_dict_fts.values(), key=lambda x: x[1], reverse=True)

        # Return just the original objects (the first item in the list)
        return [pair[0] for pair in ranked_pairs[:k]]
    