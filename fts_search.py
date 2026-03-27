"""
Full-text search engine wrapper for Python (a 'mini search engine').

Features:
    - Creates an FTS (Full-Text Search) database using SQLite FTS5.
    - Stores and indexes document chunks with metadata.
    - Supports single-mode (keyword, phrase, prefix) searches.
    - Supports multi-mode search with combined scores (score fusion).
    - Returns results as `SearchResult` objects compatible with LangGraph tools.
"""

# Standard library
import json
import sqlite3

# Local application / project imports
from pydantic_models import SearchResult


class FTSStore:
    """
    Wrapper around a SQLite FTS5 database for full-text search.

    This class provides methods to:
        - Initialize the FTS database.
        - Add document chunks with metadata.
        - Perform keyword, phrase, or prefix searches.
        - Perform multi-mode FTS searches with weighted score fusion.
    """

    def __init__(self, db_path="fts.db"):
        """
        Initialize the FTSStore instance.

        Args:
            db_path (str): Path to the SQLite database file. Defaults to "fts.db".
        """

        self.conn = sqlite3.connect(db_path, check_same_thread=False) 
        self.cur = self.conn.cursor() 
        self._init() 


    def _init(self):
        """
        Initialize the FTS5 virtual table if it does not exist.

        The table stores document chunks and associated metadata to allow
        full-text search queries.
        """

        self.cur.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS docs USING fts5(
                page_content,
                source,
                category,
                chunk_id,
                start_char,
                end_char,
                metadata 
            )
        """)
        self.conn.commit()


    def add_documents(self, chunks): 
        """
        Add a list of document chunks to the FTS database.

        Args:
            chunks (list of SearchResult): Document chunks to index.

        Notes:
            - Metadata dictionary is serialized as JSON for storage.
            - If `chunk_id` is missing, the index in the list is used as fallback.
        """

        batch = []
        for i, chunk in enumerate(chunks):
            meta = chunk.metadata
            batch.append((
                chunk.page_content,
                meta.get("source"),           
                meta.get("category"),
                meta.get("chunk_id", i),
                meta.get("start_char"),
                meta.get("end_char"),
                json.dumps(meta)
            ))

        self.cur.executemany("""
            INSERT INTO docs (
                page_content,
                source,
                category,
                chunk_id,
                start_char,
                end_char,
                metadata
            ) 
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, batch)  
        self.conn.commit() 


    def search_single(self, query, k=3, metadata_filters=None, use_phrase=False, use_prefix=False): 
        """
        Perform a single-mode full-text search over the indexed documents.

        Args:
            query (str): Search query string.
            k (int): Maximum number of results to return. Defaults to 3.
            metadata_filters (dict, optional): Key-value filters on metadata.
                Allowed keys: "source", "category", "chunk_id", "start_char", "end_char".
            use_phrase (bool): If True, search for the exact phrase.
            use_prefix (bool): If True, search using prefix matching (query*).

        Returns:
            list[SearchResult]: List of results with BM25 relevance scores.
        """

        # Build FTS query based on search mode
        fts_query = query 
        if use_phrase:
            fts_query = f'"{query}"' 
        elif use_prefix:
            fts_query = f'{query}*'
        
        # Base SQL query
        sql = """
            SELECT page_content, source, category, chunk_id, start_char, end_char, metadata, bm25(docs) as score
            FROM docs
            WHERE docs MATCH ?
        """                                

        params = [fts_query] 

        # Apply optional metadata filters
        allowed = {"source", "category", "chunk_id", "start_char", "end_char"}    
        if metadata_filters:
            for key, value in metadata_filters.items():
                if key in allowed:
                    sql += f" AND {key} = ?"
                    params.append(value)

        # Limit number of results
        sql += " LIMIT ?"  
        params.append(k)

        # Execute query and fetch results
        self.cur.execute(sql, params)
        rows = self.cur.fetchall()

        # Convert rows to SearchResult objects
        results = []
        for row in rows:
            results.append(
                SearchResult(
                    page_content = row[0], 
                    source = row[1], 
                    category = row[2],
                    chunk_id = row[3],
                    start_char = row[4],
                    end_char = row[5],
                    metadata = json.loads(row[6]) if row[6] else {},
                    score = row[7] or 0,
                    backend = "fts"
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


        # Helper: adds results into unique_dict_fts and applies weight
        def add_results(results, weight):
            for result in results:
                result_key = (result.page_content, result.chunk_id)

                # invert bm25 (lower = better → higher = better)
                fts_rank_score = -result.score * weight

                if result_key not in unique_dict_fts:
                    result.score = fts_rank_score
                    unique_dict_fts[result_key] = result
                else:
                    unique_dict_fts[result_key].score += fts_rank_score
        

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
        ranked = sorted(unique_dict_fts.values(), key=lambda x: x.score, reverse=True)

        return ranked[:k]
        