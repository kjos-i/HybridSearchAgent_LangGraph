"""
Hybrid search retriever combining full-text and vector-based search.

This module provides the `HybridRetriever` class that:
    - Integrates keyword/phrase/prefix-based FTS results.
    - Integrates semantic vector search results (Chroma DB).
    - Computes hybrid scores for ranking results across both search types.
    - Returns deduplicated and ranked `SearchResult` objects.
"""

# Local imports
from pydantic_models import SearchResult
       

class HybridRetriever:
    """
    Combines FTS and vector search results into a hybrid ranking.

    Attributes:
        fts (FTSStore): Instance of FTSStore for keyword/phrase/prefix search.
        vector (Any): Vector store instance (Chroma DB).
        fts_weight (float): Weight of FTS scores in hybrid scoring.
        vector_similarity_weight (float): Weight for vector similarity scores.
        vector_mmr_weight (float): Weight for vector MMR scores.
        vector_max_score (float): Maximum expected vector similarity score (for normalization).
    """

    def __init__(
            self, 
            fts_store, 
            vector_store, 
            fts_weight: float, 
            vector_similarity_weight: float, 
            vector_mmr_weight: float, 
            vector_max_score: float
     ):
        """
        Initialize the HybridRetriever.

        Args:
            fts_store: Instance of FTSStore.
            vector_store: Loaded vector DB instance (e.g., Chroma).
            fts_weight (float): Weight for FTS BM25 scores.
            vector_similarity_weight (float): Weight for vector similarity scores.
            vector_mmr_weight (float): Weight for vector MMR scores.
            vector_max_score (float): Maximum expected vector similarity score.
        """

        self.fts = fts_store
        self.vector = vector_store
        self.fts_weight = fts_weight
        self.vector_similarity_weight = vector_similarity_weight
        self.vector_mmr_weight = vector_mmr_weight
        self.vector_max_score = vector_max_score


    def _hybrid_score(self, doc: SearchResult) -> float:
        """
        Compute a unified hybrid score for ranking a single document.

        Scores are normalized and weighted based on the backend:
            - FTS: Inverts BM25 scores (lower BM25 → higher score)
            - Vector similarity: normalized by vector_max_score
            - Vector MMR: normalized by vector_max_score

        Args:
            doc (SearchResult): Document result to score.

        Returns:
            float: Weighted, normalized score used for hybrid ranking.
        """

        raw_score = doc.score or 0
        raw_score = max(raw_score, 0)

        if doc.backend == "fts":
            norm_score = min(1 / (raw_score + 1e-6), 1.0)
            return norm_score * self.fts_weight

        elif doc.backend == "vector_similarity":
            norm_score = min(raw_score / self.vector_max_score, 1.0)
            return norm_score * self.vector_similarity_weight

        elif doc.backend == "vector_mmr":
            norm_score = min(raw_score / self.vector_max_score, 1.0)
            return norm_score * self.vector_mmr_weight

        # fallback for unknown backend
        return 0.0


    def search(
            self,
            query: str,
            k: int = 5, 
            vector_search_method: str = "similarity", 
            use_phrase: bool = False,
            use_prefix: bool = False,
            multi_fts: bool = False,
            fts_multi_weights: dict = None,
            **metadata_filters
    ): 
        """
        Perform a hybrid search combining FTS and vector results.

        Supports:
            - Single-mode FTS (keyword, phrase, or prefix)
            - Multi-mode FTS (keyword + phrase + prefix with weighted score fusion)
            - Vector similarity search or max marginal relevance (MMR) search
            - Deduplication and hybrid scoring

        Args:
            query (str): Search query string.
            k (int): Maximum number of results to return.
            vector_search_method (str): "similarity" or "mmr" for vector search.
            use_phrase (bool): Use exact phrase matching in FTS.
            use_prefix (bool): Use prefix search in FTS.
            multi_fts (bool): If True, perform multi-mode FTS search with score fusion.
            fts_multi_weights (dict, optional): Weights per FTS mode, e.g.,
                {"phrase": 1.0, "keyword": 1.0, "prefix": 1.0}.
            **metadata_filters: Optional key-value filters for FTS or vector results.

        Returns:
            list[SearchResult]: Ranked and deduplicated hybrid search results.
        """

        # --- FTS search ---
        if multi_fts:
            fts_multi_weights = fts_multi_weights or {"phrase": 1.0, "keyword": 1.0, "prefix": 1.0}

            fts_results = self.fts.search_multi(
                query_phrases=[query] if use_phrase else [],
                query_keywords=[query] if not use_phrase else [],
                query_prefixes=[query] if use_prefix else [],
                k=k,
                metadata_filters=metadata_filters,
                fts_multi_weights=fts_multi_weights
            )
        
        else:
            fts_results = self.fts.search_single(
                query, 
                k=k, 
                use_phrase=use_phrase,
                use_prefix=use_prefix,
                metadata_filters=metadata_filters
                )

        # --- Vector search ---
        if vector_search_method == "similarity":
            vector_result_objects = self.vector.similarity_search_with_score(
                query, k=k, 
                filter=metadata_filters if metadata_filters else None
            )
            backend_label = "vector_similarity"
       
        elif vector_search_method == "mmr":
            vector_result_objects = self.vector.max_marginal_relevance_search_with_score(
                query, 
                k=k, 
                filter=metadata_filters if metadata_filters else None
            )
            backend_label = "vector_mmr"
        
        else:
            raise ValueError(f"Unknown vector search method: {vector_search_method}")

        # Convert vector results to SearchResult format
        vector_results = []
        for obj, score in vector_result_objects:       
            vector_results.append(
                SearchResult(
                    page_content = obj.page_content,
                    source = obj.metadata.get("source"),
                    category = obj.metadata.get("category"),
                    chunk_id = obj.metadata.get("chunk_id"),
                    start_char = obj.metadata.get("start_char"),
                    end_char = obj.metadata.get("end_char"),
                    metadata = obj.metadata,
                    score = score,
                    backend = backend_label
                )
            )

         # --- Combine and deduplicate results ---
        combined_results = fts_results + vector_results
        unique_dict = {}   
        
        for item in combined_results:
            item_key = item.page_content 
        
            if item_key not in unique_dict:
                unique_dict[item_key] = item  
            else:
                existing = unique_dict[item_key] 

                if self._hybrid_score(item) > self._hybrid_score(existing):    
                    unique_dict[item_key] = item    
        
        unique_list = list(unique_dict.values()) 
        ranked = sorted(unique_list, key=self._hybrid_score, reverse=True) 
        
        # --- Final ranking ---
        return ranked[:k]
